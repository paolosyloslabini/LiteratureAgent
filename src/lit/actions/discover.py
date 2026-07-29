"""`lit find` — search for papers on a topic and add them, in parallel.

Orchestrator shape: one process builds and de-duplicates the whole candidate
pool *before* any expensive work happens, then workers fan out over it. That
way two agents never read the same paper, and nothing already in the library is
proposed twice.

The pool is fed from two sources:

1. **Reference mining** (free, no LLM, no hallucination risk). Every entry
   already in the library carries a structured reference list pulled from the
   metadata APIs. Works cited by several of your papers but not yet in the
   library are exactly the gaps worth filling. These are filtered for relevance
   in one cheap LLM call before anything is read.
2. **Scout agents** searching the web in parallel, each given a different angle
   (foundational, recent, surveys/benchmarks, adjacent fields, critical work).
   One agent asked for "10 good papers" returns a monoculture; five agents with
   different briefs return a spread. Each is handed the titles already in the
   library and told not to propose them.

Candidates found by more than one source rank highest — independent agreement
is a real relevance signal. Only then does the read fan-out start, one agent
per paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm import LLMError
from ..models import normalize_arxiv, normalize_doi, slugify
from ..prompts import (
    DISCOVERY_ANGLES,
    SCOUT_SYSTEM,
    discover_prompt,
    filter_candidates_prompt,
)
from ..runner import run_parallel
from .add import AddResult, add_paper
from .context import Ctx, Target

# A work must be cited by at least this many library papers to be mined as a
# candidate, once the library is big enough for co-citation to mean anything.
MIN_COCITATIONS = 2
COCITATION_LIBRARY_THRESHOLD = 3


@dataclass
class Candidate:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    why: str = ""
    # Which scout angles proposed this (agreement signal).
    angles: list[str] = field(default_factory=list)
    # How many library papers cite this (co-citation signal).
    cocitations: int = 0
    source: str = "scout"  # scout | references | both

    @property
    def dedup_key(self) -> str:
        return self.doi or (f"arxiv:{self.arxiv_id}" if self.arxiv_id
                            else slugify(self.title, 120))

    @property
    def score(self) -> tuple:
        """Sort key, best first: agreement, then co-citation, then recency."""
        return (-(len(self.angles) + self.cocitations), -(self.year or 0))

    def to_target(self) -> Target:
        return Target(doi=self.doi, arxiv_id=self.arxiv_id, title=self.title)

    def to_dict(self) -> dict:
        return {
            "title": self.title, "authors": self.authors, "year": self.year,
            "venue": self.venue, "doi": self.doi, "arxiv_id": self.arxiv_id,
            "why": self.why, "source": self.source,
            "found_by_angles": len(self.angles), "cocitations": self.cocitations,
        }


@dataclass
class FindResult:
    candidates: list[Candidate] = field(default_factory=list)
    added: list[AddResult] = field(default_factory=list)
    scout_errors: list[str] = field(default_factory=list)
    pool_size: int = 0
    from_references: int = 0
    from_scouts: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "results": [r.to_dict() for r in self.added],
            "scout_errors": self.scout_errors,
            "pool": {
                "total": self.pool_size,
                "from_references": self.from_references,
                "from_scouts": self.from_scouts,
            },
        }


def discover(
    ctx: Ctx,
    query: str,
    *,
    limit: int = 10,
    angles: int = 3,
    per_angle: int | None = None,
    use_references: bool = True,
    use_web: bool = True,
) -> FindResult:
    """Build the de-duplicated candidate pool. Nothing is read or added here."""
    lib = ctx.library
    entries = lib.entries()
    known_titles = [e.title for e in entries]
    known_slugs = {slugify(t, 120) for t in known_titles}
    known_ids = {e.doi for e in entries if e.doi} | {
        f"arxiv:{e.arxiv_id}" for e in entries if e.arxiv_id
    }

    result = FindResult()
    pool: dict[str, Candidate] = {}

    def absorb(c: Candidate) -> None:
        """Merge a candidate into the pool, combining signals on collision."""
        if c.dedup_key in known_ids or slugify(c.title, 120) in known_slugs:
            return
        cur = pool.get(c.dedup_key)
        if cur is None:
            pool[c.dedup_key] = c
            return
        cur.angles += [a for a in c.angles if a not in cur.angles]
        cur.cocitations = max(cur.cocitations, c.cocitations)
        cur.doi = cur.doi or c.doi
        cur.arxiv_id = cur.arxiv_id or c.arxiv_id
        cur.year = cur.year or c.year
        cur.venue = cur.venue or c.venue
        cur.why = cur.why or c.why
        if cur.source != c.source:
            cur.source = "both"

    # ---- source 1: references of what we already have --------------------
    mined: list[Candidate] = []
    if use_references and entries:
        mined = mine_references(entries, known_slugs, known_ids)
        result.from_references = len(mined)
        if mined:
            ctx.log(
                f"[bold]Mining[/bold] references of {len(entries)} existing "
                f"entries → {len(mined)} uncited-by-you candidates"
            )
            mined = _filter_mined(ctx, query, mined, limit)
            for c in mined:
                absorb(c)

    # ---- source 2: web scouts, in parallel -------------------------------
    if use_web:
        angle_list = DISCOVERY_ANGLES[:max(1, min(angles, len(DISCOVERY_ANGLES)))]
        per_angle = per_angle or max(3, (limit * 2) // len(angle_list))
        # Scouts are told about the library *and* about what reference mining
        # already surfaced, so they spend their searches on genuinely new ground.
        exclude = known_titles + [c.title for c in pool.values()]

        def scout(angle: str) -> list[Candidate]:
            data = ctx.llm.json(
                discover_prompt(
                    query=query, scope=lib.settings.scope, angle=angle,
                    limit=per_angle, exclude_titles=exclude,
                ),
                system=SCOUT_SYSTEM,
                role="scout",
                tools=["WebSearch", "WebFetch"],
                max_turns=24,
                required=("papers",),
            )
            return [c for c in (_to_candidate(p, angle) for p in (data.get("papers") or [])) if c]

        ctx.log(
            f"[bold]Scouting[/bold] {len(angle_list)} angles in parallel "
            f"(up to {per_angle} papers each)…"
        )
        runs = run_parallel(
            angle_list, scout,
            workers=min(ctx.workers, len(angle_list)),
            on_done=lambda r: ctx.vlog(
                f"  angle done: {len(r.value or [])} candidates"
                if r.ok else f"  angle failed: {r.error}"
            ),
        )
        for r in runs:
            if not r.ok:
                result.scout_errors.append(str(r.error))
                continue
            result.from_scouts += len(r.value or [])
            for c in (r.value or []):
                absorb(c)

    ranked = sorted(pool.values(), key=lambda c: c.score)
    result.pool_size = len(ranked)
    result.candidates = ranked[:limit]
    return result


def mine_references(entries, known_slugs: set[str],
                    known_ids: set[str]) -> list[Candidate]:
    """Harvest works cited by library entries but not yet in the library.

    Reference lists come from Crossref/OpenAlex/S2, not from a model, so these
    candidates are real works with real identifiers.
    """
    counts: dict[str, int] = {}
    protos: dict[str, Candidate] = {}

    for e in entries:
        # One vote per citing paper, even if it lists a work twice.
        for k in {_ref_key(r) for r in e.references if r.title}:
            counts[k] = counts.get(k, 0) + 1
        for r in e.references:
            if not r.title:
                continue
            k = _ref_key(r)
            if k not in protos:
                protos[k] = Candidate(
                    title=r.title,
                    authors=list(r.authors),
                    year=r.year,
                    doi=r.doi,
                    arxiv_id=r.arxiv_id,
                    source="references",
                )

    threshold = MIN_COCITATIONS if len(entries) >= COCITATION_LIBRARY_THRESHOLD else 1
    out: list[Candidate] = []
    for k, n in counts.items():
        if n < threshold:
            continue
        c = protos[k]
        if c.dedup_key in known_ids or slugify(c.title, 120) in known_slugs:
            continue
        c.cocitations = n
        c.why = f"cited by {n} paper(s) already in this library"
        out.append(c)

    out.sort(key=lambda c: (-c.cocitations, -(c.year or 0)))
    return out


def _filter_mined(ctx: Ctx, query: str, mined: list[Candidate],
                  limit: int) -> list[Candidate]:
    """One cheap LLM call to drop tangential references before we read anything."""
    shortlist = mined[: max(40, limit * 4)]
    if not shortlist or not ctx.llm.available:
        return shortlist[:limit]
    try:
        data = ctx.llm.json(
            filter_candidates_prompt(
                query=query, scope=ctx.library.settings.scope, candidates=shortlist
            ),
            system=SCOUT_SYSTEM,
            role="filter",
            required=("keep",),
        )
    except LLMError as exc:
        ctx.vlog(f"candidate filter failed, keeping co-citation order: {exc}")
        return shortlist[:limit]

    kept: list[Candidate] = []
    for k in (data.get("keep") or []):
        try:
            i = int(k.get("index"))
            rel = float(k.get("relevance", 0.0))
        except (TypeError, ValueError):
            continue
        if not (0 <= i < len(shortlist)) or rel < 0.4:
            continue
        c = shortlist[i]
        why = str(k.get("why") or "").strip()
        if why:
            c.why = f"{why} ({c.why})"
        kept.append(c)
    ctx.vlog(f"  reference filter kept {len(kept)}/{len(shortlist)}")
    return kept


def add_candidates(ctx: Ctx, candidates: list[Candidate], *,
                   force: bool = False, extra_tags: list[str] | None = None
                   ) -> list[AddResult]:
    """Read and add candidates concurrently, one agent per paper.

    The pool was de-duplicated by the orchestrator, so no two workers here can
    end up reading the same paper.
    """
    if not candidates:
        return []

    ctx.log(
        f"[bold]Reading[/bold] {len(candidates)} papers ({ctx.workers} at a time)…"
    )

    def work(c: Candidate) -> AddResult:
        return add_paper(
            ctx, c.title, target=c.to_target(), force=force, extra_tags=extra_tags
        )

    def report(r) -> None:
        if not r.ok:
            item: Candidate = r.item  # type: ignore[assignment]
            ctx.log(f"  [red]![/red] {item.title[:70]}: {r.error}")
            return
        res: AddResult = r.value
        icon = {
            "added": "[green]+[/green]", "updated": "[green]~[/green]",
            "duplicate": "[dim]=[/dim]", "rejected": "[yellow]-[/yellow]",
            "not_found": "[red]?[/red]", "error": "[red]![/red]",
        }.get(res.status, " ")
        ctx.log(f"  {icon} {res.message}")

    runs = run_parallel(candidates, work, workers=ctx.workers, on_done=report)

    out: list[AddResult] = []
    for r in runs:
        if r.ok and r.value is not None:
            out.append(r.value)
        elif r.error is not None:
            item: Candidate = r.item  # type: ignore[assignment]
            msg = (str(r.error) if isinstance(r.error, LLMError)
                   else f"{type(r.error).__name__}: {r.error}")
            out.append(AddResult("error", f"{item.title[:70]}: {msg}"))
    return out


def _ref_key(r) -> str:
    return r.doi or (f"arxiv:{r.arxiv_id}" if r.arxiv_id else slugify(r.title, 120))


def _to_candidate(p: dict, angle: str) -> Candidate | None:
    title = str(p.get("title") or "").strip()
    if not title or len(title) < 6:
        return None
    year = p.get("year")
    try:
        year = int(year) if year not in (None, "", "null") else None
    except (TypeError, ValueError):
        year = None
    authors = p.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    return Candidate(
        title=title,
        authors=[str(a) for a in authors if a],
        year=year,
        venue=(str(p["venue"]).strip() or None) if p.get("venue") else None,
        doi=normalize_doi(p.get("doi")),
        arxiv_id=normalize_arxiv(p.get("arxiv_id")),
        why=str(p.get("why") or "").strip(),
        angles=[angle],
        source="scout",
    )
