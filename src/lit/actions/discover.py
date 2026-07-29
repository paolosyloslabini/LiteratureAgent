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

The pool is then ranked on two things, in this order of weight:

1. **Relevance** — how well the paper answers what was actually asked. This is
   the one judgement here that cannot be computed, so it comes from a single
   LLM call scoring the whole pool at once. Scoring everything in one call
   matters: a scout works from an *angle* ("foundational work", "recent work"),
   not from the query, so its proposals need checking against the query just as
   much as a mined reference does.
2. **Importance** — how much the work matters, computed in code from citation
   velocity and venue rank (the same metrics `quality.assess` uses), plus what
   this library's own papers cite. The figures come from one free OpenAlex
   lookup per candidate, made *before* ranking rather than during the read, so
   the cut is informed by them.

Only then does the read fan-out start, one agent per paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from ..fetch.metadata import from_openalex
from ..llm import LLMError
from ..models import normalize_arxiv, normalize_doi, slugify
from ..prompts import (
    DISCOVERY_ANGLES,
    SCOUT_SYSTEM,
    discover_prompt,
    rank_candidates_prompt,
)
from ..quality import citations_per_year, venue_level
from ..runner import run_parallel
from .add import AddResult, add_paper
from .context import Ctx, Target

# A work must be cited by at least this many library papers to be mined as a
# candidate, once the library is big enough for co-citation to mean anything.
MIN_COCITATIONS = 2
COCITATION_LIBRARY_THRESHOLD = 3

# How the final rank is composed. Relevance dominates: a library full of
# important papers that miss the question is a worse answer than one of
# on-topic papers of mixed stature.
RELEVANCE_WEIGHT = 0.65
IMPORTANCE_WEIGHT = 0.35
# Below this, a paper is dropped outright rather than merely ranked low, so an
# off-topic classic cannot occupy a slot on the strength of its citations.
MIN_RELEVANCE = 0.35
# Used when the relevance pass could not run (no LLM, or the call failed), so
# ranking falls back to importance order instead of discarding everything.
ASSUMED_RELEVANCE = 0.5

# Importance: what the world thinks (citations, venue) vs. what this library's
# own papers and the scouts converged on.
WORLD_WEIGHT = 0.7
LOCAL_WEIGHT = 0.3
# Assumed world standing for a paper no metadata source knows about.
ASSUMED_WORLD = 0.35
_VENUE_POINTS = {"A*": 1.0, "A": 0.8, "B": 0.55, "C": 0.3}
# Citations per year that earns full marks. Landmark papers clear this easily;
# the scale in between is logarithmic because citation counts are.
VELOCITY_FULL_MARKS = 100.0
# A paper this new has not had time to accumulate citations, so it is judged on
# venue alone — the same allowance `quality.assess` makes.
RECENCY_ALLOWANCE_YEARS = 2

# Reference mining can surface hundreds of works. Take the best-co-cited slice
# rather than letting it swamp the pool the scouts contribute to.
def _mined_intake(limit: int) -> int:
    return max(20, limit * 2)


# Backstop on how many candidates get enriched and scored, for the case where
# both sources come back unusually large.
def _pool_cap(limit: int) -> int:
    return max(40, limit * 4)


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
    # Filled by `_enrich` from OpenAlex, before ranking.
    citation_count: int | None = None
    # Filled by `_score_relevance`. None means the pass did not run.
    relevance: float | None = None

    @property
    def dedup_key(self) -> str:
        return self.doi or (f"arxiv:{self.arxiv_id}" if self.arxiv_id
                            else slugify(self.title, 120))

    @property
    def triage_key(self) -> tuple:
        """Cheap pre-ranking order: how many sources corroborate this, then age.

        Only used to decide what to spend enrichment and scoring on when the
        pool is very large. It is not the rank — corroboration says a paper is
        worth *looking at*, not that it is the best answer.
        """
        return (-(len(self.angles) + self.cocitations), -(self.year or 0))

    @property
    def velocity_score(self) -> float | None:
        """Citations per year, log-scaled to 0..1. None when unknown."""
        cpy = citations_per_year(self.citation_count, self.year)
        if cpy is None:
            return None
        return min(1.0, math.log10(1 + max(0.0, cpy)) / math.log10(1 + VELOCITY_FULL_MARKS))

    @property
    def venue_score(self) -> float | None:
        """Venue standing on 0..1, from the curated table. None when unknown."""
        hit = venue_level(self.venue)
        return _VENUE_POINTS.get(hit[0], 0.4) if hit else None

    @property
    def world_score(self) -> float | None:
        """How the field at large regards this work. None when nothing is known.

        Recent papers are judged on venue alone: two years is not enough time
        to accumulate citations, and scoring them on velocity would bury every
        new paper under the classics.
        """
        venue = self.venue_score
        recent = self.year is not None and (date.today().year - self.year) <= \
            RECENCY_ALLOWANCE_YEARS
        velocity = None if recent else self.velocity_score
        if velocity is None:
            return venue
        if venue is None:
            return velocity
        return 0.6 * velocity + 0.4 * venue

    @property
    def local_score(self) -> float:
        """What this library and its scouts say, independent of the wider field."""
        cited = min(1.0, self.cocitations / 3.0)
        # One angle proposing a paper is the baseline, not a signal; agreement
        # between independent briefs is.
        agreed = min(1.0, max(0, len(self.angles) - 1) / 2.0)
        return 0.6 * cited + 0.4 * agreed

    @property
    def importance(self) -> float:
        """0..1, from measured metrics rather than judgement.

        A paper is never punished merely for having no metadata — an unknown
        work is assumed middling rather than worthless, and lets relevance
        decide.
        """
        world = self.world_score
        return min(1.0, WORLD_WEIGHT * (ASSUMED_WORLD if world is None else world)
                   + LOCAL_WEIGHT * self.local_score)

    @property
    def rank_score(self) -> float:
        """The number the top-N cut is made on."""
        rel = ASSUMED_RELEVANCE if self.relevance is None else self.relevance
        return RELEVANCE_WEIGHT * rel + IMPORTANCE_WEIGHT * self.importance

    def to_target(self) -> Target:
        return Target(doi=self.doi, arxiv_id=self.arxiv_id, title=self.title)

    def to_dict(self) -> dict:
        return {
            "title": self.title, "authors": self.authors, "year": self.year,
            "venue": self.venue, "doi": self.doi, "arxiv_id": self.arxiv_id,
            "why": self.why, "source": self.source,
            "found_by_angles": len(self.angles), "cocitations": self.cocitations,
            "citation_count": self.citation_count,
            "relevance": self.relevance,
            "importance": round(self.importance, 3),
            "score": round(self.rank_score, 3),
        }


@dataclass
class FindResult:
    candidates: list[Candidate] = field(default_factory=list)
    added: list[AddResult] = field(default_factory=list)
    scout_errors: list[str] = field(default_factory=list)
    pool_size: int = 0
    from_references: int = 0
    from_scouts: int = 0
    dropped_off_topic: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "results": [r.to_dict() for r in self.added],
            "scout_errors": self.scout_errors,
            "pool": {
                "total": self.pool_size,
                "from_references": self.from_references,
                "from_scouts": self.from_scouts,
                "dropped_off_topic": self.dropped_off_topic,
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
            intake = _mined_intake(limit)
            if len(mined) > intake:
                ctx.log(f"  taking the {intake} most co-cited of {len(mined)}")
                mined = mined[:intake]
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

    # ---- rank: relevance first, importance second -------------------------
    considered = sorted(pool.values(), key=lambda c: c.triage_key)
    result.pool_size = len(considered)
    cap = _pool_cap(limit)
    if len(considered) > cap:
        ctx.log(f"  pool of {len(considered)} trimmed to the {cap} "
                f"best-corroborated before scoring")
        considered = considered[:cap]

    if considered:
        _enrich(ctx, considered)
        _score_relevance(ctx, query, considered)

    kept = [c for c in considered
            if c.relevance is None or c.relevance >= MIN_RELEVANCE]
    result.dropped_off_topic = len(considered) - len(kept)
    if result.dropped_off_topic:
        ctx.log(f"  {result.dropped_off_topic} dropped as off-topic "
                f"(relevance below {MIN_RELEVANCE})")

    kept.sort(key=lambda c: (-c.rank_score, -(c.year or 0)))
    result.candidates = kept[:limit]
    return result


def _enrich(ctx: Ctx, candidates: list[Candidate]) -> None:
    """Fill in citation count and venue from OpenAlex — one lookup each, no LLM.

    This is what lets importance be measured rather than guessed. It runs
    before the cut, unlike the metadata resolution during a read, because a
    citation count discovered after a paper has been dropped is no use.
    """
    ctx.log(f"[bold]Weighing[/bold] {len(candidates)} candidates "
            f"(citations, venue)…")

    def look_up(c: Candidate):
        return _lookup_meta(ctx, c)

    runs = run_parallel(candidates, look_up, workers=ctx.workers)
    found = 0
    for r in runs:
        c: Candidate = r.item  # type: ignore[assignment]
        if not r.ok:
            ctx.vlog(f"  lookup failed for {c.title[:60]}: {r.error}")
            continue
        meta = r.value
        if meta is None:
            continue
        found += 1
        if meta.citation_count is not None:
            c.citation_count = meta.citation_count
        c.venue = c.venue or meta.venue
        c.year = c.year or meta.year
        # A record found by title may be a mirror or re-registration of the
        # work rather than the work itself (see `PaperMeta.supplementary`).
        # Its citations and venue still describe the right paper and are only
        # used for scoring, but its identifiers would be handed to the reader
        # as a Target — so those are taken only from an identifier lookup.
        if not meta.title_matched:
            c.doi = c.doi or meta.doi
            c.arxiv_id = c.arxiv_id or meta.arxiv_id
    ctx.vlog(f"  metadata found for {found}/{len(candidates)}")


def _lookup_meta(ctx: Ctx, c: Candidate):
    """One OpenAlex hit for a candidate. Split out so tests can stub it."""
    return from_openalex(
        ctx.http,
        doi=c.doi,
        title=None if c.doi else c.title,
        with_references=False,
    )


def _score_relevance(ctx: Ctx, query: str, candidates: list[Candidate]) -> None:
    """One LLM call scoring the whole pool against the query.

    Leaves `relevance` as None on every candidate if it cannot run, which the
    caller reads as "fall back to importance order" rather than "drop them all".
    """
    if not ctx.llm.available:
        ctx.vlog("no LLM available, ranking on importance alone")
        return
    ctx.log(f"[bold]Scoring[/bold] {len(candidates)} candidates against the query…")
    try:
        data = ctx.llm.json(
            rank_candidates_prompt(
                query=query, scope=ctx.library.settings.scope, candidates=candidates
            ),
            system=SCOUT_SYSTEM,
            role="rank",
            required=("scores",),
        )
    except LLMError as exc:
        ctx.vlog(f"relevance scoring failed, ranking on importance alone: {exc}")
        return

    scored = 0
    for s in (data.get("scores") or []):
        try:
            i = int(s.get("index"))
            rel = float(s.get("relevance"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not (0 <= i < len(candidates)):
            continue
        c = candidates[i]
        c.relevance = min(1.0, max(0.0, rel))
        scored += 1
        why = str(s.get("why") or "").strip()
        if why:
            c.why = why
    ctx.vlog(f"  scored {scored}/{len(candidates)}")


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
