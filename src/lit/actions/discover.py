"""`lit find` — search for papers on a topic and add them, in parallel.

Orchestrator shape: one process builds and de-duplicates the whole candidate
pool *before* any expensive work happens, then workers fan out over it. That
way two agents never read the same paper, and nothing already in the library is
proposed twice.

Everything on the default path is free. The pool is fed from two sources that
cost API calls rather than tokens:

1. **Reference mining** (no LLM, no hallucination risk). Every entry already in
   the library carries a structured reference list pulled from the metadata
   APIs. Works cited by several of your papers but not yet in the library are
   exactly the gaps worth filling.
2. **Indexed search** across Crossref, OpenAlex and arXiv. The same query is
   asked several ways — best keyword match, most-cited, published recently,
   reviews only, arXiv preprints — because one query asked one way returns a
   monoculture. Each facet is a plain HTTP request that returns real works with
   real identifiers, citation counts and abstracts.

`--parallel` adds a third source on top: **scout agents** searching the web,
one per angle. They cost real tokens, so they are opt-in, and their angles are
ordered to lead with what indexed search genuinely cannot do — adjacent fields
that use different vocabulary, and critical or negative-result work, neither of
which any keyword facet will surface.

The pool is then ranked on two things, in this order of weight:

1. **Relevance** — how well the paper answers what was actually asked. This is
   the one judgement here that cannot be computed, so it comes from a single
   LLM call scoring the whole pool at once. Scoring everything in one call
   matters: a facet or a scout works from an *angle* ("most-cited work"), not
   from the query, so its proposals need checking against the query just as
   much as a mined reference does.
2. **Importance** — how much the work matters, computed in code from citation
   velocity and venue rank (the same metrics `quality.assess` uses), plus what
   this library's own papers cite. Search hits arrive carrying these figures;
   anything else gets one free OpenAlex lookup, made *before* ranking rather
   than during the read, so the cut is informed by them.

Papers are then filed from their metadata. Reading them in full is a separate,
opt-in step (`--read` here, or `lit read` later), because a section-by-section
summary of twenty papers is by far the most expensive thing this tool can do
and most of those twenty will not turn out to be worth it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from ..fetch.metadata import PaperMeta, from_openalex, search_arxiv, search_openalex, search_works
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
# rather than letting it swamp the pool the other sources contribute to.
def _mined_intake(limit: int) -> int:
    return max(20, limit * 2)


# Backstop on how many candidates get enriched and scored, for the case where
# the sources come back unusually large.
def _pool_cap(limit: int) -> int:
    return max(40, limit * 4)


# Share of the pool held open for candidates nothing else corroborates. Without
# it, `triage_key` lets co-citation decide who gets *scored at all*, and the
# library's existing shape quietly gatekeeps every new direction: a work that
# no paper you own cites and only one source proposed is exactly what a search
# for a topic you have not covered yet returns.
UNCORROBORATED_QUOTA = 0.3

# How far back "recent" reaches, for the recency facet.
RECENT_YEARS = 2

# The free facets. Each is one HTTP request; together they are the cheap
# equivalent of pointing several scouts at different angles.
SEARCH_FACETS: list[tuple[str, str]] = [
    ("relevance", "best keyword match (Crossref + OpenAlex)"),
    ("foundational", "most-cited work on the topic"),
    ("recent", f"published in the last {RECENT_YEARS} years"),
    ("surveys", "reviews and survey articles"),
    ("preprints", "arXiv preprints, including work too new to be indexed"),
]


@dataclass
class Candidate:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    why: str = ""
    # The publisher's abstract where a source supplied one. Never model-written;
    # it is what gives the ranking call something to judge beyond a bare title.
    abstract: str = ""
    # Which search facets and scout angles proposed this (agreement signal).
    angles: list[str] = field(default_factory=list)
    # How many library papers cite this (co-citation signal).
    cocitations: int = 0
    source: str = "scout"  # search | scout | references | both
    # Filled by `_enrich` from OpenAlex, before ranking.
    citation_count: int | None = None
    # Filled by `_score_relevance`. None means the pass did not run.
    relevance: float | None = None

    @property
    def dedup_key(self) -> str:
        return self.doi or (f"arxiv:{self.arxiv_id}" if self.arxiv_id
                            else slugify(self.title, 120))

    @property
    def corroborated(self) -> bool:
        """Did more than one source independently point at this work?"""
        return self.cocitations > 0 or len(self.angles) > 1

    @property
    def triage_key(self) -> tuple:
        """Cheap pre-ranking order: how many sources corroborate this, then age.

        Only used to decide what to spend enrichment and scoring on when the
        pool is very large. It is not the rank — corroboration says a paper is
        worth *looking at*, not that it is the best answer, which is why a slice
        of the pool is reserved for candidates this ordering would bury (see
        `UNCORROBORATED_QUOTA`).
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
            "found_by_angles": len(self.angles), "angles": list(self.angles),
            "cocitations": self.cocitations,
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
    from_search: int = 0
    from_scouts: int = 0
    dropped_off_topic: int = 0
    # True when the candidates were filed without being read.
    unread: bool = False

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "results": [r.to_dict() for r in self.added],
            "scout_errors": self.scout_errors,
            "unread": self.unread,
            "pool": {
                "total": self.pool_size,
                "from_references": self.from_references,
                "from_search": self.from_search,
                "from_scouts": self.from_scouts,
                "dropped_off_topic": self.dropped_off_topic,
            },
        }


def discover(
    ctx: Ctx,
    query: str,
    *,
    limit: int = 10,
    angles: int = 5,
    per_angle: int | None = None,
    use_references: bool = True,
    use_web: bool = True,
    use_scouts: bool = False,
) -> FindResult:
    """Build the de-duplicated candidate pool. Nothing is read or added here.

    `use_web` covers the free indexed search; `use_scouts` adds the LLM web
    agents on top and is what `--parallel` turns on.
    """
    lib = ctx.library
    entries = lib.entries()
    known_titles = [e.title for e in entries]
    known_slugs = {slugify(t, 120) for t in known_titles}
    known_ids = {e.doi for e in entries if e.doi} | {
        f"arxiv:{e.arxiv_id}" for e in entries if e.arxiv_id
    }

    # `--no-web` means "use only what my own library already cites". Scouts are
    # an online source like any other, so it switches them off too.
    use_scouts = use_scouts and use_web

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
        cur.abstract = cur.abstract or c.abstract
        if cur.citation_count is None:
            cur.citation_count = c.citation_count
        if not cur.authors:
            cur.authors = list(c.authors)
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

    # ---- source 2: indexed search, free ----------------------------------
    if use_web:
        per_facet = max(5, limit)
        ctx.log(
            f"[bold]Searching[/bold] {len(SEARCH_FACETS)} indexes "
            f"(Crossref, OpenAlex, arXiv) — no tokens spent…"
        )
        hits = _search_candidates(ctx, query, per_facet=per_facet)
        result.from_search = len(hits)
        for c in hits:
            absorb(c)
        ctx.vlog(f"  {len(hits)} candidates from indexed search")

    # ---- source 3: web scouts, in parallel (opt-in, costs tokens) --------
    if use_scouts:
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
        considered = _trim_pool(considered, cap)
        solo = sum(1 for c in considered if not c.corroborated)
        ctx.log(f"  pool of {result.pool_size} trimmed to {len(considered)} "
                f"before scoring ({solo} held for uncorroborated work)")

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


def _trim_pool(candidates: list[Candidate], cap: int) -> list[Candidate]:
    """Cut an oversized pool to `cap`, keeping room for uncorroborated work.

    Ordering by corroboration alone means the only candidates ever scored are
    the ones your library already points at or several sources happened to
    agree on. On a topic you have not covered yet that is precisely the wrong
    filter — the paper that opens a new direction is cited by none of your
    entries and was found by one facet. So a share of the pool is reserved for
    those, and only backfilled from the corroborated pile if too few exist.
    """
    if len(candidates) <= cap:
        return candidates
    strong = [c for c in candidates if c.corroborated]
    solo = [c for c in candidates if not c.corroborated]

    reserved = min(len(solo), int(round(cap * UNCORROBORATED_QUOTA)))
    kept = strong[:cap - reserved] + solo[:reserved]
    # Whichever pile ran short, top up from the other so the cap is filled.
    if len(kept) < cap:
        chosen = {id(c) for c in kept}
        kept += [c for c in candidates if id(c) not in chosen][:cap - len(kept)]
    return sorted(kept, key=lambda c: c.triage_key)


def _search_candidates(ctx: Ctx, query: str, *, per_facet: int) -> list[Candidate]:
    """Ask the free indexes the same question several different ways.

    Runs the facets concurrently — they are independent HTTP requests to three
    different services, so waiting on them one at a time is pure latency.
    """
    def run(facet: tuple[str, str]) -> list[Candidate]:
        metas = _search_facet(ctx, query, facet[0], per_facet)
        return [c for c in (_meta_to_candidate(m, facet[0]) for m in metas) if c]

    runs = run_parallel(
        SEARCH_FACETS, run,
        workers=min(len(SEARCH_FACETS), max(2, ctx.workers)),
        on_done=lambda r: ctx.vlog(
            f"  facet {r.item[0]}: {len(r.value or [])} hits" if r.ok
            else f"  facet {r.item[0]} failed: {r.error}"
        ),
    )
    out: list[Candidate] = []
    for r in runs:
        if r.ok and r.value:
            out += r.value
    return out


def _search_facet(ctx: Ctx, query: str, facet: str, limit: int) -> list[PaperMeta]:
    """One facet of the free search. Split out so tests can stub the network."""
    http = ctx.http
    if facet == "foundational":
        return search_openalex(http, query, limit=limit, sort="cited_by_count:desc")
    if facet == "recent":
        return search_openalex(http, query, limit=limit,
                               from_year=date.today().year - RECENT_YEARS)
    if facet == "surveys":
        return search_openalex(http, query, limit=limit, kind="review")
    if facet == "preprints":
        return search_arxiv(http, query, limit=limit)
    return search_works(http, query, limit, ctx.cfg.fetch)


def _meta_to_candidate(meta: PaperMeta, facet: str) -> Candidate | None:
    if not meta or not meta.title or len(meta.title) < 6:
        return None
    return Candidate(
        title=meta.title,
        authors=list(meta.authors),
        year=meta.year,
        venue=meta.venue,
        doi=meta.doi,
        arxiv_id=meta.arxiv_id,
        abstract=meta.abstract or "",
        citation_count=meta.citation_count,
        angles=[facet],
        source="search",
    )


def _enrich(ctx: Ctx, candidates: list[Candidate]) -> None:
    """Fill in citation count and venue from OpenAlex — one lookup each, no LLM.

    This is what lets importance be measured rather than guessed. It runs
    before the cut, unlike the metadata resolution during a read, because a
    citation count discovered after a paper has been dropped is no use.

    Candidates that came from indexed search already carry their citation count
    and venue, so they are skipped — there is nothing left to look up.
    """
    pending = [c for c in candidates if c.citation_count is None or not c.venue]
    if not pending:
        ctx.vlog(f"  all {len(candidates)} candidates already carry their metrics")
        return
    ctx.log(f"[bold]Weighing[/bold] {len(pending)} candidates "
            f"(citations, venue)…")

    def look_up(c: Candidate):
        return _lookup_meta(ctx, c)

    runs = run_parallel(pending, look_up, workers=ctx.workers)
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
        c.abstract = c.abstract or meta.abstract
    ctx.vlog(f"  metadata found for {found}/{len(pending)}")


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
                   force: bool = False, extra_tags: list[str] | None = None,
                   read: bool = False) -> list[AddResult]:
    """Add candidates concurrently, one worker per paper.

    The pool was de-duplicated by the orchestrator, so no two workers here can
    end up handling the same paper.

    With `read=False` (the default) this is metadata only — no full text is
    fetched and no reader agent runs, so a twenty-paper find costs nothing but
    API calls. `read=True` fans the reader agents out as before, on a tighter
    per-paper text budget than `lit add` uses.
    """
    if not candidates:
        return []

    if read:
        ctx.log(
            f"[bold]Reading[/bold] {len(candidates)} papers ({ctx.workers} at a time)…"
        )
    else:
        ctx.log(f"[bold]Filing[/bold] {len(candidates)} papers from their metadata "
                f"(unread — `lit read` to summarize them)…")

    budget = ctx.cfg.llm.find_read_chars if read else None

    def work(c: Candidate) -> AddResult:
        return add_paper(
            ctx, c.title, target=c.to_target(), force=force, extra_tags=extra_tags,
            read=read, max_chars=budget,
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
