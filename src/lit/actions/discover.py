"""`lit find` — search for papers on a topic and add them, in parallel.

Orchestrator shape: one process builds and de-duplicates the whole candidate
pool *before* any expensive work happens, then workers fan out over it. That
way two agents never read the same paper, and nothing already in the library is
proposed twice.

The request is read once, by the cheapest model available, into the parameters
an index can actually be given: the topic in the field's own vocabulary, a
couple of alternate phrasings, a year window, a document type (see
`SearchPlan`). "Recent surveys on RAG, nothing before 2022" is a sentence a
person understands and a keyword index does not — handed it whole, Crossref
matches `recent` and `nothing`. Everything downstream is asked what the plan
says, and the window holds for mined references and scouts too, not just for
the sources that had a filter field for it.

Apart from that one call, everything on the default path is free. The pool is
fed from two sources that cost API calls rather than tokens:

1. **Reference mining** (no LLM, no hallucination risk). Every entry already in
   the library carries a structured reference list pulled from the metadata
   APIs. Works cited by several of your papers but not yet in the library are
   exactly the gaps worth filling.
2. **Indexed search** across Crossref, Semantic Scholar and arXiv. The same
   query is asked several ways — best keyword match, most-cited, published
   recently, reviews only, arXiv preprints — because one query asked one way
   returns a monoculture. Each facet is a plain HTTP request that returns real
   works with real identifiers, citation counts and abstracts.

   When an index does not answer, that is *reported*, not absorbed. A search
   source that fails returns an empty list, which is indistinguishable from a
   source that had nothing to say — so a dead index used to cost a run its
   most-cited facet and its citation figures without a word to the user, who
   then saw a confident table of whatever the surviving sources returned. Every
   phase below counts what failed to reach an index and says so.

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
   anything else is filled in by a single batched Semantic Scholar lookup, made
   *before* ranking rather than during the read, so the cut is informed by them.

Papers are then filed from their metadata. Reading them in full is a separate,
opt-in step (`--read` here, or `lit read` later), because a section-by-section
summary of twenty papers is by far the most expensive thing this tool can do
and most of those twenty will not turn out to be worth it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from itertools import zip_longest

from ..fetch.metadata import (
    S2_REVIEW_TYPE,
    S2_SORT_CITED,
    CROSSREF_SORT_CITED,
    PaperMeta,
    batch_s2,
    s2_by_title,
    search_arxiv,
    search_crossref,
    search_semantic_scholar,
)
from ..llm import LLMError
from ..models import normalize_arxiv, normalize_doi, slugify
from ..prompts import (
    DISCOVERY_ANGLES,
    PLANNER_SYSTEM,
    SCOUT_SYSTEM,
    discover_prompt,
    plan_query_prompt,
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
#
# `foundational` is the one that answers "what are the landmark papers here",
# and it is asked of both citation-aware indexes rather than one. Crossref
# counts only citations from DOI-registered work, which in CS undercounts a
# preprint badly; Semantic Scholar counts arXiv properly but shares an
# unauthenticated rate-limit pool with the world. Either can come back empty, so
# neither is trusted to carry the facet alone.
SEARCH_FACETS: list[tuple[str, str]] = [
    ("relevance", "best keyword match (Crossref + Semantic Scholar)"),
    ("foundational", "most-cited work on the topic"),
    ("recent", f"published in the last {RECENT_YEARS} years"),
    ("surveys", "reviews and survey articles"),
    ("preprints", "arXiv preprints, including work too new to be indexed"),
]

# The indexes a run talks to, for reporting which of them went quiet.
INDEX_NAMES = "Crossref, Semantic Scholar, arXiv"


# ---- query planning -------------------------------------------------------
# How many index queries one plan may carry. Each is another request to each
# index; past a handful, paraphrases of the same topic return the same works.
MAX_PLAN_QUERIES = 3
# Tags the plan may hang on what it finds.
MAX_PLAN_TAGS = 6
# Document types a request may ask for. Each index has its own vocabulary and
# its own gaps — Crossref can filter all but `review`, Semantic Scholar only
# `review` — so this is the neutral spelling the plan uses and each search
# function translates what it can (see `_CROSSREF_KINDS`).
PLAN_KINDS = ["review", "article", "preprint", "book", "book-chapter",
              "dataset", "dissertation", "report"]
# Words people use for those types, mapped onto them.
_KIND_ALIASES = {"reviews": "review", "survey": "review", "surveys": "review",
                 "review article": "review", "journal article": "article",
                 "articles": "article", "preprints": "preprint",
                 "books": "book", "datasets": "dataset", "thesis": "dissertation",
                 "reports": "report"}
# A year outside this is a parsing accident, not a date the person meant.
EARLIEST_PLAUSIBLE_YEAR = 1600


@dataclass
class SearchPlan:
    """What a plain-language request turns out to be asking the indexes for.

    A query typed at a person — "recent surveys on RAG, nothing before 2022" —
    is not a query an index can answer. Crossref and Semantic Scholar have fields for
    two thirds of that sentence, but handed it whole they match `recent` and
    `nothing` as keywords. This is that sentence taken apart: the topic in the
    field's own vocabulary, a date window, a document type.

    `planned` is False when the parse did not happen (no LLM, a failed call,
    `--no-plan`), in which case the string is searched exactly as typed — the
    behaviour this replaced.
    """

    query: str  # what the user typed, kept verbatim for ranking and display
    topic: str = ""
    # Index queries, most direct first. Alternates re-phrase the same topic in
    # a neighbouring field's vocabulary.
    queries: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    kind: str | None = None
    intent: str = ""
    planned: bool = False

    @classmethod
    def unplanned(cls, query: str) -> "SearchPlan":
        """The query as typed, which is what every index gets asked."""
        return cls(query=query, topic=query, queries=[query])

    @property
    def primary(self) -> str:
        return self.queries[0] if self.queries else self.query

    @property
    def alternates(self) -> list[str]:
        return self.queries[1:]

    @property
    def has_window(self) -> bool:
        return bool(self.year_from or self.year_to)

    def allows(self, year: int | None) -> bool:
        """Is a paper from this year inside the window the request asked for?

        A candidate whose year nothing knows passes. The window rules out what
        the request excluded; it is not a demand that every paper prove a date.
        """
        if year is None:
            return True
        if self.year_from and year < self.year_from:
            return False
        return not (self.year_to and year > self.year_to)

    def constraints(self) -> tuple[str, ...]:
        """The hard limits, phrased for the agents that also have to honour them."""
        out = []
        if self.year_from and self.year_to:
            out.append(f"published between {self.year_from} and {self.year_to}")
        elif self.year_from:
            out.append(f"published in {self.year_from} or later")
        elif self.year_to:
            out.append(f"published in {self.year_to} or earlier")
        if self.kind:
            out.append(f"{self.kind} documents only")
        return tuple(out)

    def summary_lines(self) -> list[str]:
        """What was understood, for the user to see before anything is spent."""
        lines = [f"  topic: {self.topic}"] if self.topic else []
        if self.queries:
            lines.append("  asking: " + " · ".join(f"{q!r}" for q in self.queries))
        window = ""
        if self.year_from and self.year_to:
            window = f"{self.year_from}–{self.year_to}"
        elif self.year_from:
            window = f"{self.year_from}–"
        elif self.year_to:
            window = f"–{self.year_to}"
        extras = [f"years: {window}" if window else "",
                  f"type: {self.kind}" if self.kind else "",
                  f"tags: {', '.join(self.tags)}" if self.tags else ""]
        if any(extras):
            lines.append("  " + "   ".join(e for e in extras if e))
        return lines

    def to_dict(self) -> dict:
        return {
            "query": self.query, "planned": self.planned, "topic": self.topic,
            "queries": list(self.queries), "tags": list(self.tags),
            "year_from": self.year_from, "year_to": self.year_to,
            "kind": self.kind, "intent": self.intent,
        }


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
    # Filled by `_enrich` from Semantic Scholar, before ranking.
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
    plan: SearchPlan | None = None
    pool_size: int = 0
    from_references: int = 0
    from_search: int = 0
    from_scouts: int = 0
    dropped_off_topic: int = 0
    dropped_out_of_window: int = 0
    # True when the candidates were filed without being read.
    unread: bool = False
    # Requests that never reached an index — a timeout, a refused connection, or
    # a rate limit that outlasted the retries. Distinct from a search that ran
    # and found nothing, and the difference matters: the first means the pool is
    # missing work that exists, the second means there is none.
    search_failures: int = 0
    weigh_failures: int = 0
    # Candidates that reached the ranking with no citation count at all, so
    # their importance was assumed rather than measured.
    unweighed: int = 0

    @property
    def degraded(self) -> bool:
        """Did some part of the search fail to reach the index it needed?"""
        return bool(self.search_failures or self.weigh_failures)

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "results": [r.to_dict() for r in self.added],
            "scout_errors": self.scout_errors,
            "plan": self.plan.to_dict() if self.plan else None,
            "unread": self.unread,
            "degraded": self.degraded,
            "index_failures": {
                "search": self.search_failures,
                "weigh": self.weigh_failures,
                "ranked_without_citations": self.unweighed,
            },
            "pool": {
                "total": self.pool_size,
                "from_references": self.from_references,
                "from_search": self.from_search,
                "from_scouts": self.from_scouts,
                "dropped_off_topic": self.dropped_off_topic,
                "dropped_out_of_window": self.dropped_out_of_window,
            },
        }


def plan_query(ctx: Ctx, query: str) -> SearchPlan:
    """Read the request into index parameters with one cheap call.

    Falls back to the query as typed on any failure. A search that runs on a
    slightly worse query is a far better outcome than one that does not run.
    """
    if not ctx.llm.available:
        ctx.vlog("no LLM available, searching the query exactly as typed")
        return SearchPlan.unplanned(query)
    try:
        data = ctx.llm.json(
            plan_query_prompt(
                query=query, scope=ctx.library.settings.scope,
                this_year=date.today().year, max_queries=MAX_PLAN_QUERIES,
                kinds=PLAN_KINDS,
            ),
            system=PLANNER_SYSTEM,
            role="plan",
            required=("queries",),
        )
    except LLMError as exc:
        ctx.vlog(f"query planning failed, searching as typed: {exc}")
        return SearchPlan.unplanned(query)
    return read_plan(query, data)


def read_plan(query: str, data: dict) -> SearchPlan:
    """Turn the planner's JSON into a plan, discarding anything implausible.

    Kept separate from the call because this is where a cheap model's output
    gets held at arm's length: a year it invented, a document type that is not
    a type, an empty query list. Anything unusable falls back to the query as
    typed rather than searching on nonsense.
    """
    queries = _clean_strings(data.get("queries"), MAX_PLAN_QUERIES, min_len=2)
    topic = " ".join(str(data.get("topic") or "").split())
    if not queries and len(topic) >= 2:
        queries = [topic]
    if not queries:
        return SearchPlan.unplanned(query)

    lo = _plan_year(data.get("year_from"))
    hi = _plan_year(data.get("year_to"))
    if lo and hi and lo > hi:
        lo, hi = hi, lo

    kind = str(data.get("kind") or "").strip().lower()
    kind = _KIND_ALIASES.get(kind, kind)

    return SearchPlan(
        query=query,
        topic=topic or queries[0],
        queries=queries,
        tags=_plan_tags(data.get("tags")),
        year_from=lo,
        year_to=hi,
        kind=kind if kind in PLAN_KINDS else None,
        intent=" ".join(str(data.get("intent") or "").split())[:400],
        planned=True,
    )


def _clean_strings(raw, cap: int, *, min_len: int) -> list[str]:
    """De-duplicated, whitespace-normalized strings, in the order given."""
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    seen: set[str] = set()
    for item in (raw or []):
        s = " ".join(str(item).split())
        if len(s) < min_len or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _plan_tags(raw) -> list[str]:
    tags = _clean_strings(raw, MAX_PLAN_TAGS, min_len=2)
    return [t.lower().replace(" ", "-").strip("-") for t in tags]


def _plan_year(v) -> int | None:
    """A four-digit year, or None if what came back cannot be one."""
    try:
        year = int(str(v).strip()[:4])
    except (TypeError, ValueError):
        return None
    return year if EARLIEST_PLAUSIBLE_YEAR <= year <= date.today().year + 1 else None


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
    use_plan: bool = True,
) -> FindResult:
    """Build the de-duplicated candidate pool. Nothing is read or added here.

    `use_plan` is the cheap call that reads the request into index parameters;
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

    # Every source below is asked what the plan says the request means, not what
    # the user typed at it — including the year window, which reference mining
    # and the scouts have to honour as much as the indexes do.
    plan = SearchPlan.unplanned(query)
    if use_plan:
        ctx.log("[bold]Planning[/bold] the search from your query…")
        plan = plan_query(ctx, query)
        if plan.planned:
            for line in plan.summary_lines():
                ctx.log(f"[dim]{line}[/dim]")
        else:
            ctx.log("[dim]  (could not read the query — searching it as typed)[/dim]")

    result = FindResult(plan=plan)
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
        extra = (f" × {len(plan.queries)} phrasings" if len(plan.queries) > 1
                 else "")
        ctx.log(
            f"[bold]Searching[/bold] {len(SEARCH_FACETS)} facets{extra} "
            f"({INDEX_NAMES}) — no tokens spent…"
        )
        before = ctx.http.unavailable_count
        hits = _search_candidates(ctx, plan, per_facet=per_facet)
        result.search_failures = ctx.http.unavailable_count - before
        result.from_search = len(hits)
        for c in hits:
            absorb(c)
        ctx.vlog(f"  {len(hits)} candidates from indexed search")
        if result.search_failures:
            ctx.log(
                f"  [yellow]{result.search_failures} index request(s) never got "
                f"an answer[/yellow] — some facets searched nothing, so work "
                f"that exists may be missing from this pool"
            )

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
                    intent=plan.intent, constraints=plan.constraints(),
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

    # The window is applied twice: here, so a paper the request excluded cannot
    # take a slot in the trimmed pool, and again after enrichment, which is
    # where a candidate that arrived without a year finally gets one.
    considered, out_of_window = _apply_window(plan, considered)

    cap = _pool_cap(limit)
    if len(considered) > cap:
        considered = _trim_pool(considered, cap)
        solo = sum(1 for c in considered if not c.corroborated)
        ctx.log(f"  pool of {result.pool_size} trimmed to {len(considered)} "
                f"before scoring ({solo} held for uncorroborated work)")

    if considered:
        before = ctx.http.unavailable_count
        result.unweighed = _enrich(ctx, considered)
        result.weigh_failures = ctx.http.unavailable_count - before
        considered, late = _apply_window(plan, considered)
        out_of_window += late
        if result.weigh_failures:
            ctx.log(
                f"  [yellow]{result.weigh_failures} metadata request(s) never got "
                f"an answer[/yellow]"
            )
        if result.unweighed:
            # Importance is 65% relevance / 35% measured stature. With no
            # citation count the stature half falls back to an assumed middling
            # value, identical for every such paper — so the ranking below is
            # doing much less work than the numbers in the table suggest.
            ctx.log(
                f"  [yellow]{result.unweighed} of {len(considered)} ranked with "
                f"no citation count[/yellow] — their importance is assumed, not "
                f"measured"
            )

    result.dropped_out_of_window = out_of_window
    if out_of_window:
        ctx.log(f"  {out_of_window} dropped as outside {plan.constraints()[0]}")
    if considered:
        _score_relevance(ctx, plan, considered)

    kept = [c for c in considered
            if c.relevance is None or c.relevance >= MIN_RELEVANCE]
    result.dropped_off_topic = len(considered) - len(kept)
    if result.dropped_off_topic:
        ctx.log(f"  {result.dropped_off_topic} dropped as off-topic "
                f"(relevance below {MIN_RELEVANCE})")

    kept.sort(key=lambda c: (-c.rank_score, -(c.year or 0)))
    result.candidates = kept[:limit]

    # Said once, at the end, where the failures above have already been counted.
    # Semantic Scholar supplies the citation counts this ranking rests on, and
    # unauthenticated callers share one throttled pool with everyone else — so
    # the commonest cause of a thin, uninformed pool has a free fix, and the
    # moment it bites is the moment worth mentioning it.
    if result.degraded and not ctx.cfg.fetch.semantic_scholar_api_key:
        ctx.log(
            "  [dim]Semantic Scholar throttles callers without an API key; a "
            "free one removes most of these failures. Set "
            "fetch.semantic_scholar_api_key or $SEMANTIC_SCHOLAR_API_KEY.[/dim]"
        )
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


def _apply_window(plan: SearchPlan, candidates: list[Candidate]
                  ) -> tuple[list[Candidate], int]:
    """Drop candidates published outside the window the request asked for.

    The indexes were already told the window, but the other two sources cannot
    be: a reference mined from a paper you own and a title a scout came back
    with arrive with whatever year they have. "Nothing before 2022" has to hold
    for those too, or the filter only applies to the source that needed it least.
    """
    if not plan.has_window:
        return candidates, 0
    kept = [c for c in candidates if plan.allows(c.year)]
    return kept, len(candidates) - len(kept)


def _search_candidates(ctx: Ctx, plan: SearchPlan, *,
                       per_facet: int) -> list[Candidate]:
    """Ask the free indexes what the plan says the request means.

    Two dimensions, not one. The facets ask the same query several ways —
    best match, most-cited, recent, reviews, preprints. The plan's alternate
    phrasings ask the same *question* in another field's words, and only on the
    best-match facet: re-running "most-cited" over a paraphrase mostly returns
    the works the first phrasing already found, and each pairing is another
    round trip to a free service.

    Runs concurrently — they are independent HTTP requests to three different
    services, so waiting on them one at a time is pure latency.
    """
    tasks = [(plan.primary, facet) for facet, _ in SEARCH_FACETS]
    tasks += [(q, "relevance") for q in plan.alternates]

    def run(task: tuple[str, str]) -> list[Candidate]:
        query, facet = task
        metas = _search_facet(ctx, query, facet, per_facet, plan)
        return [c for c in (_meta_to_candidate(m, facet) for m in metas) if c]

    runs = run_parallel(
        tasks, run,
        workers=min(len(tasks), max(2, ctx.workers)),
        on_done=lambda r: ctx.vlog(
            f"  {r.item[1]} {r.item[0]!r}: {len(r.value or [])} hits" if r.ok
            else f"  {r.item[1]} {r.item[0]!r} failed: {r.error}"
        ),
    )
    out: list[Candidate] = []
    for r in runs:
        if r.ok and r.value:
            out += r.value
    return out


def _search_facet(ctx: Ctx, query: str, facet: str, limit: int,
                  plan: SearchPlan) -> list[PaperMeta]:
    """One facet of the free search. Split out so tests can stub the network.

    Facets that ask about standing rather than wording — `foundational` above
    all — go to both citation-aware indexes and pool the answers, because each
    is blind where the other sees. Crossref cannot see a preprint at all;
    Semantic Scholar can be rate-limited into silence. Asking only one means a
    facet that quietly returns nothing on the day that index is unhappy.
    """
    http = ctx.http
    key = ctx.cfg.fetch.semantic_scholar_api_key
    lo, hi, kind = plan.year_from, plan.year_to, plan.kind

    if facet == "foundational":
        # The one facet whose entire job is surfacing landmark work. Both
        # orderings are undercounts of the same thing, so take both.
        return _merge_metas(
            search_semantic_scholar(http, query, limit=limit, sort=S2_SORT_CITED,
                                    from_year=lo, to_year=hi, kind=_s2_kind(kind),
                                    api_key=key),
            search_crossref(http, query, limit, sort=CROSSREF_SORT_CITED,
                            from_year=lo, to_year=hi, kind=kind),
        )
    if facet == "recent":
        floor = date.today().year - RECENT_YEARS
        # "Recent" and a window that closes before it are contradictory; the
        # window is what the person actually asked for, so the facet stands down.
        if hi and hi < floor:
            return []
        since = max(floor, lo or floor)
        return _merge_metas(
            search_semantic_scholar(http, query, limit=limit, from_year=since,
                                    to_year=hi, kind=_s2_kind(kind), api_key=key),
            search_crossref(http, query, limit, from_year=since, to_year=hi,
                            kind=kind),
        )
    if facet == "surveys":
        # Crossref has no "review" document type — a survey is registered as an
        # ordinary journal article — so this facet rests on S2's own label.
        return search_semantic_scholar(http, query, limit=limit,
                                       kind=S2_REVIEW_TYPE, from_year=lo,
                                       to_year=hi, api_key=key)
    if facet == "preprints":
        return search_arxiv(http, query, limit=limit, from_year=lo, to_year=hi)
    return _merge_metas(
        search_crossref(http, query, limit, from_year=lo, to_year=hi, kind=kind),
        search_semantic_scholar(http, query, limit=limit, from_year=lo,
                                to_year=hi, kind=_s2_kind(kind), api_key=key),
    )


def _s2_kind(kind: str | None) -> str | None:
    """The plan's document type in Semantic Scholar's vocabulary, if it has one."""
    return S2_REVIEW_TYPE if kind == "review" else None


def _merge_metas(*groups: list[PaperMeta]) -> list[PaperMeta]:
    """Interleave results from several indexes, dropping repeats.

    Interleaved rather than concatenated so that a `per_facet` slice downstream
    takes the best of each index instead of all of whichever answered first.
    """
    out: list[PaperMeta] = []
    seen: set[str] = set()
    for row in zip_longest(*groups):
        for meta in row:
            if meta is None or not meta.title:
                continue
            key = (meta.doi or (f"arxiv:{meta.arxiv_id}" if meta.arxiv_id
                                else slugify(meta.title, 120)))
            if key in seen:
                continue
            seen.add(key)
            out.append(meta)
    return out


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


def _enrich(ctx: Ctx, candidates: list[Candidate]) -> int:
    """Fill in citation count and venue from Semantic Scholar, no LLM.

    This is what lets importance be measured rather than guessed. It runs
    before the cut, unlike the metadata resolution during a read, because a
    citation count discovered after a paper has been dropped is no use.

    Candidates carrying an identifier go in one batch request — S2 answers for
    up to 500 papers at a time, so weighing a pool of forty costs a single call
    against an index that throttles at roughly one request a second. Only the
    candidates with no identifier at all need a title search of their own, and
    those are usually a handful mined from reference lists.

    Returns the number still lacking a citation count afterwards, so the caller
    can say plainly when the pool was ranked without the numbers.
    """
    pending = [c for c in candidates if c.citation_count is None or not c.venue]
    if not pending:
        ctx.vlog(f"  all {len(candidates)} candidates already carry their metrics")
        return 0
    ctx.log(f"[bold]Weighing[/bold] {len(pending)} candidates (citations, venue)…")

    by_ident: dict[str, Candidate] = {}
    untitled: list[Candidate] = []
    for c in pending:
        ident = (f"DOI:{c.doi}" if c.doi
                 else f"arXiv:{c.arxiv_id}" if c.arxiv_id else None)
        if ident:
            by_ident.setdefault(ident, c)
        else:
            untitled.append(c)

    found = 0
    if by_ident:
        metas = _batch_lookup(ctx, list(by_ident))
        for ident, meta in metas.items():
            if (c := by_ident.get(ident)) is not None:
                _absorb_meta(c, meta)
                found += 1

    if untitled:
        runs = run_parallel(untitled, lambda c: _lookup_meta(ctx, c),
                            workers=ctx.workers)
        for r in runs:
            c: Candidate = r.item  # type: ignore[assignment]
            if not r.ok:
                ctx.vlog(f"  lookup failed for {c.title[:60]}: {r.error}")
                continue
            if r.value is not None:
                _absorb_meta(c, r.value)
                found += 1

    ctx.vlog(f"  metadata found for {found}/{len(pending)}")
    return sum(1 for c in candidates if c.citation_count is None)


def _absorb_meta(c: Candidate, meta: PaperMeta) -> None:
    """Copy what a metadata lookup found onto a candidate."""
    if meta.citation_count is not None:
        c.citation_count = meta.citation_count
    c.venue = c.venue or meta.venue
    c.year = c.year or meta.year
    # A record found by title may be a mirror or re-registration of the work
    # rather than the work itself (see `PaperMeta.supplementary`). Its citations
    # and venue still describe the right paper and are only used for scoring,
    # but its identifiers would be handed to the reader as a Target — so those
    # are taken only from an identifier lookup.
    if not meta.title_matched:
        c.doi = c.doi or meta.doi
        c.arxiv_id = c.arxiv_id or meta.arxiv_id
    c.abstract = c.abstract or meta.abstract


def _batch_lookup(ctx: Ctx, idents: list[str]) -> dict[str, PaperMeta]:
    """One batched S2 request for many papers. Split out so tests can stub it."""
    return batch_s2(ctx.http, idents, ctx.cfg.fetch.semantic_scholar_api_key,
                    with_references=False)


def _lookup_meta(ctx: Ctx, c: Candidate):
    """One title lookup for a candidate with no identifier. Stubbed in tests."""
    return s2_by_title(ctx.http, c.title,
                       ctx.cfg.fetch.semantic_scholar_api_key,
                       with_references=False)


def _score_relevance(ctx: Ctx, plan: SearchPlan,
                     candidates: list[Candidate]) -> None:
    """One LLM call scoring the whole pool against the query.

    Scored against what the user typed, not against the plan's queries: the
    plan is how the indexes were asked, and a keyword phrasing is a lossy
    rendering of a request. The plan's reading of it rides along as context so
    the same understanding decides the cut.

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
                query=plan.query, scope=ctx.library.settings.scope,
                candidates=candidates, intent=plan.intent,
                constraints=plan.constraints(),
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

    Reference lists come from Crossref and Semantic Scholar, not from a model, so these
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
        # `cocitations` is the whole signal here — the ranking card renders it
        # in words already. Writing the same sentence into `why` would spend
        # tokens saying it twice *and* take the slot the abstract goes in.
        c.cocitations = n
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
