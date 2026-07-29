"""`lit claim` — trace a claim back to the paper that originates it.

Implements the procedure in the spec:

1. Find a paper that makes the claim (from the library, or via web search).
2. Fetch its references.
3. Work out which reference the claim actually originates from.
4. Fetch and read that earlier paper to confirm it is the true source.
5. Repeat until reaching the paper that introduces the concept.
6. Report the full chain.

Step 3-4 are fused: at each hop the top candidate references are resolved, read
and analysed *concurrently*, and the winning candidate's analysis is carried
forward as the next hop. So confirming a reference and advancing the chain are
the same piece of work, done once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fetch.fulltext import fetch_fulltext, truncate_for_llm
from ..fetch.metadata import PaperMeta, resolve_metadata
from ..llm import LLMError
from ..models import Entry, Reference, slugify
from ..prompts import ANALYST_SYSTEM, SCOUT_SYSTEM, claim_origin_prompt, find_claim_source_prompt
from ..runner import run_parallel
from .add import add_paper
from .context import Ctx, Target, parse_target

MAX_CANDIDATES_PER_HOP = 3


@dataclass
class Node:
    """A paper being examined during a trace. May or may not be in the library."""

    meta: PaperMeta
    entry: Entry | None = None
    text: str = ""
    truncated: bool = False

    @property
    def title(self) -> str:
        return self.meta.title

    @property
    def year(self) -> int | None:
        return self.meta.year

    @property
    def ident(self) -> str:
        return (self.meta.doi or self.meta.arxiv_id
                or slugify(self.meta.title, 60) or "?")

    def label(self) -> str:
        first = self.meta.authors[0].split()[-1] if self.meta.authors else "Unknown"
        return f"{first}{' et al.' if len(self.meta.authors) > 1 else ''} " \
               f"({self.year or 'n.d.'})"


@dataclass
class Hop:
    title: str
    year: int | None
    label: str
    doi: str | None = None
    arxiv_id: str | None = None
    library_key: str | None = None
    states_claim: bool = False
    is_origin: bool = False
    quote: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title, "year": self.year, "label": self.label,
            "doi": self.doi, "arxiv_id": self.arxiv_id,
            "library_key": self.library_key,
            "states_claim": self.states_claim, "is_origin": self.is_origin,
            "quote": self.quote, "reasoning": self.reasoning,
        }


@dataclass
class ClaimResult:
    claim: str
    chain: list[Hop] = field(default_factory=list)
    origin: Hop | None = None
    notes: list[str] = field(default_factory=list)
    complete: bool = False

    def chain_str(self) -> str:
        """e.g. 'Paper A (2020) -> Paper B (2015) -> Paper C (2008, original)'."""
        parts = []
        for h in self.chain:
            tag = ", original" if h.is_origin else ""
            parts.append(f"{_short(h.title)} ({h.year or 'n.d.'}{tag})")
        return " -> ".join(parts)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "chain": [h.to_dict() for h in self.chain],
            "chain_str": self.chain_str(),
            "origin": self.origin.to_dict() if self.origin else None,
            "complete": self.complete,
            "notes": self.notes,
        }


def trace_claim(
    ctx: Ctx,
    claim: str,
    *,
    start: str | None = None,
    max_hops: int = 5,
    add_to_library: bool = False,
) -> ClaimResult:
    result = ClaimResult(claim=claim)

    node = _find_start(ctx, claim, start, result)
    if node is None:
        return result

    seen: set[str] = set()
    analysis: dict | None = None

    for _ in range(max_hops):
        if node.ident in seen:
            result.notes.append("stopped: the chain looped back on itself")
            break
        seen.add(node.ident)

        if not node.text:
            result.notes.append(
                f"could not read the full text of {node.title!r}; "
                "the chain stops here"
            )
            result.chain.append(_hop(node, {}))
            break

        if analysis is None:
            ctx.log(f"[bold]Reading[/bold] {_short(node.title)} ({node.year or 'n.d.'})…")
            try:
                analysis = _analyse(ctx, claim, node)
            except LLMError as exc:
                result.notes.append(f"analysis failed for {node.title!r}: {exc}")
                result.chain.append(_hop(node, {}))
                break

        hop = _hop(node, analysis)
        result.chain.append(hop)

        if add_to_library and node.entry is None:
            _adopt(ctx, node, hop)

        if hop.is_origin:
            result.origin = hop
            result.complete = True
            ctx.log(f"  [green]origin found[/green]: {_short(node.title)}")
            break

        refs = node.meta.references
        picks = _candidate_refs(analysis, refs)
        if not picks:
            result.notes.append(
                f"{_short(node.title)} states the claim but attributes it to nothing "
                "traceable; treating it as the earliest source reachable."
            )
            hop.is_origin = True
            result.origin = hop
            break

        ctx.log(
            f"  checking {len(picks)} cited source(s) in parallel: "
            + "; ".join(_short(r.title, 40) for r in picks)
        )
        nxt, nxt_analysis = _best_next(ctx, claim, picks)
        if nxt is None:
            result.notes.append(
                "none of the cited sources could be read or none stated the claim; "
                f"{_short(node.title)} is the earliest confirmed source."
            )
            hop.is_origin = True
            result.origin = hop
            break

        node, analysis = nxt, nxt_analysis
    else:
        result.notes.append(
            f"stopped after {max_hops} hops without reaching an origin "
            "(raise it with --max-hops)"
        )

    if result.origin is None and result.chain:
        result.origin = result.chain[-1]
    return result


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def _find_start(ctx: Ctx, claim: str, start: str | None,
                result: ClaimResult) -> Node | None:
    """Locate a paper that makes the claim: explicit, in-library, or via search."""
    if start:
        node = _load_node(ctx, parse_target(start))
        if node is None:
            result.notes.append(f"could not resolve the starting paper {start!r}")
        return node

    # Prefer something already in the library — it is already vetted and cached.
    from .search import search as lib_search

    hits = lib_search(ctx, claim, limit=3, pool=20, rank=False)
    for m in hits.matches:
        if m.entry.is_verified:
            ctx.log(f"[bold]Starting[/bold] from library entry [{m.entry.key}]")
            node = _load_node(ctx, Target(doi=m.entry.doi, arxiv_id=m.entry.arxiv_id,
                                          title=m.entry.title), entry=m.entry)
            if node and node.text:
                return node

    ctx.log("[bold]Searching[/bold] for a paper that makes this claim…")
    try:
        data = ctx.llm.json(
            find_claim_source_prompt(claim=claim, scope=ctx.library.settings.scope),
            system=SCOUT_SYSTEM,
            role="scout",
            tools=["WebSearch", "WebFetch"],
            max_turns=20,
            required=("papers",),
        )
    except LLMError as exc:
        result.notes.append(f"could not search for a starting paper: {exc}")
        return None

    for p in (data.get("papers") or [])[:3]:
        node = _load_node(ctx, Target(
            doi=p.get("doi"), arxiv_id=p.get("arxiv_id"), title=p.get("title"),
        ))
        if node and node.text:
            return node

    result.notes.append(
        "could not find and read any paper stating this claim. "
        "Try `lit claim \"...\" --start <doi|arxiv|title>`."
    )
    return None


def _load_node(ctx: Ctx, target: Target, entry: Entry | None = None) -> Node | None:
    """Resolve metadata and full text for one paper."""
    meta = resolve_metadata(
        ctx.http, doi=target.doi, arxiv_id=target.arxiv_id, title=target.title,
        cfg=ctx.cfg.fetch,
    )
    if meta is None or not meta.title:
        return None

    if entry is None:
        entry = ctx.library.find_duplicate(
            doi=meta.doi, arxiv_id=meta.arxiv_id, title=meta.title
        )
    cache_key = entry.key if entry else f"_trace_{slugify(meta.doi or meta.arxiv_id or meta.title, 60)}"

    ft = fetch_fulltext(
        ctx.http, meta,
        key=cache_key,
        pdf_dir=ctx.library.pdfs_dir,
        text_dir=ctx.library.text_dir,
        cfg=ctx.cfg.fetch,
    )
    node = Node(meta=meta, entry=entry)
    if ft is not None:
        # References are deliberately left in: tracing a claim means reading the
        # citation context around it, so the bibliography is the evidence here.
        node.text, node.truncated = truncate_for_llm(ft.text, ctx.read_budget)
    return node


def _analyse(ctx: Ctx, claim: str, node: Node) -> dict:
    pseudo = node.entry or Entry(
        key=node.ident, title=node.meta.title, authors=node.meta.authors,
        year=node.meta.year, venue=node.meta.venue,
    )
    return ctx.llm.json(
        claim_origin_prompt(
            claim=claim, entry=pseudo, references=node.meta.references,
            truncated=node.truncated,
        ),
        stdin_text="=== FULL TEXT BEGINS ===\n" + node.text,
        system=ANALYST_SYSTEM,
        role="analyst",
        required=("is_origin",),
    )


def _candidate_refs(analysis: dict, refs: list[Reference]) -> list[Reference]:
    """The references the model pointed at, best first."""
    scored: list[tuple[float, Reference]] = []
    for a in (analysis.get("attributed_to") or []):
        try:
            i = int(a.get("index"))
            conf = float(a.get("confidence", 0.5))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= i < len(refs):
            scored.append((conf, refs[i]))
    scored.sort(key=lambda t: -t[0])

    out: list[Reference] = []
    seen: set[str] = set()
    for _, r in scored:
        k = r.doi or r.arxiv_id or r.title.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= MAX_CANDIDATES_PER_HOP:
            break
    return out


def _best_next(ctx: Ctx, claim: str,
               refs: list[Reference]) -> tuple[Node | None, dict | None]:
    """Resolve, read and analyse candidate references concurrently; pick one."""

    def work(ref: Reference):
        node = _load_node(ctx, Target(doi=ref.doi, arxiv_id=ref.arxiv_id,
                                      title=ref.title))
        if node is None or not node.text:
            return None
        try:
            return node, _analyse(ctx, claim, node)
        except LLMError:
            return None

    runs = run_parallel(refs, work, workers=min(ctx.workers, len(refs)))

    best: tuple[Node, dict] | None = None
    best_score = (-1, 0)
    for r in runs:
        if not r.ok or r.value is None:
            continue
        node, analysis = r.value
        if not analysis.get("states_claim"):
            continue
        # Prefer a confirmed origin; among equals, prefer the older paper —
        # tracing runs backwards in time.
        score = (2 if analysis.get("is_origin") else 1, -(node.year or 9999))
        if score > best_score:
            best_score = score
            best = (node, analysis)
    return (best[0], best[1]) if best else (None, None)


def _adopt(ctx: Ctx, node: Node, hop: Hop) -> None:
    """Add a paper discovered during tracing into the library."""
    res = add_paper(
        ctx, node.meta.title,
        target=Target(doi=node.meta.doi, arxiv_id=node.meta.arxiv_id,
                      title=node.meta.title),
        meta=node.meta,
        force=True,
    )
    if res.entry:
        hop.library_key = res.entry.key
        node.entry = res.entry


def _hop(node: Node, analysis: dict) -> Hop:
    return Hop(
        title=node.meta.title,
        year=node.meta.year,
        label=node.label(),
        doi=node.meta.doi,
        arxiv_id=node.meta.arxiv_id,
        library_key=node.entry.key if node.entry else None,
        states_claim=bool(analysis.get("states_claim")),
        is_origin=bool(analysis.get("is_origin")),
        quote=str(analysis.get("evidence_quote") or "").strip(),
        reasoning=str(analysis.get("reasoning") or "").strip(),
    )


def _short(title: str, n: int = 60) -> str:
    title = (title or "").strip()
    return title if len(title) <= n else title[: n - 1] + "…"
