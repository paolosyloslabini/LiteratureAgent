"""`lit search` — find in-library sources for a query, with a short answer.

Cheap and fast by design: this works from the stored summaries only. It is the
"which of my papers are about this?" command. `lit ask` is the one that opens
the papers themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm import LLMError
from ..models import Entry
from ..prompts import ANALYST_SYSTEM, library_search_prompt
from ..store import Store, hydrate
from .context import Ctx


@dataclass
class Match:
    entry: Entry
    relevance: float
    why: str = ""


@dataclass
class SearchResult:
    query: str
    answer: str = ""
    matches: list[Match] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "matches": [
                {
                    "key": m.entry.key,
                    "title": m.entry.title,
                    "citation": m.entry.citation(),
                    "level": m.entry.level,
                    "status": m.entry.status,
                    "relevance": m.relevance,
                    "why": m.why,
                    "one_liner": m.entry.one_liner,
                }
                for m in self.matches
            ],
            "gaps": self.gaps,
        }


def search(
    ctx: Ctx,
    query: str,
    *,
    limit: int = 10,
    pool: int = 30,
    level: str | None = None,
    tag: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    rank: bool = True,
) -> SearchResult:
    """Retrieve with FTS, then optionally re-rank and answer with the LLM."""
    result = SearchResult(query=query)

    with Store(ctx.library) as store:
        hits = store.search(
            query, limit=max(pool, limit), level=level, tag=tag,
            year_from=year_from, year_to=year_to,
        )
        hits = hydrate(ctx.library, hits)

    if not hits:
        result.answer = "No entries in this library matched that query."
        return result

    if not rank or not ctx.llm.available:
        result.matches = [Match(entry=h.entry, relevance=0.0) for h in hits[:limit]]
        return result

    entries = [h.entry for h in hits]
    by_key = {e.key: e for e in entries}

    try:
        data = ctx.llm.json(
            library_search_prompt(
                query=query, scope=ctx.library.settings.scope, entries=entries
            ),
            system=ANALYST_SYSTEM,
            role="filter",
            required=("relevant",),
        )
    except LLMError as exc:
        ctx.vlog(f"ranking failed, falling back to bm25 order: {exc}")
        result.matches = [Match(entry=e, relevance=0.0) for e in entries[:limit]]
        result.answer = f"(LLM ranking unavailable: {exc})"
        return result

    result.answer = str(data.get("answer") or "").strip()
    result.gaps = [str(g).strip() for g in (data.get("gaps") or []) if str(g).strip()]

    seen: set[str] = set()
    for r in (data.get("relevant") or []):
        key = str(r.get("key") or "").strip()
        entry = by_key.get(key)
        if entry is None or key in seen:
            continue
        seen.add(key)
        try:
            rel = float(r.get("relevance", 0.0))
        except (TypeError, ValueError):
            rel = 0.0
        result.matches.append(
            Match(entry=entry, relevance=rel, why=str(r.get("why") or "").strip())
        )

    result.matches.sort(key=lambda m: -m.relevance)
    result.matches = result.matches[:limit]
    return result
