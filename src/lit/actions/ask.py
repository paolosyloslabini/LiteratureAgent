"""`lit ask` — answer a question from the library by reading the actual papers.

The spec is explicit that summaries are for navigation, not for answering. So:

1. Retrieve with FTS + LLM relevance ranking over the summaries — this picks
   *which* papers to open.
2. Open those papers' full text and extract evidence, one agent per paper, all
   in parallel. Each agent must copy quotes verbatim from the text in front of
   it.
3. Optionally follow references of the selected papers and pull in works not yet
   in the library (`--expand`).
4. Synthesize a cited answer, with a bibliography, from that evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fetch.fulltext import fetch_fulltext, truncate_for_llm
from ..fetch.metadata import PaperMeta
from ..llm import LLMError
from ..models import Entry
from ..prompts import (
    ANALYST_SYSTEM,
    extract_evidence_prompt,
    synthesize_answer_prompt,
)
from ..runner import run_parallel
from .add import add_paper
from .context import Ctx, Target
from .search import search


@dataclass
class Quote:
    text: str
    section: str | None = None


@dataclass
class Evidence:
    entry: Entry
    relevant: bool
    summary: str = ""
    quotes: list[Quote] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    error: str = ""

    def to_prompt_dict(self) -> dict:
        return {
            "key": self.entry.key,
            "title": self.entry.title,
            "citation": self.entry.citation(),
            "summary": self.summary,
            "quotes": [
                {"text": q.text, "section": q.section}
                for q in self.quotes
            ],
            "caveats": self.caveats,
        }


@dataclass
class AskResult:
    question: str
    answer: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    consulted: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    @property
    def cited(self) -> list[Evidence]:
        return [e for e in self.evidence if e.relevant and e.quotes]

    def bibliography(self) -> list[dict]:
        out = []
        for ev in self.cited:
            e = ev.entry
            out.append({
                "key": e.key, "title": e.title, "authors": e.authors,
                "year": e.year, "venue": e.venue, "doi": e.doi,
                "arxiv_id": e.arxiv_id, "url": e.url, "level": e.level,
                "citation": e.citation(), "bibtex": e.bibtex,
            })
        return out

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "bibliography": self.bibliography(),
            "evidence": [
                {
                    "key": ev.entry.key,
                    "relevant": ev.relevant,
                    "summary": ev.summary,
                    "quotes": [
                        {"text": q.text, "section": q.section}
                        for q in ev.quotes
                    ],
                    "caveats": ev.caveats,
                    "error": ev.error,
                }
                for ev in self.evidence
            ],
            "consulted": self.consulted,
            "pulled_in": self.pulled,
            "unreadable": self.unreadable,
        }


def ask(
    ctx: Ctx,
    question: str,
    *,
    read: int = 5,
    pool: int = 30,
    expand: int = 0,
    level: str | None = None,
    tag: str | None = None,
) -> AskResult:
    result = AskResult(question=question)
    lib = ctx.library

    # ---- 1. pick what to read -------------------------------------------
    ctx.log("[bold]Selecting[/bold] sources from the library…")
    # `brief` because this call only decides which papers to open. The prose
    # answer `search` normally writes is drawn from summaries and would be
    # thrown away here — the answer this command returns is synthesized in
    # step 4 from the full texts.
    sel = search(ctx, question, limit=read, pool=pool, level=level, tag=tag,
                 brief=True)
    # An entry that was filed but never read is a perfectly good source here:
    # `ask` fetches the full text itself. Only UNVERIFIED entries are skipped,
    # because for those the full text was already looked for and not found.
    chosen = [m.entry for m in sel.matches
              if m.entry.is_verified or m.entry.is_unread][:read]

    if not chosen:
        blocked = [m.entry.key for m in sel.matches if not m.entry.is_verified]
        result.answer = (
            "No readable sources in this library match that question."
            + (f" {len(blocked)} matching entries are UNVERIFIED (no full text): "
               f"{', '.join(blocked[:5])}." if blocked else "")
        )
        return result

    # ---- 2. optionally pull in cited work not yet in the library ---------
    if expand > 0:
        pulled = _expand_from_references(ctx, chosen, question, expand)
        result.pulled = [e.key for e in pulled]
        chosen += pulled

    result.consulted = [e.key for e in chosen]
    ctx.log(
        f"[bold]Reading[/bold] {len(chosen)} papers in full "
        f"({ctx.workers} at a time)…"
    )

    # ---- 3. read them, in parallel --------------------------------------
    def extract(entry: Entry) -> Evidence:
        ft = fetch_fulltext(
            ctx.http,
            _meta_from_entry(entry),
            key=entry.key,
            pdf_dir=lib.pdfs_dir,
            text_dir=lib.text_dir,
            cfg=ctx.cfg.fetch,
        )
        if ft is None:
            return Evidence(entry=entry, relevant=False,
                            error="full text no longer retrievable")
        text, truncated = truncate_for_llm(ft.text, ctx.read_budget,
                                           drop_references=True)
        data = ctx.llm.json(
            extract_evidence_prompt(
                question=question, entry=entry, truncated=truncated
            ),
            stdin_text="=== FULL TEXT BEGINS ===\n" + text,
            system=ANALYST_SYSTEM,
            role="analyst",
            required=("relevant",),
        )
        quotes = []
        for q in (data.get("quotes") or [])[:4]:
            qt = str(q.get("text") or "").strip().strip('"')
            if not qt:
                continue
            quotes.append(Quote(
                text=qt,
                section=(str(q.get("section")).strip() or None)
                if q.get("section") else None,
            ))
        return Evidence(
            entry=entry,
            relevant=bool(data.get("relevant")),
            summary=str(data.get("summary") or "").strip(),
            quotes=quotes,
            caveats=[str(c).strip() for c in (data.get("caveats") or []) if str(c).strip()],
        )

    def report(r) -> None:
        entry: Entry = r.item  # type: ignore[assignment]
        if not r.ok:
            ctx.log(f"  [red]![/red] {entry.key}: {r.error}")
            return
        ev: Evidence = r.value
        if ev.error:
            ctx.log(f"  [yellow]-[/yellow] {entry.key}: {ev.error}")
        elif ev.relevant:
            ctx.log(f"  [green]✓[/green] {entry.key}: {len(ev.quotes)} quote(s)")
        else:
            ctx.log(f"  [dim]·[/dim] {entry.key}: not relevant")

    runs = run_parallel(chosen, extract, workers=ctx.workers, on_done=report)
    for r in runs:
        if r.ok and r.value is not None:
            ev: Evidence = r.value
            result.evidence.append(ev)
            if ev.error:
                result.unreadable.append(ev.entry.key)
        else:
            entry: Entry = r.item  # type: ignore[assignment]
            result.evidence.append(
                Evidence(entry=entry, relevant=False, error=str(r.error))
            )
            result.unreadable.append(entry.key)

    # ---- 4. synthesize ---------------------------------------------------
    usable = result.cited
    if not usable:
        result.answer = (
            "None of the papers read contained evidence bearing on that question. "
            "Sources consulted: " + ", ".join(result.consulted) + "."
        )
        return result

    ctx.log("[bold]Synthesizing[/bold] the answer…")
    try:
        res = ctx.llm.run(
            synthesize_answer_prompt(
                question=question,
                scope=lib.settings.scope,
                evidence=[e.to_prompt_dict() for e in usable],
            ),
            system=ANALYST_SYSTEM,
            role="synthesis",
        )
        result.answer = res.text.strip()
    except LLMError as exc:
        result.answer = f"(synthesis failed: {exc})"
    return result


# --------------------------------------------------------------------------
# Expansion via references
# --------------------------------------------------------------------------

def _expand_from_references(ctx: Ctx, entries: list[Entry], question: str,
                            limit: int) -> list[Entry]:
    """Pull in works cited by the chosen papers that aren't in the library yet.

    Ranked by how many of the chosen papers cite them: a work that several of
    your best sources all point at is worth reading.

    They are filed from their metadata, not read. `extract` is about to fetch
    each one's full text and put it in front of an agent anyway; reading them
    here as well would put the same document through two full-text calls in a
    single command. The entry is left `unread`, and `lit read <key>` buys the
    stored summary later if it turns out to be worth one.
    """
    lib = ctx.library
    counts: dict[str, int] = {}
    proto: dict[str, object] = {}
    for e in entries:
        for ref in e.references:
            if not ref.title:
                continue
            k = ref.doi or (f"arxiv:{ref.arxiv_id}" if ref.arxiv_id else ref.title.lower())
            counts[k] = counts.get(k, 0) + 1
            proto.setdefault(k, ref)

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    picked: list[Entry] = []
    for k, n in ranked:
        if len(picked) >= limit:
            break
        ref = proto[k]
        if lib.find_duplicate(doi=ref.doi, arxiv_id=ref.arxiv_id, title=ref.title):
            continue
        if n < 2 and len(entries) > 2:
            continue  # only follow works that more than one source cites
        ctx.log(f"  [cyan]+[/cyan] pulling in (cited by {n}): {ref.title[:70]}")
        res = add_paper(
            ctx, ref.title,
            target=Target(doi=ref.doi, arxiv_id=ref.arxiv_id, title=ref.title),
            read=False,
        )
        # Unread is the expected state here; a paper whose full text cannot be
        # reached is reported by `extract` as unreadable rather than dropped
        # silently at this point.
        if res.ok and res.entry and (res.entry.is_verified or res.entry.is_unread):
            picked.append(res.entry)
    return picked


def _meta_from_entry(entry: Entry) -> PaperMeta:
    """A PaperMeta good enough to re-fetch this entry's full text."""
    return PaperMeta(
        title=entry.title,
        authors=list(entry.authors),
        year=entry.year,
        venue=entry.venue,
        type=entry.type,
        doi=entry.doi,
        arxiv_id=entry.arxiv_id,
        url=entry.url,
        is_oa=bool(entry.arxiv_id),
        oa_pdf_url=f"https://arxiv.org/pdf/{entry.arxiv_id}" if entry.arxiv_id else None,
    )
