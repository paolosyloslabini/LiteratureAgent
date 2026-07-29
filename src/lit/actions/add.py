"""`lit add` — put one paper into the library.

The spec is strict about this path, and so is the code:

* The full text must be read. Metadata alone never produces a summary.
* The agent that reads the paper is the one that fills the fields — there is a
  single LLM call holding the whole text, not a read step and a separate write
  step working off notes.
* A paper that cannot be retrieved is stored anyway, flagged UNVERIFIED, with
  its summaries left blank. Nothing is inferred from the abstract.
* A code link is recorded here only when the paper's own text prints it. A
  model's recollection of where a project lives is not evidence. Searching the
  web for one is a separate, opt-in step (`lit code`), and what it finds is
  stored marked as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..fetch.fulltext import fetch_fulltext, truncate_for_llm
from ..fetch.metadata import PaperMeta, resolve_metadata, synth_bibtex
from ..library import list_libraries
from ..llm import LLMError
from ..models import (
    CODE_FROM_PAPER,
    ENTRY_TYPES,
    STATUS_UNREAD,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    Entry,
    Section,
    make_key,
    normalize_code_url,
)
from ..prompts import (
    MAX_KEY_FINDINGS,
    MAX_SECTIONS,
    READER_SYSTEM,
    SECTION_WORDS_CEILING,
    read_paper_prompt,
)
from ..quality import LevelVerdict, assess, passes_quality_bar
from ..runner import run_parallel
from .context import Ctx, Target, parse_target


@dataclass
class AddResult:
    status: str  # added | updated | duplicate | rejected | not_found | unavailable | error
    message: str
    entry: Entry | None = None
    verdict: LevelVerdict | None = None
    meta: PaperMeta | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("added", "updated")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "key": self.entry.key if self.entry else None,
            "title": self.entry.title if self.entry else None,
            "level": self.entry.level if self.entry else None,
            "entry_status": self.entry.status if self.entry else None,
            "warnings": self.warnings,
        }


def add_paper(
    ctx: Ctx,
    query: str,
    *,
    local_pdf: Path | None = None,
    force: bool = False,
    refresh: bool = False,
    extra_tags: list[str] | None = None,
    meta: PaperMeta | None = None,
    target: Target | None = None,
    read: bool = True,
    max_chars: int | None = None,
) -> AddResult:
    """Resolve, vet, read and store one paper.

    With `read=False` the paper is filed from its metadata alone — bibliographic
    record, publisher abstract, references, level from metrics — and marked
    `unread`. No full text is fetched and no reader agent runs, so the whole
    thing costs nothing but a few API calls. `lit read <key>` fills in the
    summaries later, for the entries that turn out to be worth it.
    """
    lib = ctx.library
    warnings: list[str] = []
    target = target or parse_target(query)

    # ---- 1. metadata -----------------------------------------------------
    # Remembered so a failure to resolve can say *why*: every index being
    # unreachable looks identical to the paper not existing, and telling a user
    # to "try a DOI" when arXiv was simply rate-limiting them sends them off
    # fixing an input that was correct all along.
    unavailable_before = ctx.http.unavailable_count
    if meta is None:
        ctx.vlog(f"resolving metadata for {target.describe()!r}")
        meta = resolve_metadata(
            ctx.http,
            doi=target.doi,
            arxiv_id=target.arxiv_id,
            title=target.title,
            cfg=ctx.cfg.fetch,
        )

    if meta is None or not meta.title:
        if local_pdf and Path(local_pdf).exists():
            # No record online, but we hold the PDF — proceed off the file alone.
            meta = PaperMeta(title=target.title or Path(local_pdf).stem, type="other")
            warnings.append("no online record found; metadata taken from the filename")
        elif ctx.http.unavailable_count > unavailable_before:
            return AddResult(
                "unavailable",
                f"could not reach the metadata indexes for {target.describe()!r}. "
                "This was a network or rate-limit failure, not a missing paper — "
                "the record may well exist. Try again in a minute.",
            )
        else:
            return AddResult(
                "not_found",
                f"could not find a bibliographic record for {target.describe()!r}. "
                "Try a DOI, an arXiv id, or --pdf with a local file.",
            )

    # ---- 2. duplicates ---------------------------------------------------
    existing = lib.find_duplicate(doi=meta.doi, arxiv_id=meta.arxiv_id, title=meta.title)
    if existing and not refresh:
        return AddResult(
            "duplicate",
            f"already in the library as [{existing.key}]"
            + ("" if existing.is_verified else " (UNVERIFIED — use --refresh to retry)"),
            entry=existing,
            meta=meta,
        )

    # ---- 3. quality bar --------------------------------------------------
    verdict = assess(meta)
    if not force:
        ok, why = passes_quality_bar(
            verdict, meta, lib.settings.min_level, lib.settings.allow_non_published
        )
        if not ok:
            return AddResult(
                "rejected",
                f"{meta.title!r}: {why}. Use --force to add it anyway.",
                verdict=verdict,
                meta=meta,
            )

    # ---- 4. full text ----------------------------------------------------
    key = existing.key if existing else lib.unique_key(
        make_key(meta.authors, meta.year, meta.title)
    )

    if not read:
        entry = _base_entry(key, meta, verdict, extra_tags)
        entry.status = STATUS_UNREAD
        if existing:
            entry.added = existing.added
            # A metadata-only pass must never cost a summary that was already
            # paid for: refreshing an entry that has been read keeps its
            # reading, and only its bibliographic fields are updated.
            _carry_reading(entry, existing)
        saved = lib.save_entry(entry)
        note = ("refreshed" if entry.is_verified
                else f"unread — `lit read {key}` to summarize it")
        return AddResult(
            "updated" if existing else "added",
            f"[{key}] {entry.title} ({note})",
            entry=saved, verdict=verdict, meta=meta, warnings=warnings,
        )

    # A paper you keep in two libraries was read twice, and a read is the most
    # expensive thing here — the same PDF, the same agent, the same answer, paid
    # for again because the entry lives in a different directory. If a sibling
    # library under the same root has already read this one, take its reading.
    # `--refresh` still buys a fresh one.
    if not refresh:
        found = _reading_elsewhere(ctx, meta)
        if found is not None:
            source_lib, done = found
            entry = _base_entry(key, meta, verdict, extra_tags)
            if existing:
                entry.added = existing.added
            _carry_reading(entry, done)
            # The PDF and cached text belong to the other library's directory,
            # so this entry claims neither; `lit read --refresh` fetches its own.
            entry.pdf_path = None
            entry.tags = _norm_tags(list(done.tags) + list(extra_tags or []))
            saved = lib.save_entry(entry)
            return AddResult(
                "updated" if existing else "added",
                f"[{key}] {entry.title} (reading reused from library "
                f"'{source_lib}' — --refresh to re-read)",
                entry=saved, verdict=verdict, meta=meta, warnings=warnings,
            )

    ctx.vlog(f"[{key}] fetching full text")
    ft = fetch_fulltext(
        ctx.http, meta,
        key=key,
        pdf_dir=lib.pdfs_dir,
        text_dir=lib.text_dir,
        cfg=ctx.cfg.fetch,
        local_pdf=local_pdf,
        refresh=refresh,
    )

    entry = _base_entry(key, meta, verdict, extra_tags)
    if existing:
        entry.added = existing.added

    if ft is None:
        entry.status = STATUS_UNVERIFIED
        # A fetch that fails must not cost a reading already paid for: the
        # entry keeps it, and only the retrieval is reported as having failed.
        if existing:
            _carry_reading(entry, existing)
            _carry_code(entry, existing)
        kept = entry.is_verified
        warnings.append(
            "full text could not be retrieved — "
            + ("the reading already on record was kept."
               if kept else "summaries left blank.")
            + f" Drop the PDF in {lib.inbox_dir} and run `lit inbox`, "
            "or re-add with --pdf <file>."
        )
        saved = lib.save_entry(entry)
        return AddResult(
            "updated" if existing else "added",
            f"[{key}] kept its existing reading (no full text available)" if kept
            else f"[{key}] added as UNVERIFIED (no full text available)",
            entry=saved, verdict=verdict, meta=meta, warnings=warnings,
        )

    # ---- 5. read it ------------------------------------------------------
    budget = max_chars or ctx.read_budget
    # Bibliography and back matter both go: this call writes the library card,
    # and neither belongs on one. `ask` and `claim` keep the appendix, because
    # the evidence for a question may well be in a table there.
    text, truncated = truncate_for_llm(ft.text, budget, drop_references=True,
                                       drop_appendix=True)
    if truncated:
        warnings.append(
            f"document is {ft.chars:,} chars; the middle was omitted to fit context"
        )
    if ft.partial:
        warnings.append(
            f"long document: {ft.pages_read} of {ft.pages} pages were read "
            "(front matter, an even spread of the body, and the closing pages). "
            "The summary covers that sample, not the whole work. Raise the "
            "budget with `lit config set fetch.max_read_pages <n>`."
        )
    ctx.vlog(f"[{key}] reading {len(text):,} chars via {ft.source}")

    try:
        data = ctx.llm.json(
            read_paper_prompt(
                title=meta.title,
                scope=lib.settings.scope,
                known_venue=meta.venue,
                known_year=meta.year,
                truncated=truncated,
                needs_level=verdict.needs_judgement,
                sampled_pages=(ft.pages_read, ft.pages) if ft.partial else None,
            ),
            stdin_text="=== FULL TEXT BEGINS ===\n" + text,
            system=READER_SYSTEM,
            role="reader",
            required=("one_liner", "sections"),
        )
    except LLMError as exc:
        entry.status = STATUS_UNVERIFIED
        # The text was in hand and only the reader fell over, so the reading
        # already on record is still the best one there is: it stays. The fetch
        # itself succeeded, so its own facts below are the fresh ones.
        if existing:
            _carry_reading(entry, existing)
            _carry_code(entry, existing)
        kept = entry.is_verified
        entry.read_source = ft.source
        entry.pdf_path = _rel(ft.pdf_path, lib.path)
        entry.text_chars = ft.chars
        entry.pages = ft.pages
        entry.pages_read = ft.pages_read
        lib.save_entry(entry)
        return AddResult(
            "error",
            f"[{key}] retrieved the full text but the reading step failed: {exc}"
            + (" — the reading already on record was kept" if kept else ""),
            entry=entry, verdict=verdict, meta=meta, warnings=warnings,
        )

    _apply_reading(entry, data, meta, verdict, extra_tags, text)
    _carry_code(entry, existing)
    entry.status = STATUS_VERIFIED
    entry.read_source = ft.source
    entry.pdf_path = _rel(ft.pdf_path, lib.path)
    entry.text_chars = ft.chars
    entry.pages = ft.pages
    entry.pages_read = ft.pages_read

    # Re-check the bar now that the model may have supplied the missing level.
    if not force and verdict.needs_judgement:
        recheck = LevelVerdict(entry.level, entry.level_reason)
        ok, why = passes_quality_bar(
            recheck, meta, lib.settings.min_level, lib.settings.allow_non_published
        )
        if not ok:
            return AddResult(
                "rejected",
                f"{meta.title!r}: after reading it, {why}. Use --force to add anyway.",
                verdict=recheck, meta=meta,
            )

    saved = lib.save_entry(entry)
    return AddResult(
        "updated" if existing else "added",
        f"[{key}] {entry.title}",
        entry=saved, verdict=verdict, meta=meta, warnings=warnings,
    )


# --------------------------------------------------------------------------
# Entry assembly
# --------------------------------------------------------------------------

def _base_entry(key: str, meta: PaperMeta, verdict: LevelVerdict,
                extra_tags: list[str] | None) -> Entry:
    return Entry(
        key=key,
        title=meta.title,
        authors=list(meta.authors),
        year=meta.year,
        venue=meta.venue,
        type=meta.type,
        doi=meta.doi,
        arxiv_id=meta.arxiv_id,
        url=meta.url,
        status=STATUS_UNVERIFIED,
        level=verdict.level,
        level_reason=verdict.reason,
        # The publisher's own abstract. It comes from the metadata APIs, so an
        # entry that never gets read still carries it.
        abstract=meta.abstract,
        citation_count=meta.citation_count,
        references=list(meta.references),
        bibtex=meta.bibtex,
        tags=_norm_tags(extra_tags or []),
    )


def _reading_elsewhere(ctx: Ctx, meta: PaperMeta) -> tuple[str, Entry] | None:
    """A finished reading of this paper in another library under the same root.

    Identity is the same test `find_duplicate` uses within a library — DOI,
    arXiv id, then normalized title — so a paper is only recognised when an
    identifier or the title agrees, never on a loose match.
    """
    try:
        siblings = list_libraries(ctx.cfg)
    except Exception:
        return None
    here = ctx.library.path.resolve()
    for other in siblings:
        if other.path.resolve() == here:
            continue
        try:
            match = other.find_duplicate(
                doi=meta.doi, arxiv_id=meta.arxiv_id, title=meta.title
            )
        except Exception:
            continue
        if match is not None and match.is_verified and match.one_liner:
            return other.name, match
    return None


def _carry_reading(entry: Entry, existing: Entry) -> None:
    """Move an earlier read's output onto a freshly built entry.

    Used on the metadata-only path, where the new entry has correct
    bibliographic fields but no summaries at all.
    """
    if not existing.is_verified:
        return
    entry.status = existing.status
    entry.one_liner = existing.one_liner
    entry.sections = list(existing.sections)
    entry.key_findings = list(existing.key_findings)
    entry.tags = _norm_tags(list(existing.tags) + list(entry.tags))
    _carry_code(entry, existing)
    entry.read_source = existing.read_source
    entry.pdf_path = existing.pdf_path
    entry.text_chars = existing.text_chars
    entry.pages = existing.pages
    entry.pages_read = existing.pages_read
    if existing.venue and not entry.venue:
        entry.venue = existing.venue


def _carry_code(entry: Entry, existing: Entry | None) -> None:
    """Keep a code link this pass could not find for itself.

    `lit code` records a repository the paper's own text never printed, so a
    re-read — which only ever takes what it can see in the text — would drop
    something already paid for. A link the paper prints is the stronger evidence
    and always wins; this only fills a gap.
    """
    if existing is None or not existing.code_url or entry.code_url:
        return
    entry.code_url = existing.code_url
    entry.code_source = existing.code_source
    entry.code_reason = existing.code_reason


def _apply_reading(entry: Entry, data: dict, meta: PaperMeta,
                   verdict: LevelVerdict, extra_tags: list[str] | None,
                   text: str) -> None:
    """Copy the reader agent's output onto the entry, with light validation."""
    one = str(data.get("one_liner") or "").strip()
    entry.one_liner = one or None

    # Capped rather than trusted. The prompt asks for a bounded record, but the
    # bound is what keeps a stored summary small, and a stored summary is quoted
    # back in every later prompt that ranks or searches the library — so a
    # reader that ignores the limit would go on costing tokens after the read.
    entry.sections = [
        s for s in (Section.coerce(x) for x in (data.get("sections") or [])) if s
    ][:MAX_SECTIONS]
    for s in entry.sections:
        s.summary = _clip_words(s.summary, SECTION_WORDS_CEILING)
    entry.key_findings = [
        str(f).strip() for f in (data.get("key_findings") or []) if str(f).strip()
    ][:MAX_KEY_FINDINGS]

    tags = _norm_tags(list(data.get("tags") or []) + list(extra_tags or []))
    entry.tags = tags

    kind = str(data.get("type") or "").strip().lower()
    # Metadata APIs normally know the publication type better than the text does,
    # so the model's answer is only taken when the APIs had nothing to say.
    if kind in ENTRY_TYPES and entry.type in ("other", ""):
        entry.type = kind

    # An arXiv-only record has no venue, but the paper's own first page usually
    # names one ("31st Conference on Neural Information Processing Systems").
    # When the reader recovers it, the type follows — it read the header, and a
    # record filed as a preprint by arXiv is really the conference version. The
    # BibTeX is then rebuilt so it comes out as @inproceedings rather than @misc.
    from_text = str(data.get("venue_from_text") or "").strip()
    if not entry.venue and from_text:
        entry.venue = from_text
        if kind in ENTRY_TYPES and kind != "other" and entry.type == "preprint":
            entry.type = kind
        # Only regenerate a BibTeX entry we synthesized ourselves; a Crossref
        # one is authoritative and must not be overwritten.
        if not meta.doi:
            regenerated = replace(meta, venue=entry.venue, type=entry.type)
            entry.bibtex = synth_bibtex(regenerated)

    entry.code_url = _code_url_from_reading(data.get("code_url"), text)
    # Only stamped when the paper itself printed the link. A read that finds
    # none must also clear a stale provenance, or a re-read of a paper whose
    # repository was found by `lit code` would leave the note without the URL.
    entry.code_source = CODE_FROM_PAPER if entry.code_url else None
    entry.code_reason = ""

    if verdict.needs_judgement:
        lvl = str(data.get("level") or "").strip()
        if lvl in ("A*", "A", "B", "C"):
            entry.level = lvl
            reason = str(data.get("level_reason") or "").strip()
            entry.level_reason = (
                f"judged from content: {reason}" if reason
                else "judged from paper content (no citation metrics yet)"
            )

    # References come from the metadata APIs, never from the model — a model
    # asked to list references will happily hallucinate plausible ones.
    entry.references = list(meta.references)


def _code_url_from_reading(raw: object, text: str) -> str | None:
    """Accept the reader's repository link only if the paper really prints it.

    Asked for a repo, a model will otherwise offer a plausible one from memory —
    the right lab, a URL that never appears in the paper and often does not
    exist. So the link has to be found in the very text the model was given.
    Both sides are folded to bare alphanumerics before comparing, because PDF
    extraction routinely breaks a long URL across lines or drops its scheme.
    """
    url = normalize_code_url(raw if isinstance(raw, str) else None)
    if not url:
        return None
    needle = _fold(url.split("://", 1)[-1])
    return url if needle and needle in _fold(text) else None


def _fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _clip_words(text: str, limit: int) -> str:
    """Cut a summary to `limit` words, on a sentence boundary where there is one."""
    words = (text or "").split()
    if len(words) <= limit:
        return text
    clipped = " ".join(words[:limit])
    stop = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if stop > len(clipped) * 0.5:
        return clipped[:stop + 1]
    return clipped.rstrip(",;:") + "…"


def _norm_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for t in tags:
        t = str(t).strip().lower().replace(" ", "-").strip("-")
        if t and t not in out:
            out.append(t)
    return out[:12]


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def link_references(ctx: Ctx, entry: Entry) -> int:
    """Point this entry's references at library entries where they match."""
    lib = ctx.library
    linked = 0
    for ref in entry.references:
        if ref.key:
            continue
        match = lib.find_duplicate(doi=ref.doi, arxiv_id=ref.arxiv_id, title=ref.title)
        if match:
            ref.key = match.key
            linked += 1
    if linked:
        lib.save_entry(entry)
    return linked


def reread(ctx: Ctx, key: str, *, local_pdf: Path | None = None) -> AddResult:
    """Re-run the reading step for an entry already in the library."""
    entry = ctx.library.get(key)
    if entry is None:
        return AddResult("not_found", f"no entry with key {key!r}")
    ident = entry.doi or entry.arxiv_id or entry.title
    return add_paper(
        ctx, ident,
        local_pdf=local_pdf,
        refresh=True,
        force=True,
        target=Target(doi=entry.doi, arxiv_id=entry.arxiv_id, title=entry.title),
    )


def read_entries(ctx: Ctx, entries: list[Entry]) -> list[AddResult]:
    """Fetch and read a batch of entries concurrently, one agent per paper.

    This is the other half of the deal `find` makes: papers are filed cheaply
    from their metadata, and the expensive read is spent later, deliberately,
    on the ones that turned out to matter.
    """
    if not entries:
        return []

    ctx.log(f"[bold]Reading[/bold] {len(entries)} papers "
            f"({ctx.workers} at a time)…")

    def work(e: Entry) -> AddResult:
        return reread(ctx, e.key)

    def report(r) -> None:
        entry: Entry = r.item
        if not r.ok:
            ctx.log(f"  [red]![/red] [{entry.key}] {r.error}")
            return
        res: AddResult = r.value
        icon = "[green]+[/green]" if res.ok else "[yellow]-[/yellow]"
        ctx.log(f"  {icon} {res.message}")

    runs = run_parallel(entries, work, workers=ctx.workers, on_done=report)
    out: list[AddResult] = []
    for r in runs:
        if r.ok and r.value is not None:
            out.append(r.value)
        elif r.error is not None:
            entry: Entry = r.item  # type: ignore[assignment]
            out.append(AddResult("error", f"[{entry.key}] {r.error}"))
    return out
