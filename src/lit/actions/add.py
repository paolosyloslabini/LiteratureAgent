"""`lit add` — put one paper into the library.

The spec is strict about this path, and so is the code:

* The full text must be read. Metadata alone never produces a summary.
* The agent that reads the paper is the one that fills the fields — there is a
  single LLM call holding the whole text, not a read step and a separate write
  step working off notes.
* A paper that cannot be retrieved is stored anyway, flagged UNVERIFIED, with
  its summaries left blank. Nothing is inferred from the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from ..fetch.fulltext import fetch_fulltext, truncate_for_llm
from ..fetch.metadata import PaperMeta, resolve_metadata, synth_bibtex
from ..llm import LLMError
from ..models import (
    ENTRY_TYPES,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    Entry,
    Reference,
    Section,
    make_key,
)
from ..prompts import READER_SYSTEM, read_paper_prompt
from ..quality import LevelVerdict, assess, passes_quality_bar
from .context import Ctx, Target, parse_target

# How much paper text we hand the model in one call.
MAX_PROMPT_CHARS = 400_000


@dataclass
class AddResult:
    status: str  # added | updated | duplicate | rejected | not_found | error
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
) -> AddResult:
    """Resolve, vet, read and store one paper."""
    lib = ctx.library
    warnings: list[str] = []
    target = target or parse_target(query)

    # ---- 1. metadata -----------------------------------------------------
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
        warnings.append(
            "full text could not be retrieved — summaries left blank. "
            f"Drop the PDF in {lib.inbox_dir} and run `lit inbox`, "
            "or re-add with --pdf <file>."
        )
        saved = lib.save_entry(entry)
        return AddResult(
            "updated" if existing else "added",
            f"[{key}] added as UNVERIFIED (no full text available)",
            entry=saved, verdict=verdict, meta=meta, warnings=warnings,
        )

    # ---- 5. read it ------------------------------------------------------
    text, truncated = truncate_for_llm(ft.text, MAX_PROMPT_CHARS)
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
        entry.read_source = ft.source
        entry.pdf_path = _rel(ft.pdf_path, lib.path)
        entry.text_chars = ft.chars
        entry.pages = ft.pages
        entry.pages_read = ft.pages_read
        lib.save_entry(entry)
        return AddResult(
            "error",
            f"[{key}] retrieved the full text but the reading step failed: {exc}",
            entry=entry, verdict=verdict, meta=meta, warnings=warnings,
        )

    _apply_reading(entry, data, meta, verdict, extra_tags)
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
        citation_count=meta.citation_count,
        references=list(meta.references),
        bibtex=meta.bibtex,
        tags=_norm_tags(extra_tags or []),
    )


def _apply_reading(entry: Entry, data: dict, meta: PaperMeta,
                   verdict: LevelVerdict, extra_tags: list[str] | None) -> None:
    """Copy the reader agent's output onto the entry, with light validation."""
    one = str(data.get("one_liner") or "").strip()
    entry.one_liner = one or None

    entry.sections = [
        s for s in (Section.coerce(x) for x in (data.get("sections") or [])) if s
    ]
    entry.key_findings = [
        str(f).strip() for f in (data.get("key_findings") or []) if str(f).strip()
    ]

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


def make_reference(meta: PaperMeta) -> Reference:
    return Reference(
        title=meta.title, year=meta.year, doi=meta.doi, arxiv_id=meta.arxiv_id,
        authors=list(meta.authors),
    )
