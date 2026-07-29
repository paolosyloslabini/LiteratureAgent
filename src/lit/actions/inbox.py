"""`lit inbox` — adopt PDFs you dropped in `pdfs/inbox/`.

This is the escape hatch for paywalled work: if you have institutional access,
you download the PDF yourself, drop it in the inbox, and the agent reads it from
disk exactly as it would an open-access one.

Each PDF is identified from its own first page (DOI/arXiv id by regex, else the
title line), matched against existing UNVERIFIED entries, and either fills in
that entry or becomes a new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..fetch.fulltext import pdf_to_text
from ..models import Entry, normalize_arxiv, normalize_doi, slugify
from ..runner import run_parallel
from .add import AddResult, add_paper
from .context import Ctx, Target

# How much of the PDF to look at when working out what it is.
_SNIFF_CHARS = 6000


@dataclass
class InboxItem:
    path: Path
    doi: str | None = None
    arxiv_id: str | None = None
    title_guess: str | None = None
    matched_key: str | None = None
    result: AddResult | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "file": self.path.name,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title_guess": self.title_guess,
            "matched_entry": self.matched_key,
            "status": self.result.status if self.result else ("error" if self.error else "skipped"),
            "message": self.result.message if self.result else self.error,
        }


@dataclass
class InboxResult:
    items: list[InboxItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"processed": [i.to_dict() for i in self.items]}


def process_inbox(ctx: Ctx, *, keep: bool = False,
                  only: list[Path] | None = None) -> InboxResult:
    lib = ctx.library
    lib.inbox_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(only) if only else sorted(lib.inbox_dir.glob("*.pdf"))
    result = InboxResult()

    if not pdfs:
        return result

    ctx.log(f"[bold]Inbox[/bold]: {len(pdfs)} PDF(s) to process")

    # Identification is local and cheap; do it up front so matching can consider
    # all of them together.
    items = [_identify(p) for p in pdfs]
    unverified = [e for e in lib.iter_entries() if not e.is_verified]

    for item in items:
        match = _match_entry(item, unverified, lib)
        item.matched_key = match.key if match else None

    def work(item: InboxItem) -> InboxItem:
        if item.error:
            return item
        entry = lib.get(item.matched_key) if item.matched_key else None
        if entry is not None:
            ident = entry.doi or entry.arxiv_id or entry.title
            target = Target(doi=entry.doi, arxiv_id=entry.arxiv_id, title=entry.title)
        else:
            ident = item.doi or item.arxiv_id or item.title_guess or item.path.stem
            target = Target(doi=item.doi, arxiv_id=item.arxiv_id,
                            title=item.title_guess or item.path.stem)
        item.result = add_paper(
            ctx, ident,
            target=target,
            local_pdf=item.path,
            refresh=True,
            force=True,  # you went and got this PDF; that is the quality signal
        )
        return item

    def report(r) -> None:
        item: InboxItem = r.item  # type: ignore[assignment]
        if not r.ok:
            ctx.log(f"  [red]![/red] {item.path.name}: {r.error}")
            return
        it: InboxItem = r.value
        if it.error:
            ctx.log(f"  [red]![/red] {it.path.name}: {it.error}")
        elif it.result and it.result.ok:
            verb = "filled in" if it.matched_key else "added"
            ctx.log(f"  [green]✓[/green] {it.path.name} → {verb} {it.result.message}")
        else:
            ctx.log(f"  [yellow]-[/yellow] {it.path.name}: "
                    f"{it.result.message if it.result else 'unhandled'}")

    runs = run_parallel(items, work, workers=ctx.workers, on_done=report)
    result.items = [r.value if r.ok and r.value else r.item for r in runs]  # type: ignore[misc]

    if not keep:
        for item in result.items:
            if item.result and item.result.ok and item.path.exists():
                # The PDF now lives under pdfs/<key>.pdf; clear the inbox copy.
                try:
                    item.path.unlink()
                except OSError:
                    pass
    return result


def _identify(path: Path) -> InboxItem:
    item = InboxItem(path=path)
    # Only the first pages are needed to identify a document; extracting a
    # 400-page book to read its title page is pure waste.
    text = pdf_to_text(path, max_pages=3)
    if not text:
        item.error = "no extractable text (scanned image PDF?)"
        return item

    head = text[:_SNIFF_CHARS]
    item.doi = normalize_doi(head)
    item.arxiv_id = normalize_arxiv_from_header(head)
    item.title_guess = _guess_title(head)
    return item


def normalize_arxiv_from_header(head: str) -> str | None:
    """Find an arXiv id, preferring the stamp arXiv puts down the left margin."""
    m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", head, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s]+)", head, re.IGNORECASE)
    if m:
        return normalize_arxiv(m.group(1))
    return None


def _guess_title(head: str) -> str | None:
    """The title is usually the first substantial line that isn't boilerplate."""
    skip = re.compile(
        r"^(arxiv|doi|http|www|abstract|keywords|proceedings of|published|"
        r"preprint|under review|accepted at|copyright|\d+$|acm|ieee)",
        re.IGNORECASE,
    )
    lines = [l.strip() for l in head.splitlines()]
    buf: list[str] = []
    for line in lines[:60]:
        if not line:
            if buf:
                break
            continue
        if skip.match(line) or len(line) < 8:
            if buf:
                break
            continue
        # A line that is mostly names is the author block, not the title.
        if buf and re.search(r"\b(and|,)\b", line) and len(line.split()) < 12:
            break
        buf.append(line)
        if sum(len(b) for b in buf) > 200:
            break
    title = " ".join(buf).strip()
    title = re.sub(r"\s+", " ", title)
    return title if 8 <= len(title) <= 300 else None


def _match_entry(item: InboxItem, unverified: list[Entry], lib) -> Entry | None:
    """Match a PDF to an existing entry, preferring UNVERIFIED ones."""
    if item.doi or item.arxiv_id:
        for e in unverified:
            if item.doi and e.doi and e.doi.lower() == item.doi.lower():
                return e
            if item.arxiv_id and e.arxiv_id == item.arxiv_id:
                return e
        hit = lib.find_duplicate(doi=item.doi, arxiv_id=item.arxiv_id)
        if hit:
            return hit

    if item.title_guess:
        target = slugify(item.title_guess, 200)
        for e in unverified:
            es = slugify(e.title, 200)
            if es and (es == target or es in target or target in es):
                return e
        hit = lib.find_duplicate(title=item.title_guess)
        if hit:
            return hit

    # Filenames like "vaswani2017attention.pdf" often name the key outright.
    stem = slugify(item.path.stem, 60)
    for e in unverified:
        if stem and stem == slugify(e.key, 60):
            return e
    return None
