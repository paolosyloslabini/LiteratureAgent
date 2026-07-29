"""`lit refresh` — re-fetch bibliographic metadata without re-reading papers.

A library outlives the day it was built: citation counts move, preprints get
published, and reference lists that a source hadn't indexed yet appear later.
This walks existing entries and updates the machine-owned bibliographic fields
from the metadata APIs.

It is network-only — no LLM call, so it is fast, free and parallel. It never
touches summaries (those require reading the paper, which is `lit reread`) and
never touches user notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fetch.metadata import resolve_metadata
from ..models import Entry
from ..quality import assess
from ..runner import run_parallel
from ..venue import is_placeholder
from .context import Ctx


@dataclass
class RefreshResult:
    key: str
    updated: bool = False
    changes: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "updated": self.updated,
                "changes": self.changes, "error": self.error}


def refresh_entries(ctx: Ctx, keys: list[str] | None = None,
                    *, relevel: bool = True) -> list[RefreshResult]:
    lib = ctx.library
    entries = [e for e in (lib.get(k) for k in keys) if e] if keys else lib.entries()
    if not entries:
        return []

    ctx.log(f"[bold]Refreshing[/bold] metadata for {len(entries)} entries…")

    def work(entry: Entry) -> RefreshResult:
        res = RefreshResult(key=entry.key)
        meta = resolve_metadata(
            ctx.http,
            doi=entry.doi,
            arxiv_id=entry.arxiv_id,
            title=None if (entry.doi or entry.arxiv_id) else entry.title,
            cfg=ctx.cfg.fetch,
        )
        if meta is None or not meta.title:
            res.error = "no bibliographic record found"
            return res

        if meta.citation_count is not None and meta.citation_count != entry.citation_count:
            res.changes.append(
                f"citations {entry.citation_count or 0} → {meta.citation_count}"
            )
            entry.citation_count = meta.citation_count

        if len(meta.references) > len(entry.references):
            res.changes.append(
                f"references {len(entry.references)} → {len(meta.references)}"
            )
            entry.references = list(meta.references)

        # A preprint that has since been published gains a DOI and a real venue.
        if meta.doi and not entry.doi:
            res.changes.append(f"DOI found: {meta.doi}")
            entry.doi = meta.doi
        # A venue that names a publisher or a preprint server is replaced, not
        # just filled. Refreshing only an empty venue is why an entry stored as
        # "arXiv" stayed that way after the paper appeared at a conference: the
        # field was occupied, so the real venue never got in.
        if meta.venue and not is_placeholder(meta.venue) and (
            not entry.venue or is_placeholder(entry.venue)
        ):
            res.changes.append(
                f"venue: {meta.venue}" if not entry.venue
                else f"venue {entry.venue!r} → {meta.venue!r}"
            )
            entry.venue = meta.venue
        if meta.bibtex and meta.doi and meta.bibtex != entry.bibtex:
            res.changes.append("bibtex updated")
            entry.bibtex = meta.bibtex
        if meta.url and not entry.url:
            entry.url = meta.url
        # Backfills entries stored before the abstract was kept, and fills in
        # one that simply wasn't published yet when the entry was added.
        if meta.abstract and not entry.abstract.strip():
            res.changes.append("abstract added")
            entry.abstract = meta.abstract

        # Re-run the level scorer now that the metrics may have arrived. A level
        # the model judged from the paper's content is only overwritten once
        # real metrics exist to overwrite it with.
        if relevel:
            verdict = assess(meta)
            if not verdict.needs_judgement and verdict.level != entry.level:
                res.changes.append(f"level {entry.level} → {verdict.level}")
                entry.level = verdict.level
                entry.level_reason = verdict.reason

        if res.changes:
            # save_entry restores user notes from disk; summaries are untouched
            # because nothing above writes to them.
            lib.save_entry(entry)
            res.updated = True
        return res

    def report(r) -> None:
        if not r.ok:
            entry: Entry = r.item  # type: ignore[assignment]
            ctx.log(f"  [red]![/red] {entry.key}: {r.error}")
            return
        out: RefreshResult = r.value
        if out.error:
            ctx.log(f"  [yellow]-[/yellow] {out.key}: {out.error}")
        elif out.updated:
            ctx.log(f"  [green]~[/green] {out.key}: {'; '.join(out.changes)}")
        else:
            ctx.vlog(f"  [dim]·[/dim] {out.key}: unchanged")

    runs = run_parallel(
        entries, work,
        # Network-bound, no LLM: more concurrency than the reading fan-out.
        workers=max(ctx.workers, 8),
        on_done=report,
    )

    out: list[RefreshResult] = []
    for r in runs:
        if r.ok and r.value is not None:
            out.append(r.value)
        else:
            entry: Entry = r.item  # type: ignore[assignment]
            out.append(RefreshResult(key=entry.key, error=str(r.error)))
    return out
