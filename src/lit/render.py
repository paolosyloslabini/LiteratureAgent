"""Rich-based terminal rendering for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import STATUS_VERIFIED, Entry

LEVEL_STYLE = {
    "A*": "bold green", "A": "green", "B": "yellow", "C": "red",
    "unranked": "dim",
}


def level_text(level: str | None) -> Text:
    return Text(level or "unranked", style=LEVEL_STYLE.get(level or "", "dim"))


def status_text(status: str) -> Text:
    if status == STATUS_VERIFIED:
        return Text("read", style="green")
    return Text("UNVERIFIED", style="bold yellow")


def entries_table(entries: list[Entry], *, title: str | None = None,
                  show_why: dict[str, str] | None = None) -> Table:
    """A scannable one-line-per-entry table.

    Titles are truncated rather than wrapped: a list is for scanning, and a
    wrapped title turns every row into a paragraph. When relevance notes are
    supplied they go under the title in the same cell instead of in a column of
    their own, which would squeeze both into unreadable slivers.
    """
    # Only spend the extra vertical space when there are notes to actually show;
    # `--raw` search passes the dict with every value empty.
    has_why = bool(show_why) and any(v.strip() for v in show_why.values())
    t = Table(title=title, header_style="bold", expand=True,
              show_lines=has_why, pad_edge=False)
    t.add_column("key", style="cyan", no_wrap=True)
    t.add_column("title", ratio=4, overflow="ellipsis", no_wrap=not has_why)
    t.add_column("venue", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("yr", justify="right", no_wrap=True)
    t.add_column("lvl", no_wrap=True)
    t.add_column("cites", justify="right", no_wrap=True)
    t.add_column("status", no_wrap=True)

    for e in entries:
        title_cell = e.title
        if has_why and (why := show_why.get(e.key, "").strip()):
            title_cell = f"{e.title}\n[dim]{why}[/dim]"
        t.add_row(
            e.key,
            title_cell,
            e.venue or ("arXiv" if e.arxiv_id else "—"),
            str(e.year or "—"),
            level_text(e.level),
            str(e.citation_count) if e.citation_count is not None else "—",
            status_text(e.status),
        )
    return t


def entry_panel(entry: Entry, *, full: bool = False) -> Panel:
    lines: list[str] = []
    ident = " · ".join(filter(None, [
        f"doi:{entry.doi}" if entry.doi else "",
        f"arXiv:{entry.arxiv_id}" if entry.arxiv_id else "",
    ]))
    meta = " · ".join(filter(None, [
        entry.citation(),
        entry.type,
        f"level {entry.level}",
        f"{entry.citation_count} citations" if entry.citation_count is not None else "",
        entry.status if entry.status != STATUS_VERIFIED else "",
    ]))
    lines.append(f"**{meta}**")
    if ident:
        lines.append(ident)
    if entry.level_reason:
        lines.append(f"*Level: {entry.level_reason}*")
    if entry.url:
        lines.append(entry.url)

    if entry.one_liner:
        lines += ["", f"> {entry.one_liner}"]
    elif not entry.is_verified:
        lines += ["", "> _No summary: the full text could not be retrieved._"]

    if entry.tags:
        lines += ["", "Tags: " + ", ".join(f"`{t}`" for t in entry.tags)]

    if entry.key_findings:
        lines += ["", "**Key findings**", ""]
        lines += [f"- {f}" for f in entry.key_findings]

    if full and entry.sections:
        lines += ["", "**Detailed summary**", ""]
        for s in entry.sections:
            lines += [f"*{s.name}*", "", s.summary, ""]

    if full and entry.references:
        lines += ["", f"**References** ({len(entry.references)})", ""]
        for r in entry.references[:40]:
            mark = f" → `{r.key}`" if r.key else ""
            lines.append(f"- {r.title}" + (f" ({r.year})" if r.year else "") + mark)
        if len(entry.references) > 40:
            lines.append(f"- …and {len(entry.references) - 40} more")

    if entry.notes.strip():
        lines += ["", "**Your notes**", "", entry.notes.strip()]
    elif full:
        lines += ["", f"*No notes yet — add some with* `lit note {entry.key}`"]

    if full:
        prov = " · ".join(filter(None, [
            f"read via {entry.read_source}" if entry.read_source else "",
            f"{entry.text_chars:,} chars" if entry.text_chars else "",
            f"pdf: {entry.pdf_path}" if entry.pdf_path else "",
            f"added {entry.added}",
        ]))
        if prov:
            lines += ["", f"*{prov}*"]

    return Panel(
        Markdown("\n".join(lines)),
        title=f"[cyan]{entry.key}[/cyan] — {entry.title}",
        border_style=LEVEL_STYLE.get(entry.level, "dim").replace("bold ", ""),
    )


def print_answer(console: Console, *, question: str, answer: str,
                 bibliography: list[dict], footer: str = "") -> None:
    console.print(Panel(Text(question, style="bold"), border_style="cyan",
                        title="Question"))
    console.print()
    console.print(Markdown(answer or "_(no answer)_"))
    if bibliography:
        console.print()
        t = Table(title="Bibliography", header_style="bold", expand=True)
        t.add_column("key", style="cyan", no_wrap=True)
        t.add_column("reference", ratio=3)
        t.add_column("lvl", no_wrap=True)
        t.add_column("link", ratio=1, no_wrap=True)
        for b in bibliography:
            link = (f"doi:{b['doi']}" if b.get("doi")
                    else f"arXiv:{b['arxiv_id']}" if b.get("arxiv_id") else "—")
            authors = ", ".join(b.get("authors") or [])[:60]
            ref = f"{b['title']}\n[dim]{authors}[/dim]"
            t.add_row(b["key"], ref, level_text(b.get("level")), link)
        console.print(t)
    if footer:
        console.print(f"\n[dim]{footer}[/dim]")


def chain_tree(chain: list, claim: str) -> Panel:
    lines = [f"**Claim:** {claim}", ""]
    for i, hop in enumerate(chain):
        arrow = "" if i == 0 else "↓ cites"
        if arrow:
            lines += [f"    *{arrow}*", ""]
        tag = " **← ORIGIN**" if hop.is_origin else ""
        loc = f" `{hop.library_key}`" if hop.library_key else ""
        ident = (f"doi:{hop.doi}" if hop.doi
                 else f"arXiv:{hop.arxiv_id}" if hop.arxiv_id else "")
        lines.append(f"**{i + 1}. {hop.title}** ({hop.year or 'n.d.'}){tag}{loc}")
        if ident:
            lines.append(f"   {ident}")
        if hop.quote:
            lines.append(f'   > "{hop.quote}"')
        if hop.reasoning:
            lines.append(f"   *{hop.reasoning}*")
        lines.append("")
    return Panel(Markdown("\n".join(lines)), title="Citation chain",
                 border_style="cyan")


def _short(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"
