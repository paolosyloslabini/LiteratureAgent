"""`lit browse` — an optional interactive browser over a library.

This is a convenience layer, not a dependency: every capability it exposes is
also a subcommand, which is what scripts and agents use. Nothing here is
required to use the tool.
"""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Markdown,
    Static,
    TextArea,
)

from . import entryfile
from .config import Config
from .library import Library
from .models import Entry, level_rank
from .store import Store


class NotesEditor(ModalScreen[str | None]):
    """Full-screen editor for the user's notes on an entry."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, entry: Entry):
        super().__init__()
        self.entry = entry

    def compose(self) -> ComposeResult:
        with Vertical(id="notes-box"):
            yield Static(f"Notes — {self.entry.title[:70]}", id="notes-title")
            yield TextArea(self.entry.notes, id="notes-area", language="markdown")
            yield Static("[dim]ctrl+s save · esc cancel[/dim]", id="notes-help")

    def on_mount(self) -> None:
        self.query_one("#notes-area", TextArea).focus()

    def action_save(self) -> None:
        self.dismiss(self.query_one("#notes-area", TextArea).text)

    def action_cancel(self) -> None:
        self.dismiss(None)


class BrowserApp(App):
    """Two-pane browser: entry list on top, detail below."""

    CSS = """
    Screen { layers: base overlay; }
    #list { height: 45%; border: round $accent; }
    #detail { height: 1fr; border: round $primary; padding: 0 1; }
    #search { dock: top; display: none; }
    #search.visible { display: block; }
    #status { dock: bottom; height: 1; padding: 0 1; color: $text-muted; }
    #notes-box {
        width: 80%; height: 70%; margin: 4 8;
        border: thick $accent; background: $surface;
    }
    #notes-title { padding: 0 1; background: $accent; color: $text; }
    #notes-area { height: 1fr; }
    #notes-help { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear", show=False),
        Binding("n", "edit_notes", "Notes"),
        Binding("o", "open_link", "Open"),
        Binding("f", "cycle_filter", "Filter"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("r", "reload", "Reload"),
    ]

    FILTERS = [
        ("all", lambda e: True),
        ("A* and A", lambda e: level_rank(e.level) <= level_rank("A")),
        ("unread", lambda e: not e.is_verified),
        ("with notes", lambda e: bool(e.notes.strip())),
    ]
    SORTS = [
        ("level", lambda e: (level_rank(e.level), -(e.year or 0))),
        ("year", lambda e: -(e.year or 0)),
        ("title", lambda e: e.title.lower()),
        ("citations", lambda e: -(e.citation_count or 0)),
    ]

    def __init__(self, library: Library, cfg: Config):
        super().__init__()
        self.library = library
        self.cfg = cfg
        self.entries: list[Entry] = []
        self.shown: list[Entry] = []
        self.query = ""
        self.filter_i = 0
        self.sort_i = 0

    # ---------------- layout ----------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="search… (enter to apply, esc to clear)", id="search")
        with Vertical():
            with Horizontal(id="list"):
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Markdown("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.library.name
        self.sub_title = self.library.settings.scope or ""
        table = self.query_one("#table", DataTable)
        table.add_columns("key", "title", "venue", "yr", "lvl", "cites", "status")
        self.action_reload()

    # ---------------- data ----------------

    def action_reload(self) -> None:
        self.entries = self.library.entries()
        self.refresh_rows()

    def refresh_rows(self) -> None:
        name, pred = self.FILTERS[self.filter_i]
        rows = [e for e in self.entries if pred(e)]

        if self.query:
            with Store(self.library) as store:
                order = [h.key for h in store.search(self.query, limit=500)]
            rank = {k: i for i, k in enumerate(order)}
            rows = [e for e in rows if e.key in rank]
            rows.sort(key=lambda e: rank[e.key])
        else:
            rows.sort(key=self.SORTS[self.sort_i][1])

        self.shown = rows
        table = self.query_one("#table", DataTable)
        table.clear()
        for e in rows:
            table.add_row(
                e.key,
                e.title[:80],
                (e.venue or ("arXiv" if e.arxiv_id else "—"))[:20],
                str(e.year or "—"),
                e.level,
                str(e.citation_count) if e.citation_count is not None else "—",
                "read" if e.is_verified else "UNVERIFIED",
                key=e.key,
            )
        self.query_one("#status", Static).update(
            f"{len(rows)}/{len(self.entries)} entries · filter: {name} · "
            f"sort: {self.SORTS[self.sort_i][0]}"
            + (f" · query: {self.query!r}" if self.query else "")
        )
        if rows:
            table.move_cursor(row=0)
            self.show_detail(rows[0])
        else:
            self.query_one("#detail", Markdown).update("_Nothing matches._")

    def show_detail(self, entry: Entry) -> None:
        parts = [
            f"# {entry.title}",
            "",
            f"**{entry.citation()}** · {entry.type} · level {entry.level}"
            + (f" · {entry.citation_count} citations"
               if entry.citation_count is not None else "")
            + ("" if entry.is_verified else " · **UNVERIFIED**"),
        ]
        if entry.level_reason:
            parts.append(f"*{entry.level_reason}*")
        if entry.one_liner:
            parts += ["", f"> {entry.one_liner}"]
        elif not entry.is_verified:
            parts += ["", "> _Full text unavailable — no summary was written._"]
        if entry.tags:
            parts += ["", "Tags: " + ", ".join(f"`{t}`" for t in entry.tags)]
        if entry.code_url:
            parts += ["", f"Code: {entry.code_url}"]
        if entry.abstract.strip():
            parts += ["", "## Abstract", "", entry.abstract.strip()]
        if entry.key_findings:
            parts += ["", "## Key findings", ""]
            parts += [f"- {f}" for f in entry.key_findings]
        if entry.sections:
            parts += ["", "## Detailed summary", ""]
            for s in entry.sections:
                parts += [f"### {s.name}", "", s.summary, ""]
        if entry.notes.strip():
            parts += ["", "## Your notes", "", entry.notes.strip()]
        if entry.references:
            parts += ["", f"## References ({len(entry.references)})", ""]
            parts += [
                f"- {r.title}" + (f" ({r.year})" if r.year else "")
                + (f" → `{r.key}`" if r.key else "")
                for r in entry.references[:30]
            ]
        self.query_one("#detail", Markdown).update("\n".join(parts))

    def current(self) -> Entry | None:
        table = self.query_one("#table", DataTable)
        if not self.shown or table.cursor_row is None:
            return None
        if 0 <= table.cursor_row < len(self.shown):
            return self.shown[table.cursor_row]
        return None

    # ---------------- events ----------------

    @on(DataTable.RowHighlighted)
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.shown):
            self.show_detail(self.shown[event.cursor_row])

    @on(Input.Submitted, "#search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        self.query = event.value.strip()
        self.query_one("#search", Input).remove_class("visible")
        self.query_one("#table", DataTable).focus()
        self.refresh_rows()

    # ---------------- actions ----------------

    def action_focus_search(self) -> None:
        box = self.query_one("#search", Input)
        box.add_class("visible")
        box.focus()

    def action_clear_search(self) -> None:
        self.query = ""
        box = self.query_one("#search", Input)
        box.value = ""
        box.remove_class("visible")
        self.query_one("#table", DataTable).focus()
        self.refresh_rows()

    def action_cycle_filter(self) -> None:
        self.filter_i = (self.filter_i + 1) % len(self.FILTERS)
        self.refresh_rows()

    def action_cycle_sort(self) -> None:
        self.sort_i = (self.sort_i + 1) % len(self.SORTS)
        self.refresh_rows()

    def action_edit_notes(self) -> None:
        entry = self.current()
        if entry is None:
            return

        def save(notes: str | None) -> None:
            if notes is None:
                return
            # Goes through set_notes, the one sanctioned path for user notes.
            entryfile.set_notes(self.library.entry_path(entry.key), notes.strip())
            self.action_reload()

        self.push_screen(NotesEditor(entry), save)

    def action_open_link(self) -> None:
        import webbrowser

        entry = self.current()
        if entry is None:
            return
        url = entry.url or (
            f"https://doi.org/{entry.doi}" if entry.doi
            else f"https://arxiv.org/abs/{entry.arxiv_id}" if entry.arxiv_id else None
        )
        if url:
            webbrowser.open(url)
            self.notify(f"opened {url}")
        else:
            self.notify("no link for this entry", severity="warning")


def run_browser(library: Library, cfg: Config) -> None:
    BrowserApp(library, cfg).run()
