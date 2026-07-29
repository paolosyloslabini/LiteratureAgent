"""`lit browse` — an optional interactive browser over a library.

This is a convenience layer, not a dependency: every capability it exposes is
also a subcommand, which is what scripts and agents use. Nothing here is
required to use the tool.
"""

from __future__ import annotations

from functools import partial

from rich.console import Console
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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
from .actions.add import reread
from .actions.code import find_code
from .actions.context import Ctx
from .config import Config
from .library import Library
from .models import Entry, level_rank
from .openurl import open_url
from .store import Store


def paper_url(entry: Entry) -> str | None:
    """The best public link to the paper itself, or None if it has none."""
    return entry.url or (
        f"https://doi.org/{entry.doi}" if entry.doi
        else f"https://arxiv.org/abs/{entry.arxiv_id}" if entry.arxiv_id else None
    )


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


class ConfirmRead(ModalScreen[bool]):
    """Reading a paper fetches its full text and spends a reader agent on it.

    That is the expensive step the CLI keeps deliberate (`lit read <key>`), so
    the browser does not let one stray keypress buy it either.
    """

    BINDINGS = [
        Binding("y", "confirm", "Read"),
        Binding("enter", "confirm", "Read", show=False),
        Binding("escape", "cancel", "Cancel"),
        Binding("n", "cancel", "Cancel", show=False),
    ]

    def __init__(self, entry: Entry):
        super().__init__()
        self.entry = entry

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static("Read this paper in full?", id="confirm-title")
            yield Static(
                f"{self.entry.title[:200]}\n\n"
                f"[dim]{self.entry.citation()}[/dim]\n\n"
                "Fetches the full text and spends one reader agent writing the "
                "summaries. Runs in the background; the browser stays usable.",
                id="confirm-body",
            )
            yield Static("[dim]y read · esc cancel[/dim]", id="confirm-help")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfirmFindCode(ModalScreen[bool]):
    """Searching the web for a paper's repository spends one cheap agent.

    Far less than a read, but it is still a real call that goes out to the
    network, so it is asked for rather than assumed — and the prompt says what
    happens to an entry that already has a link.
    """

    BINDINGS = [
        Binding("y", "confirm", "Search"),
        Binding("enter", "confirm", "Search", show=False),
        Binding("escape", "cancel", "Cancel"),
        Binding("n", "cancel", "Cancel", show=False),
    ]

    def __init__(self, entry: Entry):
        super().__init__()
        self.entry = entry

    def body_text(self) -> str:
        """What the prompt says. Pure, so it can be tested on its own."""
        head = (
            f"{self.entry.title[:200]}\n\n"
            f"[dim]{self.entry.citation()}[/dim]\n\n"
            "Sends one cheap agent to search the web for the repository that "
            "implements this paper, and records it only if the repository names "
            "the paper. Runs in the background; the browser stays usable."
        )
        if self.entry.code_url:
            head += (
                f"\n\n[bold yellow]This entry already links "
                f"{self.entry.code_url} ({self.entry.code_provenance()}). "
                "A repository found now replaces it.[/bold yellow]"
            )
        return head

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static("Search the web for this paper's code?", id="confirm-title")
            yield Static(self.body_text(), id="confirm-body")
            yield Static("[dim]y search · esc cancel[/dim]", id="confirm-help")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfirmDelete(ModalScreen[bool]):
    """Deleting is the one thing in here that cannot be undone.

    So it asks first, it says what goes with the entry, and — unlike the read
    prompt — it does not accept `enter`, which is whatever the user was already
    pressing to move around the list.
    """

    BINDINGS = [
        Binding("y", "confirm", "Delete"),
        Binding("escape", "cancel", "Cancel"),
        Binding("n", "cancel", "Cancel", show=False),
    ]

    def __init__(self, entry: Entry):
        super().__init__()
        self.entry = entry

    def body_text(self) -> str:
        """What the prompt says. Pure, so it can be tested on its own."""
        return (
            f"{self.entry.title[:200]}\n\n"
            f"[dim]{self.entry.citation()}[/dim]\n\n"
            "Removes the record, its PDF and its cached text. This cannot "
            "be undone."
            + ("\n\n[bold yellow]Your notes on this entry go too.[/bold yellow]"
               if self.entry.notes.strip() else "")
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box", classes="danger"):
            yield Static("Delete this entry?", id="confirm-title", classes="danger")
            yield Static(self.body_text(), id="confirm-body")
            yield Static("[dim]y delete · esc cancel[/dim]", id="confirm-help")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class BrowserApp(App):
    """Two-pane browser: entry list on top, detail below."""

    CSS = """
    Screen { layers: base overlay; }
    #list { height: 45%; border: round $accent; }
    #detail-pane { height: 1fr; border: round $primary; padding: 0 1; }
    #detail-pane:focus { border: round $accent; }
    #detail { height: auto; }
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
    #confirm-box {
        width: 70%; height: auto; margin: 6 8;
        border: thick $accent; background: $surface;
    }
    #confirm-title { padding: 0 1; background: $accent; color: $text; }
    #confirm-body { padding: 1; }
    #confirm-help { padding: 0 1; }
    /* The destructive prompt should not look like the routine one. */
    #confirm-box.danger { border: thick $error; }
    #confirm-title.danger { background: $error; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear", show=False),
        Binding("R", "read_entry", "Read"),
        Binding("n", "edit_notes", "Notes"),
        Binding("d", "delete_entry", "Delete"),
        Binding("o", "open_link", "Open"),
        Binding("c", "open_code", "Code"),
        Binding("C", "find_code", "Find code"),
        Binding("f", "cycle_filter", "Filter"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("r", "reload", "Reload"),
        Binding("ctrl+d", "scroll_detail(1)", "Scroll ↓"),
        Binding("ctrl+u", "scroll_detail(-1)", "Scroll ↑"),
    ]

    FILTERS = [
        ("all", lambda e: True),
        ("A* and A", lambda e: level_rank(e.level) <= level_rank("A")),
        # The read backlog, which is what `R` acts on: anything carrying no
        # summary, unread and UNVERIFIED alike. Not called "unread", because
        # `lit ls --status unread` gives that word to only one of the two.
        ("not read", lambda e: e.needs_read),
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
        # Keys of the entries a reader agent is working on right now.
        self.reading: set[str] = set()
        # Keys a code scout is searching the web for right now.
        self.finding: set[str] = set()
        self._quit_warned = False

    @property
    def busy(self) -> set[str]:
        """Keys with a background agent working on them, of any kind."""
        return self.reading | self.finding

    # ---------------- layout ----------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="search… (enter to apply, esc to clear)", id="search")
        with Vertical():
            with Horizontal(id="list"):
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            # A summary runs to several screens, so the detail pane scrolls:
            # ctrl+d / ctrl+u, the mouse wheel, or tab into it and use arrows.
            with VerticalScroll(id="detail-pane"):
                yield Markdown("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.library.name
        self.sub_title = self.library.settings.scope or ""
        table = self.query_one("#table", DataTable)
        table.add_columns("key", "title", "venue", "yr", "lvl", "cites", "status")
        self.action_reload()
        # The search box is first in the DOM and would otherwise take the focus
        # while hidden, swallowing every key the browser is driven by.
        table.focus()

    # ---------------- data ----------------

    def action_reload(self) -> None:
        self.entries = self.library.entries()
        self.refresh_rows()

    def refresh_rows(self) -> None:
        # A reload can happen under the user — when a read finishes, say — so
        # the row they were on is restored rather than jumping back to the top.
        selected = self.current()
        selected_key = selected.key if selected else None

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
                self.row_status(e),
                key=e.key,
            )
        self.refresh_status(filter_name=name)
        if rows:
            index = next((i for i, e in enumerate(rows) if e.key == selected_key), 0)
            table.move_cursor(row=index)
            self.show_detail(rows[index])
        else:
            self.query_one("#detail", Markdown).update("_Nothing matches._")

    def row_status(self, entry: Entry) -> str:
        """What the status column says: unread and UNVERIFIED are not the same."""
        if entry.key in self.reading:
            return "reading…"
        if entry.key in self.finding:
            return "code…"
        if entry.is_verified:
            return "read"
        return "unread" if entry.is_unread else "UNVERIFIED"

    def refresh_status(self, filter_name: str | None = None) -> None:
        bits = [
            f"{len(self.shown)}/{len(self.entries)} entries",
            f"filter: {filter_name or self.FILTERS[self.filter_i][0]}",
            f"sort: {self.SORTS[self.sort_i][0]}",
        ]
        if self.query:
            bits.append(f"query: {self.query!r}")
        if self.reading:
            bits.append(f"reading: {', '.join(sorted(self.reading))}")
        if self.finding:
            bits.append(f"finding code: {', '.join(sorted(self.finding))}")
        self.query_one("#status", Static).update(" · ".join(bits))

    def detail_markdown(self, entry: Entry) -> str:
        """The detail pane's content. Pure, so it can be tested on its own."""
        parts = [
            f"# {entry.title}",
            "",
            f"**{entry.citation()}** · {entry.type} · level {entry.level}"
            + (f" · {entry.citation_count} citations"
               if entry.citation_count is not None else "")
            # Never read and could-not-be-read are different states, and the
            # first one is a keypress away from being fixed.
            + ("" if entry.is_verified
               else " · **unread**" if entry.is_unread else " · **UNVERIFIED**"),
        ]
        if entry.level_reason:
            parts.append(f"*{entry.level_reason}*")

        # The URLs are their own link text: a bare "paper" would be clickable
        # but tells you nothing, and these are worth reading and copying.
        links = []
        if url := paper_url(entry):
            links.append(f"- Paper: [{url}]({url})")
        if entry.code_url:
            # Which kind of link this is comes before anything else about it: a
            # repository a scout found on the web is a weaker claim than one the
            # paper printed, and the browser must not flatten the difference.
            links.append(
                f"- Code: [{entry.code_url}]({entry.code_url}) "
                f"— _{entry.code_provenance()}_"
            )
            if entry.code_reason:
                links.append(f"  - _{entry.code_reason}_")
        if links:
            parts += [""] + links

        if entry.one_liner:
            parts += ["", f"> {entry.one_liner}"]
        elif entry.is_unread:
            parts += ["", "> _Not read yet — press R to read it in full._"]
        elif not entry.is_verified:
            parts += ["", "> _Full text unavailable — no summary was written._"]
        if entry.tags:
            parts += ["", "Tags: " + ", ".join(f"`{t}`" for t in entry.tags)]
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
        return "\n".join(parts)

    def show_detail(self, entry: Entry) -> None:
        self.query_one("#detail", Markdown).update(self.detail_markdown(entry))
        # New entry, new document: start at the top whatever the last one was
        # scrolled to.
        self.query_one("#detail-pane", VerticalScroll).scroll_home(animate=False)

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

    @on(Markdown.LinkClicked)
    def _link_clicked(self, event: Markdown.LinkClicked) -> None:
        self.open_in_browser(event.href, "link")

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

    def action_scroll_detail(self, direction: int) -> None:
        pane = self.query_one("#detail-pane", VerticalScroll)
        pane.scroll_relative(y=direction * max(1, pane.size.height // 2), animate=False)

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
        entry = self.current()
        if entry is not None:
            self.open_in_browser(paper_url(entry), "link")

    def action_open_code(self) -> None:
        entry = self.current()
        if entry is None:
            return
        if not entry.code_url:
            self.notify(
                "no code repository recorded — press C to search the web for it",
                severity="warning",
            )
            return
        self.open_in_browser(entry.code_url, "code repository")

    def open_in_browser(self, url: str | None, what: str = "link") -> None:
        if not url:
            self.notify(f"no {what} for this entry", severity="warning")
            return
        how = open_url(url)
        if how:
            self.notify(f"opened in {how}: {url}")
        else:
            self.notify(f"could not open {url}", severity="error")

    # ---------------- deleting ----------------

    def action_delete_entry(self) -> None:
        """Delete the selected entry — the browser's half of `lit delete <key>`."""
        entry = self.current()
        if entry is None:
            return
        if entry.key in self.busy:
            self.notify(
                f"an agent is working on {entry.key} right now — "
                "let it finish first",
                severity="warning",
            )
            return
        self.push_screen(ConfirmDelete(entry), partial(self._delete_confirmed, entry))

    def _delete_confirmed(self, entry: Entry, confirmed: bool | None) -> None:
        if not confirmed:
            return
        table = self.query_one("#table", DataTable)
        # Where to land afterwards: the row that slides up into this one, so
        # working down a backlog doesn't throw the cursor back to the top.
        row = table.cursor_row if table.cursor_row is not None else 0
        if self.library.delete_entry(entry.key):
            self.notify(f"deleted {entry.key}")
        else:
            self.notify(f"{entry.key} was already gone", severity="warning")
        self.action_reload()
        if self.shown:
            index = max(0, min(row, len(self.shown) - 1))
            table.move_cursor(row=index)
            self.show_detail(self.shown[index])

    # ---------------- finding code ----------------

    def action_find_code(self) -> None:
        """Search the web for this paper's code — the browser's `lit code <key>`."""
        entry = self.current()
        if entry is None:
            return
        if entry.key in self.finding:
            self.notify(f"already searching for {entry.key}'s code",
                        severity="warning")
            return
        self.push_screen(ConfirmFindCode(entry), partial(self._find_confirmed, entry))

    def _find_confirmed(self, entry: Entry, confirmed: bool | None) -> None:
        if confirmed:
            self.start_find_code(entry)

    def start_find_code(self, entry: Entry) -> None:
        """Run one code search in a background thread, leaving the browser usable."""
        self.finding.add(entry.key)
        self.refresh_rows()
        self.notify(f"searching the web for {entry.key}'s code")
        self.run_worker(
            partial(self._find_code_worker, entry.key),
            thread=True, group="code", name=f"code:{entry.key}",
        )

    def _find_code_worker(self, key: str) -> None:
        ctx = Ctx(cfg=self.cfg, library=self.library, console=Console(quiet=True),
                  yes=True)
        cost = ""
        try:
            entry = self.library.get(key)
            if entry is None:
                ok, message = False, f"{key} is no longer in the library"
            else:
                # --force: the user was shown the existing link in the prompt and
                # chose to search anyway.
                results = find_code(ctx, [entry], force=True)
                res = results[0]
                ok = res.ok
                message = f"{key}: {res.message or res.status}"
                cost = ctx.usage.summary()
        except Exception as exc:
            ok, message = False, f"{key}: {exc}"
        finally:
            ctx.close()
        self.call_from_thread(self._find_code_finished, key, ok, message, cost)

    def _find_code_finished(self, key: str, ok: bool, message: str,
                            cost: str) -> None:
        self.finding.discard(key)
        if not self.busy:
            self._quit_warned = False
        self.action_reload()
        self.notify(
            message + (f"\n{cost}" if cost else ""),
            severity="information" if ok else "warning",
            timeout=12,
        )

    # ---------------- reading ----------------

    def action_read_entry(self) -> None:
        """Read the selected paper — the browser's half of `lit read <key>`."""
        entry = self.current()
        if entry is None:
            return
        if entry.key in self.reading:
            self.notify(f"{entry.key} is already being read", severity="warning")
            return
        if not entry.needs_read:
            self.notify(
                f"{entry.key} already has a summary — `lit reread {entry.key}` "
                "redoes it from scratch.",
                severity="warning",
            )
            return
        self.push_screen(ConfirmRead(entry), partial(self._read_confirmed, entry))

    def _read_confirmed(self, entry: Entry, confirmed: bool | None) -> None:
        if confirmed:
            self.start_read(entry)

    def start_read(self, entry: Entry) -> None:
        """Run one read in a background thread, leaving the browser usable."""
        self.reading.add(entry.key)
        self.refresh_rows()
        self.notify(f"reading {entry.key} — this takes a minute")
        self.run_worker(
            partial(self._read_worker, entry.key),
            thread=True, group="read", name=f"read:{entry.key}",
        )

    def _read_worker(self, key: str) -> None:
        # A quiet console: the action layer narrates its progress, and anything
        # printed to the real terminal would land on top of the running TUI.
        ctx = Ctx(cfg=self.cfg, library=self.library, console=Console(quiet=True),
                  yes=True)
        cost = ""
        try:
            result = reread(ctx, key)
            ok, message = result.ok, result.message
            cost = ctx.usage.summary()
        except Exception as exc:
            # A reader agent that dies takes this thread with it; the user
            # should hear about it in the UI, not in a traceback after quitting.
            ok, message = False, f"{key}: {exc}"
        finally:
            ctx.close()
        self.call_from_thread(self._read_finished, key, ok, message, cost)

    def _read_finished(self, key: str, ok: bool, message: str, cost: str) -> None:
        self.reading.discard(key)
        if not self.busy:
            self._quit_warned = False
        self.action_reload()
        self.notify(
            message + (f"\n{cost}" if cost else ""),
            severity="information" if ok else "error",
            timeout=12,
        )

    async def action_quit(self) -> None:
        # A thread worker cannot be cancelled, and quitting mid-run would drop
        # work that has already been paid for.
        if self.busy and not self._quit_warned:
            self._quit_warned = True
            self.notify(
                f"still working on {', '.join(sorted(self.busy))} — "
                "press q again to quit anyway",
                severity="warning",
            )
            return
        await super().action_quit()


def run_browser(library: Library, cfg: Config) -> None:
    BrowserApp(library, cfg).run()
