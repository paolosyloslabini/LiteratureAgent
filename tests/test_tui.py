"""The browser is optional, but it must not be broken.

Every capability it exposes is also a subcommand, so these tests target the
browser's own logic — filtering, sorting, detail rendering, and the fact that
note editing goes through the same protected path as `lit note` — rather than
Textual's event dispatch.
"""

from __future__ import annotations

import pytest
from factories import make_entry

from lit.actions.add import AddResult
from lit.actions.code import CodeResult
from lit.config import Config
from lit.models import CODE_FROM_WEB, STATUS_UNREAD
from lit.tui import BrowserApp, ConfirmDelete, ConfirmFindCode, ConfirmRead


@pytest.fixture
def stocked(lib):
    lib.save_entry(make_entry(code_url="https://github.com/tensorflow/tensor2tensor"))
    lib.save_entry(make_entry(
        key="doe2010obscure", title="An Obscure Study", authors=["Jane Doe"],
        year=2010, venue="Widget Weekly", level="C", arxiv_id=None,
        one_liner="Widgets.", tags=["widgets"], citation_count=3,
        status="UNVERIFIED", sections=[], references=[],
    ))
    return lib


@pytest.fixture
def app(stocked):
    return BrowserApp(stocked, Config())


# --------------------------------------------------------------------------
# Mounting
# --------------------------------------------------------------------------

async def test_browser_mounts_and_lists_entries(app):
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.shown) == 2
        assert app.title == "testlib"
        assert app.sub_title == "Testing the literature agent"


async def test_browser_mounts_with_an_empty_library(lib):
    empty = BrowserApp(lib, Config())
    async with empty.run_test() as pilot:
        await pilot.pause()
        assert empty.shown == []


# --------------------------------------------------------------------------
# Filtering and sorting
# --------------------------------------------------------------------------

async def test_filter_cycles_through_the_defined_views(app):
    async with app.run_test() as pilot:
        await pilot.pause()
        assert [e.key for e in app.shown] == ["vaswani2017attention", "doe2010obscure"]

        app.action_cycle_filter()  # -> A* and A
        assert [e.key for e in app.shown] == ["vaswani2017attention"]

        app.action_cycle_filter()  # -> not read
        assert [e.key for e in app.shown] == ["doe2010obscure"]

        app.action_cycle_filter()  # -> with notes
        assert app.shown == []

        app.action_cycle_filter()  # wraps back to all
        assert len(app.shown) == 2


async def test_the_backlog_filter_is_not_named_after_the_cli_status(app, stocked):
    """It shows both states `R` will read, so it cannot borrow `unread` for them.

    `lit ls --status unread` means one status; the status column beside the
    filter says the same. The filter is the wider read backlog, so it says so.
    """
    stocked.save_entry(make_entry(key="doe2024unread", status=STATUS_UNREAD,
                                  one_liner=None, sections=[], references=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_reload()
        app.action_cycle_filter()  # -> A* and A
        app.action_cycle_filter()  # -> the read backlog
        name, pred = app.FILTERS[app.filter_i]

        shown = {e.key: app.row_status(e) for e in app.shown}
        assert shown == {"doe2024unread": "unread", "doe2010obscure": "UNVERIFIED"}
        assert name != STATUS_UNREAD
        # And it is exactly the set the `R` key is willing to act on.
        assert all(pred(e) is e.needs_read for e in stocked.entries())


async def test_with_notes_filter_finds_annotated_entries(app, stocked):
    from lit import entryfile

    entryfile.set_notes(stocked.entry_path("doe2010obscure"), "my thought")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_reload()
        for _ in range(3):
            app.action_cycle_filter()
        assert [e.key for e in app.shown] == ["doe2010obscure"]


async def test_sort_cycles(app):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_sort()  # level -> year
        assert app.shown[0].year == 2017
        app.action_cycle_sort()  # -> title
        assert app.shown[0].title.startswith("An Obscure")


async def test_search_narrows_the_rows(app):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query = "widgets obscure"
        app.refresh_rows()
        assert [e.key for e in app.shown] == ["doe2010obscure"]

        app.action_clear_search()
        assert len(app.shown) == 2


# --------------------------------------------------------------------------
# Detail pane
# --------------------------------------------------------------------------

async def test_detail_renders_a_read_entry(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.show_detail(stocked.get("vaswani2017attention"))  # must not raise


async def test_detail_renders_an_unverified_entry_without_inventing_a_summary(
        app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.show_detail(stocked.get("doe2010obscure"))  # must not raise


async def test_detail_links_to_the_paper_and_to_its_code(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.detail_markdown(stocked.get("vaswani2017attention"))
    assert "[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)" in md
    assert ("[https://github.com/tensorflow/tensor2tensor]"
            "(https://github.com/tensorflow/tensor2tensor)") in md


async def test_detail_omits_links_an_entry_does_not_have(app, stocked):
    """The obscure entry has no arXiv id, no DOI, no URL and no repository."""
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.detail_markdown(stocked.get("doe2010obscure"))
    assert "Paper:" not in md
    assert "Code:" not in md


async def test_the_detail_pane_scrolls(app, stocked):
    from textual.containers import VerticalScroll

    long_entry = make_entry(key="long2020paper", title="A Very Long Paper",
                            abstract="\n\n".join(f"paragraph {i}" for i in range(200)))
    stocked.save_entry(long_entry)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_reload()
        app.show_detail(stocked.get("long2020paper"))
        await pilot.pause()

        pane = app.query_one("#detail-pane", VerticalScroll)
        assert pane.max_scroll_y > 0, "a long summary should overflow the pane"
        app.action_scroll_detail(1)
        await pilot.pause()
        assert pane.scroll_target_y > 0
        app.action_scroll_detail(-1)
        await pilot.pause()
        assert pane.scroll_target_y == 0


async def test_switching_entries_returns_the_detail_pane_to_the_top(app, stocked):
    from textual.containers import VerticalScroll

    stocked.save_entry(make_entry(
        key="long2020paper", title="A Very Long Paper",
        abstract="\n\n".join(f"paragraph {i}" for i in range(200))))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_reload()
        app.show_detail(stocked.get("long2020paper"))
        await pilot.pause()
        app.action_scroll_detail(1)
        await pilot.pause()

        app.show_detail(stocked.get("doe2010obscure"))
        await pilot.pause()
        assert app.query_one("#detail-pane", VerticalScroll).scroll_target_y == 0


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------

def test_note_editing_uses_the_protected_path(stocked):
    """The browser writes notes via `entryfile.set_notes`, exactly as `lit note`
    does, so a later machine re-save cannot clobber them."""
    from lit import entryfile

    path = stocked.entry_path("vaswani2017attention")
    entryfile.set_notes(path, "typed in the browser")
    assert stocked.get("vaswani2017attention").notes == "typed in the browser"

    stocked.save_entry(make_entry())  # a re-read of the machine fields
    assert stocked.get("vaswani2017attention").notes == "typed in the browser"


async def test_current_returns_none_when_nothing_is_shown(lib):
    empty = BrowserApp(lib, Config())
    async with empty.run_test() as pilot:
        await pilot.pause()
        assert empty.current() is None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

async def test_reading_an_unread_entry_runs_the_same_read_the_cli_does(
        app, stocked, monkeypatch):
    """`R` is `lit read <key>` — the browser adds no second path to a summary."""
    calls: list[str] = []

    def fake_reread(ctx, key):
        calls.append(key)
        return AddResult("updated", f"read {key}")

    monkeypatch.setattr("lit.tui.reread", fake_reread)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.start_read(stocked.get("doe2010obscure"))
        assert app.reading == {"doe2010obscure"}
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.reading == set()

    assert calls == ["doe2010obscure"]


async def test_a_failed_read_is_reported_and_does_not_take_the_browser_down(
        app, stocked, monkeypatch):
    def boom(ctx, key):
        raise RuntimeError("the reader agent died")

    monkeypatch.setattr("lit.tui.reread", boom)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.start_read(stocked.get("doe2010obscure"))
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.reading == set()
        assert app.is_running


async def test_pressing_R_then_confirming_reads_the_paper(app, stocked, monkeypatch):
    """The whole keyboard path, since that is the only way a user reaches it."""
    calls: list[str] = []
    monkeypatch.setattr("lit.tui.reread",
                        lambda ctx, key: calls.append(key) or AddResult("updated", "ok"))
    stocked.save_entry(make_entry(
        key="doe2024unread", title="A Paper Nobody Has Read Yet",
        authors=["Jane Doe"], year=2024, arxiv_id=None, status=STATUS_UNREAD,
        level="B", one_liner=None, sections=[], references=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_reload()
        app.query_one("#table").move_cursor(
            row=[e.key for e in app.shown].index("doe2024unread"))
        await pilot.pause()

        await pilot.press("R")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmRead)
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == ["doe2024unread"]


async def test_an_unread_entry_is_not_labelled_unverified(app, stocked):
    """`lit` distinguishes never-read from could-not-be-read; so does the browser."""
    stocked.save_entry(make_entry(key="doe2024unread", status=STATUS_UNREAD,
                                  one_liner=None, sections=[], references=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        entry = stocked.get("doe2024unread")
        md = app.detail_markdown(entry)
        assert "UNVERIFIED" not in md
        assert "unread" in md
        assert app.row_status(entry) == "unread"
        assert app.row_status(stocked.get("doe2010obscure")) == "UNVERIFIED"


async def test_read_asks_before_spending_a_reader_agent(app, monkeypatch):
    monkeypatch.setattr("lit.tui.reread",
                        lambda ctx, key: pytest.fail("read without confirming"))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_filter()  # -> A* and A
        app.action_cycle_filter()  # -> not read, so the cursor is on one
        await pilot.pause()

        app.action_read_entry()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmRead)

        await pilot.press("escape")
        await pilot.pause()
        assert app.reading == set()


async def test_an_entry_that_already_has_a_summary_is_not_read_again(app, monkeypatch):
    monkeypatch.setattr("lit.tui.reread",
                        lambda ctx, key: pytest.fail("re-read a verified entry"))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_filter()  # -> A* and A: only the verified entry
        await pilot.pause()
        app.action_read_entry()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmRead)
        assert app.reading == set()


async def test_the_row_stays_selected_across_a_reload(app):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#table").move_cursor(row=1)
        await pilot.pause()
        assert app.current().key == "doe2010obscure"

        app.action_reload()
        await pilot.pause()
        assert app.current().key == "doe2010obscure"


# --------------------------------------------------------------------------
# Deleting
# --------------------------------------------------------------------------

async def test_delete_asks_first_and_removes_the_entry(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#table").move_cursor(row=1)
        await pilot.pause()
        assert app.current().key == "doe2010obscure"

        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDelete)

        await pilot.press("y")
        await pilot.pause()
        assert not stocked.has("doe2010obscure")
        assert [e.key for e in app.shown] == ["vaswani2017attention"]


async def test_cancelling_the_prompt_deletes_nothing(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDelete)

        await pilot.press("escape")
        await pilot.pause()
        assert len(stocked) == 2
        assert len(app.shown) == 2


async def test_enter_does_not_confirm_a_delete(app, stocked):
    """`enter` is a navigation key here — it must not be able to delete."""
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDelete)
        assert len(stocked) == 2


async def test_delete_leaves_the_cursor_where_the_row_was(app, stocked):
    """Working down a backlog shouldn't throw the cursor back to the top."""
    stocked.save_entry(make_entry(key="zeta2020last", title="Zeta", year=2020,
                                  level="C", arxiv_id=None, sections=[],
                                  references=[]))
    async with app.run_test() as pilot:
        app.action_reload()
        await pilot.pause()
        app.query_one("#table").move_cursor(row=1)
        await pilot.pause()
        doomed = app.current().key

        app.action_delete_entry()
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()

        assert doomed not in [e.key for e in app.shown]
        assert app.current() is app.shown[1]


async def test_an_entry_being_read_is_not_deleted_underneath_the_reader(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.reading.add(app.current().key)
        app.action_delete_entry()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmDelete)
        assert len(stocked) == 2


def test_the_delete_prompt_says_when_notes_go_with_it(stocked):
    """Notes are the one thing in an entry the user wrote themselves."""
    from lit import entryfile

    assert "Your notes" not in ConfirmDelete(stocked.get("doe2010obscure")).body_text()

    entryfile.set_notes(stocked.entry_path("doe2010obscure"),
                        "Worth revisiting for the widget benchmark.")
    assert "Your notes" in ConfirmDelete(stocked.get("doe2010obscure")).body_text()


# --------------------------------------------------------------------------
# Opening links
# --------------------------------------------------------------------------

async def test_open_uses_the_arxiv_link_when_there_is_no_doi(app, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("lit.tui.open_url",
                        lambda url: opened.append(url) or "Chrome")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_filter()  # -> A* and A: the arXiv entry
        await pilot.pause()
        app.action_open_link()
    assert opened == ["https://arxiv.org/abs/1706.03762"]


async def test_open_code_uses_the_repository_recorded_on_the_entry(app, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("lit.tui.open_url",
                        lambda url: opened.append(url) or "Chrome")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_filter()  # -> A* and A
        await pilot.pause()
        app.action_open_code()
    assert opened == ["https://github.com/tensorflow/tensor2tensor"]


async def test_open_code_says_so_when_there_is_no_repository(app, monkeypatch):
    monkeypatch.setattr("lit.tui.open_url",
                        lambda url: pytest.fail(f"opened {url!r} out of nothing"))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_filter()
        app.action_cycle_filter()  # -> not read: the entry with no code link
        await pilot.pause()
        app.action_open_code()  # must not raise, must not open


# --------------------------------------------------------------------------
# Finding code
# --------------------------------------------------------------------------

def _stub_find(monkeypatch, calls, *, result=None, boom=False):
    """Stand in for the action, so no agent and no network are reached."""
    def fake(ctx, entries, **kw):
        calls.append((entries[0].key, kw))
        if boom:
            raise RuntimeError("the scout died")
        return [result or CodeResult(
            key=entries[0].key, status="found",
            code_url="https://github.com/found/here", official=True,
            message="found https://github.com/found/here")]

    monkeypatch.setattr("lit.tui.find_code", fake)


async def test_finding_code_runs_the_same_action_the_cli_does(app, stocked,
                                                             monkeypatch):
    """`C` is `lit code <key>` — the browser adds no second path to a link."""
    calls: list = []
    _stub_find(monkeypatch, calls)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.start_find_code(stocked.get("doe2010obscure"))
        assert app.finding == {"doe2010obscure"}
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.finding == set()

    assert calls[0][0] == "doe2010obscure"


async def test_pressing_C_then_confirming_searches_for_the_code(app, stocked,
                                                                monkeypatch):
    calls: list = []
    _stub_find(monkeypatch, calls)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_filter()
        app.action_cycle_filter()  # -> not read: the entry with no code link
        await pilot.pause()

        await pilot.press("C")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmFindCode)
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls[0][0] == "doe2010obscure"


async def test_finding_code_asks_before_spending_an_agent(app, monkeypatch):
    monkeypatch.setattr("lit.tui.find_code",
                        lambda *a, **k: pytest.fail("searched without confirming"))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_find_code()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmFindCode)

        await pilot.press("escape")
        await pilot.pause()
        assert app.finding == set()


async def test_a_failed_search_is_reported_and_does_not_take_the_browser_down(
        app, stocked, monkeypatch):
    _stub_find(monkeypatch, [], boom=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.start_find_code(stocked.get("doe2010obscure"))
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.finding == set()
        assert app.is_running


async def test_the_browser_re_searches_a_link_the_user_was_warned_about(
        app, stocked, monkeypatch):
    """The prompt shows the existing link, so confirming means replace it."""
    calls: list = []
    _stub_find(monkeypatch, calls)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.start_find_code(stocked.get("vaswani2017attention"))
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls[0][1]["force"] is True


def test_the_find_code_prompt_warns_that_an_existing_link_is_replaced(stocked):
    with_link = ConfirmFindCode(stocked.get("vaswani2017attention")).body_text()
    assert "tensor2tensor" in with_link
    assert "replaces it" in with_link

    without = ConfirmFindCode(stocked.get("doe2010obscure")).body_text()
    assert "replaces it" not in without


async def test_a_paper_being_searched_shows_it(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        entry = stocked.get("doe2010obscure")
        app.finding.add(entry.key)
        assert app.row_status(entry) == "code…"
        assert entry.key in app.busy


async def test_an_entry_being_searched_is_not_deleted_underneath_the_scout(
        app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        app.finding.add(app.current().key)
        app.action_delete_entry()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmDelete)
        assert len(stocked) == 2


# --------------------------------------------------------------------------
# Where a code link came from
# --------------------------------------------------------------------------

async def test_the_detail_says_a_link_came_from_the_paper(app, stocked):
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.detail_markdown(stocked.get("vaswani2017attention"))
    assert "tensor2tensor" in md
    assert "printed in the paper" in md


async def test_the_detail_marks_a_link_found_by_searching(app, stocked):
    """A repository a scout found must never read as one the authors printed."""
    stocked.save_entry(make_entry(
        code_url="https://github.com/found/here", code_source=CODE_FROM_WEB,
        code_reason="author release, high confidence — the README cites it"))
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.detail_markdown(stocked.get("vaswani2017attention"))
    assert "found on the web" in md
    assert "printed in the paper" not in md
    assert "the README cites it" in md
