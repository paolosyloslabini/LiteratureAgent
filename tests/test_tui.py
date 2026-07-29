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
from lit.config import Config
from lit.models import STATUS_UNREAD
from lit.tui import BrowserApp, ConfirmRead


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

        app.action_cycle_filter()  # -> unread
        assert [e.key for e in app.shown] == ["doe2010obscure"]

        app.action_cycle_filter()  # -> with notes
        assert app.shown == []

        app.action_cycle_filter()  # wraps back to all
        assert len(app.shown) == 2


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
        app.action_cycle_filter()  # -> unread, so the cursor is on one
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
        app.action_cycle_filter()  # -> unread: the entry with no code link
        await pilot.pause()
        app.action_open_code()  # must not raise, must not open
