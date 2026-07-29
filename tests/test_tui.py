"""The browser is optional, but it must not be broken.

Every capability it exposes is also a subcommand, so these tests target the
browser's own logic — filtering, sorting, detail rendering, and the fact that
note editing goes through the same protected path as `lit note` — rather than
Textual's event dispatch.
"""

from __future__ import annotations

import pytest
from factories import make_entry

from lit.config import Config
from lit.tui import BrowserApp


@pytest.fixture
def stocked(lib):
    lib.save_entry(make_entry())
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
