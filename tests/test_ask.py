"""`lit ask` when nothing is quotable: why it failed, not a verdict on the library.

There are three ways `ask` reaches the end with no quotes, and they are not the
same news: every source failed to load, every source loaded and held nothing, or
some of each. Only the last two say anything about the library. A run where the
`claude` CLI was missing or every PDF was paywalled used to read as "your
library has nothing on this", with no reason and no next step.
"""

from __future__ import annotations

import pytest
from factories import make_entry
from rich.console import Console

from lit.actions.ask import ask
from lit.actions.context import Ctx
from lit.fetch.fulltext import FullText
from lit.llm import LLMError


@pytest.fixture
def ctx(lib, cfg):
    for i in range(2):
        lib.save_entry(make_entry(
            key=f"paper{i}", title=f"Paper {i} on causal agents",
            arxiv_id=None, doi=f"10.1/{i}", status="verified",
            one_liner="A causal account of agents.", tags=["causal"],
        ))
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True))


class StubLLM:
    """Picks both papers to read, then reports whatever the reader is given."""

    available = True

    def __init__(self, extract=None, extract_error=None):
        self.extract = extract or {"relevant": False}
        self.extract_error = extract_error
        self.selected = False

    def json(self, prompt, **kw):
        if not self.selected:  # the first call is the selection over summaries
            self.selected = True
            return {"relevant": [{"key": "paper0", "relevance": 0.9},
                                 {"key": "paper1", "relevance": 0.8}]}
        if self.extract_error:
            raise self.extract_error
        return self.extract

    def run(self, prompt, **kw):
        return type("R", (), {"text": "an answer"})()


READABLE = FullText("t" * 9000, "arxiv")


def unreadable(*keys):
    """Full text for every paper except `keys`."""
    def fetch(http, meta, *, key, **kw):
        return None if key in keys else READABLE
    return fetch


def test_sources_that_could_not_be_read_are_not_reported_as_an_empty_library(
        ctx, monkeypatch):
    """Every paywalled source is a failure of the fetch, not a verdict."""
    ctx._llm = StubLLM()
    monkeypatch.setattr("lit.actions.ask.fetch_fulltext",
                        unreadable("paper0", "paper1"))

    res = ask(ctx, "causal", read=2)

    assert res.unreadable == ["paper0", "paper1"]
    assert "None of the sources consulted could be read" in res.answer
    assert "contained evidence bearing" not in res.answer
    assert "Could not read: paper0, paper1" in res.answer


def test_a_failed_llm_call_is_reported_as_a_failure_too(ctx, monkeypatch):
    """A missing `claude` CLI reads nothing — it does not disprove anything."""
    ctx._llm = StubLLM(extract_error=LLMError(
        "the `claude` CLI was not found on PATH"))
    monkeypatch.setattr("lit.actions.ask.fetch_fulltext", unreadable())

    res = ask(ctx, "causal", read=2)

    assert res.unreadable == ["paper0", "paper1"]
    assert "None of the sources consulted could be read" in res.answer
    assert "contained evidence bearing" not in res.answer
    assert all("claude" in ev.error for ev in res.evidence)


def test_papers_that_were_read_and_held_nothing_are_still_reported_that_way(
        ctx, monkeypatch):
    """The fix must not turn a real "nothing on this" into an excuse."""
    ctx._llm = StubLLM(extract={"relevant": False})
    monkeypatch.setattr("lit.actions.ask.fetch_fulltext", unreadable())

    res = ask(ctx, "causal", read=2)

    assert res.unreadable == []
    assert "contained evidence bearing on that question" in res.answer
    assert "could not be read" not in res.answer
    assert "Sources read: paper0, paper1" in res.answer


def test_a_mixed_run_says_which_were_read_and_which_failed(ctx, monkeypatch):
    """One read and empty, one unreachable: both halves have to be in there."""
    ctx._llm = StubLLM(extract={"relevant": False})
    monkeypatch.setattr("lit.actions.ask.fetch_fulltext", unreadable("paper0"))

    res = ask(ctx, "causal", read=2)

    assert res.unreadable == ["paper0"]
    assert "contained evidence bearing on that question" in res.answer
    assert "Sources read: paper1" in res.answer
    assert "1 could not be read: paper0" in res.answer
