"""`lit claim` — a trace survives a reply that came back the wrong shape.

Every hop of a trace is paid for: metadata lookups, a full text fetch and an
analyst call per candidate. So one model reply that names its attributions as
strings instead of objects must cost that reply, not the whole chain.
"""

from __future__ import annotations

import pytest
from factories import make_meta
from rich.console import Console

from lit.actions.claim import _candidate_refs, trace_claim
from lit.actions.context import Ctx
from lit.models import Reference


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


class StubLLM:
    """Replays a fixed list of replies, in order."""

    available = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def json(self, prompt, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else {}


REFS = [Reference(title="An Earlier Paper", year=2015, doi="10.1/earlier")]


def test_candidate_refs_ignores_attributions_that_are_not_objects():
    analysis = {"attributed_to": ["Smith 2020 [0]"]}
    assert _candidate_refs(analysis, REFS) == []


def test_candidate_refs_still_reads_the_objects_beside_a_bad_one():
    analysis = {"attributed_to": ["Smith 2020 [0]", {"index": 0, "confidence": 0.9}]}
    assert _candidate_refs(analysis, REFS) == REFS


def test_a_badly_shaped_attribution_does_not_abort_the_trace(monkeypatch, ctx):
    """The model answered in strings; the hops already paid for still report."""
    from lit.fetch.fulltext import FullText

    monkeypatch.setattr("lit.actions.claim.resolve_metadata",
                        lambda *a, **k: make_meta(title="The Citing Paper",
                                                  doi="10.1/citing",
                                                  references=REFS))
    monkeypatch.setattr("lit.actions.claim.fetch_fulltext",
                        lambda *a, **k: FullText("t" * 9000, "arxiv"))
    ctx._llm = StubLLM([{"is_origin": False, "states_claim": True,
                         "attributed_to": ["Smith 2020 [0]"]}])

    res = trace_claim(ctx, "a claim", start="10.1/citing", max_hops=3)

    assert [h.title for h in res.chain] == ["The Citing Paper"]
    assert res.origin is not None
    assert res.origin.title == "The Citing Paper"
    assert not res.complete, "an unparseable attribution is not a confirmed origin"
    assert any("nothing traceable" in n for n in res.notes), res.notes
