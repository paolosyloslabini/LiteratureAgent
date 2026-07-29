"""The `find` orchestrator: candidate pooling, de-duplication, reference mining.

The point of the orchestrator is that the pool is built and de-duplicated once,
before any expensive read happens, so no two workers read the same paper and
nothing already in the library is proposed again.
"""

from __future__ import annotations

import pytest
from factories import make_entry
from rich.console import Console

from lit.actions.context import Ctx
from lit.actions.discover import discover, mine_references
from lit.models import Reference, slugify
from lit.runner import run_parallel


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


class ScoutLLM:
    """Returns a scripted set of papers per angle, and records exclusion lists."""

    available = True

    def __init__(self, per_angle):
        self.per_angle = per_angle
        self.prompts: list[str] = []
        self.calls = 0

    def json(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        if kw.get("role") == "filter":
            # Keep everything the filter is shown.
            n = prompt.count("\n  [")
            return {"keep": [{"index": i, "relevance": 0.9, "why": "fits"}
                             for i in range(n)]}
        idx = len(self.prompts) - 1
        return {"papers": self.per_angle[min(idx, len(self.per_angle) - 1)]}


def paper(title, **kw):
    return {"title": title, "authors": ["A B"], "year": 2020, "why": "relevant", **kw}


# --------------------------------------------------------------------------
# De-duplication across sources
# --------------------------------------------------------------------------

def test_same_paper_from_two_angles_appears_once(ctx):
    ctx._llm = ScoutLLM([
        [paper("Shared Paper"), paper("Only From Angle One")],
        [paper("Shared Paper"), paper("Only From Angle Two")],
    ])
    res = discover(ctx, "topic", limit=10, angles=2, use_references=False)
    titles = [c.title for c in res.candidates]
    assert titles.count("Shared Paper") == 1


def test_agreement_across_angles_ranks_higher(ctx):
    ctx._llm = ScoutLLM([
        [paper("Agreed On"), paper("Solo A")],
        [paper("Agreed On"), paper("Solo B")],
    ])
    res = discover(ctx, "topic", limit=10, angles=2, use_references=False)
    assert res.candidates[0].title == "Agreed On"


def test_dedup_matches_on_doi_across_differing_titles(ctx):
    ctx._llm = ScoutLLM([
        [paper("Attention Is All You Need", doi="10.1234/abcd")],
        [paper("ATTENTION IS ALL YOU NEED (v2)", doi="10.1234/abcd")],
    ])
    res = discover(ctx, "topic", limit=10, angles=2, use_references=False)
    assert len(res.candidates) == 1


def test_papers_already_in_the_library_are_dropped(ctx):
    ctx.library.save_entry(make_entry())
    ctx._llm = ScoutLLM([[paper("Attention Is All You Need"), paper("Something New")]])
    res = discover(ctx, "topic", limit=10, angles=1, use_references=False)
    assert [c.title for c in res.candidates] == ["Something New"]


def test_scouts_are_told_what_is_already_present(ctx):
    ctx.library.save_entry(make_entry())
    llm = ScoutLLM([[paper("New Thing")]])
    ctx._llm = llm
    discover(ctx, "topic", limit=5, angles=1, use_references=False)
    assert "Attention Is All You Need" in llm.prompts[0]
    assert "do not propose them again" in llm.prompts[0]


def test_limit_is_respected(ctx):
    ctx._llm = ScoutLLM([[paper(f"Paper {i}") for i in range(20)]])
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False)
    assert len(res.candidates) == 5


def test_a_failing_scout_does_not_sink_the_run(ctx):
    class Flaky(ScoutLLM):
        def json(self, prompt, **kw):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                raise RuntimeError("angle blew up")
            return {"papers": [paper("Survivor")]}

    ctx._llm = Flaky([])
    res = discover(ctx, "topic", limit=5, angles=2, use_references=False)
    assert [c.title for c in res.candidates] == ["Survivor"]
    assert res.scout_errors


# --------------------------------------------------------------------------
# Reference mining
# --------------------------------------------------------------------------

def build(entries):
    return [make_entry(key=f"e{i}", title=f"Paper {i}", arxiv_id=f"200{i}.0000{i}",
                       references=refs)
            for i, refs in enumerate(entries)]


def test_mine_references_ranks_by_cocitation():
    entries = build([
        [Reference(title="Cited By All"), Reference(title="Cited Once")],
        [Reference(title="Cited By All")],
        [Reference(title="Cited By All"), Reference(title="Cited Twice")],
    ])
    entries[2].references.append(Reference(title="Cited Twice"))
    mined = mine_references(entries, set(), set())
    assert mined[0].title == "Cited By All"
    assert mined[0].cocitations == 3


def test_mine_references_needs_two_citations_once_the_library_is_big_enough():
    entries = build([[Reference(title="Only Once")], [], []])
    assert mine_references(entries, set(), set()) == []


def test_mine_references_accepts_single_citations_in_a_small_library():
    entries = build([[Reference(title="Only Once")]])
    assert [c.title for c in mine_references(entries, set(), set())] == ["Only Once"]


def test_mine_references_skips_works_already_in_the_library():
    entries = build([[Reference(title="Already Here")], [Reference(title="Already Here")]])
    known = {slugify("Already Here", 120)}
    assert mine_references(entries, known, set()) == []


def test_mine_references_counts_a_duplicate_citation_once():
    entries = build([
        [Reference(title="Twice In One Paper"), Reference(title="Twice In One Paper")],
    ])
    assert mine_references(entries, set(), set())[0].cocitations == 1


def test_mined_and_scouted_candidates_merge(ctx):
    for i in range(2):
        ctx.library.save_entry(make_entry(
            key=f"e{i}", title=f"Existing {i}", arxiv_id=f"100{i}.0000{i}",
            references=[Reference(title="Overlapping Work", doi="10.1234/over")],
        ))
    ctx._llm = ScoutLLM([[paper("Overlapping Work", doi="10.1234/over")]])
    res = discover(ctx, "topic", limit=10, angles=1, use_references=True)
    matches = [c for c in res.candidates if c.title == "Overlapping Work"]
    assert len(matches) == 1
    assert matches[0].source == "both"


def test_scouts_are_also_told_about_mined_candidates(ctx):
    for i in range(2):
        ctx.library.save_entry(make_entry(
            key=f"e{i}", title=f"Existing {i}", arxiv_id=f"100{i}.0000{i}",
            references=[Reference(title="Mined Work")],
        ))
    llm = ScoutLLM([[paper("Fresh Web Result")]])
    ctx._llm = llm
    discover(ctx, "topic", limit=10, angles=1, use_references=True)
    scout_prompt = [p for p in llm.prompts if "Search from this specific angle" in p][0]
    assert "Mined Work" in scout_prompt


def test_web_can_be_disabled(ctx):
    ctx.library.save_entry(make_entry(
        references=[Reference(title="From References")]))
    ctx._llm = ScoutLLM([[paper("Should Not Appear")]])
    res = discover(ctx, "t", limit=5, angles=2, use_references=True, use_web=False)
    assert [c.title for c in res.candidates] == ["From References"]


def test_references_can_be_disabled(ctx):
    ctx.library.save_entry(make_entry(references=[Reference(title="From References")]))
    ctx._llm = ScoutLLM([[paper("From Web")]])
    res = discover(ctx, "t", limit=5, angles=1, use_references=False)
    assert [c.title for c in res.candidates] == ["From Web"]


# --------------------------------------------------------------------------
# The parallel runner itself
# --------------------------------------------------------------------------

def test_runner_preserves_input_order():
    res = run_parallel([1, 2, 3, 4], lambda x: x * 10, workers=4)
    assert [r.value for r in res] == [10, 20, 30, 40]


def test_runner_isolates_failures():
    def f(x):
        if x == 2:
            raise ValueError("bad")
        return x

    res = run_parallel([1, 2, 3], f, workers=3)
    assert res[0].ok and res[2].ok
    assert not res[1].ok
    assert isinstance(res[1].error, ValueError)


def test_runner_handles_empty_input():
    assert run_parallel([], lambda x: x, workers=4) == []


def test_runner_single_worker_path():
    res = run_parallel([1, 2], lambda x: x + 1, workers=1)
    assert [r.value for r in res] == [2, 3]
