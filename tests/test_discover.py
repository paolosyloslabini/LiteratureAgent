"""The `find` orchestrator: candidate pooling, de-duplication, reference mining.

The point of the orchestrator is that the pool is built and de-duplicated once,
before any expensive read happens, so no two workers read the same paper and
nothing already in the library is proposed again. It is then ranked on
relevance to the query first and measured importance second.
"""

from __future__ import annotations

import re

import pytest
from factories import make_entry
from rich.console import Console

from lit.actions import discover as D
from lit.actions.context import Ctx
from lit.actions.discover import Candidate, discover, mine_references
from lit.fetch.metadata import PaperMeta
from lit.models import Reference, slugify
from lit.runner import run_parallel


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Both free sources reach the network; tests that care stub them explicitly.

    `_lookup_meta` is the enrichment lookup, `_search_facet` the indexed search.
    """
    monkeypatch.setattr(D, "_lookup_meta", lambda ctx, c: None)
    monkeypatch.setattr(D, "_search_facet", lambda ctx, q, facet, limit: [])


def stub_metadata(monkeypatch, by_title: dict):
    """Give named candidates a citation count / venue, as OpenAlex would."""
    def fake(ctx, c):
        spec = by_title.get(c.title)
        return PaperMeta(**spec) if spec else None
    monkeypatch.setattr(D, "_lookup_meta", fake)


def stub_search(monkeypatch, by_facet: dict):
    """Script what each free search facet returns, as PaperMeta records."""
    def fake(ctx, query, facet, limit):
        return [PaperMeta(**spec) if isinstance(spec, dict) else spec
                for spec in by_facet.get(facet, [])][:limit]
    monkeypatch.setattr(D, "_search_facet", fake)


def hit(title, **kw):
    """A search hit as the indexes return it."""
    return {"title": title, "authors": ["A B"], "year": 2021, **kw}


def scouted(ctx, query="topic", **kw):
    """`discover` with the scout swarm turned on, as `--parallel` does."""
    kw.setdefault("use_scouts", True)
    kw.setdefault("use_references", False)
    return discover(ctx, query, **kw)


_LISTED = re.compile(r"^ {2}\[(\d+)\] (.+)$", re.M)


def _titles_in(prompt: str) -> dict[int, str]:
    """Recover the titles the ranking prompt listed, with their indices."""
    out = {}
    for idx, rest in _LISTED.findall(prompt):
        title = rest.split(" — ")[0]
        title = re.sub(r" \(\d{4}\)$", "", title)
        out[int(idx)] = title
    return out


class ScoutLLM:
    """Scripted papers per angle; records prompts; scores relevance on request.

    `relevance` maps title -> score for the ranking call. Anything unlisted
    scores 0.9, so a test only has to name the papers it cares about.
    """

    available = True

    def __init__(self, per_angle, relevance: dict | None = None):
        self.per_angle = per_angle
        self.relevance = relevance or {}
        self.prompts: list[str] = []
        self.rank_prompts: list[str] = []
        self.calls = 0

    def json(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        if kw.get("role") == "rank":
            self.rank_prompts.append(prompt)
            return {"scores": [
                {"index": i, "relevance": self.relevance.get(t, 0.9), "why": "fits"}
                for i, t in _titles_in(prompt).items()
            ]}
        idx = len([p for p in self.prompts if "Search from this specific angle" in p]) - 1
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
    res = discover(ctx, "topic", limit=10, angles=2, use_references=False, use_scouts=True)
    titles = [c.title for c in res.candidates]
    assert titles.count("Shared Paper") == 1


def test_agreement_across_angles_ranks_higher(ctx):
    ctx._llm = ScoutLLM([
        [paper("Agreed On"), paper("Solo A")],
        [paper("Agreed On"), paper("Solo B")],
    ])
    res = discover(ctx, "topic", limit=10, angles=2, use_references=False, use_scouts=True)
    assert res.candidates[0].title == "Agreed On"


def test_dedup_matches_on_doi_across_differing_titles(ctx):
    ctx._llm = ScoutLLM([
        [paper("Attention Is All You Need", doi="10.1234/abcd")],
        [paper("ATTENTION IS ALL YOU NEED (v2)", doi="10.1234/abcd")],
    ])
    res = discover(ctx, "topic", limit=10, angles=2, use_references=False, use_scouts=True)
    assert len(res.candidates) == 1


def test_papers_already_in_the_library_are_dropped(ctx):
    ctx.library.save_entry(make_entry())
    ctx._llm = ScoutLLM([[paper("Attention Is All You Need"), paper("Something New")]])
    res = discover(ctx, "topic", limit=10, angles=1, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates] == ["Something New"]


def test_scouts_are_told_what_is_already_present(ctx):
    ctx.library.save_entry(make_entry())
    llm = ScoutLLM([[paper("New Thing")]])
    ctx._llm = llm
    discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert "Attention Is All You Need" in llm.prompts[0]
    assert "do not propose them again" in llm.prompts[0]


def test_limit_is_respected(ctx):
    ctx._llm = ScoutLLM([[paper(f"Paper {i}") for i in range(20)]])
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
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
    res = discover(ctx, "topic", limit=5, angles=2, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates] == ["Survivor"]
    assert res.scout_errors


# --------------------------------------------------------------------------
# The default path: free indexed search, no scout agents
# --------------------------------------------------------------------------

def test_scouts_do_not_run_unless_asked_for(ctx, monkeypatch):
    """The whole point of --parallel: no agent spends a token without it."""
    stub_search(monkeypatch, {"relevance": [hit("Found For Free")]})
    llm = ScoutLLM([[paper("Cost Money")]])
    ctx._llm = llm
    res = discover(ctx, "topic", limit=5, use_references=False)

    assert [c.title for c in res.candidates] == ["Found For Free"]
    assert not any("Search from this specific angle" in p for p in llm.prompts)
    # One call, and it is the ranking pass — nothing else is model-driven.
    assert llm.calls == 1
    assert len(llm.rank_prompts) == 1


def test_parallel_adds_scouts_on_top_of_the_free_search(ctx, monkeypatch):
    stub_search(monkeypatch, {"relevance": [hit("Found For Free")]})
    ctx._llm = ScoutLLM([[paper("Found By A Scout")]])
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False,
                   use_scouts=True)

    assert {c.title for c in res.candidates} == {"Found For Free", "Found By A Scout"}
    assert res.from_search == 1
    assert res.from_scouts == 1


def test_every_facet_is_asked_and_merged(ctx, monkeypatch):
    ctx._llm = ScoutLLM([])
    stub_search(monkeypatch, {
        "relevance": [hit("Best Match")],
        "foundational": [hit("The Classic", year=1998)],
        "recent": [hit("Brand New", year=2025)],
        "surveys": [hit("A Survey Of Everything")],
        "preprints": [hit("Fresh Preprint", arxiv_id="2501.00001")],
    })
    res = discover(ctx, "topic", limit=10, use_references=False)
    assert {c.title for c in res.candidates} == {
        "Best Match", "The Classic", "Brand New", "A Survey Of Everything",
        "Fresh Preprint",
    }


def test_a_paper_found_by_several_facets_appears_once_and_ranks_up(ctx, monkeypatch):
    stub_search(monkeypatch, {
        "relevance": [hit("Agreed On", doi="10.1/x"), hit("Solo", doi="10.1/y")],
        "foundational": [hit("Agreed On", doi="10.1/x")],
        "surveys": [hit("Agreed On", doi="10.1/x")],
    })
    ctx._llm = ScoutLLM([])
    res = discover(ctx, "topic", limit=10, use_references=False)
    titles = [c.title for c in res.candidates]
    assert titles.count("Agreed On") == 1
    assert titles[0] == "Agreed On"


def test_search_hits_carry_their_own_metrics_and_skip_enrichment(ctx, monkeypatch):
    """An indexed hit already knows its citations; looking them up again is waste."""
    stub_search(monkeypatch, {
        "relevance": [hit("Measured", citation_count=4321, venue="ICML")],
    })
    ctx._llm = ScoutLLM([])
    def explode(_ctx, _c):
        raise AssertionError("enrichment ran for a candidate that needed nothing")
    monkeypatch.setattr(D, "_lookup_meta", explode)

    res = discover(ctx, "topic", limit=5, use_references=False)
    assert res.candidates[0].citation_count == 4321
    assert res.candidates[0].venue == "ICML"


def test_abstracts_from_search_reach_the_ranking_call(ctx, monkeypatch):
    """Without a scout's note, the abstract is what the ranker has to judge on."""
    stub_search(monkeypatch, {
        "relevance": [hit("Opaque Title", abstract="We introduce a benchmark for "
                                                   "causal discovery in agents.")],
    })
    llm = ScoutLLM([])
    ctx._llm = llm
    discover(ctx, "topic", limit=5, use_references=False)
    assert "causal discovery in agents" in llm.rank_prompts[0]


def test_a_failing_facet_does_not_sink_the_run(ctx, monkeypatch):
    ctx._llm = ScoutLLM([])
    def flaky(ctx_, query, facet, limit):
        if facet == "foundational":
            raise RuntimeError("openalex is down")
        return [PaperMeta(**hit("Survivor"))] if facet == "relevance" else []
    monkeypatch.setattr(D, "_search_facet", flaky)

    res = discover(ctx, "topic", limit=5, use_references=False)
    assert [c.title for c in res.candidates] == ["Survivor"]


# --------------------------------------------------------------------------
# Pool triage: corroboration decides who is scored, but not exclusively
# --------------------------------------------------------------------------

def test_uncorroborated_candidates_keep_a_share_of_the_pool():
    """A new direction is cited by nothing you own and found by one source."""
    corroborated = [Candidate(title=f"Co-Cited {i}", cocitations=3) for i in range(40)]
    solo = [Candidate(title=f"Novel {i}", angles=["relevance"]) for i in range(20)]
    kept = D._trim_pool(sorted(corroborated + solo, key=lambda c: c.triage_key), 20)

    assert len(kept) == 20
    assert sum(1 for c in kept if not c.corroborated) == 6  # 30% of 20


def test_the_quota_is_not_a_reservation_when_nothing_needs_it():
    """No uncorroborated candidates means the cap goes entirely to the rest."""
    only_strong = [Candidate(title=f"Co-Cited {i}", cocitations=2) for i in range(30)]
    kept = D._trim_pool(only_strong, 10)
    assert len(kept) == 10
    assert all(c.corroborated for c in kept)


def test_the_quota_backfills_when_too_few_uncorroborated_exist():
    pool = ([Candidate(title=f"Co-Cited {i}", cocitations=2) for i in range(30)]
            + [Candidate(title="Lonely", angles=["relevance"])])
    kept = D._trim_pool(sorted(pool, key=lambda c: c.triage_key), 10)
    assert len(kept) == 10
    assert "Lonely" in [c.title for c in kept]


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
    res = discover(ctx, "topic", limit=10, angles=1, use_references=True, use_scouts=True)
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
    discover(ctx, "topic", limit=10, angles=1, use_references=True, use_scouts=True)
    scout_prompt = [p for p in llm.prompts if "Search from this specific angle" in p][0]
    assert "Mined Work" in scout_prompt


def test_web_can_be_disabled(ctx, monkeypatch):
    """`--no-web` is offline-except-my-library: it silences the scouts too."""
    stub_search(monkeypatch, {"relevance": [hit("From The Index")]})
    ctx.library.save_entry(make_entry(
        references=[Reference(title="From References")]))
    ctx._llm = ScoutLLM([[paper("Should Not Appear")]])
    res = discover(ctx, "t", limit=5, angles=2, use_references=True, use_web=False,
                   use_scouts=True)
    assert [c.title for c in res.candidates] == ["From References"]


def test_references_can_be_disabled(ctx):
    ctx.library.save_entry(make_entry(references=[Reference(title="From References")]))
    ctx._llm = ScoutLLM([[paper("From Web")]])
    res = discover(ctx, "t", limit=5, angles=1, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates] == ["From Web"]


# --------------------------------------------------------------------------
# Ranking: relevance to the query first, measured importance second
# --------------------------------------------------------------------------

def test_relevance_outweighs_importance(ctx, monkeypatch):
    """A celebrated paper that does not answer the question loses to one that does."""
    ctx._llm = ScoutLLM(
        [[paper("Famous But Off Topic", year=2015), paper("On Topic And Modest")]],
        relevance={"Famous But Off Topic": 0.4, "On Topic And Modest": 0.95},
    )
    stub_metadata(monkeypatch, {
        "Famous But Off Topic": {"citation_count": 50_000, "venue": "NeurIPS"},
    })
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates][0] == "On Topic And Modest"


def test_importance_decides_between_equally_relevant_papers(ctx, monkeypatch):
    ctx._llm = ScoutLLM([[paper("Well Cited"), paper("Never Cited")]])
    stub_metadata(monkeypatch, {
        "Well Cited": {"citation_count": 20_000, "venue": "ICML"},
    })
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates] == ["Well Cited", "Never Cited"]


def test_off_topic_candidates_are_dropped_not_just_ranked_low(ctx):
    ctx._llm = ScoutLLM(
        [[paper("Fits The Query"), paper("Shares Only Vocabulary")]],
        relevance={"Shares Only Vocabulary": 0.1},
    )
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates] == ["Fits The Query"]
    assert res.dropped_off_topic == 1


def test_mined_references_no_longer_automatically_outrank_scouts(ctx):
    """The old ranking put every co-cited reference above every web result."""
    for i in range(3):
        ctx.library.save_entry(make_entry(
            key=f"e{i}", title=f"Existing {i}", arxiv_id=f"100{i}.0000{i}",
            references=[Reference(title="Tangential But Co-Cited")],
        ))
    ctx._llm = ScoutLLM(
        [[paper("Exactly What Was Asked")]],
        relevance={"Tangential But Co-Cited": 0.5, "Exactly What Was Asked": 0.95},
    )
    res = discover(ctx, "topic", limit=5, angles=1, use_references=True, use_scouts=True)
    assert res.candidates[0].title == "Exactly What Was Asked"


def test_every_source_is_scored_by_the_same_call(ctx):
    for i in range(2):
        ctx.library.save_entry(make_entry(
            key=f"e{i}", title=f"Existing {i}", arxiv_id=f"100{i}.0000{i}",
            references=[Reference(title="From References")],
        ))
    llm = ScoutLLM([[paper("From The Web")]])
    ctx._llm = llm
    discover(ctx, "topic", limit=5, angles=1, use_references=True, use_scouts=True)
    assert len(llm.rank_prompts) == 1
    listed = set(_titles_in(llm.rank_prompts[0]).values())
    assert {"From References", "From The Web"} <= listed


def test_enrichment_fills_citations_before_the_cut(ctx, monkeypatch):
    ctx._llm = ScoutLLM([[paper("Measured")]])
    stub_metadata(monkeypatch, {"Measured": {"citation_count": 1234, "venue": "ICLR"}})
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert res.candidates[0].citation_count == 1234
    assert res.candidates[0].venue == "ICLR"


def test_identifiers_from_a_title_match_are_not_adopted(ctx, monkeypatch):
    """A title lookup can land on a re-registration; its DOI must not be reused."""
    ctx._llm = ScoutLLM([[paper("Attention Is All You Need")]])
    def fake(_ctx, c):
        return PaperMeta(citation_count=99, doi="10.9999/wrong", title_matched=True)
    monkeypatch.setattr(D, "_lookup_meta", fake)
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert res.candidates[0].citation_count == 99
    assert res.candidates[0].doi is None


def test_ranking_survives_a_failed_relevance_pass(ctx, monkeypatch):
    from lit.llm import LLMError

    class NoRanking(ScoutLLM):
        def json(self, prompt, **kw):
            if kw.get("role") == "rank":
                raise LLMError("scoring blew up")
            return super().json(prompt, **kw)

    ctx._llm = NoRanking([[paper("Widely Cited Work"), paper("Never Cited Work")]])
    stub_metadata(monkeypatch,
                  {"Widely Cited Work": {"citation_count": 9_000, "venue": "ACL"}})
    res = discover(ctx, "topic", limit=5, angles=1, use_references=False, use_scouts=True)
    assert [c.title for c in res.candidates] == ["Widely Cited Work",
                                                 "Never Cited Work"]
    assert all(c.relevance is None for c in res.candidates)


def test_recent_papers_are_judged_on_venue_not_on_citations():
    from datetime import date

    year = date.today().year
    fresh = Candidate(title="New", year=year, venue="NeurIPS", citation_count=3)
    older = Candidate(title="Old", year=year - 20, venue="NeurIPS", citation_count=3)
    assert fresh.world_score == fresh.venue_score
    assert older.world_score < fresh.world_score


def test_a_paper_with_no_metadata_is_middling_not_worthless():
    assert 0 < Candidate(title="Obscure").importance < 0.5


def test_agreement_between_angles_is_a_bonus_not_a_floor():
    solo = Candidate(title="A", angles=["one"])
    agreed = Candidate(title="B", angles=["one", "two"])
    assert agreed.importance > solo.importance


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
