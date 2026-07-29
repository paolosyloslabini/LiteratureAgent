"""`lit check`: verifying metadata that looks wrong.

The audit is pure, so it is tested directly. Everything else is tested for
policy: which corrections are believed, which are thrown away, and what is
actually written to the entry. The failure mode this command exists to avoid is
a confident replacement value that no source supports — a venue that sounds
right for the topic — so most of these are about refusing one.
"""

from __future__ import annotations

from datetime import date

import pytest
from factories import make_entry, make_meta
from rich.console import Console

from lit.actions.check import (
    STALE_CITATION_YEARS,
    arxiv_year,
    audit,
    check_entries,
)
from lit.actions.context import Ctx
from lit import entryfile
from lit.llm import LLMError

THIS_YEAR = date.today().year
SOURCE = "https://proceedings.neurips.cc/paper/2017"

CONFIRMED_VENUE = {
    "fields": [{
        "field": "venue",
        "verdict": "wrong",
        "proposed": "Advances in Neural Information Processing Systems",
        "evidence": "the proceedings front matter lists this paper in NeurIPS 2017",
        "source_url": SOURCE,
    }],
}


class StubLLM:
    available = True

    def __init__(self, payload=None, fail=False):
        self.payload = payload
        self.fail = fail
        self.calls = 0
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def json(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        self.kwargs.append(kw)
        if self.fail:
            raise LLMError("model unavailable")
        return dict(self.payload if self.payload is not None else CONFIRMED_VENUE)

    def run(self, prompt, **kw):  # pragma: no cover - unused here
        raise NotImplementedError


class StubHttp:
    def __init__(self, live=(SOURCE,), unavailable=()):
        self.live = set(live)
        self.unavailable = set(unavailable)
        self.asked: list[str] = []
        self._unavailable_count = 0

    @property
    def unavailable_count(self) -> int:
        return self._unavailable_count

    def get(self, url, **kw):
        self.asked.append(url)
        if url in self.unavailable:
            self._unavailable_count += 1
            return None
        return object() if url in self.live else None

    def close(self):
        pass


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


def wire(ctx, monkeypatch, payload=None, *, fail=False, live=(SOURCE,),
         unavailable=(), meta=None, crossref=None):
    """Stub the model, the network and both metadata entry points."""
    llm = StubLLM(payload, fail=fail)
    ctx._llm = llm
    ctx._http = StubHttp(live=live, unavailable=unavailable)
    monkeypatch.setattr("lit.actions.check.resolve_metadata",
                        lambda *a, **kw: meta)
    monkeypatch.setattr("lit.actions.check.from_crossref",
                        lambda *a, **kw: crossref)
    return llm


# --------------------------------------------------------------------------
# The audit — no network, no tokens
# --------------------------------------------------------------------------

def test_a_complete_entry_looks_clean():
    assert audit(make_entry()) == []


@pytest.mark.parametrize("venue", [
    "Elsevier BV",
    "Springer Science and Business Media LLC",
    "Association for Computing Machinery",
    "Nature Portfolio",
    "Wiley",
])
def test_a_publisher_in_the_venue_field_is_suspicious(venue):
    problems = audit(make_entry(venue=venue))
    assert any(s.field == "venue" and "publisher" in s.problem for s in problems)


@pytest.mark.parametrize("venue", ["arXiv", "bioRxiv", "Research Square", "SSRN"])
def test_a_preprint_server_in_the_venue_field_is_suspicious(venue):
    problems = audit(make_entry(venue=venue))
    assert any(s.field == "venue" and "preprint server" in s.problem
               for s in problems)


def test_a_published_paper_with_no_venue_is_suspicious():
    problems = audit(make_entry(venue=None, type="journal article"))
    assert any(s.field == "venue" for s in problems)


def test_a_preprint_with_no_venue_is_not_suspicious():
    """Having no venue is the normal, correct state for a preprint."""
    problems = audit(make_entry(venue=None, type="preprint"))
    assert not any(s.field == "venue" for s in problems)


def test_a_workshop_venue_filed_as_a_conference_paper_is_suspicious():
    problems = audit(make_entry(venue="NeurIPS 2023 Workshop on Instruction Tuning"))
    assert any(s.field == "type" and "workshop" in s.problem for s in problems)


def test_an_old_paper_with_no_citations_is_suspicious():
    old = THIS_YEAR - STALE_CITATION_YEARS - 1
    problems = audit(make_entry(year=old, citation_count=0, arxiv_id=None))
    assert any(s.field == "citation_count" for s in problems)


def test_a_new_paper_with_no_citations_is_not_suspicious():
    problems = audit(make_entry(year=THIS_YEAR, citation_count=0, arxiv_id=None))
    assert not any(s.field == "citation_count" for s in problems)


def test_a_year_before_the_papers_own_preprint_is_suspicious():
    problems = audit(make_entry(year=2015, arxiv_id="1706.03762"))
    assert any(s.field == "year" and "predates" in s.problem for s in problems)


def test_a_year_after_the_preprint_is_fine():
    """A preprint published two years later is the ordinary case."""
    problems = audit(make_entry(year=2019, arxiv_id="1706.03762"))
    assert not any(s.field == "year" for s in problems)


@pytest.mark.parametrize("year", [None, 3999, 1200])
def test_a_missing_or_impossible_year_is_suspicious(year):
    assert any(s.field == "year" for s in audit(make_entry(year=year)))


def test_a_collapsed_author_list_is_suspicious():
    problems = audit(make_entry(authors=["Vaswani et al."]))
    assert any(s.field == "authors" for s in problems)


def test_no_authors_at_all_is_suspicious():
    assert any(s.field == "authors" for s in audit(make_entry(authors=[])))


def test_unrendered_markup_in_a_title_is_suspicious():
    problems = audit(make_entry(title="&lt;p&gt;Attention Is All You Need"))
    assert any(s.field == "title" for s in problems)


@pytest.mark.parametrize("arxiv_id,year", [
    ("1706.03762", 2017),
    ("2301.00234", 2023),
    ("cs.CL/0701001", 2007),
    ("hep-th/9201001", 1992),
    (None, None),
    ("not-an-id", None),
])
def test_arxiv_year_comes_from_the_identifier(arxiv_id, year):
    assert arxiv_year(arxiv_id) == year


# --------------------------------------------------------------------------
# What gets checked at all
# --------------------------------------------------------------------------

def test_an_entry_that_looks_fine_is_still_checked(ctx, lib, monkeypatch):
    """The point of the command: code does not get to decide it looks fine.

    `make_entry()` passes the local audit cleanly, and is checked anyway —
    running `check` is the statement that the stored record is not trusted.
    """
    llm = wire(ctx, monkeypatch, {"fields": []})
    entry = lib.save_entry(make_entry())
    assert audit(entry) == []

    res = check_entries(ctx, [entry])[0]

    assert llm.calls == 1
    assert res.asked_model
    assert res.status != "clean"


def test_the_agent_is_asked_even_when_the_indexes_already_fixed_it(
        ctx, lib, monkeypatch):
    """A free index correction is not a reason to skip the second opinion."""
    llm = wire(ctx, monkeypatch, {"fields": []},
               meta=make_meta(venue="Advances in Neural Information Processing "
                                    "Systems"))
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    check_entries(ctx, [entry], fix=True)

    assert llm.calls == 1


def test_the_checker_gets_web_tools_and_the_cheap_model(ctx, lib, monkeypatch):
    llm = wire(ctx, monkeypatch)
    check_entries(ctx, [lib.save_entry(make_entry(venue="Elsevier BV"))])

    kw = llm.kwargs[0]
    assert kw["tools"] == ["WebSearch", "WebFetch"]
    assert kw["role"] == "check"
    assert ctx.cfg.llm.model_for("check") == "haiku"
    assert ctx.cfg.llm.effort_for("check") == "low"


def test_the_prompt_names_the_stored_value_and_the_problem(ctx, lib, monkeypatch):
    llm = wire(ctx, monkeypatch)
    check_entries(ctx, [lib.save_entry(make_entry(venue="Elsevier BV"))])

    prompt = llm.prompts[0]
    assert "Elsevier BV" in prompt
    assert "publisher" in prompt
    assert "Attention Is All You Need" in prompt


def test_nothing_to_do_costs_nothing(ctx, monkeypatch):
    llm = wire(ctx, monkeypatch)
    assert check_entries(ctx, []) == []
    assert llm.calls == 0


# --------------------------------------------------------------------------
# The indexes get asked before the model
# --------------------------------------------------------------------------

def test_a_publisher_venue_is_fixed_from_the_indexes(ctx, lib, monkeypatch):
    """The indexes are asked first because they are free and authoritative."""
    wire(ctx, monkeypatch, {"fields": []}, meta=make_meta(
        venue="Advances in Neural Information Processing Systems"))
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "fixed"
    assert lib.get(entry.key).venue == (
        "Advances in Neural Information Processing Systems")


def test_an_index_venue_that_is_also_a_publisher_is_not_adopted(
        ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": []}, meta=make_meta(venue="Springer Nature"))
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).venue == "Elsevier BV"  # unchanged, still flagged


def test_a_real_venue_is_not_overwritten_by_the_indexes(ctx, lib, monkeypatch):
    """Only a placeholder venue is replaced; a good one is left alone."""
    wire(ctx, monkeypatch, {"fields": []}, meta=make_meta(venue="ICML"))
    entry = lib.save_entry(make_entry(venue="NeurIPS", authors=[]))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).venue == "NeurIPS"


def test_a_stale_citation_count_is_refreshed_from_the_indexes(
        ctx, lib, monkeypatch):
    old = THIS_YEAR - STALE_CITATION_YEARS - 1
    wire(ctx, monkeypatch, {"fields": []},
         meta=make_meta(year=old, citation_count=4200))
    entry = lib.save_entry(make_entry(year=old, citation_count=0, arxiv_id=None))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).citation_count == 4200


# --------------------------------------------------------------------------
# Refusing what cannot be believed
# --------------------------------------------------------------------------

def test_a_proposed_publisher_name_is_refused(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": [{
        "field": "venue", "verdict": "wrong", "proposed": "Springer Nature",
        "evidence": "the page says Springer", "source_url": SOURCE,
    }]})
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "unresolved"
    assert any("publisher" in u for u in res.unresolved)
    assert lib.get(entry.key).venue == "Elsevier BV"


def test_a_correction_with_no_evidence_is_refused(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": [{
        "field": "venue", "verdict": "wrong", "proposed": "NeurIPS",
        "evidence": "", "source_url": SOURCE,
    }]})
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "unresolved"
    assert any("no evidence" in u for u in res.unresolved)
    assert lib.get(entry.key).venue == "Elsevier BV"


def test_a_correction_whose_source_does_not_resolve_is_refused(
        ctx, lib, monkeypatch):
    """An invented venue comes with an invented page to have read it on."""
    wire(ctx, monkeypatch, live=())
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "unresolved"
    assert any("does not resolve" in u for u in res.unresolved)
    assert lib.get(entry.key).venue == "Elsevier BV"


def test_an_unreachable_source_does_not_count_against_the_answer(
        ctx, lib, monkeypatch):
    """Our network failing is not evidence the model was wrong."""
    wire(ctx, monkeypatch, live=(), unavailable=(SOURCE,))
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "fixed"
    assert "Neural Information Processing" in lib.get(entry.key).venue


def test_a_citation_count_is_never_taken_from_the_model(ctx, lib, monkeypatch):
    old = THIS_YEAR - STALE_CITATION_YEARS - 1
    wire(ctx, monkeypatch, {"fields": [{
        "field": "citation_count", "verdict": "wrong", "proposed": 99999,
        "evidence": "Google Scholar says 99999", "source_url": SOURCE,
    }]})
    entry = lib.save_entry(make_entry(year=old, citation_count=0, arxiv_id=None))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).citation_count == 0  # untouched by the model


def test_an_author_list_is_never_taken_from_the_model(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": [{
        "field": "authors", "verdict": "wrong",
        "proposed": ["A Ghostwriter"],
        "evidence": "the page lists one author", "source_url": SOURCE,
    }]})
    entry = lib.save_entry(make_entry(authors=["Vaswani et al."]))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).authors == ["Vaswani et al."]


def test_an_implausible_proposed_year_is_refused(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": [{
        "field": "year", "verdict": "wrong", "proposed": 1204,
        "evidence": "the page says 1204", "source_url": SOURCE,
    }]})
    entry = lib.save_entry(make_entry(year=None))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "unresolved"
    assert lib.get(entry.key).year is None


def test_a_proposed_year_before_the_preprint_is_refused(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": [{
        "field": "year", "verdict": "wrong", "proposed": 2010,
        "evidence": "the page says 2010", "source_url": SOURCE,
    }]})
    entry = lib.save_entry(make_entry(year=None, arxiv_id="1706.03762"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert any("predates" in u for u in res.unresolved)
    assert lib.get(entry.key).year is None


def test_a_failed_check_leaves_the_entry_alone(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, fail=True)
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "unresolved"
    assert lib.get(entry.key).venue == "Elsevier BV"


# --------------------------------------------------------------------------
# A DOI is confirmed against Crossref, not taken on trust
# --------------------------------------------------------------------------

def test_a_proposed_doi_is_adopted_when_crossref_agrees(ctx, lib, monkeypatch):
    """The indexes answer nothing here, so the agent's DOI is the only lead."""
    wire(
        ctx, monkeypatch,
        {"fields": [], "doi": "10.5555/3295222.3295349"},
        crossref=make_meta(title="Attention Is All You Need"),
        meta=None,
    )
    entry = lib.save_entry(make_entry(venue="Elsevier BV", doi=None))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "fixed"
    assert lib.get(entry.key).doi == "10.5555/3295222.3295349"


def test_a_doi_for_a_different_paper_is_refused(ctx, lib, monkeypatch):
    """The most expensive mistake available: adopting another work's identity."""
    wire(
        ctx, monkeypatch,
        {"fields": [], "doi": "10.1000/some-other-paper"},
        crossref=make_meta(title="A Completely Unrelated Study of Soil Nitrogen"),
    )
    entry = lib.save_entry(make_entry(venue="Elsevier BV", doi=None))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).doi is None


def test_a_doi_crossref_does_not_know_is_refused(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, {"fields": [], "doi": "10.1000/nonexistent"},
         crossref=None)
    entry = lib.save_entry(make_entry(venue="Elsevier BV", doi=None))

    check_entries(ctx, [entry], fix=True)

    assert lib.get(entry.key).doi is None


# --------------------------------------------------------------------------
# Writing, and not writing
# --------------------------------------------------------------------------

def test_without_fix_nothing_is_written(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, meta=make_meta(venue="ICML"))
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    res = check_entries(ctx, [entry])[0]

    assert res.status == "proposed"
    assert res.changes
    assert "--fix" in res.message
    assert lib.get(entry.key).venue == "Elsevier BV"


def test_an_applied_correction_records_where_it_came_from(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch)
    entry = lib.save_entry(make_entry(venue="Elsevier BV"))

    check_entries(ctx, [entry], fix=True)

    note = lib.get(entry.key).check_note
    assert "venue" in note
    assert "web search" in note
    assert "proceedings front matter" in note


def test_a_correction_never_touches_the_summaries_or_notes(ctx, lib, monkeypatch):
    wire(ctx, monkeypatch, meta=make_meta(venue="ICML"))
    entry = make_entry(venue="Elsevier BV")
    lib.save_entry(entry)
    entryfile.set_notes(lib.entry_path(entry.key), "my own reading of this paper")

    check_entries(ctx, [lib.get(entry.key)], fix=True)

    saved = lib.get(entry.key)
    assert saved.notes == "my own reading of this paper"
    assert saved.one_liner == entry.one_liner
    assert [s.name for s in saved.sections] == [s.name for s in entry.sections]


def test_an_entry_that_checks_out_is_reported_as_confirmed(ctx, lib, monkeypatch):
    """Looking wrong and being wrong are different; saying so stops a re-check.

    A book's imprint really is what a citation names, so "MIT Press" trips the
    publisher heuristic and is nonetheless correct.
    """
    wire(ctx, monkeypatch, {"fields": [{
        "field": "venue", "verdict": "correct", "proposed": None,
        "evidence": "the title page gives the imprint as MIT Press",
        "source_url": SOURCE,
    }]}, meta=None)
    entry = lib.save_entry(make_entry(venue="MIT Press", type="book"))

    res = check_entries(ctx, [entry], fix=True)[0]

    assert res.status == "confirmed"
    assert res.confirmed == ["venue"]
    assert not res.unresolved
    assert lib.get(entry.key).venue == "MIT Press"


def test_several_entries_are_checked_in_one_run(ctx, lib, monkeypatch):
    llm = wire(ctx, monkeypatch)
    entries = [
        lib.save_entry(make_entry(venue="Elsevier BV")),
        lib.save_entry(make_entry(key="doe2010obscure", title="An Obscure Study",
                                  venue="Wiley")),
    ]

    results = check_entries(ctx, entries, fix=True)

    assert llm.calls == 2
    assert {r.key for r in results} == {"vaswani2017attention", "doe2010obscure"}
