"""Search index: query sanitization, ranking and filters."""

import pytest
from factories import make_entry

from lit.models import Reference, Section
from lit.store import Store, to_fts_query


# --------------------------------------------------------------------------
# FTS query construction — users type questions, not FTS5 syntax.
# --------------------------------------------------------------------------

def test_question_becomes_a_valid_fts_query():
    q = to_fts_query("What are the current capabilities in causal discovery?")
    assert "?" not in q
    assert '"causal"' in q and '"discovery"' in q
    assert '"what"' not in q  # stopword


def test_quoted_phrase_is_preserved():
    q = to_fts_query('papers about "chain of thought" prompting')
    assert '"chain of thought"' in q


def test_punctuation_that_would_break_fts_is_neutralized():
    for raw in ["C++ (templates)", "a AND b", 'quote " unbalanced', "NEAR/2", "*"]:
        to_fts_query(raw)  # must not raise


def test_empty_query():
    assert to_fts_query("") == ""
    assert to_fts_query("the of and") == ""


def test_terms_are_deduplicated():
    q = to_fts_query("attention attention attention")
    assert q.count('"attention"') == 1


# --------------------------------------------------------------------------
# Indexing and search
# --------------------------------------------------------------------------

@pytest.fixture
def stocked(lib):
    lib.save_entry(make_entry())
    lib.save_entry(make_entry(
        key="brown2020language", title="Language Models are Few-Shot Learners",
        authors=["Tom Brown"], year=2020, venue="NeurIPS", level="A*",
        one_liner="GPT-3 shows few-shot learning emerges from scale.",
        tags=["llm", "few-shot"], citation_count=30000,
        sections=[Section("Approach", "Scale the model to 175B parameters.")],
        references=[Reference(title="Attention Is All You Need", year=2017,
                              arxiv_id="1706.03762")],
    ))
    lib.save_entry(make_entry(
        key="doe2010obscure", title="An Obscure Study of Widgets",
        authors=["Jane Doe"], year=2010, venue="Widget Weekly", level="C",
        one_liner="Widgets are studied.", tags=["widgets"], citation_count=3,
        sections=[], references=[], status="UNVERIFIED",
    ))
    return lib


def test_search_finds_by_title(stocked):
    with Store(stocked) as s:
        hits = s.search("few-shot learners")
    assert hits[0].key == "brown2020language"


def test_search_finds_by_summary_content(stocked):
    with Store(stocked) as s:
        hits = s.search("175B parameters")
    assert "brown2020language" in [h.key for h in hits]


def test_search_finds_by_abstract(stocked):
    path = stocked.entry_path("doe2010obscure")
    entry = stocked.get("doe2010obscure")
    entry.abstract = "A study of hydrodynamic widget lubrication."
    stocked.save_entry(entry)
    assert path.exists()
    with Store(stocked) as s:
        hits = s.search("hydrodynamic lubrication")
    assert "doe2010obscure" in [h.key for h in hits]


def test_search_finds_by_user_notes(stocked):
    from lit import entryfile

    entryfile.set_notes(stocked.entry_path("doe2010obscure"), "relevant to my thesis")
    with Store(stocked) as s:
        hits = s.search("thesis")
    assert "doe2010obscure" in [h.key for h in hits]


def test_search_filters_by_level(stocked):
    with Store(stocked) as s:
        hits = s.search("", level="A")
    assert "doe2010obscure" not in [h.key for h in hits]


def test_search_filters_by_tag(stocked):
    with Store(stocked) as s:
        assert [h.key for h in s.search("", tag="widgets")] == ["doe2010obscure"]


def test_search_filters_by_status(stocked):
    with Store(stocked) as s:
        hits = s.search("", status="UNVERIFIED")
    assert [h.key for h in hits] == ["doe2010obscure"]


def test_search_filters_by_year(stocked):
    with Store(stocked) as s:
        hits = s.search("", year_from=2015)
    assert "doe2010obscure" not in [h.key for h in hits]


def test_empty_query_lists_best_first(stocked):
    with Store(stocked) as s:
        hits = s.search("")
    assert hits[-1].key == "doe2010obscure"  # level C sorts last


def test_index_tracks_deletions(stocked):
    with Store(stocked) as s:
        s.sync()
        stocked.delete_entry("doe2010obscure")
        s.sync()
        assert "doe2010obscure" not in [h.key for h in s.search("")]


def test_index_tracks_edits(stocked):
    with Store(stocked) as s:
        s.sync()
        e = stocked.get("doe2010obscure")
        e.one_liner = "Now mentions quantum entanglement."
        stocked.save_entry(e)
        assert "doe2010obscure" in [h.key for h in s.search("quantum entanglement")]


def test_stats(stocked):
    with Store(stocked) as s:
        st = s.stats()
    assert st["entries"] == 3
    assert st["by_level"]["A*"] == 2
    assert st["by_status"]["UNVERIFIED"] == 1
    assert st["references"] >= 2


def test_citing_entries(stocked):
    with Store(stocked) as s:
        s.sync()
        citing = s.citing_entries(doi=None, arxiv_id="1706.03762", title=None)
    assert "brown2020language" in citing


def test_index_is_rebuildable(stocked):
    with Store(stocked) as s:
        s.sync()
    stocked.index_path.unlink()
    with Store(stocked) as s:
        assert len(s.search("")) == 3
