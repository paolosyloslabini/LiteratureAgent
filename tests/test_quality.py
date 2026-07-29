"""Level assignment must be explainable and driven by the stated metrics."""

from datetime import date

import pytest
from factories import make_meta

from lit.quality import (
    LANDMARK_VELOCITY,
    assess,
    citations_per_year,
    passes_quality_bar,
    venue_level,
)

THIS_YEAR = date.today().year


@pytest.mark.parametrize("venue,level", [
    ("NeurIPS", "A*"),
    ("Advances in Neural Information Processing Systems", "A*"),
    ("Proceedings of the 25th ACM SIGKDD International Conference", "A*"),
    ("International Conference on Machine Learning", "A*"),
    ("Nature", "A*"),
    ("AISTATS", "A"),
    ("Transactions on Machine Learning Research", "A"),
    ("PLOS ONE", "B"),
    ("Journal of Irreproducible Results", None),
    (None, None),
])
def test_venue_level(venue, level):
    hit = venue_level(venue)
    assert (hit[0] if hit else None) == level


def test_workshops_do_not_inherit_venue_rank():
    assert venue_level("NeurIPS 2023 Workshop on Instruction Tuning") is None


def test_citations_per_year():
    assert citations_per_year(1000, THIS_YEAR - 10) == pytest.approx(100.0)
    assert citations_per_year(None, 2017) is None
    assert citations_per_year(10, None) is None
    # Never divides by zero for a paper published this year.
    assert citations_per_year(5, THIS_YEAR) == 5.0


def test_ranked_venue_sets_level():
    v = assess(make_meta(venue="NeurIPS", citation_count=None, year=2015))
    assert v.level == "A*"
    assert "NeurIPS" in v.reason
    assert not v.needs_judgement


def test_high_velocity_promotes_an_A_venue_to_A_star():
    v = assess(make_meta(venue="AISTATS", year=THIS_YEAR - 10, citation_count=2000))
    assert v.level == "A*"


def test_velocity_alone_ranks_an_unlisted_venue():
    v = assess(make_meta(venue="Some Regional Symposium", type="conference paper",
                         year=THIS_YEAR - 10, citation_count=300))
    assert v.level == "A"
    assert "citations/year" in v.reason


def test_preprint_is_capped_at_A():
    v = assess(make_meta(venue=None, type="preprint",
                         year=THIS_YEAR - 10, citation_count=1200))
    assert v.level == "A"


def test_landmark_preprint_escapes_the_cap():
    cites = int(LANDMARK_VELOCITY * 10) + 100
    v = assess(make_meta(venue=None, type="preprint",
                         year=THIS_YEAR - 10, citation_count=cites))
    assert v.level == "A*"
    assert "landmark" in v.reason


def test_recent_paper_is_not_punished_for_having_no_citations():
    v = assess(make_meta(venue=None, type="preprint", year=THIS_YEAR, citation_count=0))
    assert v.level == "unranked"
    assert v.needs_judgement  # handed to the reader agent instead


def test_thin_metrics_go_to_llm_judgement_rather_than_rejection():
    v = assess(make_meta(venue="Unknown Venue", type="conference paper",
                         year=THIS_YEAR - 8, citation_count=3))
    assert v.needs_judgement


def test_known_venue_with_weak_uptake_is_decided_without_the_llm():
    v = assess(make_meta(venue="PLOS ONE", type="journal article",
                         year=THIS_YEAR - 8, citation_count=2))
    assert not v.needs_judgement


def test_no_metadata_at_all_asks_for_judgement():
    v = assess(make_meta(venue=None, year=None, citation_count=None, type="other"))
    assert v.needs_judgement


def test_quality_bar_rejects_below_minimum():
    v = assess(make_meta(venue="PLOS ONE", type="journal article",
                         year=THIS_YEAR - 8, citation_count=2))
    ok, why = passes_quality_bar(v, make_meta(), "A", allow_non_published=False)
    assert not ok
    assert "below" in why


def test_quality_bar_admits_unjudged_entries():
    v = assess(make_meta(year=THIS_YEAR, citation_count=None, venue=None))
    ok, _ = passes_quality_bar(v, make_meta(), "A*", allow_non_published=False)
    assert ok  # never rejected on missing data alone


def test_quality_bar_rejects_video_unless_allowed():
    m = make_meta(type="video")
    v = assess(m)
    assert not passes_quality_bar(v, m, "C", allow_non_published=False)[0]
    assert passes_quality_bar(v, m, "C", allow_non_published=True)[0]


def test_default_bar_is_permissive():
    """A weak-but-real paper should still get in at the default min_level."""
    m = make_meta(venue="PLOS ONE", type="journal article",
                  year=THIS_YEAR - 8, citation_count=2)
    v = assess(m)
    assert v.level == "B"
    assert passes_quality_bar(v, m, "C", allow_non_published=False)[0]


def test_default_bar_admits_a_barely_cited_unknown_paper():
    m = make_meta(venue="Some Regional Symposium", type="conference paper",
                  year=THIS_YEAR - 8, citation_count=1)
    v = assess(m)
    assert passes_quality_bar(v, m, "C", allow_non_published=False)[0]
