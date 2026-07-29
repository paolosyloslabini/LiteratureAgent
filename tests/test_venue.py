"""Venue strings: telling a venue from what an index handed us instead.

`venue` is both what a bibliography prints and what the level scorer ranks on, so
these are the rules that stop a publisher's legal name being cited as a
conference and stop a workshop paper inheriting the main track's rank.
"""

from __future__ import annotations

import pytest

from lit.venue import (
    clean_venue,
    is_placeholder,
    is_preprint_server,
    is_publisher_name,
    is_workshop,
)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("NeurIPS", "NeurIPS"),
    ("  Advances in   Neural Information\nProcessing Systems  ",
     "Advances in Neural Information Processing Systems"),
    # Crossref serves some fields as escaped markup.
    ("&lt;i&gt;Nature&lt;/i&gt;", "Nature"),
    ("&amp;nbsp;ICML", "ICML"),
    ("<i>Cell</i>", "Cell"),
    ("ICML.", "ICML"),
    (None, None),
    ("", None),
    ("   ", None),
    ("n/a", None),
    ("unknown", None),
    ("-", None),
])
def test_clean_venue(raw, expected):
    assert clean_venue(raw) == expected


# --------------------------------------------------------------------------
# Publishers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("venue", [
    "Elsevier BV",
    "Springer Science and Business Media LLC",
    "Wiley",
    "Taylor & Francis",
    "Informa UK Limited Ltd",
    "Association for Computing Machinery",
    "Institute of Electrical and Electronics Engineers",
    "Nature Portfolio",
    "Nature Publishing Group",
    "MDPI",
    "Oxford University Press",
    "MIT Press",
    "American Chemical Society",
    "Some Random Publishing",
])
def test_publisher_names_are_recognised(venue):
    assert is_publisher_name(venue)


@pytest.mark.parametrize("venue", [
    "NeurIPS",
    "Advances in Neural Information Processing Systems",
    "Nature",
    "Nature Communications",
    "Cell",
    "Science",
    "Proceedings of the 25th ACM SIGKDD International Conference",
    "Transactions on Machine Learning Research",
    "Cold Spring Harbor Perspectives in Biology",
    "Computational Linguistics",
])
def test_real_venues_are_not_mistaken_for_publishers(venue):
    assert not is_publisher_name(venue)


def test_nature_the_journal_is_not_nature_the_publisher():
    """The bug this guards: `^nature\\s` matched a publisher into an A* venue."""
    assert not is_publisher_name("Nature")
    assert not is_placeholder("Nature")
    assert is_placeholder("Nature Portfolio")


# --------------------------------------------------------------------------
# Preprint servers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("venue", [
    "arXiv", "arXiv.org", "bioRxiv", "medRxiv", "SSRN", "Research Square",
    "Preprints.org", "TechRxiv", "OSF Preprints", "Zenodo",
])
def test_preprint_servers_are_recognised(venue):
    assert is_preprint_server(venue)
    assert is_placeholder(venue)


def test_a_venue_merely_mentioning_a_server_is_not_one():
    """Matched at the start, so a real venue containing the word survives."""
    assert not is_preprint_server("Journal of Open Source Software")
    assert not is_preprint_server("Proceedings of the arXiv Workshop on Metadata")


# --------------------------------------------------------------------------
# Workshops
# --------------------------------------------------------------------------

@pytest.mark.parametrize("venue", [
    "NeurIPS 2023 Workshop on Instruction Tuning",
    "ICML Workshops",
    "Proceedings of the Workshop on Machine Translation",
    "CHI Conference Companion",
    "Companion Volume of ACL",
    "Late Breaking Results",
    "Doctoral Consortium",
    "ACL Student Research Workshop",
    "Adjunct Proceedings of UIST",
    "Work-shop on Something",
])
def test_workshops_and_satellites_are_recognised(venue):
    assert is_workshop(venue)


@pytest.mark.parametrize("venue", [
    "NeurIPS",
    "International Conference on Machine Learning",
    "Nature",
    None,
    "",
])
def test_main_tracks_are_not_workshops(venue):
    assert not is_workshop(venue)


def test_the_plural_is_caught():
    """`\\bworkshop\\b` does not match "Workshops" — which is how a workshop
    paper at "NeurIPS 2023 Workshops" came to be ranked A*."""
    assert is_workshop("NeurIPS 2023 Workshops")


# --------------------------------------------------------------------------
# Placeholders overall
# --------------------------------------------------------------------------

def test_a_string_that_cleans_away_to_nothing_is_a_placeholder():
    assert is_placeholder("n/a")
    assert is_placeholder("<i></i>")


def test_no_venue_at_all_is_not_a_placeholder():
    """Absent and wrong are different: only one of them is worth correcting."""
    assert not is_placeholder(None)
    assert not is_placeholder("")
