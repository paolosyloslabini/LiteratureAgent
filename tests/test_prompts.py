"""Prompt construction — mainly the entry card, which is paid for 30x a search.

`lit search` renders every candidate as a card and concatenates the lot into one
ranking call, so a few hundred characters per card is tens of thousands of
tokens per query. These tests pin the budget and, just as importantly, pin what
must survive inside it.
"""

from factories import make_entry

from lit.models import Section
from lit.prompts import _entry_card, library_search_prompt


def _filler(n: int, word: str = "part") -> str:
    """Prose `n` characters long. `word` keeps one field's text out of another."""
    base = (f"This {word} reports the setup, the baselines it was compared "
            "against, and the outcome measured in each condition. ")
    return (base * (n // len(base) + 2))[:n]


def _saturated():
    """An entry pushing every card field to or past its cap."""
    return make_entry(
        one_liner=_filler(200, "opener"),
        tags=[f"tag-number-{i}" for i in range(10)],
        key_findings=[_filler(160, f"finding-{i}") for i in range(5)],
        sections=[Section(f"Section Number {i}", _filler(400, f"section-{i}"))
                  for i in range(12)],
        notes=_filler(700, "note"),
        abstract=_filler(2000, "abstract"),
    )


# --------------------------------------------------------------------------
# Card size — the whole reason the section budget is small
# --------------------------------------------------------------------------

def test_saturated_card_fits_the_ranking_budget():
    # Thirty of these go into a single prompt. At the old 12-sections-by-300
    # budget this card ran to ~5,800 chars, i.e. ~43k tokens for one search.
    assert len(_entry_card(_saturated())) < 3_000


def test_a_full_pool_stays_well_under_a_hundred_thousand_characters():
    entries = [_saturated() for _ in range(30)]
    prompt = library_search_prompt(query="state space models", scope="ml",
                                   entries=entries)
    assert len(prompt) < 100_000


def test_sections_are_the_field_that_gets_cut():
    e = _saturated()
    card = _entry_card(e)
    assert "Section Number 3" in card       # four openers survive
    assert "Section Number 4" not in card   # the rest do not
    assert e.sections[0].summary not in card  # and each one is clipped
    assert e.sections[0].summary[:150] in card


def test_the_high_signal_fields_are_left_whole():
    """Title, one-liner, tags and findings are why the card exists at all."""
    e = _saturated()
    card = _entry_card(e)
    assert e.title in card
    assert e.one_liner in card
    for finding in e.key_findings:
        assert finding in card
    for tag in e.tags:
        assert tag in card


# --------------------------------------------------------------------------
# Unread entries — the card has to show what FTS matched on
# --------------------------------------------------------------------------

def test_unread_card_shows_abstract_evidence_past_the_old_cap():
    """FTS indexes the whole abstract, so the card must show more than its head.

    Without this, an entry retrieved on a term 900 characters into its abstract
    reaches the ranker with no visible reason for being there — and the prompt
    tells the ranker to drop entries that merely share vocabulary.
    """
    e = make_entry(
        one_liner=None, sections=[], key_findings=[],
        abstract=_filler(900) + " hydrodynamic lubrication " + _filler(400),
    )
    assert "hydrodynamic lubrication" in _entry_card(e)


def test_unread_card_is_still_bounded():
    e = make_entry(one_liner=None, sections=[], key_findings=[],
                   abstract=_filler(20_000))
    assert len(_entry_card(e)) < 2_200
