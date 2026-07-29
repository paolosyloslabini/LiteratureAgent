"""Prompt construction — mainly the entry card, which is paid for 30x a search.

`lit search` renders every candidate as a card and concatenates the lot into one
ranking call, so a few hundred characters per card is tens of thousands of
tokens per query. These tests pin the budget and, just as importantly, pin what
must survive inside it.
"""

from factories import make_entry

from lit.models import Reference, Section
from lit.prompts import (
    _entry_card,
    claim_origin_prompt,
    library_search_prompt,
    read_paper_prompt,
)


def _filler(n: int, word: str = "part") -> str:
    """Prose `n` characters long. `word` keeps one field's text out of another."""
    base = (f"This {word} reports the setup, the baselines it was compared "
            "against, and the outcome measured in each condition. ")
    return (base * (n // len(base) + 2))[:n]


def _saturated():
    """An entry pushing every card field to or past its cap."""
    return make_entry(
        # Stored one-liners are stripped on the way in (`add._apply_reading`),
        # and the card renders on one line.
        one_liner=_filler(200, "opener").strip(),
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


# --------------------------------------------------------------------------
# Reading — don't pay to re-derive metadata the caller already has
# --------------------------------------------------------------------------

def _read_prompt(**kw) -> str:
    base = dict(title="A Paper", scope="ml", known_venue=None, known_year=None,
                truncated=False, needs_level=False)
    return read_paper_prompt(**{**base, **kw})


def test_venue_is_asked_for_when_the_metadata_apis_had_none():
    assert "venue_from_text" in _read_prompt(known_venue=None)


def test_venue_is_not_asked_for_when_it_is_already_known():
    """`_apply_reading` drops this field unless the venue slot is empty.

    On the common path — a record the metadata APIs resolved — asking for it
    buys a search through the header for a string that is then discarded.
    """
    prompt = _read_prompt(known_venue="NeurIPS", known_year=2017)
    assert "venue_from_text" not in prompt
    assert "header/footer" not in prompt
    assert "NeurIPS" in prompt  # still told what the venue is


def test_a_verbose_reading_cannot_inflate_every_later_ranking_prompt():
    """Findings and the one-liner are stored raw, so the card is where they stop.

    Section summaries are clipped when the reading is saved; these two are not,
    which means one long-winded read goes on being paid for in every search and
    every `ask` selection until the entry is re-read.
    """
    e = make_entry(one_liner=_filler(4_000, "opener"),
                   key_findings=[_filler(4_000, f"finding-{i}") for i in range(5)],
                   sections=[], notes="")
    assert len(_entry_card(e)) < 2_000
    assert len(_entry_card(e, brief=True)) < 1_200


# --------------------------------------------------------------------------
# Claim tracing — the reference list is quoted at every hop
# --------------------------------------------------------------------------

def _refs(n: int) -> list[Reference]:
    return [Reference(title=f"Cited Work Number {i}", year=1990 + i % 30,
                      doi=f"10.1/ref{i}") for i in range(n)]


def test_claim_prompt_caps_a_survey_sized_reference_list():
    """A survey's bibliography is bought at every hop and read at most 3 deep."""
    prompt = claim_origin_prompt(claim="attention beats recurrence",
                                 entry=make_entry(), references=_refs(300),
                                 truncated=False)
    assert len(prompt) < 10_000
    assert "Cited Work Number 299" not in prompt


def test_claim_prompt_indices_still_address_the_caller_s_list():
    """The model answers with an index; `claim._candidate_refs` looks it up.

    The cap is a prefix slice, so every index the model can see names the same
    reference in the full list the caller holds — and no index it can see is
    out of range for that list.
    """
    refs = _refs(300)
    prompt = claim_origin_prompt(claim="a claim", entry=make_entry(),
                                 references=refs, truncated=False)
    for i in (0, 1, 79):
        assert f"  [{i}] {refs[i].title}" in prompt
    assert "  [80] " not in prompt
