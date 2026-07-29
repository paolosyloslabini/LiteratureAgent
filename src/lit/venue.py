"""What a venue string is, and what an index hands you instead of one.

`venue` is the field the level scorer ranks on and the field a bibliography
prints, so a wrong one is expensive twice over. The indexes make three kinds of
mistake, and all three arrive as a well-formed string:

* A **publisher** where a venue belongs. Crossref records a `container-title`
  for work that has one and nothing at all for work that does not, so reaching
  for `publisher` as a fallback fills the field with "Elsevier BV" or "Springer
  Science and Business Media LLC". Those are not venues, and one of them is
  worse than noise: the ranked table matches `^nature\\s`, so a paper whose
  publisher reads "Nature Portfolio" would inherit A* without ever naming a
  venue.
* A **preprint server**. A posted-content DOI gives publisher "arXiv", which
  says only what the entry's `arxiv_id` already said.
* A **workshop** presented as its parent conference. "NeurIPS 2023 Workshop on
  Instruction Tuning" contains the A* pattern for NeurIPS, and a workshop paper
  must not inherit the main track's rank.

These live here rather than beside either caller because both `fetch.metadata`
(which fills the field) and `quality` (which ranks it) need the same answer, and
`quality` already imports `fetch.metadata` — a helper in either one would have
to be duplicated in the other to avoid a cycle. `actions.check` uses them a
third time, to decide which stored venues are worth an agent's attention.
"""

from __future__ import annotations

import html
import re

# Corporate suffixes and imprint words. A venue name does not end in a company
# form; a publisher's legal name almost always does.
_PUBLISHER_TAIL = re.compile(
    r"\b(?:bv|b\.v\.|llc|l\.l\.c\.|ltd|ltd\.|inc|inc\.|gmbh|ag|a\.g\.|sa|s\.a\.|"
    r"kg|plc|co|corp|corporation|company|press|publishing|publishers|"
    r"publications|media|group|portfolio|holdings)$",
    re.IGNORECASE,
)

# Publishing houses, matched anywhere in the string. Naming them explicitly
# catches the ones whose legal name carries no company suffix at all.
_PUBLISHER_NAMES = re.compile(
    r"\b(?:elsevier|springer|wiley|taylor\s*&?\s*francis|informa|sage|"
    r"de\s*gruyter|emerald|hindawi|frontiers\s+media|mdpi|"
    r"oxford\s+university\s+press|cambridge\s+university\s+press|"
    r"association\s+for\s+computing\s+machinery|"
    r"institute\s+of\s+electrical\s+and\s+electronics\s+engineers|"
    r"nature\s+(?:portfolio|publishing\s+group)|"
    r"american\s+(?:chemical|physical|medical)\s+society)\b",
    re.IGNORECASE,
)

# Preprint servers. Real hosts of real work, but not venues: a paper on one has
# not been through review, and the entry's own identifier already records it.
_PREPRINT_SERVERS = re.compile(
    r"^(?:arxiv|biorxiv|medrxiv|chemrxiv|techrxiv|engrxiv|psyarxiv|socarxiv|"
    r"ssrn|osf(?:\s+preprints)?|preprints(?:\.org)?|research\s*square|"
    r"authorea|hal|zenodo)\b",
    re.IGNORECASE,
)

# Strings an index uses to mean "no venue". Stored verbatim they read as one.
_NULL_VENUES = {
    "", "n/a", "na", "none", "null", "unknown", "unpublished", "no venue",
    "not available", "-", "--", "—",
}

# Venue-shaped words that mean this was a satellite event, not the main track.
# Plural included: `\bworkshop\b` does not match "Workshops", which is how a
# workshop paper at "NeurIPS 2023 Workshops" came to be ranked A*.
_WORKSHOP = re.compile(
    r"\b(?:work-?shops?|companion(?:\s+volume)?|late[-\s]?breaking|"
    r"doctoral\s+(?:consortium|symposium)|student\s+research\s+workshop|"
    r"adjunct\s+proceedings)\b",
    re.IGNORECASE,
)


def clean_venue(raw: str | None) -> str | None:
    """Normalize a venue string, or None if it does not carry one.

    Crossref serves some fields as escaped markup, so the same two-pass unescape
    the title and abstract get is applied here: left alone, a venue reaches a
    bibliography as "&lt;i&gt;Nature&lt;/i&gt;" and — worse — two spellings of
    one venue stop comparing equal.
    """
    if raw is None:
        return None
    s = re.sub(r"<[^>]+>", " ", html.unescape(str(raw)))
    s = re.sub(r"\s+", " ", html.unescape(s)).strip()
    # Publishers submit these wrapped in their own punctuation often enough.
    s = s.strip(" ,;:.").strip()
    if s.lower() in _NULL_VENUES or len(s) < 2:
        return None
    return s


def is_workshop(venue: str | None) -> bool:
    """Is this a workshop, companion or satellite track rather than a main one?"""
    return bool(_WORKSHOP.search(venue or ""))


def is_publisher_name(venue: str | None) -> bool:
    """Does this name a publishing house rather than a venue?"""
    v = (venue or "").strip().strip(" ,;:.")
    if not v:
        return False
    if _PUBLISHER_NAMES.search(v):
        return True
    # Compare on the last word so "Journal of Vision Research Ltd" is caught but
    # "Cold Spring Harbor Perspectives" is not.
    return bool(_PUBLISHER_TAIL.search(v))


def is_preprint_server(venue: str | None) -> bool:
    """Is this a preprint host? Says nothing an identifier does not already."""
    return bool(_PREPRINT_SERVERS.match((venue or "").strip()))


def is_placeholder(venue: str | None) -> bool:
    """True for a venue string that names something other than a venue.

    What the level scorer must not rank, `refresh` should replace when a real
    venue appears, and `check` should treat as suspicious.
    """
    v = clean_venue(venue)
    if v is None:
        return bool(venue)  # a non-empty string that cleaned away to nothing
    return is_publisher_name(v) or is_preprint_server(v)
