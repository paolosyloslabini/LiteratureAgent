"""Core data types for library entries.

An entry is the unit of a library: one published paper, book, or document.
Entries are persisted as Markdown files with YAML frontmatter (see `entryfile`);
these dataclasses are the in-memory view.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Levels are ordered best-first. "unranked" means we had no basis to judge.
LEVELS = ["A*", "A", "B", "C", "unranked"]

ENTRY_TYPES = [
    "conference paper",
    "journal article",
    "workshop paper",
    "preprint",
    "book",
    "book chapter",
    "thesis",
    "report",
    "video",
    "other",
]

STATUS_VERIFIED = "verified"
STATUS_UNVERIFIED = "UNVERIFIED"
# Where a `code_url` came from. The paper's own text printing a link is the
# strong claim; a repository found by searching the web is a weaker one, and the
# two must never be presented as the same thing. An entry written before this
# field existed carries None, which could only have come from the paper — that
# was the only writer at the time.
CODE_FROM_PAPER = "paper"
CODE_FROM_WEB = "web"
# Stored deliberately without a read: the metadata and the publisher's abstract
# are there, the summaries are not, and nothing was attempted. Distinct from
# UNVERIFIED, which means a read *was* attempted and no full text could be
# found. An unread entry is one `lit read <key>` away from a full record.
STATUS_UNREAD = "unread"


def level_rank(level: str | None) -> int:
    """Sort key: lower is better. Unknown levels sort last."""
    try:
        return LEVELS.index(level or "unranked")
    except ValueError:
        return len(LEVELS)


def meets_min_level(level: str | None, minimum: str | None) -> bool:
    """True if `level` is at least as good as `minimum`. No minimum accepts all."""
    if not minimum or minimum == "any":
        return True
    return level_rank(level) <= level_rank(minimum)


@dataclass
class Section:
    """One section of the section-by-section detailed summary."""

    name: str
    summary: str

    @classmethod
    def coerce(cls, raw: Any) -> "Section | None":
        if isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("section") or "").strip()
            summary = str(raw.get("summary") or raw.get("text") or "").strip()
            if name or summary:
                return cls(name=name or "Section", summary=summary)
        return None


@dataclass
class Reference:
    """A work cited BY this entry. Populated from metadata APIs, not guessed."""

    title: str
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    authors: list[str] = field(default_factory=list)
    # Set when this reference is itself an entry in the library.
    key: str | None = None

    @classmethod
    def coerce(cls, raw: Any) -> "Reference | None":
        if isinstance(raw, dict):
            title = str(raw.get("title") or "").strip()
            if not title:
                return None
            year = raw.get("year")
            try:
                year = int(year) if year not in (None, "") else None
            except (TypeError, ValueError):
                year = None
            authors = raw.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            return cls(
                title=title,
                year=year,
                doi=normalize_doi(raw.get("doi")),
                arxiv_id=raw.get("arxiv_id") or raw.get("arxivId"),
                authors=[str(a) for a in authors if a],
                key=raw.get("key"),
            )
        if isinstance(raw, str) and raw.strip():
            return cls(title=raw.strip())
        return None


@dataclass
class Entry:
    """A single library entry."""

    key: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    type: str = "other"
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    # The work's code/artifact repository: printed in its own text, or found on
    # the web by `lit code`. `code_source` says which, and `code_reason` records
    # the evidence, so the two are never confused for each other.
    code_url: str | None = None
    code_source: str | None = None
    code_reason: str = ""

    status: str = STATUS_UNVERIFIED
    level: str = "unranked"
    level_reason: str = ""
    # What `lit check` corrected on this entry and what confirmed it. Set only
    # by that command, and only when it actually changed a field — an entry that
    # has never needed correcting carries nothing here.
    check_note: str = ""

    # The publisher's abstract, verbatim. Bibliographic metadata, not a summary:
    # it is copied from Crossref/arXiv/S2 and is never written by a model, so it
    # is present even on UNVERIFIED entries whose full text was never retrieved.
    abstract: str = ""
    one_liner: str | None = None
    sections: list[Section] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)

    citation_count: int | None = None
    references: list[Reference] = field(default_factory=list)
    bibtex: str | None = None

    # Provenance: where the full text actually came from, and the local copy.
    read_source: str | None = None
    pdf_path: str | None = None
    text_chars: int | None = None
    # Page accounting for PDF sources. `pages_read < pages` marks a partial
    # read: a book or long report that was sampled rather than read end to end.
    pages: int | None = None
    pages_read: int | None = None

    added: str = field(default_factory=lambda: date.today().isoformat())
    updated: str = field(default_factory=lambda: date.today().isoformat())

    # User-authored. The agent never writes this field; `entryfile.save` always
    # re-reads it from disk so an LLM-produced Entry cannot clobber it.
    notes: str = ""

    @property
    def is_verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    @property
    def is_unread(self) -> bool:
        """Stored without a read, and a read is still worth attempting."""
        return self.status == STATUS_UNREAD

    @property
    def needs_read(self) -> bool:
        """True for anything carrying no agent-written summary yet.

        Covers both the deliberately unread and the entry whose full text could
        not be retrieved at the time — `lit read` is willing to try either.
        """
        return not self.is_verified

    @property
    def is_partial_read(self) -> bool:
        """True when the summary came from a sample, not the whole document."""
        return bool(self.pages and self.pages_read and self.pages_read < self.pages)

    def coverage(self) -> str:
        """Human-readable read coverage, e.g. '10 of 412 pages (sampled)'."""
        if not self.pages:
            return ""
        if not self.is_partial_read:
            return f"{self.pages} pages (complete)"
        return f"{self.pages_read} of {self.pages} pages (sampled)"

    def code_provenance(self) -> str:
        """Short label for where `code_url` came from. Empty when there is none.

        The evidence behind it lives in `code_reason`; this is the one-glance
        version. Nothing displays a code link without it — a repository found by
        searching the web must never read as one the authors printed.
        """
        if not self.code_url:
            return ""
        if self.code_source == CODE_FROM_WEB:
            return "found on the web"
        return "printed in the paper"

    def detailed_summary_md(self) -> str:
        """Section-by-section summary rendered as Markdown."""
        if not self.sections:
            return ""
        return "\n\n".join(f"### {s.name}\n\n{s.summary}".strip() for s in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def citation(self) -> str:
        """Short human-readable citation, e.g. 'Vaswani et al., NeurIPS 2017'."""
        who = "Unknown"
        if self.authors:
            first = self.authors[0].split(",")[0].split()[-1] if self.authors[0] else ""
            who = f"{first} et al." if len(self.authors) > 1 else (first or self.authors[0])
        where = self.venue or ("arXiv" if self.arxiv_id else "")
        bits = [b for b in (who, where, str(self.year) if self.year else "") if b]
        return ", ".join(bits)


# --------------------------------------------------------------------------
# Identifier normalization
# --------------------------------------------------------------------------

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9<>\[\]]+", re.IGNORECASE)
_ARXIV_NEW_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_ARXIV_OLD_RE = re.compile(r"([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


def normalize_doi(raw: str | None) -> str | None:
    """Extract a bare, lowercased DOI from a URL or string. None if absent."""
    if not raw:
        return None
    m = _DOI_RE.search(str(raw))
    if not m:
        return None
    doi = m.group(0).lower().rstrip(".,;)")
    # Trailing punctuation from prose is common; brackets must stay balanced.
    while doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    return doi


def normalize_arxiv(raw: str | None) -> str | None:
    """Extract a bare arXiv id (no version suffix) from a URL or string."""
    if not raw:
        return None
    s = str(raw)
    m = _ARXIV_NEW_RE.search(s)
    if m:
        return m.group(1)
    m = _ARXIV_OLD_RE.search(s)
    if m:
        return m.group(1)
    return None


# Hosts that serve code or research artifacts rather than prose. A paper's
# "our code is available at ..." line almost always points at one of these, and
# restricting to them keeps project landing pages, DOI links and stray URLs
# from the reference list out of the field.
CODE_HOSTS = {
    "github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "gitee.com",
    "sourceforge.net", "git.sr.ht", "savannah.gnu.org",
    "huggingface.co", "zenodo.org", "figshare.com", "osf.io",
    "codeocean.com", "softwareheritage.org", "dataverse.harvard.edu",
}

# Forges where a repository needs both an owner and a project to exist:
# "github.com/deepmind" is an organization page, not a repo.
_OWNER_REPO_HOSTS = {
    "github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "gitee.com",
}


def normalize_code_url(raw: str | None) -> str | None:
    """Clean a code/artifact repository URL. None if it is not one.

    Papers print these inside prose ("code at https://github.com/foo/bar."), so
    trailing punctuation has to come off, and a bare host or an owner page is
    not a repository.
    """
    if not raw:
        return None
    # One pass over the whole set, so a URL wrapped *and* punctuated — the
    # "(see https://github.com/foo/bar)." of a footnote — comes out clean.
    s = str(raw).strip().strip("<>()[]{}.,;:'\"")
    if not s or " " in s:
        return None
    if not s.lower().startswith(("http://", "https://")):
        s = "https://" + s.lstrip("/")

    try:
        parts = urlsplit(s)
    except ValueError:
        return None

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in CODE_HOSTS and not any(host.endswith(f".{h}") for h in CODE_HOSTS):
        return None

    segments = [p for p in parts.path.split("/") if p]
    if len(segments) < (2 if host in _OWNER_REPO_HOSTS else 1):
        return None

    # Always https, no fragment: the fragment is a scroll position, not part of
    # the repository's identity.
    return urlunsplit(("https", host, "/" + "/".join(segments), parts.query, ""))


def slugify(text: str, max_len: int = 40) -> str:
    """ASCII-fold and slug a string for use in filenames and citation keys."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "", text)
    return text.lower()[:max_len]


def make_key(authors: list[str], year: int | None, title: str) -> str:
    """Build a BibTeX-style citation key: surname + year + first title word."""
    surname = ""
    if authors:
        # Handle both "Ashish Vaswani" and "Vaswani, Ashish".
        a = authors[0]
        surname = a.split(",")[0].strip() if "," in a else a.split()[-1] if a.split() else ""
    surname = slugify(surname, 20) or "anon"

    stop = {
        "a", "an", "the", "on", "in", "of", "for", "and", "to", "with",
        "is", "are", "at", "by", "from", "via", "using",
    }
    # Fold accents first, or a title like "Théorie des jeux" truncates at the
    # first non-ASCII letter and yields a key of "th".
    ascii_title = unicodedata.normalize("NFKD", title or "")
    ascii_title = ascii_title.encode("ascii", "ignore").decode("ascii")
    word = ""
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", ascii_title):
        if w.lower() not in stop:
            word = slugify(w, 20)
            break

    return f"{surname}{year or ''}{word}" or "entry"
