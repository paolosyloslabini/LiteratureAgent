"""Bibliographic metadata lookup across Crossref, arXiv and Semantic Scholar.

Design notes:
* Semantic Scholar is the primary source for citation counts and reference
  lists. It is the only free index that counts citations to arXiv-only work
  properly, and it answers for up to 500 papers in a single batch request.
  Its unauthenticated pool is shared and bursts earn a 429, so every call goes
  through the host throttle in `http.py` and bulk work uses `batch_s2`.
* Crossref is authoritative for the published version, gives ready BibTeX, and
  has no practical rate limit for polite callers — which makes it the source
  that is always reachable when the others are not.
* No index is authoritative for citation counts; they are combined by taking
  the largest, which `merge` explains.
* A published version always wins over a preprint, per the spec.

OpenAlex used to sit between these two and is deliberately gone. It moved to a
metered credit model: once the small daily allowance is spent every request
comes back 429 "Insufficient budget" until midnight UTC. Because a dead index
returns an empty list rather than an error, that failure was invisible — a
`find` would quietly lose its most-cited facet, its reviews facet and every
citation lookup, then rank whatever Crossref and arXiv happened to return. A
source that can silently stop answering for the rest of the day is worse than
no source at all.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

from ..config import FetchConfig
from ..models import Reference, normalize_arxiv, normalize_doi
from ..venue import clean_venue, is_placeholder, is_workshop
from .http import HttpClient

CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"
# Relevance search tops out at 100 per page; the bulk endpoint is the only one
# that can sort, which is what makes a "most-cited work on this topic" facet
# possible at all.
S2_SEARCH_API = f"{S2_API}/search"
S2_BULK_API = f"{S2_API}/search/bulk"
S2_BATCH_API = f"{S2_API}/batch"
# The batch endpoint accepts 500 ids per request.
S2_BATCH_MAX = 500

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class PaperMeta:
    """Normalized bibliographic record assembled from one or more sources."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    type: str = "other"
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    abstract: str = ""
    citation_count: int | None = None
    is_oa: bool = False
    oa_pdf_url: str | None = None
    pmcid: str | None = None
    bibtex: str | None = None
    references: list[Reference] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # True when this record was located by title rather than by an identifier.
    # Such a record describes the right *work* but is not authoritative about
    # its *identity* — see `supplementary()`.
    title_matched: bool = False

    def supplementary(self) -> "PaperMeta":
        """The parts of a title-matched record that are safe to trust.

        A title lookup can land on a mirror or re-registration of the same work.
        "Attention Is All You Need" resolves to a 2025 DOI minted by a medical
        academy — same authors, same title, but adopting its DOI, venue or year
        would put "2025, Shenzhen Medical Academy" into a bibliography.

        Citation counts, reference lists and OA download links describe the work
        itself and survive that; identity fields do not.
        """
        return PaperMeta(
            citation_count=self.citation_count,
            references=list(self.references),
            is_oa=self.is_oa,
            oa_pdf_url=self.oa_pdf_url,
            pmcid=self.pmcid,
            sources=[f"{s}(title-match)" for s in self.sources],
        )

    def merge(self, other: "PaperMeta") -> "PaperMeta":
        """Fill blanks from `other` without overwriting what we already trust."""
        for f in ("title", "doi", "arxiv_id", "url", "abstract",
                  "oa_pdf_url", "pmcid", "bibtex"):
            if not getattr(self, f) and getattr(other, f):
                setattr(self, f, getattr(other, f))
        # Venue is deliberately not first-wins. Crossref is the primary record
        # and is asked first, but it is also the source most likely to hold no
        # usable venue for a work — a posted-content DOI has no
        # `container-title` at all. Semantic Scholar is very often the one that
        # knows the paper appeared at NeurIPS. Filling only when the field was
        # empty meant whichever source answered first could pin it to a
        # placeholder permanently.
        if other.venue and (
            not self.venue
            or (is_placeholder(self.venue) and not is_placeholder(other.venue))
        ):
            self.venue = other.venue
        if self.year is None:
            self.year = other.year
        if not self.authors:
            self.authors = other.authors
        # Citation counts are lower bounds rather than measurements: an index
        # can only count the citing works it has itself indexed, and it splits
        # what it does have across the preprint and published records of the
        # same paper. They undercount, never the reverse, so the largest number
        # anyone reports is the best estimate available — first-wins would pin
        # the result to whichever source happened to answer first.
        if other.citation_count is not None and (
            self.citation_count is None or other.citation_count > self.citation_count
        ):
            self.citation_count = other.citation_count
        if self.type in ("other", "") and other.type not in ("other", ""):
            self.type = other.type
        self.is_oa = self.is_oa or other.is_oa
        if len(other.references) > len(self.references):
            self.references = other.references
        self.sources += [s for s in other.sources if s not in self.sources]
        return self


# --------------------------------------------------------------------------
# Individual sources
# --------------------------------------------------------------------------

def from_crossref(http: HttpClient, doi: str) -> PaperMeta | None:
    data = http.get_json(f"{CROSSREF_API}/{doi}")
    if not data or "message" not in data:
        return None
    m = data["message"]
    meta = PaperMeta(sources=["crossref"])
    meta.title = _first(m.get("title")) or ""
    meta.authors = [_crossref_author(a) for a in (m.get("author") or [])]
    meta.authors = [a for a in meta.authors if a]
    meta.doi = normalize_doi(m.get("DOI"))
    meta.venue = _crossref_venue(m)
    meta.year = _crossref_year(m)
    meta.type = _map_crossref_type(m.get("type", ""), meta.venue)
    meta.url = m.get("URL")
    meta.citation_count = m.get("is-referenced-by-count")
    meta.abstract = _strip_tags(m.get("abstract") or "")

    for r in (m.get("reference") or []):
        title = r.get("article-title") or r.get("volume-title") or r.get("unstructured")
        if not title:
            continue
        year = r.get("year")
        try:
            year = int(str(year)[:4]) if year else None
        except ValueError:
            year = None
        meta.references.append(
            Reference(title=str(title)[:400], year=year, doi=normalize_doi(r.get("DOI")))
        )
    return meta


def crossref_bibtex(http: HttpClient, doi: str) -> str | None:
    r = http.get(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"})
    if r is None:
        return None
    text = (r.text or "").strip()
    return text if text.startswith("@") else None


def from_arxiv(http: HttpClient, arxiv_id: str) -> PaperMeta | None:
    r = http.get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1})
    if r is None:
        return None
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None
    entry = root.find("a:entry", _ATOM)
    if entry is None:
        return None
    return _arxiv_entry(entry, fallback_id=arxiv_id)


def _arxiv_entry(entry, *, fallback_id: str = "") -> PaperMeta | None:
    """One <entry> from an arXiv Atom feed. Shared by id lookup and search."""
    if entry.find("a:title", _ATOM) is None:
        return None

    meta = PaperMeta(sources=["arxiv"], type="preprint")
    meta.title = _clean(_text(entry.find("a:title", _ATOM)))
    if not meta.title:
        return None
    meta.abstract = _clean(_text(entry.find("a:summary", _ATOM)))
    meta.authors = [
        _clean(_text(a.find("a:name", _ATOM)))
        for a in entry.findall("a:author", _ATOM)
    ]
    meta.authors = [a for a in meta.authors if a]
    meta.arxiv_id = normalize_arxiv(_text(entry.find("a:id", _ATOM))) or fallback_id
    if not meta.arxiv_id:
        return None
    published = _text(entry.find("a:published", _ATOM))
    if published[:4].isdigit():
        meta.year = int(published[:4])
    meta.url = f"https://arxiv.org/abs/{meta.arxiv_id}"
    meta.is_oa = True
    meta.oa_pdf_url = f"https://arxiv.org/pdf/{meta.arxiv_id}"

    doi_el = entry.find("arxiv:doi", _ATOM)
    if doi_el is not None:
        meta.doi = normalize_doi(_text(doi_el))
    jref = entry.find("arxiv:journal_ref", _ATOM)
    if jref is not None and _text(jref):
        # The one field an arXiv record has that names where the work was
        # actually published. Authors fill it in by hand, so it arrives in every
        # shape: "NeurIPS 2017", a full proceedings title, or a bare "to appear".
        meta.venue = clean_venue(_text(jref))
    return meta


# What a paper record has to carry to be worth ranking: identity, the numbers
# `quality.assess` grades on, and an abstract so a relevance pass is judging
# content rather than guessing from a title.
_S2_FIELDS = ["title", "year", "venue", "publicationTypes", "citationCount",
              "externalIds", "abstract", "openAccessPdf", "authors.name"]
_S2_REF_FIELDS = ["references.title", "references.year",
                  "references.externalIds", "references.authors"]


def _s2_headers(api_key: str) -> dict | None:
    return {"x-api-key": api_key} if api_key else None


def _s2_meta(data: dict, *, title_matched: bool = False) -> PaperMeta | None:
    """One Semantic Scholar paper object as a `PaperMeta`.

    Shared by identifier lookup, batch lookup and search, so a record means the
    same thing however it was found.
    """
    if not data or not data.get("title"):
        return None
    meta = PaperMeta(sources=["semanticscholar"], title_matched=title_matched)
    meta.title = _clean(data.get("title") or "")
    meta.authors = [a.get("name", "") for a in (data.get("authors") or [])]
    meta.authors = [a for a in meta.authors if a]
    meta.year = data.get("year")
    meta.venue = clean_venue(data.get("venue"))
    meta.citation_count = data.get("citationCount")
    meta.abstract = _clean(data.get("abstract") or "")
    ext = data.get("externalIds") or {}
    meta.doi = normalize_doi(ext.get("DOI"))
    meta.arxiv_id = normalize_arxiv(ext.get("ArXiv"))
    meta.pmcid = _pmcid(ext.get("PubMedCentral"))
    oa = data.get("openAccessPdf") or {}
    if oa.get("url"):
        meta.is_oa = True
        meta.oa_pdf_url = oa["url"]
    meta.type = _map_s2_type(data.get("publicationTypes") or [], meta.venue)
    if meta.arxiv_id and not meta.url:
        meta.url = f"https://arxiv.org/abs/{meta.arxiv_id}"

    for r in (data.get("references") or []):
        if not r.get("title"):
            continue
        rext = r.get("externalIds") or {}
        meta.references.append(Reference(
            title=_clean(r["title"])[:400],
            year=r.get("year"),
            doi=normalize_doi(rext.get("DOI")),
            arxiv_id=normalize_arxiv(rext.get("ArXiv")),
            authors=[a.get("name", "") for a in (r.get("authors") or [])[:5]],
        ))
    return meta


def from_semantic_scholar(http: HttpClient, ident: str, api_key: str = "",
                          with_references: bool = True) -> PaperMeta | None:
    """`ident` is an S2-style id: 'DOI:10.x/y', 'arXiv:1706.03762', or a hash."""
    fields = _S2_FIELDS + (_S2_REF_FIELDS if with_references else [])
    data = http.get_json(
        f"{S2_API}/{ident}", params={"fields": ",".join(fields)},
        headers=_s2_headers(api_key), retries=2,
    )
    return _s2_meta(data) if data else None


def batch_s2(http: HttpClient, idents: list[str], api_key: str = "",
             with_references: bool = False) -> dict[str, PaperMeta]:
    """Look up many papers at once, keyed by the identifier that was asked for.

    This is the whole reason `find` can afford to weigh a large pool: one
    request answers for up to 500 papers, where a per-paper lookup would be 500
    round trips against an index that throttles at roughly one a second.

    Identifiers nothing knows come back as `null` and are simply absent from the
    result, so a caller can tell "no such record" from "the index never
    answered" — the latter leaves the whole batch missing, not one entry.
    """
    out: dict[str, PaperMeta] = {}
    idents = [i for i in idents if i]
    if not idents:
        return out
    fields = _S2_FIELDS + (_S2_REF_FIELDS if with_references else [])
    for i in range(0, len(idents), S2_BATCH_MAX):
        chunk = idents[i:i + S2_BATCH_MAX]
        data = http.post_json(
            S2_BATCH_API, json={"ids": chunk},
            params={"fields": ",".join(fields)},
            headers=_s2_headers(api_key), retries=2,
        )
        if not isinstance(data, list):
            continue
        # The response is positional: entry n answers for id n, or is null.
        # Not strict — a short or over-long response should cost us the entries
        # we cannot line up, not the whole batch.
        for ident, item in zip(chunk, data, strict=False):
            meta = _s2_meta(item) if isinstance(item, dict) else None
            if meta is not None:
                out[ident] = meta
    return out


def s2_by_title(http: HttpClient, title: str, api_key: str = "",
                with_references: bool = True) -> PaperMeta | None:
    """Find a work by title when it has no identifier we can look it up by.

    Marked `title_matched`, because a title search can land on a mirror or a
    different paper of the same name: the citation count and reference list it
    returns describe the right work, but its identifiers must not be adopted.
    See `PaperMeta.supplementary`.
    """
    hits = search_semantic_scholar(http, title, limit=8, api_key=api_key,
                                   with_references=with_references)
    best = _best_title_match(title, hits, threshold=0.85)
    if best is None:
        return None
    best.title_matched = True
    return best


# --------------------------------------------------------------------------
# Search + orchestration
# --------------------------------------------------------------------------

# Semantic Scholar's own name for a survey. Its `publicationTypes` vocabulary is
# the only document-type filter `find` can push down into this index.
S2_REVIEW_TYPE = "Review"
# What the bulk endpoint is asked to order by for the most-cited facet.
S2_SORT_CITED = "citationCount:desc"


def search_semantic_scholar(http: HttpClient, query: str, *, limit: int = 10,
                            sort: str | None = None,
                            from_year: int | None = None,
                            to_year: int | None = None,
                            kind: str | None = None,
                            api_key: str = "",
                            with_references: bool = False) -> list[PaperMeta]:
    """Keyword search against Semantic Scholar, with the facets `find` needs.

    Every hit is a real indexed work carrying its own identifiers, venue and
    citation count — the numbers that let importance be measured rather than
    assumed. S2 also indexes arXiv preprints as first-class works, which is what
    Crossref cannot do: in a field where the landmark result stays a preprint
    for its first two years, an index that only knows DOI-registered work has no
    opinion about the papers that matter most.

    `sort` switches to the bulk endpoint, the only one that can order results.
    That is what makes a "most-cited work on this topic" facet possible, and it
    is the single most useful question to ask when someone wants the landmark
    papers in an area rather than the closest keyword match.
    """
    fields = _S2_FIELDS + (_S2_REF_FIELDS if with_references else [])
    params: dict = {"query": query, "fields": ",".join(fields)}
    window = _s2_year_window(from_year, to_year)
    if window:
        params["year"] = window
    if kind == S2_REVIEW_TYPE:
        params["publicationTypes"] = S2_REVIEW_TYPE

    if sort:
        # The bulk endpoint pages at 1000 and has no `limit`; the slice at the
        # end of this function is what actually bounds it.
        params["sort"] = sort
        url = S2_BULK_API
    else:
        params["limit"] = max(1, min(100, limit))
        url = S2_SEARCH_API

    # Retried as hard as anything else here: a search that gives up is a whole
    # facet contributing nothing, and unauthenticated callers share one throttled
    # pool with the world, so a 429 on the first try is routine rather than a
    # sign the index is down.
    data = http.get_json(url, params=params, headers=_s2_headers(api_key))
    out: list[PaperMeta] = []
    for w in (data or {}).get("data") or []:
        meta = _s2_meta(w) if isinstance(w, dict) else None
        if meta is not None:
            out.append(meta)
    return out[:limit]


def _s2_year_window(from_year: int | None, to_year: int | None) -> str:
    """S2 spells a year window as '2020-2024', '2020-' or '-2024'."""
    if from_year and to_year:
        return f"{from_year}-{to_year}"
    if from_year:
        return f"{from_year}-"
    if to_year:
        return f"-{to_year}"
    return ""


def search_arxiv(http: HttpClient, query: str, *, limit: int = 10,
                 from_year: int | None = None,
                 to_year: int | None = None) -> list[PaperMeta]:
    """Keyword search against arXiv. Catches preprints the indexes lag on.

    Relevance order only. arXiv's `all:` treats a multi-word query loosely, so
    sorting those matches by date returns whatever was submitted this morning
    rather than recent work on the topic — recency is the `recent` facet's job,
    where it is a filter on a real query rather than a sort over everything.

    A year window, when the query implies one, goes in as a `submittedDate`
    range: still relevance-ordered, but only over the window.
    """
    search = f"all:{query}"
    if from_year or to_year:
        lo = f"{from_year or 1991}01010000"
        hi = f"{to_year or date.today().year}12312359"
        search = f"({search}) AND submittedDate:[{lo} TO {hi}]"
    r = http.get(ARXIV_API, params={
        "search_query": search,
        "max_results": max(1, limit),
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    if r is None:
        return []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []
    out: list[PaperMeta] = []
    for entry in root.findall("a:entry", _ATOM):
        meta = _arxiv_entry(entry)
        if meta is not None:
            out.append(meta)
    return out[:limit]


# Crossref's own name for ordering by how often a work has been cited. It only
# counts citations from DOI-registered works, so it undercounts badly in CS —
# but it is the one most-cited ordering that is always available, and ordering
# by an undercount still puts the heavily-cited work at the top.
CROSSREF_SORT_CITED = "is-referenced-by-count"

# Crossref document types, keyed by the vocabulary a search plan speaks.
_CROSSREF_KINDS = {
    "article": "journal-article",
    "preprint": "posted-content",
    "book": "book",
    "book-chapter": "book-chapter",
    "dataset": "dataset",
    "dissertation": "dissertation",
    "report": "report",
}


def search_crossref(http: HttpClient, query: str, limit: int = 10, *,
                    sort: str | None = None,
                    from_year: int | None = None,
                    to_year: int | None = None,
                    kind: str | None = None) -> list[PaperMeta]:
    """Keyword search against Crossref.

    Crossref has no daily budget and no shared throttle, so this is the one
    index that answers whatever else is failing. It knows only DOI-registered
    work, which means it cannot see an arXiv preprint until a publisher picks it
    up — `search_semantic_scholar` and `search_arxiv` cover that gap.
    """
    out: list[PaperMeta] = []
    params: dict = {
        "query.bibliographic": query, "rows": max(1, limit),
        # The abstract rides along because ranking a candidate pool from
        # bare titles is guesswork; it costs nothing extra to ask for.
        "select": "DOI,title,author,issued,container-title,type,"
                  "is-referenced-by-count,URL,abstract",
    }
    filters = [f"from-pub-date:{from_year}-01-01" if from_year else "",
               f"until-pub-date:{to_year}-12-31" if to_year else ""]
    if kind and (cr_kind := _CROSSREF_KINDS.get(kind)):
        filters.append(f"type:{cr_kind}")
    if any(filters):
        params["filter"] = ",".join(f for f in filters if f)
    if sort:
        params["sort"] = sort
        params["order"] = "desc"

    data = http.get_json(CROSSREF_API, params=params)
    for m in ((data or {}).get("message") or {}).get("items") or []:
        title = _first(m.get("title"))
        if not title:
            continue
        meta = PaperMeta(sources=["crossref"])
        # Crossref stores titles as submitted by the publisher, which for some
        # of them means escaped HTML: a title arrives as "&lt;p&gt;Agent…" or
        # carries a leading "&nbsp;". Left alone those reach the candidate table
        # verbatim, and — worse — make two registrations of one paper hash to
        # two different dedup keys, so the same work is offered twice.
        meta.title = _strip_tags(title)
        meta.authors = [a for a in (_crossref_author(a) for a in m.get("author") or []) if a]
        meta.doi = normalize_doi(m.get("DOI"))
        meta.venue = _crossref_venue(m)
        meta.year = _crossref_year(m)
        meta.type = _map_crossref_type(m.get("type", ""), meta.venue)
        meta.citation_count = m.get("is-referenced-by-count")
        meta.url = m.get("URL")
        meta.abstract = _strip_tags(m.get("abstract") or "")
        if meta.title:
            out.append(meta)
    return out[:limit]


def search_works(http: HttpClient, query: str, limit: int = 10,
                 cfg: FetchConfig | None = None, *,
                 from_year: int | None = None,
                 to_year: int | None = None) -> list[PaperMeta]:
    """Title/keyword search. Crossref first, Semantic Scholar as a fallback."""
    cfg = cfg or FetchConfig()
    out = search_crossref(http, query, limit, from_year=from_year, to_year=to_year)
    if len(out) < limit:
        seen = {m.doi for m in out if m.doi}
        for meta in search_semantic_scholar(
            http, query, limit=limit, from_year=from_year, to_year=to_year,
            api_key=cfg.semantic_scholar_api_key,
        ):
            if meta.doi and meta.doi in seen:
                continue
            out.append(meta)
    return out[:limit]


def resolve_metadata(
    http: HttpClient,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
    cfg: FetchConfig | None = None,
    with_references: bool = True,
) -> PaperMeta | None:
    """Assemble the best record we can for one work.

    Prefers the published version over the preprint: if an arXiv entry carries a
    DOI, the Crossref record for that DOI becomes the primary record.
    """
    cfg = cfg or FetchConfig()
    doi = normalize_doi(doi)
    arxiv_id = normalize_arxiv(arxiv_id)
    meta: PaperMeta | None = None

    if arxiv_id:
        meta = from_arxiv(http, arxiv_id)
        if meta and meta.doi and not doi:
            doi = meta.doi  # published version exists — prefer it

    if not doi and not arxiv_id and title:
        found = search_works(http, title, limit=5, cfg=cfg)
        # Strict, because what comes back becomes the work's identity: a loose
        # match here files a different paper under the requested title.
        best = _best_title_match(title, found, threshold=0.85)
        if best:
            doi = best.doi
            meta = best if meta is None else meta.merge(best)
            if not doi and best.arxiv_id:
                arxiv_id = best.arxiv_id

    if doi:
        # Crossref is authoritative for a DOI and usually sufficient on its own:
        # title, authors, venue, year, type, reference list and ready BibTeX.
        cr = from_crossref(http, doi)
        if cr:
            meta = cr.merge(meta) if meta else cr

    if meta is None and arxiv_id:
        meta = from_arxiv(http, arxiv_id)

    if meta is None:
        return None

    if arxiv_id and not meta.arxiv_id:
        meta.arxiv_id = arxiv_id

    # Semantic Scholar last, and always: for the citation count, for an
    # open-access PDF link Crossref does not carry, and for a reference list
    # when nothing else supplied one.
    #
    # The count is worth a request per paper on its own. Crossref counts only
    # citations from DOI-registered works, which in CS means a landmark preprint
    # reads as barely cited — and `quality.assess` grades a preprint on citation
    # velocity and nothing else, so an undercount puts it two or three levels
    # low, or under the B threshold entirely.
    ident = (f"DOI:{meta.doi}" if meta.doi
             else f"arXiv:{meta.arxiv_id}" if meta.arxiv_id else None)
    if ident:
        s2 = from_semantic_scholar(
            http, ident, cfg.semantic_scholar_api_key,
            with_references=with_references and not meta.references,
        )
        if s2:
            meta.merge(s2)
    elif meta.title:
        # No identifier to look the work up by — all we have is a title, so the
        # match is supplementary rather than authoritative: it fills in the
        # citation count and references and never overwrites identity, because a
        # near-title match can land on a different paper of the same name and a
        # wrong DOI would poison the whole entry. See `PaperMeta.supplementary`.
        s2 = s2_by_title(http, meta.title, cfg.semantic_scholar_api_key,
                         with_references=with_references and not meta.references)
        if s2:
            meta.merge(s2.supplementary())

    if meta.doi and not meta.bibtex:
        meta.bibtex = crossref_bibtex(http, meta.doi)
    if not meta.bibtex:
        meta.bibtex = synth_bibtex(meta)
    if meta.arxiv_id and not meta.oa_pdf_url:
        meta.oa_pdf_url = f"https://arxiv.org/pdf/{meta.arxiv_id}"
        meta.is_oa = True
    return meta


def synth_bibtex(meta: PaperMeta) -> str:
    """Build a BibTeX entry when Crossref has none (e.g. arXiv-only papers)."""
    from ..models import make_key

    key = make_key(meta.authors, meta.year, meta.title)
    kind = {
        "conference paper": "inproceedings",
        "workshop paper": "inproceedings",
        "journal article": "article",
        "book": "book",
        "book chapter": "incollection",
        "thesis": "phdthesis",
        "preprint": "misc",
    }.get(meta.type, "misc")

    fields = [f"  title = {{{meta.title}}}"]
    if meta.authors:
        fields.append("  author = {" + " and ".join(meta.authors) + "}")
    if meta.year:
        fields.append(f"  year = {{{meta.year}}}")
    if meta.venue:
        label = "journal" if kind == "article" else "booktitle"
        fields.append(f"  {label} = {{{meta.venue}}}")
    if meta.doi:
        fields.append(f"  doi = {{{meta.doi}}}")
    if meta.arxiv_id:
        fields.append(f"  eprint = {{{meta.arxiv_id}}}")
        fields.append("  archivePrefix = {arXiv}")
    if meta.url:
        fields.append(f"  url = {{{meta.url}}}")
    return f"@{kind}{{{key},\n" + ",\n".join(fields) + "\n}"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _first(v) -> str | None:
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v) if v else None


def _text(el) -> str:
    return (el.text or "") if el is not None else ""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _strip_tags(s: str) -> str:
    """Plain text from a field a publisher may have filled with escaped markup.

    Crossref returns some titles and most abstracts as HTML, and a fair number
    of those are *escaped* HTML — "&lt;p&gt;Agentic and…" or a leading "&nbsp;".
    Unescaping has to come first and then again after the tags are gone, since
    one pass only turns `&amp;nbsp;` into `&nbsp;`.
    """
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    return _clean(html.unescape(s))


def _crossref_author(a: dict) -> str:
    given, family = a.get("given", ""), a.get("family", "")
    if family:
        return f"{given} {family}".strip()
    return a.get("name", "")


def _crossref_year(m: dict) -> int | None:
    for field_name in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(field_name) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def _pmcid(v) -> str | None:
    if not v:
        return None
    m = re.search(r"PMC\d+", str(v))
    return m.group(0) if m else None


def _map_crossref_type(t: str, venue: str | None) -> str:
    t = (t or "").lower()
    if t == "proceedings-article":
        return "workshop paper" if is_workshop(venue) else "conference paper"
    return {
        "journal-article": "journal article",
        "posted-content": "preprint",
        "book": "book",
        "monograph": "book",
        "book-chapter": "book chapter",
        "dissertation": "thesis",
        "report": "report",
    }.get(t, "other")


def _map_s2_type(types: list[str], venue: str | None) -> str:
    types = [t.lower() for t in (types or [])]
    if "conference" in types:
        return "workshop paper" if is_workshop(venue) else "conference paper"
    if "journalarticle" in types:
        return "journal article"
    if "book" in types:
        return "book"
    if "review" in types:
        return "journal article"
    return "other"


# Crossref types where the publisher's imprint is what a citation names. For a
# monograph or a chapter "MIT Press" is the right answer; for a journal article
# or a preprint it is the wrong one, which is what `_crossref_venue` turns on.
_CROSSREF_BOOKISH = {
    "book", "monograph", "book-chapter", "book-part", "book-section",
    "book-series", "book-set", "edited-book", "reference-book",
}


def _crossref_venue(m: dict) -> str | None:
    """The venue of a Crossref record, or None when it does not carry one.

    `container-title` is the journal or the proceedings volume, and it is the
    only field that answers this question. `publisher` used to serve as a
    fallback, which filled the field with things that are not venues: a
    posted-content DOI reads "arXiv", and a journal article whose container is
    missing reads "Elsevier BV" or "Springer Science and Business Media LLC".

    That was worse than an empty field in three ways. It reached bibliographies
    verbatim; it made `synth_bibtex` emit `journal = {Elsevier BV}`; and the
    ranked venue table matches `^nature\\s`, so a paper whose publisher read
    "Nature Portfolio" inherited A* without ever naming a venue.
    """
    venue = clean_venue(_first(m.get("container-title")))
    if venue:
        return venue
    if (m.get("type") or "").lower() in _CROSSREF_BOOKISH:
        return clean_venue(m.get("publisher"))
    return None


def _best_title_match(query: str, cands: list[PaperMeta], *,
                      threshold: float = 0.5) -> PaperMeta | None:
    """Pick the candidate whose title best overlaps the query (token F1).

    Ties break toward the most-cited, which is the record the rest of the
    literature actually points at when an index holds several copies of one
    work. `threshold` is raised by callers that will treat the answer as the
    work itself: "Attention Is All You Need" must not match "Channel Attention
    Is All You Need for Video Frames".
    """
    if not re.findall(r"[a-z0-9]+", query.lower()):
        return cands[0] if cands else None
    scored: list[tuple[float, int, PaperMeta]] = []
    for c in cands:
        if not c.title:
            continue
        score = title_similarity(query, c.title)
        if score >= threshold:
            scored.append((score, c.citation_count or 0, c))
    if not scored:
        return None
    scored.sort(key=lambda s: (-s[0], -s[1]))
    return scored[0][2]


def title_similarity(a: str | None, b: str | None) -> float:
    """How closely two titles agree, from 0 to 1 (token F1, slugs as a fast path).

    Public because deciding "are these the same work?" is not only a ranking
    question: `actions.check` uses it to refuse a DOI whose Crossref record
    turns out to describe a different paper, which is the one mistake there that
    would rewrite every other field with another work's metadata.
    """
    from ..models import slugify

    if not a or not b:
        return 0.0
    if slugify(a, 200) == slugify(b, 200):
        return 1.0
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))
