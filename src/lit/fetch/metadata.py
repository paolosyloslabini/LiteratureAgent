"""Bibliographic metadata lookup across Crossref, arXiv, OpenAlex and S2.

Design notes:
* OpenAlex is the primary source for reference lists: it is free, has no hard
  rate limit for polite users, and exposes `referenced_works`.
* Semantic Scholar is richer for reference titles, and is the only source that
  counts citations to arXiv-only work properly. It rate-limits aggressively
  without an API key, so it is asked for one record per paper, no more.
* No index is authoritative for citation counts; they are combined by taking
  the largest, which `merge` explains.
* Crossref is authoritative for the published version and gives ready BibTeX.
* A published version always wins over a preprint, per the spec.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from ..config import FetchConfig
from ..models import Reference, normalize_arxiv, normalize_doi
from .http import HttpClient

CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"

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
        for f in ("title", "venue", "doi", "arxiv_id", "url", "abstract",
                  "oa_pdf_url", "pmcid", "bibtex"):
            if not getattr(self, f) and getattr(other, f):
                setattr(self, f, getattr(other, f))
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
    meta.venue = _first(m.get("container-title")) or m.get("publisher")
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
        meta.venue = _clean(_text(jref))
    return meta


def from_openalex(http: HttpClient, *, doi: str | None = None,
                  openalex_id: str | None = None, title: str | None = None,
                  with_references: bool = True) -> PaperMeta | None:
    title_matched = False
    if doi:
        url = f"{OPENALEX_API}/doi:{doi}"
        data = http.get_json(url)
    elif openalex_id:
        data = http.get_json(f"{OPENALEX_API}/{openalex_id.rsplit('/', 1)[-1]}")
    elif title:
        # OpenAlex holds several records for the same work (a preprint copy, a
        # proceedings copy, near-title-matches on other papers). Pull a handful
        # and pick the best title match, breaking ties toward the most-cited —
        # that is the canonical record for the work.
        res = http.get_json(
            OPENALEX_API,
            params={"filter": f"title.search:{title}", "per-page": 8,
                    "select": "id,title,display_name,publication_year,doi,"
                              "cited_by_count,type,primary_location,open_access,"
                              "best_oa_location,authorships,ids,referenced_works"},
        )
        data = _pick_openalex_record(title, (res or {}).get("results") or [])
        title_matched = True
    else:
        return None
    if not data or not data.get("id"):
        return None

    meta = PaperMeta(sources=["openalex"], title_matched=title_matched)
    meta.title = data.get("title") or data.get("display_name") or ""
    meta.authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (data.get("authorships") or [])
    ]
    meta.authors = [a for a in meta.authors if a]
    meta.year = data.get("publication_year")
    meta.doi = normalize_doi(data.get("doi"))
    meta.citation_count = data.get("cited_by_count")

    loc = data.get("primary_location") or {}
    src = loc.get("source") or {}
    meta.venue = src.get("display_name")
    meta.type = _map_openalex_type(data.get("type", ""), src.get("type"))

    oa = data.get("open_access") or {}
    meta.is_oa = bool(oa.get("is_oa"))
    best = data.get("best_oa_location") or {}
    meta.oa_pdf_url = best.get("pdf_url") or oa.get("oa_url")
    meta.url = loc.get("landing_page_url") or data.get("id")

    ids = data.get("ids") or {}
    meta.pmcid = _pmcid(ids.get("pmcid"))
    meta.arxiv_id = normalize_arxiv(meta.oa_pdf_url) if "arxiv" in str(meta.oa_pdf_url) else None

    if with_references:
        meta.references = _openalex_references(http, data.get("referenced_works") or [])
    return meta


def _openalex_references(http: HttpClient, ids: list[str], cap: int = 200) -> list[Reference]:
    """Resolve OpenAlex work ids to titles in batches (50 ids per request)."""
    out: list[Reference] = []
    ids = [i.rsplit("/", 1)[-1] for i in ids[:cap]]
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        data = http.get_json(
            OPENALEX_API,
            params={
                "filter": f"openalex_id:{'|'.join(batch)}",
                "per-page": 50,
                "select": "id,title,publication_year,doi,authorships",
            },
        )
        for w in (data or {}).get("results") or []:
            authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:5]
            ]
            out.append(Reference(
                title=w.get("title") or "",
                year=w.get("publication_year"),
                doi=normalize_doi(w.get("doi")),
                authors=[a for a in authors if a],
            ))
    return [r for r in out if r.title]


def from_semantic_scholar(http: HttpClient, ident: str, api_key: str = "",
                          with_references: bool = True) -> PaperMeta | None:
    """`ident` is an S2-style id: 'DOI:10.x/y', 'arXiv:1706.03762', or a hash."""
    fields = ["title", "year", "venue", "publicationTypes", "citationCount",
              "externalIds", "abstract", "openAccessPdf", "authors.name"]
    if with_references:
        fields += ["references.title", "references.year", "references.externalIds",
                   "references.authors"]
    headers = {"x-api-key": api_key} if api_key else None
    data = http.get_json(
        f"{S2_API}/{ident}", params={"fields": ",".join(fields)},
        headers=headers, retries=2,
    )
    if not data or not data.get("title"):
        return None

    meta = PaperMeta(sources=["semanticscholar"])
    meta.title = data.get("title") or ""
    meta.authors = [a.get("name", "") for a in (data.get("authors") or [])]
    meta.authors = [a for a in meta.authors if a]
    meta.year = data.get("year")
    meta.venue = data.get("venue") or None
    meta.citation_count = data.get("citationCount")
    meta.abstract = data.get("abstract") or ""
    ext = data.get("externalIds") or {}
    meta.doi = normalize_doi(ext.get("DOI"))
    meta.arxiv_id = normalize_arxiv(ext.get("ArXiv"))
    meta.pmcid = _pmcid(ext.get("PubMedCentral"))
    oa = data.get("openAccessPdf") or {}
    if oa.get("url"):
        meta.is_oa = True
        meta.oa_pdf_url = oa["url"]
    types = data.get("publicationTypes") or []
    meta.type = _map_s2_type(types, meta.venue)

    for r in (data.get("references") or []):
        if not r.get("title"):
            continue
        rext = r.get("externalIds") or {}
        meta.references.append(Reference(
            title=r["title"],
            year=r.get("year"),
            doi=normalize_doi(rext.get("DOI")),
            arxiv_id=normalize_arxiv(rext.get("ArXiv")),
            authors=[a.get("name", "") for a in (r.get("authors") or [])[:5]],
        ))
    return meta


# --------------------------------------------------------------------------
# Search + orchestration
# --------------------------------------------------------------------------

_OA_SEARCH_SELECT = (
    "id,title,publication_year,doi,cited_by_count,primary_location,"
    "open_access,best_oa_location,type,authorships,abstract_inverted_index"
)


def search_openalex(http: HttpClient, query: str, *, limit: int = 10,
                    sort: str | None = None, from_year: int | None = None,
                    kind: str | None = None) -> list[PaperMeta]:
    """Keyword search against OpenAlex, with the facets `find` discovers on.

    Free, structured, and hallucination-proof: every hit is a real indexed work
    with its own identifiers and citation count. `sort`, `from_year` and `kind`
    are what let one query be asked several different ways — most-cited,
    published since a date, reviews only — which is the cheap equivalent of
    pointing several scout agents at different angles.
    """
    params: dict = {"search": query, "per-page": max(1, limit),
                    "select": _OA_SEARCH_SELECT}
    filters = []
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    if kind:
        filters.append(f"type:{kind}")
    if filters:
        params["filter"] = ",".join(filters)
    if sort:
        params["sort"] = sort

    data = http.get_json(OPENALEX_API, params=params)
    out: list[PaperMeta] = []
    for w in (data or {}).get("results") or []:
        meta = _openalex_search_hit(w)
        if meta is not None:
            out.append(meta)
    return out[:limit]


def _openalex_search_hit(w: dict) -> PaperMeta | None:
    title = w.get("title") or w.get("display_name") or ""
    if not title:
        return None
    meta = PaperMeta(sources=["openalex"])
    meta.title = title
    meta.authors = [
        a for a in (
            (a.get("author") or {}).get("display_name", "")
            for a in (w.get("authorships") or [])[:10]
        ) if a
    ]
    meta.year = w.get("publication_year")
    meta.doi = normalize_doi(w.get("doi"))
    meta.citation_count = w.get("cited_by_count")
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    meta.venue = src.get("display_name")
    meta.type = _map_openalex_type(w.get("type", ""), src.get("type"))
    oa = w.get("open_access") or {}
    meta.is_oa = bool(oa.get("is_oa"))
    best = w.get("best_oa_location") or {}
    meta.oa_pdf_url = best.get("pdf_url") or oa.get("oa_url")
    meta.url = loc.get("landing_page_url") or w.get("id")
    if meta.oa_pdf_url and "arxiv" in str(meta.oa_pdf_url):
        meta.arxiv_id = normalize_arxiv(meta.oa_pdf_url)
    meta.abstract = _deinvert_abstract(w.get("abstract_inverted_index"))
    return meta


def _deinvert_abstract(index: dict | None) -> str:
    """Rebuild OpenAlex's inverted abstract index into running text.

    OpenAlex ships abstracts as {word: [positions]} rather than as a string.
    Reassembling it costs nothing and gives the ranking call something to judge
    beyond a bare title.
    """
    if not isinstance(index, dict) or not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        words += [(int(p), str(word)) for p in positions if isinstance(p, int)]
    if not words:
        return ""
    words.sort()
    return _clean(" ".join(w for _, w in words))


def search_arxiv(http: HttpClient, query: str, *, limit: int = 10) -> list[PaperMeta]:
    """Keyword search against arXiv. Catches preprints the indexes lag on.

    Relevance order only. arXiv's `all:` treats a multi-word query loosely, so
    sorting those matches by date returns whatever was submitted this morning
    rather than recent work on the topic — recency is the OpenAlex facet's job,
    where it is a filter on a real query rather than a sort over everything.
    """
    r = http.get(ARXIV_API, params={
        "search_query": f"all:{query}",
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


def search_works(http: HttpClient, query: str, limit: int = 10,
                 cfg: FetchConfig | None = None) -> list[PaperMeta]:
    """Title/keyword search. Crossref first, OpenAlex as a fallback."""
    out: list[PaperMeta] = []
    data = http.get_json(
        CROSSREF_API,
        params={"query.bibliographic": query, "rows": limit,
                # The abstract rides along because ranking a candidate pool from
                # bare titles is guesswork; it costs nothing extra to ask for.
                "select": "DOI,title,author,issued,container-title,type,"
                          "is-referenced-by-count,URL,abstract"},
    )
    for m in ((data or {}).get("message") or {}).get("items") or []:
        title = _first(m.get("title"))
        if not title:
            continue
        meta = PaperMeta(sources=["crossref"])
        meta.title = title
        meta.authors = [a for a in (_crossref_author(a) for a in m.get("author") or []) if a]
        meta.doi = normalize_doi(m.get("DOI"))
        meta.venue = _first(m.get("container-title"))
        meta.year = _crossref_year(m)
        meta.type = _map_crossref_type(m.get("type", ""), meta.venue)
        meta.citation_count = m.get("is-referenced-by-count")
        meta.url = m.get("URL")
        meta.abstract = _strip_tags(m.get("abstract") or "")
        out.append(meta)

    if len(out) < limit:
        seen = {m.doi for m in out if m.doi}
        for meta in search_openalex(http, query, limit=limit):
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
        best = _best_title_match(title, found)
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

        # OpenAlex then adds what Crossref does not do well: an open-access PDF
        # link, and a citation count that counts citations from works with no
        # DOI of their own (Crossref only counts DOI-registered ones, which
        # undercounts in CS). Neither source is the last word on the count —
        # S2 is consulted below and the largest figure wins.
        # If Crossref already gave us references, skip resolving OpenAlex's —
        # that is several extra requests for data we have.
        need_refs = with_references and not (meta and meta.references)
        oa = from_openalex(http, doi=doi, with_references=need_refs)
        if oa:
            meta = meta.merge(oa) if meta else oa

    elif arxiv_id and meta is not None:
        # arXiv-only work: the arXiv API reports no venue and no citation count,
        # which would leave the level scorer with nothing to work from. Look the
        # title up in OpenAlex to recover citations and references.
        #
        # This match is made on title alone, so it is treated as supplementary,
        # not authoritative: it fills blanks in the arXiv record and never
        # overwrites it. In particular the DOI OpenAlex reports here is not
        # chased through Crossref — a near-title match can land on a different
        # paper of the same name, and a wrong DOI would poison the whole entry.
        oa = from_openalex(http, title=meta.title, with_references=with_references)
        if oa:
            meta.merge(oa.supplementary())

    if meta is None and arxiv_id:
        meta = from_arxiv(http, arxiv_id)

    if meta is None:
        return None

    if arxiv_id and not meta.arxiv_id:
        meta.arxiv_id = arxiv_id

    # Semantic Scholar last: for a reference list when nothing else supplied
    # one, and — always — for a citation count.
    #
    # OpenAlex keeps arXiv-only work as a bare preprint record that the citing
    # literature never points at, so its count for those papers is close to
    # meaningless: 9 for GAIA and 53 for AgentBench, against 1032 and 1090 at
    # S2. `quality.assess` grades a preprint on citations per year and nothing
    # else, so those figures put landmark benchmarks two to three levels low or
    # under the B threshold entirely, where they fell through to the reader
    # agent's judgement. A count is worth one request per paper on its own.
    ident = (f"DOI:{meta.doi}" if meta.doi
             else f"arXiv:{meta.arxiv_id}" if meta.arxiv_id else None)
    if ident:
        s2 = from_semantic_scholar(
            http, ident, cfg.semantic_scholar_api_key,
            with_references=with_references and not meta.references,
        )
        if s2:
            meta.merge(s2)

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
    return _clean(re.sub(r"<[^>]+>", " ", s or ""))


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
        return "workshop paper" if _is_workshop(venue) else "conference paper"
    return {
        "journal-article": "journal article",
        "posted-content": "preprint",
        "book": "book",
        "monograph": "book",
        "book-chapter": "book chapter",
        "dissertation": "thesis",
        "report": "report",
    }.get(t, "other")


def _map_openalex_type(t: str, source_type: str | None) -> str:
    t = (t or "").lower()
    if t == "article":
        return "preprint" if source_type == "repository" else "journal article"
    return {
        "preprint": "preprint",
        "book": "book",
        "book-chapter": "book chapter",
        "dissertation": "thesis",
        "report": "report",
        "proceedings-article": "conference paper",
    }.get(t, "other")


def _map_s2_type(types: list[str], venue: str | None) -> str:
    types = [t.lower() for t in (types or [])]
    if "conference" in types:
        return "workshop paper" if _is_workshop(venue) else "conference paper"
    if "journalarticle" in types:
        return "journal article"
    if "book" in types:
        return "book"
    if "review" in types:
        return "journal article"
    return "other"


def _is_workshop(venue: str | None) -> bool:
    return "workshop" in (venue or "").lower()


def _pick_openalex_record(title: str, results: list[dict]) -> dict | None:
    """Choose the canonical OpenAlex record for a work from title-search hits.

    Requires a close title match first — "Attention Is All You Need" must not
    match "Channel Attention Is All You Need for Video Frames" — then prefers
    the most-cited of the survivors, which is the record other works point at.
    """
    from ..models import slugify

    target = slugify(title, 200)
    q = set(re.findall(r"[a-z0-9]+", title.lower()))
    scored: list[tuple[float, int, dict]] = []
    for w in results:
        wt = w.get("title") or w.get("display_name") or ""
        if not wt:
            continue
        if slugify(wt, 200) == target:
            sim = 1.0
        else:
            t = set(re.findall(r"[a-z0-9]+", wt.lower()))
            if not t:
                continue
            sim = 2 * len(q & t) / (len(q) + len(t))
        if sim < 0.85:
            continue
        scored.append((sim, w.get("cited_by_count") or 0, w))
    if not scored:
        return None
    scored.sort(key=lambda s: (-s[0], -s[1]))
    return scored[0][2]


def _best_title_match(query: str, cands: list[PaperMeta]) -> PaperMeta | None:
    """Pick the candidate whose title best overlaps the query (token F1)."""
    from ..models import slugify

    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q:
        return cands[0] if cands else None
    best, best_score = None, 0.0
    for c in cands:
        if slugify(c.title, 200) == slugify(query, 200):
            return c
        t = set(re.findall(r"[a-z0-9]+", c.title.lower()))
        if not t:
            continue
        inter = len(q & t)
        score = 2 * inter / (len(q) + len(t))
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None
