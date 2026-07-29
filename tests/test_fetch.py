"""Full-text extraction helpers and inbox PDF identification."""

from __future__ import annotations

from lit.actions.inbox import _guess_title, normalize_arxiv_from_header
from lit.fetch.fulltext import (
    MIN_USABLE_CHARS,
    html_to_text,
    strip_reference_list,
    truncate_for_llm,
)
from lit.fetch.metadata import (
    _best_title_match,
    _strip_tags,
    search_arxiv,
    search_crossref,
    search_semantic_scholar,
    search_works,
    synth_bibtex,
)

from factories import make_meta


# --------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------

def test_html_to_text_drops_scripts_and_styles():
    html = "<html><head><style>p{color:red}</style></head><body>" \
           "<script>alert('x')</script><p>Real content here.</p></body></html>"
    text = html_to_text(html)
    assert "Real content here." in text
    assert "alert" not in text and "color:red" not in text


def test_html_to_text_preserves_paragraph_breaks():
    text = html_to_text("<p>First para.</p><p>Second para.</p>")
    assert "First para." in text and "Second para." in text
    assert "First para.Second" not in text


def test_html_to_text_unescapes_entities():
    assert "Knowledge & Data" in html_to_text("<p>Knowledge &amp; Data</p>")


def test_html_to_text_handles_empty():
    assert html_to_text("") == ""


# --------------------------------------------------------------------------
# Prompt truncation
# --------------------------------------------------------------------------

def test_short_text_is_not_truncated():
    text = "x" * 100
    out, truncated = truncate_for_llm(text, 1000)
    assert out == text and not truncated


def test_long_text_keeps_head_and_tail():
    text = "HEAD" + "x" * 100_000 + "TAIL"
    out, truncated = truncate_for_llm(text, 1000)
    assert truncated
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")  # the reference list lives at the end
    assert len(out) < 1200


# --------------------------------------------------------------------------
# Dropping the paper's own bibliography before a read
# --------------------------------------------------------------------------

def _paper(body: str, refs: str = "") -> str:
    return f"Abstract\n\n{body}\n\nConclusion\n\nWe conclude.\n{refs}"


def test_reference_list_is_dropped():
    refs = "\nReferences\n\n" + "\n".join(f"[{i}] Someone. A paper. 2020."
                                          for i in range(200))
    text = _paper("Method. " * 400, refs)
    out, removed = strip_reference_list(text)
    assert removed > 0
    assert "Someone. A paper." not in out
    assert "We conclude." in out
    assert out.endswith("[... reference list omitted ...]")


def test_bibliography_and_works_cited_headings_are_recognized():
    for heading in ("Bibliography", "WORKS CITED", "7. References"):
        text = _paper("Body. " * 400, f"\n{heading}\n\n" + "Ref line.\n" * 200)
        out, removed = strip_reference_list(text)
        assert removed > 0, heading
        assert "Ref line." not in out, heading


def test_a_paper_with_no_reference_list_is_left_alone():
    text = _paper("Body. " * 400)
    out, removed = strip_reference_list(text)
    assert removed == 0 and out == text


def test_the_word_references_in_a_sentence_is_not_a_heading():
    text = _paper("The paper references prior work throughout. " * 200)
    out, removed = strip_reference_list(text)
    assert removed == 0 and out == text


def test_an_early_heading_does_not_take_the_body_with_it():
    """A contents entry near the front must not truncate the whole paper."""
    text = "References\n\n" + _paper("Body. " * 800)
    out, removed = strip_reference_list(text)
    assert removed == 0 and out == text


def test_truncate_only_drops_references_when_asked():
    refs = "\nReferences\n\n" + "Ref line.\n" * 300
    text = _paper("Body. " * 400, refs)
    kept, _ = truncate_for_llm(text, 1_000_000)
    assert "Ref line." in kept
    dropped, _ = truncate_for_llm(text, 1_000_000, drop_references=True)
    assert "Ref line." not in dropped


# --------------------------------------------------------------------------
# BibTeX synthesis for records Crossref has no entry for
# --------------------------------------------------------------------------

def test_synth_bibtex_for_a_conference_paper():
    bib = synth_bibtex(make_meta(type="conference paper"))
    assert bib.startswith("@inproceedings{vaswani2017attention,")
    assert "booktitle = {NeurIPS}" in bib
    assert "author = {Ashish Vaswani and Noam Shazeer}" in bib


def test_synth_bibtex_for_a_journal_article():
    bib = synth_bibtex(make_meta(type="journal article", venue="Nature"))
    assert bib.startswith("@article{")
    assert "journal = {Nature}" in bib


def test_synth_bibtex_includes_eprint_for_arxiv():
    bib = synth_bibtex(make_meta(type="preprint"))
    assert "eprint = {1706.03762}" in bib
    assert "archivePrefix = {arXiv}" in bib


def test_synth_bibtex_survives_missing_fields():
    bib = synth_bibtex(make_meta(authors=[], year=None, venue=None, arxiv_id=None))
    assert bib.startswith("@inproceedings{")


# --------------------------------------------------------------------------
# Choosing the canonical record for a work among duplicate registrations
# --------------------------------------------------------------------------

def test_title_match_picks_the_most_cited_exact_match():
    """An index holds several copies of one work; the canonical one is the
    record the rest of the literature actually points at."""
    cands = [
        make_meta(title="Attention Is All You Need", citation_count=10),
        make_meta(title="Attention Is All You Need", citation_count=6597),
    ]
    best = _best_title_match("Attention Is All You Need", cands, threshold=0.85)
    assert best.citation_count == 6597


def test_title_match_rejects_a_similar_but_different_paper():
    cands = [make_meta(
        title="Channel Attention Is All You Need for Video Frame Interpolation",
        citation_count=900)]
    assert _best_title_match("Attention Is All You Need", cands,
                             threshold=0.85) is None


def test_title_match_returns_none_for_no_results():
    assert _best_title_match("Anything", [], threshold=0.85) is None


# --------------------------------------------------------------------------
# Publisher-escaped markup in Crossref fields
# --------------------------------------------------------------------------

def test_escaped_html_is_decoded_out_of_a_title():
    """Crossref returns some titles as escaped HTML. Left alone they reach the
    candidate table verbatim and split one work across two dedup keys."""
    assert _strip_tags("&lt;p&gt;Agentic and Multi-agent Systems&lt;/p&gt;") \
        == "Agentic and Multi-agent Systems"


def test_double_escaped_entities_are_decoded():
    # One unescape pass turns "&amp;nbsp;" into "&nbsp;", which is still not text.
    assert _strip_tags("&amp;nbsp;Agent Brain") == "Agent Brain"


def test_crossref_search_decodes_titles():
    http = RecordingHttp({"message": {"items": [
        {"title": ["&lt;p&gt;A Benchmark&lt;/p&gt;"], "DOI": "10.1/x"},
    ]}})
    assert search_crossref(http, "topic")[0].title == "A Benchmark"


# --------------------------------------------------------------------------
# Inbox identification
# --------------------------------------------------------------------------

def test_arxiv_stamp_is_found():
    head = "arXiv:1706.03762v5  [cs.CL]  6 Dec 2017\n\nAttention Is All You Need"
    assert normalize_arxiv_from_header(head) == "1706.03762"


def test_arxiv_url_is_found():
    assert normalize_arxiv_from_header("see https://arxiv.org/abs/2301.00234") == \
        "2301.00234"


def test_no_arxiv_id_returns_none():
    assert normalize_arxiv_from_header("just some text") is None


def test_guess_title_from_a_typical_first_page():
    head = "arXiv:1706.03762v5 [cs.CL] 6 Dec 2017\n\nAttention Is All You Need\n\n" \
           "Ashish Vaswani, Noam Shazeer\n\nAbstract\n\nThe dominant sequence..."
    assert _guess_title(head) == "Attention Is All You Need"


def test_guess_title_skips_boilerplate():
    head = "Preprint. Under review.\n\nDeep Residual Learning for Image Recognition\n\n" \
           "Kaiming He and Xiangyu Zhang\n"
    assert _guess_title(head) == "Deep Residual Learning for Image Recognition"


def test_guess_title_returns_none_for_junk():
    assert _guess_title("1\n2\n3\n") is None


# --------------------------------------------------------------------------
# The usability threshold
# --------------------------------------------------------------------------

def test_min_usable_chars_rejects_an_abstract_sized_extraction():
    # A landing page with only an abstract must not count as "read the paper".
    assert len("An abstract of a paper. " * 50) < MIN_USABLE_CHARS


# --------------------------------------------------------------------------
# Title-matched records must not be trusted for identity
# --------------------------------------------------------------------------

def test_supplementary_keeps_metrics_and_drops_identity():
    """A title lookup can land on a mirror of the same work. "Attention Is All
    You Need" resolves to a 2025 DOI minted by a medical academy: same authors,
    same title, wrong identity. Adopting it would corrupt the bibliography."""
    from lit.models import Reference

    mirror = make_meta(
        title="Attention Is All You Need",
        doi="10.65215/2q58a426",
        venue="Shenzhen Medical Academy of Research and Translation",
        year=2025,
        citation_count=6597,
        references=[Reference(title="A cited work")],
    )
    mirror.title_matched = True
    mirror.oa_pdf_url = "https://example.org/paper.pdf"

    safe = mirror.supplementary()
    assert safe.doi is None
    assert safe.venue is None
    assert safe.year is None
    assert safe.title == ""
    # ...but the metrics, which describe the work itself, survive.
    assert safe.citation_count == 6597
    assert len(safe.references) == 1
    assert safe.oa_pdf_url == "https://example.org/paper.pdf"


def test_merging_a_supplementary_record_preserves_the_authoritative_one():
    arxiv_record = make_meta(doi=None, venue=None, year=2017, citation_count=None,
                             references=[])
    mirror = make_meta(doi="10.65215/2q58a426", venue="Some Mirror", year=2025,
                       citation_count=6597)
    mirror.title_matched = True

    arxiv_record.merge(mirror.supplementary())
    assert arxiv_record.doi is None          # identity untouched
    assert arxiv_record.year == 2017
    assert arxiv_record.venue is None
    assert arxiv_record.citation_count == 6597   # metrics adopted


# --------------------------------------------------------------------------
# Search parameters: the year window and document type a query implies
#
# Each index spells these differently, and getting one wrong fails silently —
# the request still returns papers, just not the ones that were asked for.
# --------------------------------------------------------------------------

class RecordingHttp:
    """Enough of HttpClient to see what a search actually asked the index."""

    def __init__(self, payload: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.payload = payload or {}

    def get_json(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        return self.payload

    def get(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        return None

    @property
    def params(self) -> dict:
        return self.calls[0][1]


def test_s2_search_sends_the_year_window():
    http = RecordingHttp()
    search_semantic_scholar(http, "topic", from_year=2020, to_year=2024)
    assert http.params["year"] == "2020-2024"


def test_s2_search_sends_an_open_ended_window():
    http = RecordingHttp()
    search_semantic_scholar(http, "topic", from_year=2020)
    assert http.params["year"] == "2020-"
    http = RecordingHttp()
    search_semantic_scholar(http, "topic", to_year=2024)
    assert http.params["year"] == "-2024"


def test_s2_search_sends_the_document_type():
    http = RecordingHttp()
    search_semantic_scholar(http, "topic", kind="Review")
    assert http.params["publicationTypes"] == "Review"


def test_s2_search_without_constraints_sends_no_window():
    http = RecordingHttp()
    search_semantic_scholar(http, "topic")
    assert "year" not in http.params
    assert "publicationTypes" not in http.params


def test_s2_sorted_search_uses_the_bulk_endpoint():
    """Only the bulk endpoint can order results, and ordering by citation count
    is the whole of the most-cited facet."""
    http = RecordingHttp()
    search_semantic_scholar(http, "topic", sort="citationCount:desc")
    url, params = http.calls[0]
    assert url.endswith("/search/bulk")
    assert params["sort"] == "citationCount:desc"


def test_s2_unsorted_search_uses_the_relevance_endpoint():
    http = RecordingHttp()
    search_semantic_scholar(http, "topic")
    assert http.calls[0][0].endswith("/paper/search")


def test_crossref_search_sends_the_year_window():
    http = RecordingHttp()
    search_works(http, "topic", 5, from_year=2020, to_year=2024)
    assert http.params["filter"] == "from-pub-date:2020-01-01,until-pub-date:2024-12-31"


def test_crossref_search_sends_an_open_ended_window():
    http = RecordingHttp()
    search_works(http, "topic", 5, from_year=2020)
    assert http.params["filter"] == "from-pub-date:2020-01-01"


def test_crossref_search_without_a_window_sends_no_filter():
    http = RecordingHttp()
    search_works(http, "topic", 5)
    assert "filter" not in http.params


def test_crossref_search_sends_the_document_type():
    http = RecordingHttp()
    search_crossref(http, "topic", 5, kind="preprint")
    assert "type:posted-content" in http.params["filter"]


def test_crossref_search_can_order_by_citation_count():
    """Crossref's count undercounts CS badly, but ordering by an undercount
    still puts the heavily-cited work on top — and it is always reachable."""
    http = RecordingHttp()
    search_crossref(http, "topic", 5, sort="is-referenced-by-count")
    assert http.params["sort"] == "is-referenced-by-count"
    assert http.params["order"] == "desc"


def test_the_window_reaches_the_s2_fallback_too():
    """Crossref coming back short must not silently drop the constraint."""
    http = RecordingHttp()
    search_works(http, "topic", 5, from_year=2020)
    fallback = [p for url, p in http.calls if "semanticscholar" in url]
    assert fallback and fallback[0]["year"] == "2020-"


def test_arxiv_search_sends_the_year_window_as_a_submission_range():
    http = RecordingHttp()
    search_arxiv(http, "topic", from_year=2020, to_year=2024)
    assert http.params["search_query"] == \
        "(all:topic) AND submittedDate:[202001010000 TO 202412312359]"


def test_arxiv_search_without_a_window_stays_a_plain_query():
    http = RecordingHttp()
    search_arxiv(http, "topic")
    assert http.params["search_query"] == "all:topic"


# --------------------------------------------------------------------------
# Citation counts: indexes undercount, so the largest figure wins
# --------------------------------------------------------------------------

def test_merge_takes_the_larger_citation_count():
    """Crossref counts only citations from DOI-registered work, so for a paper
    the field cites as a preprint its count runs orders of magnitude low: 9 for
    GAIA against 1032 at S2. Filling only when the field is empty would keep
    whichever source answered first."""
    crossref = make_meta(citation_count=9)
    s2 = make_meta(citation_count=1032)

    assert crossref.merge(s2).citation_count == 1032


def test_merge_does_not_let_a_thinner_index_lower_the_count():
    s2 = make_meta(citation_count=1032)
    crossref = make_meta(citation_count=9)

    assert s2.merge(crossref).citation_count == 1032


def test_merge_still_adopts_a_count_when_it_has_none():
    assert make_meta(citation_count=None).merge(make_meta(citation_count=42)) \
        .citation_count == 42


def test_resolve_asks_s2_for_a_count_even_when_references_are_in_hand(monkeypatch):
    """The count is worth its own request. S2 must not be treated as a
    references-only fallback: a paper whose references another source already
    supplied still needs the count, or GAIA stays on 9 citations — three a year,
    below the B threshold in `quality.assess`."""
    from lit.models import Reference
    import lit.fetch.metadata as md

    calls = []

    def fake_arxiv(http, arxiv_id):
        return make_meta(title="GAIA: a benchmark for General AI Assistants",
                         doi=None, venue=None, year=2023, type="preprint",
                         arxiv_id=arxiv_id, citation_count=None,
                         references=[Reference(title="A cited work")])

    def fake_s2(http, ident, api_key="", with_references=True):
        calls.append(("s2", ident, with_references))
        return make_meta(citation_count=1032, references=[])

    monkeypatch.setattr(md, "from_arxiv", fake_arxiv)
    monkeypatch.setattr(md, "from_semantic_scholar", fake_s2)

    meta = md.resolve_metadata(http=None, arxiv_id="2311.12983")

    assert meta.citation_count == 1032
    # ...and we did not pay for a reference list we already had.
    assert ("s2", "arXiv:2311.12983", False) in calls


def test_resolve_falls_back_to_a_title_search_with_no_identifier(monkeypatch):
    """A work with neither DOI nor arXiv id still needs a citation count, but a
    title match can land on a different paper of the same name — so what it
    returns is supplementary and must never overwrite identity."""
    import lit.fetch.metadata as md

    def fake_search(http, query, limit=10, cfg=None, **kw):
        return [make_meta(title="Some Paper", doi=None, arxiv_id=None,
                          citation_count=None)]

    def fake_by_title(http, title, api_key="", with_references=True):
        return make_meta(title="Some Paper", doi="10.9999/mirror",
                         citation_count=77, title_matched=True)

    monkeypatch.setattr(md, "search_works", fake_search)
    monkeypatch.setattr(md, "s2_by_title", fake_by_title)

    meta = md.resolve_metadata(http=None, title="Some Paper")

    assert meta.citation_count == 77
    assert meta.doi is None  # the mirror's DOI was not adopted
