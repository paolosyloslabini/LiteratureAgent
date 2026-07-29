"""Full-text extraction helpers and inbox PDF identification."""

from __future__ import annotations

from lit.actions.inbox import _guess_title, normalize_arxiv_from_header
from lit.fetch.fulltext import MIN_USABLE_CHARS, html_to_text, truncate_for_llm
from lit.fetch.metadata import _pick_openalex_record, synth_bibtex

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
# Choosing the canonical OpenAlex record among duplicates
# --------------------------------------------------------------------------

def test_openalex_picks_the_most_cited_exact_title_match():
    results = [
        {"title": "Attention Is All You Need", "cited_by_count": 10},
        {"title": "Attention Is All You Need", "cited_by_count": 6597},
    ]
    assert _pick_openalex_record("Attention Is All You Need", results)["cited_by_count"] \
        == 6597


def test_openalex_rejects_a_similar_but_different_paper():
    results = [
        {"title": "Channel Attention Is All You Need for Video Frame Interpolation",
         "cited_by_count": 900},
    ]
    assert _pick_openalex_record("Attention Is All You Need", results) is None


def test_openalex_returns_none_for_no_results():
    assert _pick_openalex_record("Anything", []) is None


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
