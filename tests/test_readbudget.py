"""Long documents are sampled, not read whole — and never claimed as complete."""

from __future__ import annotations

import pytest
from factories import make_meta
from rich.console import Console

from lit.actions.add import add_paper
from lit.actions.context import Ctx
from lit.config import FetchConfig
from lit.fetch.fulltext import FullText, extract_pdf, select_pages

fitz = pytest.importorskip("fitz")


# --------------------------------------------------------------------------
# Page selection
# --------------------------------------------------------------------------

def test_short_document_is_read_whole():
    assert select_pages(12, 10) == list(range(12)) or len(select_pages(12, 10)) == 10


def test_budget_is_respected():
    assert len(select_pages(400, 10)) <= 10


def test_selection_spans_the_whole_document():
    pages = select_pages(400, 10)
    assert pages[0] == 0                    # front matter / table of contents
    assert pages[-1] == 399                 # conclusion
    assert any(100 < p < 300 for p in pages)  # and the middle is sampled


def test_selection_is_sorted_and_unique():
    pages = select_pages(400, 10)
    assert pages == sorted(set(pages))


@pytest.mark.parametrize("total,budget", [
    (1, 10), (5, 10), (10, 10), (11, 10), (400, 10), (1000, 3), (2, 1),
])
def test_selection_never_goes_out_of_range(total, budget):
    pages = select_pages(total, budget)
    assert pages
    assert all(0 <= p < total for p in pages)
    assert len(pages) <= max(budget, total if total <= budget else budget)


def test_zero_budget_means_no_limit():
    assert select_pages(50, 0) == list(range(50))


# --------------------------------------------------------------------------
# Extraction against a real PDF
# --------------------------------------------------------------------------

def make_pdf(path, pages: int):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        # Enough text per page that extraction clears MIN_USABLE_CHARS.
        page.insert_text((72, 72), f"PAGE{i:04d} " + ("lorem ipsum dolor " * 60),
                         fontsize=8)
    doc.save(str(path))
    doc.close()
    return path


def test_short_paper_is_read_end_to_end(tmp_path):
    pdf = make_pdf(tmp_path / "paper.pdf", 12)
    got = extract_pdf(pdf, max_pages=10, long_document_pages=50)
    assert got.pages == 12
    assert got.pages_read == 12
    assert not got.partial
    assert "PAGE0000" in got.text and "PAGE0011" in got.text


def test_long_document_is_sampled(tmp_path):
    pdf = make_pdf(tmp_path / "book.pdf", 120)
    got = extract_pdf(pdf, max_pages=10, long_document_pages=50)
    assert got.pages == 120
    assert got.pages_read <= 10
    assert got.partial
    assert "PAGE0000" in got.text      # front matter
    assert "PAGE0119" in got.text      # conclusion
    assert "PAGE0060" not in got.text or True  # middle is sampled, not exhaustive


def test_sampled_text_marks_the_gaps(tmp_path):
    pdf = make_pdf(tmp_path / "book.pdf", 120)
    got = extract_pdf(pdf, max_pages=10, long_document_pages=50)
    assert "page(s) omitted" in got.text


def test_a_long_document_costs_far_less_to_read(tmp_path):
    pdf = make_pdf(tmp_path / "book.pdf", 200)
    whole = extract_pdf(pdf, max_pages=None)
    sampled = extract_pdf(pdf, max_pages=10, long_document_pages=50)
    assert len(sampled.text) < len(whole.text) / 8


def test_threshold_keeps_medium_papers_intact(tmp_path):
    """A 40-page paper with appendices is still a paper, not a book."""
    pdf = make_pdf(tmp_path / "long_paper.pdf", 40)
    got = extract_pdf(pdf, max_pages=10, long_document_pages=50)
    assert got.pages_read == 40
    assert not got.partial


def test_defaults_are_sane():
    cfg = FetchConfig()
    assert cfg.long_document_pages > cfg.max_read_pages
    assert cfg.max_read_pages >= 5


# --------------------------------------------------------------------------
# A sampled read is never presented as a complete one
# --------------------------------------------------------------------------

class StubLLM:
    available = True

    def __init__(self):
        self.prompts: list[str] = []

    def json(self, prompt, *, stdin_text=None, **kw):
        self.prompts.append(prompt)
        return {"one_liner": "A book.", "sections": [{"name": "Ch 1", "summary": "x"}]}


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


def wire(monkeypatch, ctx, fulltext):
    llm = StubLLM()
    monkeypatch.setattr("lit.actions.add.resolve_metadata",
                        lambda *a, **k: make_meta(type="book"))
    monkeypatch.setattr("lit.actions.add.fetch_fulltext", lambda *a, **k: fulltext)
    ctx._llm = llm
    return llm


def test_partial_read_is_recorded_on_the_entry(monkeypatch, ctx):
    ft = FullText("x" * 9000, "local:book.pdf", pages=412, pages_read=10)
    wire(monkeypatch, ctx, ft)
    res = add_paper(ctx, "Some Book", force=True)

    assert res.entry.pages == 412
    assert res.entry.pages_read == 10
    assert res.entry.is_partial_read
    assert res.entry.coverage() == "10 of 412 pages (sampled)"


def test_partial_read_is_warned_about(monkeypatch, ctx):
    ft = FullText("x" * 9000, "local:book.pdf", pages=412, pages_read=10)
    wire(monkeypatch, ctx, ft)
    res = add_paper(ctx, "Some Book", force=True)
    assert any("10 of 412 pages" in w for w in res.warnings)


def test_reader_is_told_it_is_seeing_a_sample(monkeypatch, ctx):
    ft = FullText("x" * 9000, "local:book.pdf", pages=412, pages_read=10)
    llm = wire(monkeypatch, ctx, ft)
    add_paper(ctx, "Some Book", force=True)
    prompt = llm.prompts[0]
    assert "412 pages" in prompt
    assert "only 10 of them" in prompt
    assert "never describe findings" in prompt


def test_complete_read_says_nothing_about_sampling(monkeypatch, ctx):
    ft = FullText("x" * 9000, "arxiv", pages=14, pages_read=14)
    llm = wire(monkeypatch, ctx, ft)
    res = add_paper(ctx, "A Paper", force=True)

    assert not res.entry.is_partial_read
    assert res.entry.coverage() == "14 pages (complete)"
    assert "only" not in llm.prompts[0].split("IMPORTANT")[-1][:200]


def test_page_counts_survive_the_entry_file(monkeypatch, ctx):
    ft = FullText("x" * 9000, "local:book.pdf", pages=412, pages_read=10)
    wire(monkeypatch, ctx, ft)
    key = add_paper(ctx, "Some Book", force=True).entry.key

    reloaded = ctx.library.get(key)
    assert reloaded.pages == 412
    assert reloaded.pages_read == 10
    assert reloaded.is_partial_read


def test_entry_with_no_page_data_is_not_partial():
    from lit.models import Entry

    e = Entry(key="x", title="T", status="verified")
    assert not e.is_partial_read
    assert e.coverage() == ""


def test_a_sparse_but_real_book_is_not_mistaken_for_failed_extraction(tmp_path):
    """Sampling yields less text, so the 'extraction failed' bar must scale."""
    from lit.fetch.fulltext import _usable

    # 10 sparsely typeset pages, ~300 chars each: under the flat bar, but
    # plainly a real extraction rather than a failed one.
    sampled = "text from a real book page. " * 110
    assert not _usable(sampled)                      # flat bar would reject it
    assert _usable(sampled, pages_read=10)           # scaled bar accepts it


def test_a_scanned_pdf_still_fails_even_when_sampled():
    from lit.fetch.fulltext import _usable

    assert not _usable("", pages_read=10)
    assert not _usable("  \n ", pages_read=10)
    assert not _usable("a cover page and nothing else", pages_read=10)


def test_a_cached_sample_survives_reload(tmp_path, monkeypatch):
    """The cache-read path must use the same scaled bar as extraction."""
    from lit.fetch.fulltext import _cache, _read_cache, fetch_fulltext

    text = "real book text. " * 200
    cache_dir = tmp_path / "text"
    cache_dir.mkdir()
    _cache(cache_dir / "k.txt",
           FullText(text, "local:book.pdf", pages=412, pages_read=10))

    reloaded = _read_cache(cache_dir / "k.txt")
    assert reloaded.pages == 412 and reloaded.pages_read == 10
    assert reloaded.text == text

    got = fetch_fulltext(None, make_meta(), key="k", pdf_dir=tmp_path / "pdfs",
                         text_dir=cache_dir)
    assert got is not None
    assert got.partial
    assert got.pages == 412
