"""The `add` path: target parsing, UNVERIFIED handling, and field assembly.

The network and the LLM are both stubbed here so the *policy* is what gets
tested: what happens when a paper can't be reached, when it fails the bar, when
it's already present, and that summaries only ever come from full text.
"""

from __future__ import annotations

import pytest
from factories import make_meta
from rich.console import Console

from lit.actions.add import add_paper
from lit.actions.context import Ctx, parse_target
from lit.fetch.fulltext import FullText
from lit.llm import LLMError
from lit.models import STATUS_UNREAD, STATUS_UNVERIFIED, STATUS_VERIFIED

READING = {
    "one_liner": "Introduces the Transformer.",
    "sections": [{"name": "Introduction", "summary": "Recurrence limits parallelism."}],
    "key_findings": ["28.4 BLEU on WMT 2014"],
    "tags": ["Transformers", "self attention"],
    "type": "conference paper",
}


# --------------------------------------------------------------------------
# Target parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,field,value", [
    ("10.1145/3292500.3330701", "doi", "10.1145/3292500.3330701"),
    ("doi:10.1145/3292500.3330701", "doi", "10.1145/3292500.3330701"),
    ("https://doi.org/10.1145/3292500.3330701", "doi", "10.1145/3292500.3330701"),
    ("arxiv:1706.03762", "arxiv_id", "1706.03762"),
    ("1706.03762", "arxiv_id", "1706.03762"),
    ("https://arxiv.org/abs/1706.03762v3", "arxiv_id", "1706.03762"),
    ("Attention Is All You Need", "title", "Attention Is All You Need"),
])
def test_parse_target(raw, field, value):
    assert getattr(parse_target(raw), field) == value


def test_title_containing_digits_is_not_mistaken_for_an_arxiv_id():
    t = parse_target("GPT-4 and the 1706 problem")
    assert t.title == "GPT-4 and the 1706 problem"
    assert t.arxiv_id is None


def test_url_target_is_kept():
    assert parse_target("https://example.org/paper").url == "https://example.org/paper"


# --------------------------------------------------------------------------
# add_paper
# --------------------------------------------------------------------------

class StubLLM:
    """Stands in for ClaudeCLI. Records what it was asked to read."""

    available = True

    def __init__(self, payload=None, fail=False):
        self.payload = payload if payload is not None else dict(READING)
        self.fail = fail
        self.seen_text: list[str] = []
        self.calls = 0

    def json(self, prompt, *, stdin_text=None, **kw):
        self.calls += 1
        self.seen_text.append(stdin_text or "")
        if self.fail:
            raise LLMError("model unavailable")
        return dict(self.payload)

    def run(self, prompt, **kw):  # pragma: no cover - unused here
        raise NotImplementedError


@pytest.fixture
def ctx(lib, cfg):
    c = Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)
    return c


def wire(monkeypatch, ctx, *, meta, fulltext, llm=None):
    llm = llm or StubLLM()
    monkeypatch.setattr("lit.actions.add.resolve_metadata",
                        lambda *a, **k: meta)
    monkeypatch.setattr("lit.actions.add.fetch_fulltext",
                        lambda *a, **k: fulltext)
    ctx._llm = llm
    return llm


def test_add_reads_full_text_and_fills_fields(monkeypatch, ctx):
    text = "FULL PAPER TEXT " * 500
    llm = wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText(text, "arxiv"))
    res = add_paper(ctx, "Attention Is All You Need")

    assert res.status == "added"
    assert res.entry.status == STATUS_VERIFIED
    assert res.entry.one_liner == "Introduces the Transformer."
    assert res.entry.sections[0].name == "Introduction"
    assert res.entry.key == "vaswani2017attention"
    # The whole text was handed to the reader, not an abstract.
    assert "FULL PAPER TEXT" in llm.seen_text[0]
    assert len(llm.seen_text[0]) > 5000


def test_tags_are_normalized(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"))
    res = add_paper(ctx, "t", extra_tags=["Extra Tag"])
    assert res.entry.tags == ["transformers", "self-attention", "extra-tag"]


def test_unreachable_paper_is_unverified_with_no_summary(monkeypatch, ctx):
    llm = wire(monkeypatch, ctx, meta=make_meta(), fulltext=None)
    res = add_paper(ctx, "Attention Is All You Need")

    assert res.ok
    assert res.entry.status == STATUS_UNVERIFIED
    assert res.entry.one_liner is None
    assert res.entry.sections == []
    assert llm.calls == 0  # the model was never asked to guess
    assert any("full text" in w for w in res.warnings)


def test_metadata_is_still_recorded_when_unverified(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=None)
    res = add_paper(ctx, "x")
    assert res.entry.year == 2017
    assert res.entry.venue == "NeurIPS"
    assert res.entry.arxiv_id == "1706.03762"


# --------------------------------------------------------------------------
# Filing without reading — the cheap path `find` takes by default
# --------------------------------------------------------------------------

def test_read_false_files_metadata_and_never_fetches_or_reads(monkeypatch, ctx):
    llm = wire(monkeypatch, ctx, meta=make_meta(abstract="We propose a new model."),
               fulltext=FullText("x" * 9000, "arxiv"))
    def no_fetch(*a, **k):
        raise AssertionError("the full text was fetched on the unread path")
    monkeypatch.setattr("lit.actions.add.fetch_fulltext", no_fetch)

    res = add_paper(ctx, "Attention Is All You Need", read=False)

    assert res.status == "added"
    assert res.entry.status == STATUS_UNREAD
    assert res.entry.is_unread and res.entry.needs_read
    assert llm.calls == 0
    # Bibliographic record and the publisher's abstract are all there.
    assert res.entry.year == 2017 and res.entry.venue == "NeurIPS"
    assert res.entry.abstract == "We propose a new model."
    # But nothing a reader would have written.
    assert res.entry.one_liner is None and res.entry.sections == []


def test_the_unread_message_says_how_to_read_it(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=None)
    res = add_paper(ctx, "x", read=False)
    assert "lit read vaswani2017attention" in res.message


def test_a_metadata_refresh_does_not_discard_a_summary_already_paid_for(
        monkeypatch, ctx):
    """`read=False` over an entry that was read keeps its reading."""
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"))
    first = add_paper(ctx, "Attention Is All You Need")
    assert first.entry.status == STATUS_VERIFIED

    again = add_paper(ctx, "Attention Is All You Need", read=False, refresh=True)
    assert again.entry.status == STATUS_VERIFIED
    assert again.entry.one_liner == "Introduces the Transformer."
    assert again.entry.sections[0].name == "Introduction"


def test_reading_an_unread_entry_later_fills_it_in(monkeypatch, ctx):
    from lit.actions.add import read_entries

    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"))
    filed = add_paper(ctx, "Attention Is All You Need", read=False)
    assert filed.entry.needs_read

    [res] = read_entries(ctx, [filed.entry])
    assert res.ok
    assert res.entry.status == STATUS_VERIFIED
    assert res.entry.one_liner == "Introduces the Transformer."


def test_the_reference_list_is_not_sent_to_the_reader(monkeypatch, ctx):
    body = "Method and results. " * 500
    text = f"Abstract\n\n{body}\n\nReferences\n\n" + "[1] Someone. Paper. 2020.\n" * 200
    llm = wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText(text, "arxiv"))
    add_paper(ctx, "x")

    sent = llm.seen_text[0]
    assert "Method and results." in sent
    assert "Someone. Paper. 2020." not in sent


def test_find_reads_on_a_tighter_budget_than_add(monkeypatch, ctx):
    huge = "WORD " * 60_000  # 300k chars: over find's budget, under add's
    llm = wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText(huge, "arxiv"))

    add_paper(ctx, "x")
    generous = len(llm.seen_text[-1])
    add_paper(ctx, "x", refresh=True, max_chars=ctx.cfg.llm.find_read_chars)
    tight = len(llm.seen_text[-1])

    assert tight < generous
    assert tight <= ctx.cfg.llm.find_read_chars + 200


# --------------------------------------------------------------------------
# Code / artifact links, and the abstract
# --------------------------------------------------------------------------

def code_reading(url):
    return dict(READING, code_url=url)


def test_code_url_is_kept_when_the_paper_prints_it(monkeypatch, ctx):
    text = "x" * 9000 + "\nCode: https://github.com/google/flax\n"
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText(text, "arxiv"),
         llm=StubLLM(code_reading("https://github.com/google/flax")))
    res = add_paper(ctx, "t")
    assert res.entry.code_url == "https://github.com/google/flax"


def test_code_url_survives_a_line_break_in_the_extracted_text(monkeypatch, ctx):
    # PDF extraction routinely snaps a long URL across lines.
    text = "x" * 9000 + "\nour code is at https://github.com/\ngoogle/flax today"
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText(text, "arxiv"),
         llm=StubLLM(code_reading("https://github.com/google/flax")))
    assert add_paper(ctx, "t").entry.code_url == "https://github.com/google/flax"


def test_a_code_url_absent_from_the_text_is_dropped(monkeypatch, ctx):
    # The model recalling a plausible repo is exactly what must not be stored.
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"),
         llm=StubLLM(code_reading("https://github.com/google/attention")))
    assert add_paper(ctx, "t").entry.code_url is None


def test_a_non_repository_url_is_dropped(monkeypatch, ctx):
    text = "x" * 9000 + "\nSee https://sites.google.com/view/our-project\n"
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText(text, "arxiv"),
         llm=StubLLM(code_reading("https://sites.google.com/view/our-project")))
    assert add_paper(ctx, "t").entry.code_url is None


def test_a_paper_with_no_code_has_no_code_url(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"),
         llm=StubLLM(code_reading(None)))
    assert add_paper(ctx, "t").entry.code_url is None


def test_the_abstract_is_taken_from_metadata_not_the_model(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(abstract="The publisher's abstract."),
         fulltext=FullText("x" * 9000, "arxiv"),
         llm=StubLLM(dict(READING, abstract="A model-written abstract.")))
    assert add_paper(ctx, "t").entry.abstract == "The publisher's abstract."


def test_an_unverified_entry_still_carries_its_abstract(monkeypatch, ctx):
    # Nothing was read, but the abstract is metadata, so it is still on record.
    wire(monkeypatch, ctx, meta=make_meta(abstract="The publisher's abstract."),
         fulltext=None)
    res = add_paper(ctx, "t")
    assert res.entry.status == STATUS_UNVERIFIED
    assert res.entry.abstract == "The publisher's abstract."


def test_no_record_found_is_reported(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=None, fulltext=None)
    res = add_paper(ctx, "A Paper That Does Not Exist")
    assert res.status == "not_found"
    assert not ctx.library.keys()


def test_an_unreachable_index_is_not_reported_as_a_missing_paper(monkeypatch, ctx):
    """A rate-limited arXiv must not be dressed up as "no such paper".

    The advice attached to `not_found` is "try a DOI, an arXiv id, or --pdf" —
    all useless when the identifier was already correct and the index was
    simply unreachable, and all of it sends the user off changing a correct
    input.
    """
    def resolve_but_upstream_is_down(*a, **k):
        ctx.http._note_unavailable()
        return None

    monkeypatch.setattr("lit.actions.add.resolve_metadata",
                        resolve_but_upstream_is_down)
    monkeypatch.setattr("lit.actions.add.fetch_fulltext", lambda *a, **k: None)

    res = add_paper(ctx, "arxiv:2601.11868")
    assert res.status == "unavailable"
    assert "not a missing paper" in res.message
    assert not ctx.library.keys()


def test_duplicate_is_detected(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"))
    add_paper(ctx, "t")
    res = add_paper(ctx, "t")
    assert res.status == "duplicate"
    assert len(ctx.library) == 1


def test_refresh_rereads_an_existing_entry(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"))
    add_paper(ctx, "t")
    res = add_paper(ctx, "t", refresh=True)
    assert res.status == "updated"
    assert len(ctx.library) == 1


def test_quality_bar_rejects_and_writes_nothing(monkeypatch, ctx):
    ctx.library.settings.min_level = "A*"
    meta = make_meta(venue="PLOS ONE", type="journal article",
                     year=2010, citation_count=2)
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"))
    res = add_paper(ctx, "t")
    assert res.status == "rejected"
    assert not ctx.library.keys()


def test_force_overrides_the_quality_bar(monkeypatch, ctx):
    ctx.library.settings.min_level = "A*"
    meta = make_meta(venue="PLOS ONE", type="journal article",
                     year=2010, citation_count=2)
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"))
    assert add_paper(ctx, "t", force=True).ok


def test_reading_failure_keeps_the_entry_unverified(monkeypatch, ctx):
    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"),
         llm=StubLLM(fail=True))
    res = add_paper(ctx, "t")
    assert res.status == "error"
    assert ctx.library.get(res.entry.key).status == STATUS_UNVERIFIED


def test_references_come_from_metadata_not_the_model(monkeypatch, ctx):
    from lit.models import Reference

    meta = make_meta(references=[Reference(title="Real Reference", year=2014)])
    llm = StubLLM({**READING, "references": [{"title": "HALLUCINATED"}]})
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"), llm=llm)
    res = add_paper(ctx, "t")
    titles = [r.title for r in res.entry.references]
    assert titles == ["Real Reference"]


def test_llm_level_is_used_only_when_metrics_are_thin(monkeypatch, ctx):
    meta = make_meta(venue=None, citation_count=None, year=None, type="other")
    llm = StubLLM({**READING, "level": "A", "level_reason": "strong evaluation"})
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"), llm=llm)
    res = add_paper(ctx, "t")
    assert res.entry.level == "A"
    assert "judged from content" in res.entry.level_reason


def test_llm_cannot_override_a_metric_based_level(monkeypatch, ctx):
    meta = make_meta(venue="NeurIPS", year=2015, citation_count=5000)
    llm = StubLLM({**READING, "level": "C", "level_reason": "I disagree"})
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"), llm=llm)
    res = add_paper(ctx, "t")
    assert res.entry.level == "A*"


def test_user_notes_survive_a_reread(monkeypatch, ctx):
    from lit import entryfile

    wire(monkeypatch, ctx, meta=make_meta(), fulltext=FullText("x" * 9000, "arxiv"))
    res = add_paper(ctx, "t")
    entryfile.set_notes(ctx.library.entry_path(res.entry.key), "MY NOTE")
    add_paper(ctx, "t", refresh=True)
    assert ctx.library.get(res.entry.key).notes == "MY NOTE"


def test_venue_recovered_from_text_upgrades_type_and_bibtex(monkeypatch, ctx):
    """An arXiv-only record is filed as a preprint, but the paper's first page
    names the conference. The reader recovers it, and the BibTeX follows."""
    meta = make_meta(venue=None, type="preprint", doi=None)
    meta.bibtex = "@misc{k, title = {T}}"
    llm = StubLLM({
        **READING,
        "type": "conference paper",
        "venue_from_text": "31st Conference on Neural Information Processing Systems",
    })
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"), llm=llm)
    res = add_paper(ctx, "t")

    assert res.entry.venue.startswith("31st Conference")
    assert res.entry.type == "conference paper"
    assert res.entry.bibtex.startswith("@inproceedings{")
    assert "booktitle" in res.entry.bibtex


def test_crossref_bibtex_is_never_overwritten(monkeypatch, ctx):
    meta = make_meta(venue=None, type="preprint", doi="10.1234/abcd")
    meta.bibtex = "@inproceedings{authoritative, title = {From Crossref}}"
    llm = StubLLM({**READING, "venue_from_text": "Some Conference"})
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"), llm=llm)
    assert add_paper(ctx, "t").entry.bibtex == meta.bibtex


def test_metadata_venue_beats_the_text(monkeypatch, ctx):
    meta = make_meta(venue="NeurIPS", type="conference paper")
    llm = StubLLM({**READING, "venue_from_text": "Some Workshop Nobody Heard Of"})
    wire(monkeypatch, ctx, meta=meta, fulltext=FullText("x" * 9000, "arxiv"), llm=llm)
    assert add_paper(ctx, "t").entry.venue == "NeurIPS"
