"""`lit refresh` updates bibliographic fields and nothing else."""

from __future__ import annotations

import pytest
from factories import make_entry, make_meta
from rich.console import Console

from lit import entryfile
from lit.actions.context import Ctx
from lit.actions.refresh import refresh_entries
from lit.models import Reference


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


def wire(monkeypatch, meta):
    monkeypatch.setattr("lit.actions.refresh.resolve_metadata", lambda *a, **k: meta)


def test_citation_count_is_updated(monkeypatch, ctx):
    ctx.library.save_entry(make_entry(citation_count=100))
    wire(monkeypatch, make_meta(citation_count=5000))
    results = refresh_entries(ctx)
    assert results[0].updated
    assert ctx.library.get("vaswani2017attention").citation_count == 5000


def test_references_are_backfilled(monkeypatch, ctx):
    ctx.library.save_entry(make_entry(references=[]))
    wire(monkeypatch, make_meta(references=[
        Reference(title="A"), Reference(title="B"),
    ]))
    refresh_entries(ctx)
    assert len(ctx.library.get("vaswani2017attention").references) == 2


def test_a_shorter_reference_list_does_not_replace_a_longer_one(monkeypatch, ctx):
    ctx.library.save_entry(make_entry(
        references=[Reference(title=f"R{i}") for i in range(20)]))
    wire(monkeypatch, make_meta(references=[Reference(title="only one")]))
    refresh_entries(ctx)
    assert len(ctx.library.get("vaswani2017attention").references) == 20


def test_a_published_version_fills_in_doi_and_venue(monkeypatch, ctx):
    ctx.library.save_entry(make_entry(doi=None, venue=None, type="preprint"))
    wire(monkeypatch, make_meta(doi="10.1234/abcd", venue="NeurIPS"))
    refresh_entries(ctx)
    e = ctx.library.get("vaswani2017attention")
    assert e.doi == "10.1234/abcd"
    assert e.venue == "NeurIPS"


def test_level_is_recomputed_once_metrics_arrive(monkeypatch, ctx):
    ctx.library.save_entry(make_entry(
        level="unranked", level_reason="too recent", citation_count=None))
    wire(monkeypatch, make_meta(venue="NeurIPS", year=2015, citation_count=5000))
    refresh_entries(ctx)
    e = ctx.library.get("vaswani2017attention")
    assert e.level == "A*"
    assert "NeurIPS" in e.level_reason


def test_relevel_can_be_disabled(monkeypatch, ctx):
    ctx.library.save_entry(make_entry(level="B", level_reason="hand set"))
    wire(monkeypatch, make_meta(venue="NeurIPS", year=2015, citation_count=5000))
    refresh_entries(ctx, relevel=False)
    assert ctx.library.get("vaswani2017attention").level == "B"


def test_summaries_are_never_touched(monkeypatch, ctx):
    ctx.library.save_entry(make_entry())
    wire(monkeypatch, make_meta(citation_count=999))
    refresh_entries(ctx)
    e = ctx.library.get("vaswani2017attention")
    assert e.one_liner == "Introduces the Transformer, an attention-only sequence model."
    assert len(e.sections) == 2
    assert e.key_findings == ["28.4 BLEU on WMT 2014 EN-DE"]


def test_user_notes_are_never_touched(monkeypatch, ctx):
    ctx.library.save_entry(make_entry())
    entryfile.set_notes(ctx.library.entry_path("vaswani2017attention"), "MY NOTE")
    wire(monkeypatch, make_meta(citation_count=999))
    refresh_entries(ctx)
    assert ctx.library.get("vaswani2017attention").notes == "MY NOTE"


def test_unchanged_entry_is_not_rewritten(monkeypatch, ctx):
    ctx.library.save_entry(make_entry())
    wire(monkeypatch, make_meta())
    assert not refresh_entries(ctx)[0].updated


def test_unresolvable_entry_is_reported_not_fatal(monkeypatch, ctx):
    ctx.library.save_entry(make_entry())
    ctx.library.save_entry(make_entry(key="other", title="Other", arxiv_id="1234.56789"))
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else make_meta(citation_count=42)

    monkeypatch.setattr("lit.actions.refresh.resolve_metadata", flaky)
    results = refresh_entries(ctx)
    assert any(r.error for r in results)
    assert any(r.updated for r in results)


def test_specific_keys_only(monkeypatch, ctx):
    ctx.library.save_entry(make_entry())
    ctx.library.save_entry(make_entry(key="other", title="Other", arxiv_id="1234.56789"))
    wire(monkeypatch, make_meta(citation_count=999))
    assert [r.key for r in refresh_entries(ctx, ["other"])] == ["other"]


def test_empty_library_is_a_noop(ctx):
    assert refresh_entries(ctx) == []
