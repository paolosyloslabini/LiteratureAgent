"""Export/import round-trips and merge safety."""

import pytest
from factories import make_entry

from lit import bundle, entryfile
from lit.library import Library, LibraryError


def test_export_import_round_trip(lib, root, tmp_path):
    lib.save_entry(make_entry())
    lib.save_entry(make_entry(key="brown2020language", title="Language Models",
                              arxiv_id="2005.14165"))
    out = bundle.export_library(lib, tmp_path / "b.litlib")
    assert out.entries == 2
    assert out.path.exists()

    res = bundle.import_bundle(out.path, root=root, name="copy")
    assert sorted(res.added) == ["brown2020language", "vaswani2017attention"]

    copy = Library.open(root / "copy")
    assert copy.get("vaswani2017attention").title == "Attention Is All You Need"
    assert copy.settings.scope == lib.settings.scope


def test_export_preserves_summaries_and_references(lib, root, tmp_path):
    lib.save_entry(make_entry())
    out = bundle.export_library(lib, tmp_path / "b.litlib")
    bundle.import_bundle(out.path, root=root, name="copy")
    e = Library.open(root / "copy").get("vaswani2017attention")
    assert len(e.sections) == 2
    assert e.key_findings == ["28.4 BLEU on WMT 2014 EN-DE"]
    assert e.references[0].arxiv_id == "1409.0473"
    assert e.bibtex.startswith("@inproceedings")


def test_export_excludes_the_derived_index(lib, tmp_path):
    import zipfile

    from lit.store import Store

    lib.save_entry(make_entry())
    with Store(lib) as s:
        s.sync()
    out = bundle.export_library(lib, tmp_path / "b.litlib")
    with zipfile.ZipFile(out.path) as z:
        names = z.namelist()
    assert not any(".index.db" in n or n.startswith(".text") for n in names)


def test_export_selected_keys_only(lib, tmp_path):
    lib.save_entry(make_entry())
    lib.save_entry(make_entry(key="other", title="Other Paper", arxiv_id="1234.5678"))
    out = bundle.export_library(lib, tmp_path / "b.litlib", keys=["other"])
    assert out.entries == 1


def test_export_with_pdfs(lib, tmp_path):
    lib.save_entry(make_entry())
    (lib.pdfs_dir / "vaswani2017attention.pdf").write_bytes(b"%PDF-1.4 fake")
    out = bundle.export_library(lib, tmp_path / "b.litlib", with_pdfs=True)
    assert out.pdfs == 1


def test_peek_reads_the_manifest(lib, tmp_path):
    lib.save_entry(make_entry())
    out = bundle.export_library(lib, tmp_path / "b.litlib")
    m = bundle.peek(out.path)
    assert m["name"] == "testlib"
    assert m["entries"] == 1
    assert m["scope"] == lib.settings.scope


def test_peek_rejects_a_non_bundle(tmp_path):
    bad = tmp_path / "bad.litlib"
    bad.write_text("not a zip", encoding="utf-8")
    with pytest.raises(LibraryError):
        bundle.peek(bad)


# --------------------------------------------------------------------------
# Merge safety — importing must never cost you your own notes.
# --------------------------------------------------------------------------

def test_import_preserves_my_notes_and_appends_theirs(lib, root, tmp_path):
    theirs = Library.create(root, "theirs", "their scope")
    e = make_entry()
    e.updated = "2030-01-01"  # newer, so it would otherwise overwrite
    theirs.save_entry(e)
    entryfile.set_notes(theirs.entry_path(e.key), "THEIR NOTE")
    out = bundle.export_library(theirs, tmp_path / "b.litlib")

    lib.save_entry(make_entry())
    entryfile.set_notes(lib.entry_path("vaswani2017attention"), "MY NOTE")

    res = bundle.import_bundle(out.path, root=root, into=lib, strategy="replace")
    notes = lib.get("vaswani2017attention").notes
    assert "MY NOTE" in notes
    assert "THEIR NOTE" in notes
    assert "theirs" in notes  # attributed
    assert "vaswani2017attention" in res.notes_preserved


def test_import_skip_strategy_leaves_local_alone(lib, root, tmp_path):
    theirs = Library.create(root, "theirs", "s")
    theirs.save_entry(make_entry(one_liner="Their version."))
    out = bundle.export_library(theirs, tmp_path / "b.litlib")

    lib.save_entry(make_entry(one_liner="My version."))
    res = bundle.import_bundle(out.path, root=root, into=lib, strategy="skip")
    assert res.skipped == ["vaswani2017attention"]
    assert lib.get("vaswani2017attention").one_liner == "My version."


def test_import_never_replaces_a_read_entry_with_an_unread_one(lib, root, tmp_path):
    theirs = Library.create(root, "theirs", "s")
    theirs.save_entry(make_entry(status="UNVERIFIED", one_liner=None,
                                 sections=[], updated="2030-01-01"))
    out = bundle.export_library(theirs, tmp_path / "b.litlib")

    lib.save_entry(make_entry(one_liner="A real summary."))
    bundle.import_bundle(out.path, root=root, into=lib, strategy="newer")
    assert lib.get("vaswani2017attention").one_liner == "A real summary."


def test_import_key_collision_on_different_works_keeps_both(lib, root, tmp_path):
    theirs = Library.create(root, "theirs", "s")
    theirs.save_entry(make_entry(title="A Completely Different Paper",
                                 arxiv_id="9999.99999", doi=None))
    out = bundle.export_library(theirs, tmp_path / "b.litlib")

    lib.save_entry(make_entry())
    bundle.import_bundle(out.path, root=root, into=lib)
    assert len(lib) == 2


def test_import_rejects_unknown_strategy(lib, root, tmp_path):
    lib.save_entry(make_entry())
    out = bundle.export_library(lib, tmp_path / "b.litlib")
    with pytest.raises(LibraryError, match="strategy"):
        bundle.import_bundle(out.path, root=root, into=lib, strategy="nonsense")
