"""Entry file round-trips, and the user-notes guarantee."""

from __future__ import annotations

import pytest

from factories import make_entry

from lit import entryfile
from lit.models import Entry, Section


def test_round_trip_preserves_fields(entry):
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert parsed.key == entry.key
    assert parsed.title == entry.title
    assert parsed.authors == entry.authors
    assert parsed.year == entry.year
    assert parsed.venue == entry.venue
    assert parsed.arxiv_id == entry.arxiv_id
    assert parsed.level == entry.level
    assert parsed.tags == entry.tags
    assert parsed.one_liner == entry.one_liner
    assert parsed.key_findings == entry.key_findings
    assert parsed.citation_count == entry.citation_count
    assert parsed.bibtex == entry.bibtex


def test_round_trip_preserves_sections(entry):
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert [s.name for s in parsed.sections] == ["Introduction", "Model Architecture"]
    assert "self-attention" in parsed.sections[1].summary


def test_round_trip_preserves_references(entry):
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert len(parsed.references) == 1
    assert parsed.references[0].arxiv_id == "1409.0473"
    assert parsed.references[0].year == 2014


def test_notes_round_trip(entry, tmp_path):
    entry.notes = "This is *my* note.\n\n- with bullets\n- and more"
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert parsed.notes == entry.notes


# --------------------------------------------------------------------------
# The core guarantee: an LLM-built Entry can never clobber user notes.
# --------------------------------------------------------------------------

def test_save_preserves_existing_notes(tmp_path):
    path = tmp_path / "e.md"
    original = make_entry()
    original.notes = "MY IRREPLACEABLE NOTES"
    entryfile.save(path, original)

    # Simulate a re-read: a fresh entry built from LLM output, notes empty, and
    # for good measure carrying hostile content in the notes field.
    incoming = make_entry()
    incoming.notes = "agent-written garbage"
    incoming.one_liner = "A new summary from a re-read."
    saved = entryfile.save(path, incoming)

    assert saved.notes == "MY IRREPLACEABLE NOTES"
    assert entryfile.load(path).notes == "MY IRREPLACEABLE NOTES"
    # ...while the machine-owned fields did update.
    assert entryfile.load(path).one_liner == "A new summary from a re-read."


def test_set_notes_is_the_only_way_to_change_notes(tmp_path):
    path = tmp_path / "e.md"
    entryfile.save(path, make_entry())
    entryfile.set_notes(path, "written by the user")
    assert entryfile.load(path).notes == "written by the user"

    entryfile.save(path, make_entry())  # a re-read must not undo it
    assert entryfile.load(path).notes == "written by the user"


def test_save_salvages_notes_from_a_corrupt_file(tmp_path):
    path = tmp_path / "e.md"
    path.write_text(
        "this file has no frontmatter at all\n"
        f"{entryfile.NOTES_MARKER} -->\n\n## Notes\n\nsalvage me\n",
        encoding="utf-8",
    )
    saved = entryfile.save(path, make_entry())
    assert saved.notes == "salvage me"
    assert path.with_suffix(".md.bak").exists()  # original kept


def test_unverified_entry_has_no_invented_summary():
    e = Entry(key="x", title="Unreachable Paper", status="UNVERIFIED")
    text = entryfile.dumps(e)
    parsed = entryfile.loads(text)
    assert parsed.one_liner is None
    assert parsed.sections == []
    assert "Not available" in text


def test_multiline_bibtex_survives(entry):
    entry.bibtex = "@inproceedings{k,\n  title = {A},\n  year = {2017}\n}"
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert parsed.bibtex == entry.bibtex


def test_missing_frontmatter_raises():
    with pytest.raises(ValueError):
        entryfile.loads("no frontmatter here")


def test_sections_without_headings_are_kept():
    e = Entry(key="x", title="T", one_liner="one", status="verified",
              sections=[Section("Summary", "a flat summary")])
    parsed = entryfile.loads(entryfile.dumps(e))
    assert parsed.sections[0].summary == "a flat summary"
