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


def test_round_trip_preserves_the_code_url(entry):
    entry.code_url = "https://github.com/google/flax"
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert parsed.code_url == "https://github.com/google/flax"


def test_round_trip_preserves_a_multi_paragraph_abstract(entry):
    entry.abstract = "First paragraph.\n\nSecond paragraph, with a ## in it."
    text = entryfile.dumps(entry)
    assert "## Abstract" in text
    parsed = entryfile.loads(text)
    assert parsed.abstract == entry.abstract


def test_an_absent_abstract_writes_no_heading(entry):
    entry.abstract = ""
    assert "## Abstract" not in entryfile.dumps(entry)
    assert entryfile.loads(entryfile.dumps(entry)).abstract == ""


def test_the_abstract_does_not_leak_into_the_detailed_summary(entry):
    entry.abstract = "ABSTRACTTOKEN"
    parsed = entryfile.loads(entryfile.dumps(entry))
    assert "ABSTRACTTOKEN" not in parsed.detailed_summary_md()
    assert "ABSTRACTTOKEN" not in (parsed.one_liner or "")
    assert parsed.key_findings == entry.key_findings


def test_a_multi_line_key_finding_survives_a_save_round_trip(entry, tmp_path):
    # A finding is written as one `- ` bullet and read back the same way, so a
    # newline inside one used to drop the caveat after it — and the next save
    # wrote the truncated version back to disk.
    path = tmp_path / "e.md"
    entry.key_findings = ["Accuracy rose 3%.\nOnly on ImageNet."]
    entryfile.save(path, entry)
    assert entryfile.load(path).key_findings == ["Accuracy rose 3%. Only on ImageNet."]


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
