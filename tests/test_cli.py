"""CLI smoke tests: commands run, --json is parseable, exit codes are sane."""

from __future__ import annotations

import json

import pytest
from factories import make_entry
from typer.testing import CliRunner

from lit.cli import app, state
from lit.library import Library

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point the CLI at a throwaway root and reset its module-level state."""
    root = tmp_path / "libs"
    root.mkdir()
    monkeypatch.setenv("LIT_ROOT", str(root))
    monkeypatch.setenv("LIT_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    state._cfg = None
    state.library_name = None
    state.json_mode = False
    state.model = None
    state.parallel = None
    yield root
    state._cfg = None


def run(*args):
    return runner.invoke(app, list(args))


def js(result):
    return json.loads(result.stdout)


def test_new_creates_a_library(isolated):
    r = run("new", "mylib", "--scope", "A topic")
    assert r.exit_code == 0
    assert (isolated / "mylib" / "library.toml").exists()


def test_new_json(isolated):
    r = run("--json", "new", "mylib", "--scope", "A topic")
    assert js(r)["created"] == "mylib"


def test_new_rejects_a_bad_min_level(isolated):
    assert run("new", "l", "--scope", "s", "--min-level", "Z").exit_code != 0


def test_libs_lists_nothing_gracefully(isolated):
    assert run("--json", "libs").exit_code == 0
    assert js(run("--json", "libs"))["libraries"] == []


def test_libs_lists_created_libraries(isolated):
    run("new", "alpha", "--scope", "a")
    run("new", "beta", "--scope", "b")
    names = [l["name"] for l in js(run("--json", "libs"))["libraries"]]
    assert names == ["alpha", "beta"]


def test_commands_fail_clearly_without_a_library(isolated):
    r = run("--json", "ls")
    assert r.exit_code != 0
    assert "error" in js(r)


@pytest.fixture
def stocked(isolated):
    r = run("new", "mylib", "--scope", "Transformers")
    assert r.exit_code == 0
    lib = Library.open(isolated / "mylib")
    lib.save_entry(make_entry())
    lib.save_entry(make_entry(
        key="doe2010obscure", title="An Obscure Study", authors=["Jane Doe"],
        year=2010, venue="Widget Weekly", level="C", arxiv_id=None,
        one_liner="Widgets.", tags=["widgets"], citation_count=3,
        status="UNVERIFIED", sections=[], references=[], bibtex=None,
    ))
    return lib


def test_ls(stocked):
    data = js(run("--json", "ls"))
    assert data["count"] == 2
    assert {e["key"] for e in data["entries"]} == {"vaswani2017attention", "doe2010obscure"}


def test_ls_filters_by_level(stocked):
    data = js(run("--json", "ls", "--level", "A"))
    assert [e["key"] for e in data["entries"]] == ["vaswani2017attention"]


def test_ls_filters_by_tag(stocked):
    data = js(run("--json", "ls", "--tag", "widgets"))
    assert [e["key"] for e in data["entries"]] == ["doe2010obscure"]


def test_ls_filters_by_status(stocked):
    data = js(run("--json", "ls", "--status", "UNVERIFIED"))
    assert [e["key"] for e in data["entries"]] == ["doe2010obscure"]


def test_show(stocked):
    data = js(run("--json", "show", "vaswani2017attention"))
    assert data["title"] == "Attention Is All You Need"
    assert data["sections"][0]["name"] == "Introduction"
    assert data["references"][0]["arxiv_id"] == "1409.0473"


def test_show_reports_the_abstract_and_code_url(stocked):
    stocked.save_entry(make_entry(
        abstract="The publisher's abstract.",
        code_url="https://github.com/google/flax"))
    data = js(run("--json", "show", "vaswani2017attention"))
    assert data["abstract"] == "The publisher's abstract."
    assert data["code_url"] == "https://github.com/google/flax"


def test_abstract_command_prints_the_abstract(stocked):
    stocked.save_entry(make_entry(abstract="The publisher's abstract."))
    r = run("abstract", "vaswani2017attention")
    assert r.exit_code == 0
    assert r.stdout.strip() == "The publisher's abstract."


def test_abstract_command_labels_multiple_entries(stocked):
    stocked.save_entry(make_entry(abstract="First abstract."))
    out = run("abstract", "vaswani2017attention", "doe2010obscure").stdout
    assert "First abstract." in out
    assert "vaswani2017attention" in out and "doe2010obscure" in out


def test_abstract_command_says_so_when_there_is_none(stocked):
    assert "lit refresh" in run("abstract", "doe2010obscure").stdout


def test_abstract_command_json(stocked):
    stocked.save_entry(make_entry(abstract="The publisher's abstract."))
    data = js(run("--json", "abstract", "vaswani2017attention"))
    assert data[0]["abstract"] == "The publisher's abstract."


def test_abstract_command_missing_key_fails(stocked):
    assert run("abstract", "nope").exit_code != 0


def test_show_missing_key_fails(stocked):
    assert run("--json", "show", "nope").exit_code != 0


def test_show_raw_prints_the_file(stocked):
    out = run("show", "vaswani2017attention", "--raw").stdout
    assert "---" in out and "## One-line summary" in out


def test_info(stocked):
    data = js(run("--json", "info"))
    assert data["entries"] == 2
    assert data["scope"] == "Transformers"
    assert data["by_status"]["UNVERIFIED"] == 1


def test_note_writes_and_show_reads_back(stocked):
    assert run("note", "vaswani2017attention", "my thought").exit_code == 0
    assert js(run("--json", "show", "vaswani2017attention"))["notes"] == "my thought"


def test_note_appends_by_default(stocked):
    run("note", "vaswani2017attention", "first")
    run("note", "vaswani2017attention", "second")
    notes = js(run("--json", "show", "vaswani2017attention"))["notes"]
    assert "first" in notes and "second" in notes


def test_note_replace(stocked):
    run("note", "vaswani2017attention", "first")
    run("note", "vaswani2017attention", "second", "--replace")
    assert js(run("--json", "show", "vaswani2017attention"))["notes"] == "second"


def test_notes_survive_reindex(stocked):
    run("note", "vaswani2017attention", "durable")
    run("reindex", "--rebuild")
    assert js(run("--json", "show", "vaswani2017attention"))["notes"] == "durable"


def test_cite_bibtex(stocked):
    out = run("cite", "vaswani2017attention").stdout
    assert out.strip().startswith("@inproceedings")


def test_cite_markdown(stocked):
    out = run("cite", "--format", "markdown").stdout
    assert "**[vaswani2017attention]**" in out


def test_cite_keys(stocked):
    assert set(run("cite", "--format", "keys").stdout.split()) == {
        "vaswani2017attention", "doe2010obscure"}


def test_cite_filtered_by_level(stocked):
    assert run("cite", "--level", "A", "--format", "keys").stdout.strip() == \
        "vaswani2017attention"


def test_cite_unknown_key_fails(stocked):
    assert run("cite", "ghost").exit_code != 0


def test_cite_to_file(stocked, tmp_path):
    out = tmp_path / "refs.bib"
    run("cite", "--format", "bibtex", "-o", str(out))
    assert "@inproceedings" in out.read_text()


def test_rm(stocked):
    assert run("--json", "rm", "doe2010obscure", "-y").exit_code == 0
    assert js(run("--json", "ls"))["count"] == 1


def test_reindex(stocked):
    assert js(run("--json", "reindex"))["reindexed"] == 2
    data = js(run("--json", "reindex", "--rebuild"))
    assert data["entries"] == 2


def test_search_raw_needs_no_llm(stocked):
    data = js(run("--json", "search", "attention transformer", "--raw"))
    assert data["matches"][0]["key"] == "vaswani2017attention"


def test_search_question_does_not_break_fts(stocked):
    r = run("--json", "search", "What is attention, exactly? (2017)", "--raw")
    assert r.exit_code == 0


def test_path(stocked):
    assert run("path").stdout.strip().endswith("mylib")
    assert run("path", "vaswani2017attention").stdout.strip().endswith(
        "vaswani2017attention.md")


def test_export_import_round_trip(stocked, tmp_path, isolated):
    out = tmp_path / "b.litlib"
    assert run("export", "-o", str(out)).exit_code == 0
    assert out.exists()
    r = run("--json", "import", str(out), "--name", "copy")
    assert len(js(r)["added"]) == 2
    assert (isolated / "copy" / "library.toml").exists()


def test_import_show_manifest(stocked, tmp_path):
    out = tmp_path / "b.litlib"
    run("export", "-o", str(out))
    data = js(run("--json", "import", str(out), "--show"))
    assert data["entries"] == 2
    assert data["name"] == "mylib"


def test_use_sets_the_default(stocked, isolated):
    run("new", "other", "--scope", "x", "--no-use")
    run("use", "other")
    assert js(run("--json", "libs"))["default"] == "other"


def test_library_flag_targets_a_specific_library(stocked, isolated):
    run("new", "empty", "--scope", "x", "--no-use")
    assert js(run("--json", "-L", "empty", "ls"))["count"] == 0
    assert js(run("--json", "-L", "mylib", "ls"))["count"] == 2


def test_config_show(isolated):
    data = js(run("--json", "config", "show"))
    assert data["llm"]["models"]["reader"] == "sonnet"
    assert data["llm"]["models"]["scout"] == "haiku"


def test_config_set_role_model(isolated):
    run("config", "set", "llm.models.reader", "opus")
    assert js(run("--json", "config", "show"))["llm"]["models"]["reader"] == "opus"


def test_config_set_nested_value(isolated):
    run("config", "set", "fetch.email", "me@example.org")
    assert js(run("--json", "config", "show"))["fetch"]["email"] == "me@example.org"


def test_config_set_library_setting(stocked):
    run("config", "set", "--library", "min_level", "A")
    assert js(run("--json", "info"))["min_level"] == "A"


def test_config_rejects_unknown_key(isolated):
    assert run("config", "set", "nope.nope", "x").exit_code != 0


def test_skill_install(isolated, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = run("--json", "skill", "install", "--project")
    assert r.exit_code == 0
    dest = tmp_path / ".claude" / "skills" / "literature" / "SKILL.md"
    assert dest.exists()
    assert "name: literature" in dest.read_text()


def test_skill_show(isolated):
    assert "lit ask" in run("skill", "show").stdout


def test_help_lists_every_verb():
    out = runner.invoke(app, ["--help"]).stdout
    for verb in ["new", "add", "find", "search", "ask", "claim", "cite",
                 "export", "import", "skill", "browse", "note", "inbox"]:
        assert verb in out


def test_refresh_with_no_arguments(stocked, monkeypatch):
    """A variadic Argument arrives as None when omitted, not []."""
    monkeypatch.setattr("lit.actions.refresh.resolve_metadata", lambda *a, **k: None)
    r = run("--json", "refresh")
    assert r.exit_code == 0
    assert len(js(r)["refreshed"]) == 2


def test_refresh_specific_key(stocked, monkeypatch):
    monkeypatch.setattr("lit.actions.refresh.resolve_metadata", lambda *a, **k: None)
    data = js(run("--json", "refresh", "vaswani2017attention"))
    assert [r["key"] for r in data["refreshed"]] == ["vaswani2017attention"]


def test_cite_with_no_arguments(stocked):
    assert run("cite", "--format", "keys").exit_code == 0


def test_config_show_includes_the_fallback_resolver(isolated):
    fetch = js(run("--json", "config", "show"))["fetch"]
    assert fetch["fallback_url_template"] == ""
    assert fetch["fallback_cmd"] == ""


def test_config_set_fallback_round_trips(isolated):
    run("config", "set", "fetch.fallback_url_template", "https://example.org/{doi}")
    fetch = js(run("--json", "config", "show"))["fetch"]
    assert fetch["fallback_url_template"] == "https://example.org/{doi}"
