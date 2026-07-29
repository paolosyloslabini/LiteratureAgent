"""The Claude Code skill: valid frontmatter and the rules that matter."""

from __future__ import annotations

import re

import yaml

from lit.skillfile import SKILL_BODY, install_skill

# Declared on `@app.callback()`, so Typer only accepts them before the subcommand.
GLOBAL_FLAGS = ["--json", "-L", "--library", "-v", "--verbose", "-y", "--yes",
                "--model", "--effort", "--workers", "-j"]

POST_SUBCOMMAND = re.compile(
    r"^lit\s+(?!-)\S+.*?\s(" + "|".join(re.escape(f) for f in GLOBAL_FLAGS) + r")(?=\s|$)",
    re.MULTILINE,
)


def frontmatter() -> dict:
    _, fm, _ = SKILL_BODY.split("---", 2)
    return yaml.safe_load(fm)


def test_frontmatter_is_valid_yaml_with_required_fields():
    fm = frontmatter()
    assert fm["name"] == "literature"
    assert isinstance(fm["description"], str)
    assert len(fm["description"]) > 40


def test_description_carries_trigger_phrases():
    desc = frontmatter()["description"].lower()
    for phrase in ["my library", "find papers", "cite", "bibliography"]:
        assert phrase in desc


def test_skill_documents_every_verb_an_agent_needs():
    for verb in ["libs", "info", "ls", "show", "search", "ask", "add", "find",
                 "cite", "claim", "export", "import", "inbox"]:
        assert re.search(rf"lit (--json )?{verb}\b", SKILL_BODY), verb


def test_skill_states_the_json_contract():
    assert "--json" in SKILL_BODY


def test_skill_writes_global_flags_before_the_subcommand():
    """`lit ls --json` exits 2 with "No such option"; only `lit --json ls` runs."""
    bad = [m.group(0).strip() for m in POST_SUBCOMMAND.finditer(SKILL_BODY)]
    assert not bad, f"global flag written after the subcommand: {bad}"
    assert "go before the subcommand" in SKILL_BODY


def test_skill_protects_user_notes():
    assert "Never write to `## Notes`" in SKILL_BODY
    assert "lit note" in SKILL_BODY


def test_skill_forbids_inventing_citations():
    assert "Never invent a citation" in SKILL_BODY


def test_skill_explains_unverified():
    assert "UNVERIFIED" in SKILL_BODY
    assert "lit inbox" in SKILL_BODY


def test_install_writes_the_skill(tmp_path):
    dest = install_skill(root=tmp_path)
    assert dest == tmp_path / "literature" / "SKILL.md"
    assert dest.read_text(encoding="utf-8") == SKILL_BODY


def test_install_is_idempotent(tmp_path):
    install_skill(root=tmp_path)
    dest = install_skill(root=tmp_path)
    assert dest.exists()


def test_skill_documents_refresh():
    assert "lit refresh" in SKILL_BODY
    assert "lit reread" in SKILL_BODY


def test_skill_explains_partial_reads():
    assert "partial_read" in SKILL_BODY
    assert "sampled" in SKILL_BODY


def test_skill_does_not_claim_a_pypi_install():
    """The package is not on PyPI; a bare `pip install literature-agent` fails."""
    assert "pipx install literature-agent" not in SKILL_BODY
    assert "uv tool install literature-agent" not in SKILL_BODY
    assert "git+https://github.com/" in SKILL_BODY
