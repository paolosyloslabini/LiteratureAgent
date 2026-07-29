"""The Claude Code skill: valid frontmatter and the rules that matter."""

from __future__ import annotations

import yaml

from lit.skillfile import SKILL_BODY, install_skill


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
    for cmd in ["lit libs", "lit info", "lit ls", "lit show", "lit search",
                "lit ask", "lit add", "lit find", "lit cite", "lit claim",
                "lit export", "lit import", "lit inbox"]:
        assert cmd in SKILL_BODY


def test_skill_states_the_json_contract():
    assert "--json" in SKILL_BODY


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
