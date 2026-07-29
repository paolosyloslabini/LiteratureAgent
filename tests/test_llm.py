"""JSON salvage from model replies, and CLI invocation shape."""

import json

import pytest

from lit.config import DEFAULT_ROLE_MODELS, LLMConfig
from lit.llm import ClaudeCLI, LLMError, extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_markdown_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_fence_without_language():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_wrapped_in_prose():
    reply = 'Sure! Here is the record:\n\n{"a": 1, "b": [2, 3]}\n\nHope that helps.'
    assert extract_json(reply) == {"a": 1, "b": [2, 3]}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    reply = 'text {"quote": "he said {this} and \\"that\\"", "n": 2} tail'
    assert extract_json(reply)["n"] == 2


def test_nested_objects():
    data = {"sections": [{"name": "Intro", "summary": "x"}], "tags": ["a"]}
    assert extract_json(f"prose {json.dumps(data)} more") == data


def test_top_level_array_is_wrapped():
    assert extract_json("[1, 2]") == {"items": [1, 2]}


def test_empty_reply_raises():
    with pytest.raises(ValueError):
        extract_json("")


def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("I could not complete that request.")


def test_unterminated_object_raises():
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


# --------------------------------------------------------------------------
# Per-role model selection
# --------------------------------------------------------------------------

def test_roles_have_distinct_defaults():
    cfg = LLMConfig()
    assert cfg.model_for("scout") == "haiku"
    assert cfg.model_for("filter") == "haiku"
    assert cfg.model_for("reader") == "sonnet"
    assert cfg.model_for("analyst") == "sonnet"


def test_unknown_role_falls_back_to_the_default_model():
    assert LLMConfig(model="opus").model_for("nonexistent") == "opus"
    assert LLMConfig(model="opus").model_for(None) == "opus"


def test_override_wins_over_every_role():
    cfg = LLMConfig(override_model="opus")
    assert all(cfg.model_for(r) == "opus" for r in DEFAULT_ROLE_MODELS)


# --------------------------------------------------------------------------
# Subprocess invocation
# --------------------------------------------------------------------------

class FakeProc:
    def __init__(self, payload, returncode=0):
        self.stdout = json.dumps(payload) if isinstance(payload, dict) else payload
        self.stderr = ""
        self.returncode = returncode


# These exercise the subprocess plumbing itself, so they opt out of the
# conftest guard and stub `subprocess.run` directly.
pytestmark = pytest.mark.real_cli_layer


def _patch_run(monkeypatch, payload, capture, returncode=0):
    def fake_run(cmd, **kwargs):
        capture["cmd"] = cmd
        capture["input"] = kwargs.get("input")
        capture["cwd"] = kwargs.get("cwd")
        return FakeProc(payload, returncode)

    monkeypatch.setattr("lit.llm.subprocess.run", fake_run)
    monkeypatch.setattr("lit.llm.shutil.which", lambda _: "/usr/bin/claude")


def test_run_uses_the_role_model(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "ok", "total_cost_usd": 0.1}, cap)
    llm = ClaudeCLI(LLMConfig())
    res = llm.run("hi", role="scout")
    assert res.text == "ok"
    assert "haiku" in cap["cmd"]
    assert res.model == "haiku"


def test_toolless_calls_disallow_every_tool(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "{}"}, cap)
    ClaudeCLI(LLMConfig()).run("hi")
    assert "--disallowed-tools" in cap["cmd"]
    disallowed = cap["cmd"][cap["cmd"].index("--disallowed-tools") + 1]
    for tool in ("Bash", "Write", "Edit", "WebFetch"):
        assert tool in disallowed


def test_tool_calls_allowlist_only_what_was_asked(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "{}"}, cap)
    ClaudeCLI(LLMConfig()).run("hi", tools=["WebSearch", "WebFetch"], max_turns=5)
    assert cap["cmd"][cap["cmd"].index("--allowed-tools") + 1] == "WebSearch WebFetch"
    assert "--disallowed-tools" not in cap["cmd"]


def test_long_input_goes_over_stdin_not_argv(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "{}"}, cap)
    body = "x" * 500_000
    ClaudeCLI(LLMConfig()).run("summarize", stdin_text=body)
    assert cap["input"] == body
    assert all(len(part) < 10_000 for part in cap["cmd"])


def test_nonzero_exit_raises(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, "boom", cap, returncode=1)
    with pytest.raises(LLMError, match="exited 1"):
        ClaudeCLI(LLMConfig()).run("hi")


def test_error_payload_raises(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"is_error": True, "subtype": "x", "result": "nope"}, cap)
    with pytest.raises(LLMError):
        ClaudeCLI(LLMConfig()).run("hi")


def test_json_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        text = "sorry, no json" if calls["n"] == 1 else '{"one_liner": "x"}'
        return FakeProc({"result": text})

    monkeypatch.setattr("lit.llm.subprocess.run", fake_run)
    monkeypatch.setattr("lit.llm.shutil.which", lambda _: "/usr/bin/claude")
    data = ClaudeCLI(LLMConfig()).json("go", required=("one_liner",))
    assert data == {"one_liner": "x"}
    assert calls["n"] == 2


def test_json_gives_up_after_max_retries(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "never json"}, cap)
    cfg = LLMConfig(max_retries=1)
    with pytest.raises(LLMError, match="usable JSON"):
        ClaudeCLI(cfg).json("go")


def test_missing_required_key_is_retried(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": '{"other": 1}'}, cap)
    with pytest.raises(LLMError, match="missing required key"):
        ClaudeCLI(LLMConfig(max_retries=0)).json("go", required=("needed",))


def test_usage_totals_accumulate(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "ok", "total_cost_usd": 0.25}, cap)
    llm = ClaudeCLI(LLMConfig())
    llm.run("a", role="scout")
    llm.run("b", role="reader")
    assert llm.usage.calls == 2
    assert llm.usage.cost_usd == pytest.approx(0.5)
    assert "haiku" in llm.usage.summary() and "sonnet" in llm.usage.summary()


def test_missing_cli_is_reported_clearly(monkeypatch):
    monkeypatch.setattr("lit.llm.shutil.which", lambda _: None)
    llm = ClaudeCLI(LLMConfig())
    assert not llm.available
    with pytest.raises(LLMError, match="claude"):
        llm.run("hi")


def test_transient_cli_failure_is_retried(monkeypatch):
    """A reader call is minutes of work plus a PDF fetch; a session that dies on
    a stray tool_use must not throw that away."""
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeProc('{"is_error": true, "subtype": "error_max_turns"}')
        return FakeProc({"result": '{"one_liner": "recovered"}'})

    monkeypatch.setattr("lit.llm.subprocess.run", fake_run)
    monkeypatch.setattr("lit.llm.shutil.which", lambda _: "/usr/bin/claude")
    assert ClaudeCLI(LLMConfig()).json("go")["one_liner"] == "recovered"
    assert calls["n"] == 2


def test_persistent_cli_failure_still_raises(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, "boom", cap, returncode=1)
    with pytest.raises(LLMError, match="exited 1"):
        ClaudeCLI(LLMConfig(max_retries=1)).json("go")


def test_sealed_calls_get_turn_headroom(monkeypatch):
    cap = {}
    _patch_run(monkeypatch, {"result": "{}"}, cap)
    ClaudeCLI(LLMConfig()).run("hi")
    turns = int(cap["cmd"][cap["cmd"].index("--max-turns") + 1])
    assert turns >= 2
    assert "--disallowed-tools" in cap["cmd"]  # still no tool is reachable
