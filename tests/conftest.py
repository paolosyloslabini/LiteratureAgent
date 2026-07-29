from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from factories import make_entry  # noqa: E402

from lit.config import Config  # noqa: E402
from lit.library import Library  # noqa: E402


@pytest.fixture(autouse=True)
def never_call_the_real_claude(request, monkeypatch):
    """Fail loudly instead of shelling out to the user's `claude` CLI.

    Every LLM call in this tool is a subprocess, so a test that forgets to stub
    one does not fail — it quietly runs a real agent, taking minutes and real
    money and returning something different each time.

    The guard sits on `ClaudeCLI.run` rather than on `subprocess.run`, because
    `lit.llm.subprocess` is the stdlib module every other importer shares and
    patching its `run` would reach far beyond the LLM layer. Tests that mean to
    exercise the subprocess plumbing opt out with `@pytest.mark.real_cli_layer`
    and stub `subprocess.run` themselves.
    """
    if request.node.get_closest_marker("real_cli_layer"):
        return

    def refuse(*_a, **_kw):
        raise AssertionError(
            "a test reached the real `claude` CLI — stub ctx._llm, or mark the "
            "test @pytest.mark.real_cli_layer if the subprocess plumbing is "
            "what you mean to test."
        )
    monkeypatch.setattr("lit.llm.ClaudeCLI.run", refuse)


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "libs"
    d.mkdir()
    return d


@pytest.fixture
def cfg(root) -> Config:
    c = Config(root=str(root))
    c.fetch.email = ""
    return c


@pytest.fixture
def lib(root) -> Library:
    return Library.create(root, "testlib", "Testing the literature agent")


@pytest.fixture
def entry():
    return make_entry()
