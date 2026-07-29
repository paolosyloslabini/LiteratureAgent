from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from factories import make_entry  # noqa: E402

from lit.config import Config  # noqa: E402
from lit.library import Library  # noqa: E402


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
