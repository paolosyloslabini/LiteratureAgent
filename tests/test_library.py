"""Library lifecycle, resolution and duplicate detection."""

import pytest

from lit.library import Library, LibraryError, list_libraries, normalize_name, resolve_library


def test_create_makes_the_expected_layout(root):
    lib = Library.create(root, "My Library!", "A topic")
    assert lib.name == "my-library"
    assert (lib.path / "library.toml").exists()
    assert lib.entries_dir.is_dir()
    assert lib.inbox_dir.is_dir()


def test_create_twice_fails(root):
    Library.create(root, "dup", "x")
    with pytest.raises(LibraryError):
        Library.create(root, "dup", "x")


def test_settings_round_trip(root):
    lib = Library.create(root, "l", "the scope", min_level="A",
                         allow_non_published=True)
    reopened = Library.open(lib.path)
    assert reopened.settings.scope == "the scope"
    assert reopened.settings.min_level == "A"
    assert reopened.settings.allow_non_published is True


def test_default_min_level_is_permissive(root):
    assert Library.create(root, "l", "x").settings.min_level == "C"


def test_open_missing_library_fails(tmp_path):
    with pytest.raises(LibraryError):
        Library.open(tmp_path / "nope")


@pytest.mark.parametrize("raw,expected", [
    ("LLM Benchmarks", "llm-benchmarks"),
    ("  spaced  out  ", "spaced-out"),
    ("already-fine", "already-fine"),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_name_rejects_empty():
    with pytest.raises(LibraryError):
        normalize_name("  ")


def test_save_and_get_entry(lib, entry):
    lib.save_entry(entry)
    got = lib.get(entry.key)
    assert got.title == entry.title
    assert lib.has(entry.key)
    assert len(lib) == 1


def test_delete_entry(lib, entry):
    lib.save_entry(entry)
    assert lib.delete_entry(entry.key)
    assert not lib.delete_entry(entry.key)


def test_unique_key_disambiguates(lib, entry):
    lib.save_entry(entry)
    assert lib.unique_key(entry.key) == entry.key + "a"


def test_find_duplicate_by_arxiv(lib, entry):
    lib.save_entry(entry)
    assert lib.find_duplicate(arxiv_id="1706.03762").key == entry.key


def test_find_duplicate_by_doi(lib, entry):
    entry.doi = "10.1234/abcd"
    lib.save_entry(entry)
    assert lib.find_duplicate(doi="10.1234/ABCD").key == entry.key


def test_find_duplicate_by_title_ignores_punctuation(lib, entry):
    lib.save_entry(entry)
    assert lib.find_duplicate(title="attention is all you need!").key == entry.key


def test_find_duplicate_returns_none_for_new_work(lib, entry):
    lib.save_entry(entry)
    assert lib.find_duplicate(title="Something Else Entirely") is None


def test_iter_entries_skips_broken_files(lib, entry):
    lib.save_entry(entry)
    (lib.entries_dir / "broken.md").write_text("not an entry", encoding="utf-8")
    assert [e.key for e in lib.iter_entries()] == [entry.key]


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def test_resolve_by_explicit_name(cfg, root):
    Library.create(root, "alpha", "x")
    assert resolve_library(cfg, "alpha").name == "alpha"


def test_resolve_unknown_name_explains_how_to_create(cfg):
    with pytest.raises(LibraryError, match="lit new"):
        resolve_library(cfg, "ghost")


def test_resolve_falls_back_to_the_only_library(cfg, root):
    Library.create(root, "solo", "x")
    assert resolve_library(cfg).name == "solo"


def test_resolve_uses_configured_default(cfg, root):
    Library.create(root, "alpha", "x")
    Library.create(root, "beta", "x")
    cfg.default_library = "beta"
    assert resolve_library(cfg).name == "beta"


def test_resolve_is_ambiguous_with_several_libraries(cfg, root, monkeypatch, tmp_path):
    Library.create(root, "alpha", "x")
    Library.create(root, "beta", "x")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LibraryError, match="several libraries"):
        resolve_library(cfg)


def test_resolve_finds_library_in_cwd(cfg, root, monkeypatch):
    Library.create(root, "alpha", "x")
    lib = Library.create(root, "beta", "x")
    monkeypatch.chdir(lib.path)
    assert resolve_library(cfg).name == "beta"


def test_list_libraries(cfg, root):
    Library.create(root, "alpha", "x")
    Library.create(root, "beta", "y")
    assert [l.name for l in list_libraries(cfg)] == ["alpha", "beta"]


def test_list_libraries_ignores_stray_directories(cfg, root):
    Library.create(root, "alpha", "x")
    (root / "not-a-library").mkdir()
    assert [l.name for l in list_libraries(cfg)] == ["alpha"]
