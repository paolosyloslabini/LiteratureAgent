"""A library: a directory holding a scope, entries, PDFs and a derived index."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

import tomli_w

from . import entryfile
from .config import Config
from .models import Entry, slugify

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 path
    import tomli as tomllib  # type: ignore[no-redef]

LIBRARY_TOML = "library.toml"


class LibraryError(Exception):
    """Raised for user-facing library problems (missing library, bad name...)."""


@dataclass
class LibrarySettings:
    name: str = ""
    # The topic this library investigates. Steers relevance and quality checks.
    scope: str = ""
    created: str = field(default_factory=lambda: date.today().isoformat())
    # Entries below this level are rejected by `add` unless --force is passed.
    # Deliberately permissive by default: missing a paper that matters costs
    # more than carrying one that doesn't, and the level is recorded on every
    # entry anyway, so you can always filter at read time with `ls --level A`.
    # Tighten with `lit config set --library min_level A`.
    min_level: str = "C"
    # Allow videos, lecture notes, blog posts and other non-published material.
    allow_non_published: bool = False
    # When false, `find` asks before writing each entry.
    auto_approve: bool = True


@dataclass
class Library:
    path: Path
    settings: LibrarySettings

    # ---------------- paths ----------------

    @property
    def name(self) -> str:
        return self.settings.name or self.path.name

    @property
    def entries_dir(self) -> Path:
        return self.path / "entries"

    @property
    def pdfs_dir(self) -> Path:
        return self.path / "pdfs"

    @property
    def inbox_dir(self) -> Path:
        return self.pdfs_dir / "inbox"

    @property
    def text_dir(self) -> Path:
        """Cached extracted plain text, so re-reads don't re-parse PDFs."""
        return self.path / ".text"

    @property
    def index_path(self) -> Path:
        return self.path / ".index.db"

    def entry_path(self, key: str) -> Path:
        return self.entries_dir / f"{key}.md"

    # ---------------- lifecycle ----------------

    @classmethod
    def create(cls, root: Path, name: str, scope: str, **kwargs) -> "Library":
        name = normalize_name(name)
        path = root.expanduser() / name
        if (path / LIBRARY_TOML).exists():
            raise LibraryError(f"library '{name}' already exists at {path}")
        settings = LibrarySettings(name=name, scope=scope, **kwargs)
        lib = cls(path=path, settings=settings)
        for d in (lib.entries_dir, lib.pdfs_dir, lib.inbox_dir, lib.text_dir):
            d.mkdir(parents=True, exist_ok=True)
        (lib.text_dir / ".gitignore").write_text("*\n", encoding="utf-8")
        lib.save_settings()
        return lib

    @classmethod
    def open(cls, path: Path) -> "Library":
        path = path.expanduser()
        toml = path / LIBRARY_TOML
        if not toml.exists():
            raise LibraryError(f"no library at {path} (missing {LIBRARY_TOML})")
        raw = tomllib.loads(toml.read_text(encoding="utf-8"))
        lib_raw = raw.get("library", raw)
        settings = LibrarySettings(
            name=str(lib_raw.get("name") or path.name),
            scope=str(lib_raw.get("scope") or ""),
            created=str(lib_raw.get("created") or ""),
            min_level=str(lib_raw.get("min_level") or "C"),
            allow_non_published=bool(lib_raw.get("allow_non_published", False)),
            auto_approve=bool(lib_raw.get("auto_approve", True)),
        )
        lib = cls(path=path, settings=settings)
        lib.entries_dir.mkdir(parents=True, exist_ok=True)
        return lib

    def save_settings(self) -> None:
        data = {"library": {
            "name": self.settings.name,
            "scope": self.settings.scope,
            "created": self.settings.created,
            "min_level": self.settings.min_level,
            "allow_non_published": self.settings.allow_non_published,
            "auto_approve": self.settings.auto_approve,
        }}
        (self.path / LIBRARY_TOML).write_text(tomli_w.dumps(data), encoding="utf-8")

    # ---------------- entries ----------------

    def keys(self) -> list[str]:
        if not self.entries_dir.exists():
            return []
        return sorted(p.stem for p in self.entries_dir.glob("*.md"))

    def __len__(self) -> int:
        return len(self.keys())

    def iter_entries(self) -> Iterator[Entry]:
        for p in sorted(self.entries_dir.glob("*.md")):
            try:
                yield entryfile.load(p)
            except Exception:
                continue  # a broken file shouldn't break the whole listing

    def entries(self) -> list[Entry]:
        return list(self.iter_entries())

    def has(self, key: str) -> bool:
        return self.entry_path(key).exists()

    def get(self, key: str) -> Entry | None:
        p = self.entry_path(key)
        return entryfile.load(p) if p.exists() else None

    def save_entry(self, entry: Entry) -> Entry:
        entry.updated = date.today().isoformat()
        return entryfile.save(self.entry_path(entry.key), entry)

    def delete_entry(self, key: str) -> bool:
        """Remove an entry and everything downloaded to produce it.

        The Markdown file is the entry; the PDF and the cached text are only
        what was fetched to write it, so they go with it. The search index is a
        derived cache and drops the row on its next sync. One path, so `lit
        delete` and the browser cannot leave different debris behind.
        """
        p = self.entry_path(key)
        if not p.exists():
            return False
        p.unlink()
        (self.pdfs_dir / f"{key}.pdf").unlink(missing_ok=True)
        (self.text_dir / f"{key}.txt").unlink(missing_ok=True)
        return True

    def unique_key(self, base: str) -> str:
        """Return `base`, or base+a/b/c... if that key is already taken."""
        if not self.has(base):
            return base
        for suffix in "abcdefghijklmnopqrstuvwxyz":
            cand = f"{base}{suffix}"
            if not self.has(cand):
                return cand
        n = 2
        while self.has(f"{base}{n}"):
            n += 1
        return f"{base}{n}"

    def find_duplicate(self, *, doi=None, arxiv_id=None, title=None) -> Entry | None:
        """Locate an existing entry for the same work, by id then by title."""
        norm_title = slugify(title or "", 200)
        for e in self.iter_entries():
            if doi and e.doi and e.doi.lower() == doi.lower():
                return e
            if arxiv_id and e.arxiv_id and e.arxiv_id == arxiv_id:
                return e
            if norm_title and slugify(e.title, 200) == norm_title:
                return e
        return None


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "").strip()).strip("-").lower()
    if not slug:
        raise LibraryError("library name cannot be empty")
    return slug


def list_libraries(cfg: Config) -> list[Library]:
    root = cfg.root_path
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if (child / LIBRARY_TOML).exists():
            try:
                out.append(Library.open(child))
            except LibraryError:
                continue
    return out


def resolve_library(cfg: Config, name: str | None = None) -> Library:
    """Find the library to act on.

    Order: explicit name/path -> LIT_LIBRARY / config default -> a library.toml
    in the current directory or an ancestor -> the only library that exists.
    """
    if name:
        p = Path(name).expanduser()
        if (p / LIBRARY_TOML).exists():
            return Library.open(p)
        cand = cfg.root_path / normalize_name(name)
        if (cand / LIBRARY_TOML).exists():
            return Library.open(cand)
        raise LibraryError(
            f"no library named '{name}' in {cfg.root_path}. "
            f"Create it with: lit new {normalize_name(name)} --scope \"...\""
        )

    if cfg.default_library:
        cand = cfg.root_path / normalize_name(cfg.default_library)
        if (cand / LIBRARY_TOML).exists():
            return Library.open(cand)

    here = Path.cwd()
    for d in [here, *here.parents]:
        if (d / LIBRARY_TOML).exists():
            return Library.open(d)

    libs = list_libraries(cfg)
    if len(libs) == 1:
        return libs[0]
    if not libs:
        raise LibraryError(
            f"no libraries found in {cfg.root_path}. "
            'Create one with: lit new <name> --scope "<topic>"'
        )
    names = ", ".join(l.name for l in libs)
    raise LibraryError(f"several libraries exist ({names}); pick one with -L <name>")
