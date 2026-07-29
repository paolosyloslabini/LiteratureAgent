"""Export/import a library as a portable `.litlib` bundle, for trading with others.

A bundle is an ordinary zip containing `manifest.json`, `library.toml`, every
entry Markdown file, and optionally the PDFs. The SQLite index and extracted
text cache are never included — both are derived and rebuild themselves.

Because entries are plain Markdown, a library directory is also perfectly
shareable over git; bundles just make "send me your library" a single file.

Import never destroys your own work: on a key collision the incoming entry is
matched by DOI/arXiv/title, and your own `## Notes` section always survives —
incoming notes are appended under an attributed heading rather than replacing
yours.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import entryfile
from .library import LIBRARY_TOML, Library, LibraryError, normalize_name
from .models import Entry

BUNDLE_SUFFIX = ".litlib"
MANIFEST = "manifest.json"
FORMAT_VERSION = 1

# Merge strategies for entries that already exist locally.
STRATEGIES = ("skip", "replace", "newer")


@dataclass
class ExportResult:
    path: Path
    entries: int
    pdfs: int
    bytes: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path), "entries": self.entries,
            "pdfs": self.pdfs, "bytes": self.bytes,
        }


@dataclass
class ImportResult:
    library: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes_preserved: list[str] = field(default_factory=list)
    pdfs: int = 0
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "library": self.library,
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "notes_preserved": self.notes_preserved,
            "pdfs": self.pdfs,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export_library(lib: Library, dest: Path, *, with_pdfs: bool = False,
                   keys: list[str] | None = None) -> ExportResult:
    dest = Path(dest).expanduser()
    if dest.is_dir():
        dest = dest / f"{lib.name}{BUNDLE_SUFFIX}"
    if not dest.suffix:
        dest = dest.with_suffix(BUNDLE_SUFFIX)
    dest.parent.mkdir(parents=True, exist_ok=True)

    selected = keys or lib.keys()
    entries = [e for e in (lib.get(k) for k in selected) if e]
    n_pdfs = 0

    manifest = {
        "format": FORMAT_VERSION,
        "tool": "literature-agent",
        "name": lib.name,
        "scope": lib.settings.scope,
        "min_level": lib.settings.min_level,
        "allow_non_published": lib.settings.allow_non_published,
        "exported": date.today().isoformat(),
        "entries": len(entries),
        "includes_pdfs": bool(with_pdfs),
        "keys": [e.key for e in entries],
    }

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2))
        toml = lib.path / LIBRARY_TOML
        if toml.exists():
            z.write(toml, LIBRARY_TOML)
        for e in entries:
            p = lib.entry_path(e.key)
            if p.exists():
                z.write(p, f"entries/{e.key}.md")
            if with_pdfs:
                pdf = lib.pdfs_dir / f"{e.key}.pdf"
                if pdf.exists():
                    z.write(pdf, f"pdfs/{e.key}.pdf")
                    n_pdfs += 1

    return ExportResult(
        path=dest, entries=len(entries), pdfs=n_pdfs, bytes=dest.stat().st_size
    )


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------

def peek(path: Path) -> dict:
    """Read a bundle's manifest without unpacking it."""
    path = Path(path).expanduser()
    if not path.exists():
        raise LibraryError(f"no such bundle: {path}")
    try:
        with zipfile.ZipFile(path) as z:
            if MANIFEST in z.namelist():
                return json.loads(z.read(MANIFEST))
            keys = [
                Path(n).stem for n in z.namelist()
                if n.startswith("entries/") and n.endswith(".md")
            ]
            return {"format": 0, "name": path.stem, "entries": len(keys), "keys": keys}
    except zipfile.BadZipFile as exc:
        raise LibraryError(f"{path.name} is not a valid .litlib bundle") from exc


def import_bundle(
    path: Path,
    *,
    root: Path,
    into: Library | None = None,
    name: str | None = None,
    strategy: str = "newer",
    with_pdfs: bool = True,
) -> ImportResult:
    """Import a bundle into an existing library, or create a new one from it."""
    if strategy not in STRATEGIES:
        raise LibraryError(f"unknown merge strategy {strategy!r}; "
                           f"expected one of {', '.join(STRATEGIES)}")

    path = Path(path).expanduser()
    manifest = peek(path)
    lib = into

    if lib is None:
        target_name = normalize_name(name or manifest.get("name") or path.stem)
        target_path = root.expanduser() / target_name
        if (target_path / LIBRARY_TOML).exists():
            lib = Library.open(target_path)
        else:
            lib = Library.create(
                root.expanduser(), target_name,
                manifest.get("scope") or "",
                min_level=str(manifest.get("min_level") or "C"),
                allow_non_published=bool(manifest.get("allow_non_published", False)),
            )

    result = ImportResult(library=lib.name, source=manifest)
    lib.entries_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for name_in_zip in names:
            if not (name_in_zip.startswith("entries/") and name_in_zip.endswith(".md")):
                continue
            try:
                incoming = entryfile.loads(z.read(name_in_zip).decode("utf-8"))
            except Exception:
                continue
            if not incoming.key:
                incoming.key = Path(name_in_zip).stem

            outcome, key = _merge_entry(lib, incoming, strategy, manifest, result)
            if outcome == "added":
                result.added.append(key)
            elif outcome == "updated":
                result.updated.append(key)
            else:
                result.skipped.append(key)

            if with_pdfs and outcome != "skipped":
                src_pdf = f"pdfs/{Path(name_in_zip).stem}.pdf"
                if src_pdf in names:
                    lib.pdfs_dir.mkdir(parents=True, exist_ok=True)
                    (lib.pdfs_dir / f"{key}.pdf").write_bytes(z.read(src_pdf))
                    result.pdfs += 1

    return result


def _merge_entry(lib: Library, incoming: Entry, strategy: str,
                 manifest: dict, result: ImportResult) -> tuple[str, str]:
    """Place one incoming entry. Returns (outcome, final key)."""
    existing = lib.find_duplicate(
        doi=incoming.doi, arxiv_id=incoming.arxiv_id, title=incoming.title
    )
    if existing is None and lib.has(incoming.key):
        # Same key, different work — give the newcomer a fresh key.
        incoming.key = lib.unique_key(incoming.key)

    if existing is None:
        lib.save_entry(incoming)
        return "added", incoming.key

    if strategy == "skip":
        return "skipped", existing.key
    if strategy == "newer" and (existing.updated or "") >= (incoming.updated or ""):
        return "skipped", existing.key
    if strategy == "newer" and existing.is_verified and not incoming.is_verified:
        # Never trade a read entry for an unread one.
        return "skipped", existing.key

    incoming.key = existing.key
    incoming.added = existing.added

    # Your notes are yours. `Library.save_entry` re-reads them off disk and
    # restores them, so the incoming notes would simply be dropped; instead we
    # keep yours and append theirs, attributed.
    if incoming.notes.strip() and incoming.notes.strip() != existing.notes.strip():
        origin = manifest.get("name") or "imported library"
        mine = existing.notes.strip()
        block = f"### Note from {origin}\n\n{incoming.notes.strip()}"
        # Held in a local: `save_entry` restores the on-disk notes onto the
        # entry it is given (that is what protects user notes), so this string
        # must survive that call rather than ride in on `incoming`.
        merged = f"{mine}\n\n{block}".strip() if mine else block
        result.notes_preserved.append(existing.key)
        lib.save_entry(incoming)
        entryfile.set_notes(lib.entry_path(existing.key), merged)
    else:
        lib.save_entry(incoming)

    return "updated", existing.key
