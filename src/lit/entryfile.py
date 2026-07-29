"""Read/write entry files: Markdown body with YAML frontmatter.

Layout of an entry file:

    ---
    <machine-owned metadata: ids, level, tags, references, bibtex>
    ---

    ## One-line summary
    ## Key findings
    ## Detailed summary
    ### <section> ...

    <!-- lit:user-notes ... -->

    ## Notes
    <user-authored prose>

Everything above the ``lit:user-notes`` marker is regenerated from the `Entry`
on every save. Everything below it belongs to the user.

The user-notes guarantee is enforced structurally in `save()`: notes are always
re-read from the file on disk and copied onto the outgoing entry, so an `Entry`
built from LLM output physically cannot overwrite them. Only `set_notes()`
changes them, and only the `lit note` command calls it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import Entry, Reference, Section

NOTES_MARKER = "<!-- lit:user-notes"
NOTES_BANNER = (
    f"{NOTES_MARKER} — everything below this line belongs to you.\n"
    "     The agent never reads from or writes to this section. -->"
)

# Frontmatter key order; anything not listed is appended alphabetically.
_FM_ORDER = [
    "key", "title", "authors", "year", "venue", "type",
    "doi", "arxiv_id", "url",
    "status", "level", "level_reason",
    "tags", "citation_count",
    "read_source", "pdf_path", "text_chars", "pages", "pages_read",
    "added", "updated",
    "bibtex", "references",
]

# Fields that live in the Markdown body rather than the frontmatter.
_BODY_FIELDS = {"one_liner", "key_findings", "sections", "notes"}


class _BlockDumper(yaml.SafeDumper):
    """YAML dumper that renders multi-line strings as literal blocks."""


def _str_representer(dumper: yaml.SafeDumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _str_representer)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def dumps(entry: Entry) -> str:
    """Render an entry as the full text of its file."""
    data = entry.to_dict()
    for f in _BODY_FIELDS:
        data.pop(f, None)

    # Drop empties so files stay readable, but keep explicit falsey status flags.
    always = {"key", "title", "status", "level", "type"}
    data = {k: v for k, v in data.items() if k in always or v not in (None, "", [], {})}

    ordered: dict[str, Any] = {k: data.pop(k) for k in _FM_ORDER if k in data}
    ordered.update(dict(sorted(data.items())))

    fm = yaml.dump(
        ordered, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=100
    )

    parts = [f"---\n{fm}---\n"]

    if entry.one_liner:
        parts.append(f"\n## One-line summary\n\n{entry.one_liner.strip()}\n")

    if entry.key_findings:
        findings = "\n".join(f"- {f.strip()}" for f in entry.key_findings if f.strip())
        parts.append(f"\n## Key findings\n\n{findings}\n")

    if entry.sections:
        body = "\n\n".join(
            f"### {s.name.strip()}\n\n{s.summary.strip()}" for s in entry.sections
        )
        parts.append(f"\n## Detailed summary\n\n{body}\n")
    elif not entry.is_verified:
        parts.append(
            "\n## Detailed summary\n\n"
            "_Not available — the full text of this work could not be retrieved, "
            "so no summary was written. Drop a PDF in `pdfs/inbox/` and run "
            "`lit inbox` to fill this in._\n"
        )

    parts.append(f"\n{NOTES_BANNER}\n\n## Notes\n\n{entry.notes.strip()}\n")
    return "".join(parts)


def loads(text: str) -> Entry:
    """Parse the full text of an entry file back into an `Entry`."""
    fm_raw, body = _split_frontmatter(text)
    data = yaml.safe_load(fm_raw) or {}
    if not isinstance(data, dict):
        raise ValueError("entry frontmatter is not a mapping")

    machine_body, notes = _split_notes(body)

    entry = Entry(
        key=str(data.get("key") or ""),
        title=str(data.get("title") or ""),
    )
    for f in ("venue", "doi", "arxiv_id", "url", "level_reason", "read_source",
              "pdf_path", "bibtex", "added", "updated", "status", "level", "type"):
        if data.get(f) is not None:
            setattr(entry, f, data[f])

    entry.authors = [str(a) for a in (data.get("authors") or [])]
    entry.tags = [str(t) for t in (data.get("tags") or [])]
    entry.year = _as_int(data.get("year"))
    entry.citation_count = _as_int(data.get("citation_count"))
    entry.text_chars = _as_int(data.get("text_chars"))
    entry.pages = _as_int(data.get("pages"))
    entry.pages_read = _as_int(data.get("pages_read"))
    entry.references = [
        r for r in (Reference.coerce(x) for x in (data.get("references") or [])) if r
    ]

    entry.one_liner = _section_text(machine_body, "One-line summary") or None
    entry.key_findings = _bullets(_section_text(machine_body, "Key findings"))
    entry.sections = _parse_sections(_section_text(machine_body, "Detailed summary"))
    entry.notes = notes
    return entry


# --------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------

def load(path: Path) -> Entry:
    return loads(path.read_text(encoding="utf-8"))


def save(path: Path, entry: Entry) -> Entry:
    """Write `entry` to `path`, always preserving the user notes already on disk.

    Returns the entry as actually written (with on-disk notes restored).
    """
    if path.exists():
        try:
            entry.notes = loads(path.read_text(encoding="utf-8")).notes
        except Exception:
            # A malformed file must never cost the user their notes: keep the
            # old file around and salvage whatever sat below the marker.
            raw = path.read_text(encoding="utf-8", errors="replace")
            _, notes = _split_notes(raw)
            entry.notes = notes
            path.with_suffix(path.suffix + ".bak").write_text(raw, encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(entry), encoding="utf-8")
    tmp.replace(path)
    return entry


def set_notes(path: Path, notes: str) -> Entry:
    """The only sanctioned way to change user notes. Called by `lit note` only."""
    entry = load(path)
    entry.notes = notes
    path.write_text(dumps(entry), encoding="utf-8")
    return entry


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def _split_frontmatter(text: str) -> tuple[str, str]:
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        raise ValueError("entry file is missing YAML frontmatter")
    rest = text[3:].lstrip("\r\n")
    end = rest.find("\n---")
    if end == -1:
        raise ValueError("entry file frontmatter is not terminated")
    fm = rest[:end]
    body = rest[end + 4:]
    return fm, body.lstrip("\r\n")


def _split_notes(body: str) -> tuple[str, str]:
    """Split a body into (machine part, user notes)."""
    idx = body.find(NOTES_MARKER)
    if idx == -1:
        return body, ""
    machine = body[:idx]
    tail = body[idx:]
    # Skip past the marker comment itself, then past a leading "## Notes".
    close = tail.find("-->")
    tail = tail[close + 3:] if close != -1 else tail
    stripped = tail.lstrip()
    if stripped.lower().startswith("## notes"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return machine, stripped.strip()


def _section_text(body: str, heading: str) -> str:
    """Return the text under a given `## heading`, up to the next `## `."""
    lines = body.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith("## "):
            if capturing:
                break
            capturing = line[3:].strip().lower() == heading.lower()
            continue
        if capturing:
            out.append(line)
    return "\n".join(out).strip()


def _bullets(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")):
            out.append(line[2:].strip())
    return out


def _parse_sections(text: str) -> list[Section]:
    if not text.strip():
        return []
    sections: list[Section] = []
    name: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if name is not None:
                sections.append(Section(name=name, summary="\n".join(buf).strip()))
            name = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    if name is not None:
        sections.append(Section(name=name, summary="\n".join(buf).strip()))
    # A summary with no ### headings at all is still worth keeping.
    if not sections and text.strip() and not text.strip().startswith("_Not available"):
        sections.append(Section(name="Summary", summary=text.strip()))
    return sections


def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
