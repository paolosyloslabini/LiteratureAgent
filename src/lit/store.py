"""Derived SQLite/FTS5 index over a library.

The Markdown entry files are the source of truth; this index is a cache that can
be deleted and rebuilt at any time with `lit reindex`. It exists so search stays
fast and ranked as a library grows.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .library import Library
from .models import Entry, level_rank

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS entries(
    key            TEXT PRIMARY KEY,
    title          TEXT,
    authors        TEXT,
    year           INTEGER,
    venue          TEXT,
    type           TEXT,
    doi            TEXT,
    arxiv_id       TEXT,
    url            TEXT,
    status         TEXT,
    level          TEXT,
    level_rank     INTEGER,
    tags           TEXT,
    one_liner      TEXT,
    citation_count INTEGER,
    updated        TEXT,
    mtime          REAL
);
CREATE INDEX IF NOT EXISTS idx_entries_level ON entries(level_rank);
CREATE INDEX IF NOT EXISTS idx_entries_year  ON entries(year);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    key UNINDEXED,
    title,
    authors,
    one_liner,
    abstract,
    tags,
    summary,
    findings,
    notes,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS refs(
    citing      TEXT,
    cited_title TEXT,
    cited_doi   TEXT,
    cited_arxiv TEXT,
    cited_year  INTEGER,
    cited_key   TEXT
);
CREATE INDEX IF NOT EXISTS idx_refs_citing ON refs(citing);
CREATE INDEX IF NOT EXISTS idx_refs_doi    ON refs(cited_doi);
"""

# bm25 column weights, in declared FTS column order. Title and one-liner carry
# the most signal; notes are the user's own words so they matter a lot too. The
# abstract and the detailed summary are long and repeat the rest, so they rank
# low — they are there to catch the query nothing else matches.
_BM25_WEIGHTS = (0.0, 8.0, 2.0, 6.0, 2.0, 4.0, 1.0, 3.0, 5.0)


@dataclass
class SearchHit:
    key: str
    title: str
    score: float
    entry: Entry | None = None


class Store:
    def __init__(self, library: Library):
        self.library = library
        self.path = library.index_path
        self._conn: sqlite3.Connection | None = None

    # ---------------- connection ----------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            cur = self._conn.execute("SELECT v FROM meta WHERE k='schema_version'")
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(k,v) VALUES('schema_version',?)", (str(SCHEMA_VERSION),)
                )
                self._conn.commit()
            elif int(row["v"]) != SCHEMA_VERSION:
                self._conn.close()
                self.path.unlink(missing_ok=True)
                self._conn = None
                return self.conn  # rebuild from scratch on schema change
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- indexing ----------------

    def sync(self, force: bool = False) -> int:
        """Bring the index in line with the entry files. Returns rows touched."""
        conn = self.conn
        known = {
            r["key"]: r["mtime"]
            for r in conn.execute("SELECT key, mtime FROM entries")
        }
        seen: set[str] = set()
        touched = 0

        for path in sorted(self.library.entries_dir.glob("*.md")):
            key = path.stem
            seen.add(key)
            mtime = path.stat().st_mtime
            if not force and known.get(key) is not None and abs(known[key] - mtime) < 1e-6:
                continue
            try:
                from . import entryfile

                entry = entryfile.load(path)
            except Exception:
                continue
            entry.key = entry.key or key
            self._upsert(entry, mtime)
            touched += 1

        for stale in set(known) - seen:
            self._remove(stale)
            touched += 1

        conn.commit()
        return touched

    def _upsert(self, e: Entry, mtime: float) -> None:
        conn = self.conn
        self._remove(e.key, commit=False)
        conn.execute(
            """INSERT INTO entries(key,title,authors,year,venue,type,doi,arxiv_id,url,
                                   status,level,level_rank,tags,one_liner,citation_count,
                                   updated,mtime)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                e.key, e.title, ", ".join(e.authors), e.year, e.venue, e.type,
                e.doi, e.arxiv_id, e.url, e.status, e.level, level_rank(e.level),
                ", ".join(e.tags), e.one_liner, e.citation_count, e.updated, mtime,
            ),
        )
        conn.execute(
            "INSERT INTO entries_fts(key,title,authors,one_liner,abstract,tags,"
            "summary,findings,notes) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                e.key, e.title, ", ".join(e.authors), e.one_liner or "",
                e.abstract, " ".join(e.tags), e.detailed_summary_md(),
                "\n".join(e.key_findings), e.notes,
            ),
        )
        for r in e.references:
            conn.execute(
                "INSERT INTO refs(citing,cited_title,cited_doi,cited_arxiv,cited_year,cited_key)"
                " VALUES(?,?,?,?,?,?)",
                (e.key, r.title, r.doi, r.arxiv_id, r.year, r.key),
            )

    def _remove(self, key: str, commit: bool = True) -> None:
        conn = self.conn
        conn.execute("DELETE FROM entries WHERE key=?", (key,))
        conn.execute("DELETE FROM entries_fts WHERE key=?", (key,))
        conn.execute("DELETE FROM refs WHERE citing=?", (key,))
        if commit:
            conn.commit()

    # ---------------- querying ----------------

    def search(
        self,
        query: str,
        limit: int = 20,
        level: str | None = None,
        tag: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        status: str | None = None,
        entry_type: str | None = None,
    ) -> list[SearchHit]:
        """Ranked full-text search with optional structured filters."""
        self.sync()
        where: list[str] = []
        params: list[object] = []

        match = to_fts_query(query)
        if match:
            sql = (
                "SELECT f.key AS key, e.title AS title,"
                f" bm25(entries_fts, {', '.join(str(w) for w in _BM25_WEIGHTS)}) AS score"
                " FROM entries_fts f JOIN entries e ON e.key = f.key"
                " WHERE entries_fts MATCH ?"
            )
            params.append(match)
        else:
            # Empty query: fall back to listing, best level and newest first.
            sql = (
                "SELECT e.key AS key, e.title AS title,"
                " (e.level_rank * 1000 - COALESCE(e.year,0)) AS score"
                " FROM entries e WHERE 1=1"
            )

        if level:
            where.append("e.level_rank <= ?")
            params.append(level_rank(level))
        if tag:
            where.append("(',' || REPLACE(e.tags, ', ', ',') || ',') LIKE ?")
            params.append(f"%,{tag.strip().lower()},%")
        if year_from:
            where.append("e.year >= ?")
            params.append(year_from)
        if year_to:
            where.append("e.year <= ?")
            params.append(year_to)
        if status:
            where.append("e.status = ?")
            params.append(status)
        if entry_type:
            where.append("e.type = ?")
            params.append(entry_type)

        if where:
            sql += " AND " + " AND ".join(where)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [SearchHit(key=r["key"], title=r["title"], score=r["score"]) for r in rows]

    def stats(self) -> dict:
        self.sync()
        c = self.conn
        total = c.execute("SELECT COUNT(*) n FROM entries").fetchone()["n"]
        by_level = {
            r["level"]: r["n"]
            for r in c.execute(
                "SELECT level, COUNT(*) n FROM entries GROUP BY level ORDER BY level_rank"
            )
        }
        by_status = {
            r["status"]: r["n"]
            for r in c.execute("SELECT status, COUNT(*) n FROM entries GROUP BY status")
        }
        by_type = {
            r["type"]: r["n"]
            for r in c.execute(
                "SELECT type, COUNT(*) n FROM entries GROUP BY type ORDER BY n DESC"
            )
        }
        refs = c.execute("SELECT COUNT(*) n FROM refs").fetchone()["n"]
        years = c.execute(
            "SELECT MIN(year) lo, MAX(year) hi FROM entries WHERE year IS NOT NULL"
        ).fetchone()
        tags: dict[str, int] = {}
        for r in c.execute("SELECT tags FROM entries WHERE tags != ''"):
            for t in (r["tags"] or "").split(","):
                t = t.strip()
                if t:
                    tags[t] = tags.get(t, 0) + 1
        return {
            "entries": total,
            "by_level": by_level,
            "by_status": by_status,
            "by_type": by_type,
            "references": refs,
            "year_range": [years["lo"], years["hi"]] if years and years["lo"] else None,
            "top_tags": sorted(tags.items(), key=lambda kv: -kv[1])[:20],
        }

    def citing_entries(self, *, doi: str | None, arxiv_id: str | None,
                       title: str | None) -> list[str]:
        """Keys of library entries whose reference list contains this work."""
        out: set[str] = set()
        c = self.conn
        if doi:
            for r in c.execute("SELECT citing FROM refs WHERE LOWER(cited_doi)=?", (doi.lower(),)):
                out.add(r["citing"])
        if arxiv_id:
            for r in c.execute("SELECT citing FROM refs WHERE cited_arxiv=?", (arxiv_id,)):
                out.add(r["citing"])
        if title:
            for r in c.execute(
                "SELECT citing FROM refs WHERE LOWER(cited_title)=?", (title.lower(),)
            ):
                out.add(r["citing"])
        return sorted(out)


# --------------------------------------------------------------------------
# FTS5 query construction
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "is", "are",
    "what", "which", "how", "does", "do", "can", "with", "by", "from", "at",
    "be", "been", "was", "were", "that", "this", "these", "those", "it", "its",
    "as", "we", "you", "i", "about", "current", "state", "into", "there", "s",
}


def to_fts_query(query: str) -> str:
    """Turn free-form user text into a safe FTS5 MATCH expression.

    Users type questions ("What are current capabilities in causal discovery?"),
    which are not valid FTS5 syntax. Terms are quoted and OR-joined for recall;
    bm25 then does the ranking. An explicitly quoted phrase is passed through.
    """
    query = (query or "").strip()
    if not query:
        return ""

    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]*"', " ", query)

    terms = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+#.-]*", rest)]
    terms = [t for t in terms if t.lower() not in _STOPWORDS and len(t) > 1]

    parts = [f'"{p}"' for p in phrases if p.strip()]
    parts += [f'"{t}"' for t in dict.fromkeys(terms)]  # de-dup, keep order
    if not parts:
        return ""
    return " OR ".join(parts)


def hydrate(library: Library, hits: list[SearchHit]) -> list[SearchHit]:
    """Attach the full `Entry` to each hit."""
    for h in hits:
        h.entry = library.get(h.key)
    return [h for h in hits if h.entry]
