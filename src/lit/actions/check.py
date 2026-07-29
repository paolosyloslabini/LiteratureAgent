"""`lit check` — verify metadata that looks wrong.

An index answering at all is not an index answering correctly. What comes back is
well-formed and quietly wrong often enough to matter: a publisher's legal name
where a venue belongs, a citation count of zero for a paper the field has cited
for a decade, a year that predates the paper's own preprint, an author list
collapsed to one name. None of that fails a schema, so nothing upstream catches
it, and `lit refresh` re-fetches the same wrong answer.

Asking for a check is itself the statement that the stored record is not
trusted, so every entry passed in gets an agent that goes and looks. There is no
cheap path that skips one: a code-side audit deciding an entry "looks fine"
would answer the user's doubt with the same code whose output they are doubting.
What the caller controls is the scope — one key, or `--level` / `--tag` / `-n` —
not whether the call happens.

Each entry therefore goes through two steps:

1. **Re-resolve from the indexes.** Free and authoritative, and worth doing
   first: a field the indexes can settle with a better identifier is one the
   agent should be verifying in its corrected form rather than its stale one.
2. **Ask a cheap agent.** Always, with web search, about venue, year and type.
   `audit()` rides along as a hint — "these also look implausible from here" —
   which sharpens the prompt without gating it.

What the agent proposes is checked before it is believed, on the same principle
as `lit code`: a model asked for a venue will produce a plausible one. A proposed
venue must not itself be a publisher name, must come with evidence, and must
come with a source that resolves. Numbers and author lists are never taken from
the model at all — those come from step 2 or not at all, because a fabricated
citation count is indistinguishable from a real one.

Nothing is written without `--fix`, and every applied correction records the
source that confirmed it in `check_note`.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date

from ..fetch.metadata import from_crossref, resolve_metadata, title_similarity
from ..llm import LLMError
from ..models import ENTRY_TYPES, Entry, normalize_doi
from ..prompts import CHECK_SYSTEM, check_metadata_prompt
from ..runner import run_parallel
from ..venue import clean_venue, is_placeholder, is_publisher_name, is_workshop
from .context import Ctx

# The agent opens a publisher page or a DBLP entry and reads the venue off it.
# Fewer turns than the code scout needs: there is a specific field to look up,
# not a repository to hunt for.
MAX_TURNS = 12

# A paper this old with no recorded citations is a metadata gap, not a verdict on
# the work. Below this, zero is simply the truth.
STALE_CITATION_YEARS = 3

# Titles must agree this well before a DOI the agent proposes is adopted. The
# same bar `resolve_metadata` uses for a title lookup, for the same reason: a
# near match lands on a different paper of the same name.
DOI_TITLE_THRESHOLD = 0.85

# Fields the indexes own. `check` will rewrite these and nothing else — never a
# summary, never the abstract, never user notes.
FIXABLE = ("venue", "year", "type", "doi", "authors", "citation_count", "title")

# Only these may be taken from what the agent says, and only after checking.
# A number or an author list from a model is unverifiable, so neither is here.
MODEL_FIXABLE = ("venue", "year", "type")


@dataclass
class Suspicion:
    """One thing that looks wrong about one entry, and why."""

    field: str
    problem: str

    def to_dict(self) -> dict:
        return {"field": self.field, "problem": self.problem}


@dataclass
class CheckResult:
    key: str
    title: str = ""
    # fixed     — corrections were confirmed and written
    # proposed  — corrections were confirmed but --fix was not given
    # confirmed — looked wrong, checked out as correct after all
    # unresolved— still looks wrong and nothing could confirm a correction
    # error     — the check itself failed
    status: str = "clean"
    suspicions: list[Suspicion] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    # Fields a source positively backed up as already correct. Worth recording
    # separately from "changed nothing": a field that checked out is answered,
    # and saying so is what keeps the next run from paying to ask again.
    confirmed: list[str] = field(default_factory=list)
    asked_model: bool = False
    evidence: str = ""
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("clean", "fixed", "confirmed")

    def to_dict(self) -> dict:
        return {
            "key": self.key, "title": self.title, "status": self.status,
            "suspicions": [s.to_dict() for s in self.suspicions],
            "changes": self.changes, "unresolved": self.unresolved,
            "confirmed": self.confirmed,
            "asked_model": self.asked_model, "evidence": self.evidence,
            "message": self.message,
        }


# --------------------------------------------------------------------------
# The audit — pure, and the reason this command is cheap
# --------------------------------------------------------------------------

def audit(entry: Entry, *, today_year: int | None = None) -> list[Suspicion]:
    """Everything that looks wrong about one entry's metadata, from here.

    A hint, not a verdict: it points the agent at fields worth a closer look and
    it is what `--fix` reports against, but an empty list does not mean an entry
    goes unchecked. Deliberately conservative all the same, since a false
    positive sends the agent looking for a problem that is not there — it fires
    on metadata that is *implausible*, not merely unflattering. A paper with
    genuinely few citations is not suspicious; a decade-old paper with none
    recorded at all is a gap in what we fetched.
    """
    today_year = today_year or date.today().year
    out: list[Suspicion] = []
    v_raw = entry.venue
    v = clean_venue(v_raw)

    if v_raw and v is None:
        out.append(Suspicion("venue", f"venue {v_raw!r} is not a usable venue name"))
    elif v and is_publisher_name(v):
        out.append(Suspicion("venue", f"venue {v!r} names a publisher, not a venue"))
    elif v and is_placeholder(v):
        out.append(Suspicion(
            "venue", f"venue {v!r} names a preprint server, not a venue"))
    elif not v and entry.type in ("conference paper", "journal article"):
        out.append(Suspicion("venue", f"a {entry.type} with no venue recorded"))

    # A workshop paper filed as a main-track paper inherits the wrong rank, and
    # the venue string is the evidence of it.
    if v and is_workshop(v) and entry.type not in ("workshop paper", "other"):
        out.append(Suspicion(
            "type", f"venue names a workshop but type is {entry.type!r}"))

    if entry.year is None:
        out.append(Suspicion("year", "no year recorded"))
    elif entry.year > today_year + 1 or entry.year < 1500:
        out.append(Suspicion("year", f"year {entry.year} is implausible"))
    else:
        ay = arxiv_year(entry.arxiv_id)
        if ay and entry.year < ay:
            out.append(Suspicion(
                "year",
                f"year {entry.year} predates the paper's own arXiv posting ({ay})",
            ))

    age = (today_year - entry.year) if entry.year else None
    if not entry.citation_count and age is not None and age >= STALE_CITATION_YEARS:
        out.append(Suspicion(
            "citation_count",
            f"no citations recorded for a paper {age} years old",
        ))

    if not entry.authors:
        out.append(Suspicion("authors", "no authors recorded"))
    elif len(entry.authors) == 1 and re.search(
        r"\bet\.?\s*al\.?", entry.authors[0], re.IGNORECASE
    ):
        out.append(Suspicion(
            "authors", f"author list collapsed to {entry.authors[0]!r}"))

    if re.search(r"<[^>]+>|&[a-z]+;|&#\d+;", entry.title or "", re.IGNORECASE):
        out.append(Suspicion("title", "title still carries unrendered markup"))

    return out


def arxiv_year(arxiv_id: str | None) -> int | None:
    """The year an arXiv id was posted, from the id itself.

    New-style ids are `YYMM.NNNNN`; the old scheme was `archive/YYMMNNN` and ran
    from 1991 until 2007, which is what splits the two-digit year.
    """
    if not arxiv_id:
        return None
    m = re.match(r"^(\d{2})(\d{2})\.", str(arxiv_id))
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"/(\d{2})(\d{2})\d{3}$", str(arxiv_id))
    if m:
        yy = int(m.group(1))
        return (1900 + yy) if yy >= 91 else (2000 + yy)
    return None


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------

def check_entries(ctx: Ctx, entries: list[Entry], *,
                  fix: bool = False) -> list[CheckResult]:
    """Have an agent verify each entry's record, and optionally correct it.

    Every entry passed in gets its agent call. There is deliberately no cheap
    path that skips one: asking for a check *is* the statement that the stored
    metadata is not trusted, and letting `audit()` decide an entry looks fine
    would answer that doubt with the same code whose output is in doubt.

    `audit()` still runs, but as a hint carried into the prompt — "these fields
    also look implausible from here" — rather than as a gate in front of it. So
    the caller controls the bill by choosing what to pass in (`--level`,
    `--tag`, `-n`, or one key), not by having the audit quietly decline.
    """
    if not entries:
        return []

    todo = list(entries)
    ctx.log(
        f"[bold]Checking[/bold] the record of {len(todo)} "
        f"entr{'y' if len(todo) == 1 else 'ies'} — one cheap agent call each…"
    )

    def work(entry: Entry) -> CheckResult:
        return _check_one(ctx, entry, fix=fix)

    def report(r) -> None:
        if not r.ok:
            entry: Entry = r.item  # type: ignore[assignment]
            ctx.log(f"  [red]![/red] {entry.key}: {r.error}")
            return
        out: CheckResult = r.value
        if out.status in ("fixed", "proposed"):
            mark = "[green]~[/green]" if out.status == "fixed" else "[cyan]?[/cyan]"
            ctx.log(f"  {mark} {out.key}: {'; '.join(out.changes)}")
        elif out.status == "unresolved":
            ctx.log(f"  [yellow]-[/yellow] {out.key}: {'; '.join(out.unresolved)}")
        else:
            ctx.vlog(f"  [dim]·[/dim] {out.key}: {out.message or 'no change'}")

    runs = run_parallel(todo, work, workers=ctx.workers, on_done=report)

    out: list[CheckResult] = []
    for r in runs:
        entry: Entry = r.item  # type: ignore[assignment]
        if r.ok and r.value is not None:
            out.append(r.value)
        else:
            out.append(CheckResult(key=entry.key, title=entry.title,
                                   status="error", message=str(r.error)))
    return out


# --------------------------------------------------------------------------
# One entry
# --------------------------------------------------------------------------

def _check_one(ctx: Ctx, entry: Entry, *, fix: bool) -> CheckResult:
    res = CheckResult(key=entry.key, title=entry.title, suspicions=audit(entry))

    # Corrections are staged on a copy, so a run without --fix can report exactly
    # what it would have written without writing any of it.
    draft = copy.deepcopy(entry)
    sources: list[str] = []

    # The indexes are asked first because they are free and authoritative, and
    # because a field they can correct with a better identifier is one the agent
    # should be verifying in its corrected form rather than its stale one. This
    # is not a shortcut past the agent — it runs either way, immediately below.
    if _resolve_from_indexes(ctx, draft, res):
        sources.append("metadata indexes")

    res.asked_model = True
    if _ask_the_model(ctx, draft, audit(draft), res):
        sources.append("web search")

    # Appended, not assigned: `_ask_the_model` has already recorded *why* it
    # refused a correction ("the source given for it does not resolve"), and that
    # is the more useful half of the report. Overwriting it left the user with
    # only the original complaint and no sign that an answer had been rejected.
    still_wrong = [s for s in audit(draft) if s.field not in res.confirmed]
    res.unresolved += [s.problem for s in still_wrong]

    if not res.changes:
        if res.unresolved:
            res.status = "unresolved"
            res.message = "nothing could confirm a correction"
        else:
            res.status = "confirmed"
            res.message = "the stored metadata checks out"
        return res

    if not fix:
        res.status = "proposed"
        res.message = "not applied — re-run with --fix"
        return res

    draft.check_note = _note(res, sources)
    draft.updated = date.today().isoformat()
    ctx.library.save_entry(draft)
    res.status = "fixed"
    res.message = "; ".join(res.changes)
    return res


def _resolve_from_indexes(ctx: Ctx, draft: Entry, res: CheckResult) -> bool:
    """Fix what the indexes can settle for free. True if anything changed.

    This runs before the model on purpose. Most of what the audit flags is not a
    disagreement about the world — it is a field we filled from whichever source
    answered first, and asking again with the identifier we now have settles it.
    """
    if not (draft.doi or draft.arxiv_id or draft.title):
        return False
    meta = resolve_metadata(
        ctx.http,
        doi=draft.doi,
        arxiv_id=draft.arxiv_id,
        title=None if (draft.doi or draft.arxiv_id) else draft.title,
        cfg=ctx.cfg.fetch,
        with_references=False,
    )
    if meta is None:
        return False

    changed = False
    fresh = clean_venue(meta.venue)
    # A real venue replaces a placeholder one, which is exactly the case
    # `lit refresh` cannot fix: it only fills a venue that is empty.
    if fresh and not is_placeholder(fresh) and fresh != draft.venue:
        if not draft.venue or is_placeholder(draft.venue):
            res.changes.append(f"venue {draft.venue or '(none)'!r} → {fresh!r}")
            draft.venue = fresh
            changed = True

    if meta.citation_count and meta.citation_count != draft.citation_count:
        res.changes.append(
            f"citations {draft.citation_count or 0} → {meta.citation_count}")
        draft.citation_count = meta.citation_count
        changed = True

    if meta.year and meta.year != draft.year:
        ay = arxiv_year(draft.arxiv_id)
        implausible = draft.year is None or draft.year < 1500 or (
            ay is not None and draft.year < ay)
        if implausible:
            res.changes.append(f"year {draft.year or '(none)'} → {meta.year}")
            draft.year = meta.year
            changed = True

    if meta.doi and not draft.doi:
        res.changes.append(f"DOI found: {meta.doi}")
        draft.doi = meta.doi
        changed = True

    if meta.authors and len(meta.authors) > len(draft.authors):
        res.changes.append(
            f"authors {len(draft.authors)} → {len(meta.authors)}")
        draft.authors = list(meta.authors)
        changed = True

    # A type that disagrees with the venue is worth taking from the index, which
    # derived it from the record rather than from the venue string.
    if meta.type not in ("other", "") and meta.type != draft.type:
        if is_workshop(draft.venue) or draft.type in ("other", ""):
            res.changes.append(f"type {draft.type!r} → {meta.type!r}")
            draft.type = meta.type
            changed = True

    if meta.title and re.search(r"<[^>]+>|&[a-z]+;", draft.title or ""):
        res.changes.append("title markup stripped")
        draft.title = meta.title
        changed = True

    return changed


def _ask_the_model(ctx: Ctx, draft: Entry, suspicions: list[Suspicion],
                   res: CheckResult) -> bool:
    """Put what the indexes could not settle to a cheap agent. True if applied."""
    try:
        data = ctx.llm.json(
            check_metadata_prompt(entry=draft, suspicions=[
                (s.field, s.problem) for s in suspicions
            ]),
            system=CHECK_SYSTEM,
            role="check",
            tools=["WebSearch", "WebFetch"],
            max_turns=MAX_TURNS,
            required=("fields",),
        )
    except LLMError as exc:
        res.unresolved.append(f"check failed: {exc}")
        return False

    applied = False

    # A DOI is the one answer worth more than the field it corrects: confirmed
    # against Crossref it re-resolves everything else authoritatively, so it is
    # tried before any of the model's own field values.
    proposed_doi = normalize_doi(data.get("doi"))
    if proposed_doi and proposed_doi != draft.doi:
        if _confirm_doi(ctx, draft, proposed_doi):
            res.changes.append(f"DOI confirmed: {proposed_doi}")
            draft.doi = proposed_doi
            _resolve_from_indexes(ctx, draft, res)
            applied = True

    for raw in (data.get("fields") or []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("field") or "").strip().lower()
        if name not in MODEL_FIXABLE:
            continue
        verdict = str(raw.get("verdict") or "").strip().lower()
        evidence = str(raw.get("evidence") or "").strip()
        source = str(raw.get("source_url") or "").strip()
        # "I checked and it is right" answers the suspicion as surely as a
        # correction does, provided it was checked against something.
        if verdict == "correct" and evidence and name not in res.confirmed:
            res.confirmed.append(name)
            continue
        if verdict != "wrong":
            continue
        proposed = raw.get("proposed")
        if not evidence:
            res.unresolved.append(f"{name}: a correction with no evidence")
            continue

        ok, value, why = _vet(ctx, name, proposed, source, draft)
        if not ok:
            res.unresolved.append(f"{name}: {why}")
            continue

        res.changes.append(f"{name} {getattr(draft, name)!r} → {value!r}")
        setattr(draft, name, value)
        if evidence and not res.evidence:
            res.evidence = evidence[:300]
        applied = True

    return applied


def _vet(ctx: Ctx, name: str, proposed, source: str, draft: Entry):
    """Is this correction believable? Returns (ok, value, why-not)."""
    if name == "venue":
        v = clean_venue(proposed if isinstance(proposed, str) else None)
        if not v:
            return False, None, "proposed venue is empty"
        if is_placeholder(v):
            return False, None, f"proposed venue {v!r} is a publisher or preprint server"
        if v == draft.venue:
            return False, None, "proposed venue is the one already stored"
        # The one mechanical check available for a string: the page it was read
        # off has to exist. An invented citation comes with an invented source.
        if not _resolves(ctx, source):
            return False, None, "the source given for it does not resolve"
        return True, v, ""

    if name == "year":
        try:
            y = int(str(proposed).strip()[:4])
        except (TypeError, ValueError):
            return False, None, f"proposed year {proposed!r} is not a year"
        if y < 1500 or y > date.today().year + 1:
            return False, None, f"proposed year {y} is implausible"
        ay = arxiv_year(draft.arxiv_id)
        if ay and y < ay:
            return False, None, f"proposed year {y} predates the arXiv posting ({ay})"
        if y == draft.year:
            return False, None, "proposed year is the one already stored"
        return True, y, ""

    if name == "type":
        t = str(proposed or "").strip().lower()
        if t not in ENTRY_TYPES:
            return False, None, f"{t!r} is not one of the recorded entry types"
        if t == draft.type:
            return False, None, "proposed type is the one already stored"
        return True, t, ""

    return False, None, "not a field this command will rewrite"


def _confirm_doi(ctx: Ctx, draft: Entry, doi: str) -> bool:
    """Does this DOI really belong to this paper?

    Crossref is asked, not the model. A DOI that resolves to a different work is
    the most expensive mistake available here — it would replace every other
    field with that work's — so the titles have to agree.
    """
    cr = from_crossref(ctx.http, doi)
    if cr is None or not cr.title:
        return False
    return title_similarity(draft.title, cr.title) >= DOI_TITLE_THRESHOLD


def _resolves(ctx: Ctx, url: str) -> bool:
    """Does the cited source exist? Unreachable is not the same as absent."""
    if not url.lower().startswith(("http://", "https://")):
        return False
    before = ctx.http.unavailable_count
    if ctx.http.get(url, retries=2) is not None:
        return True
    # Our network failing is not evidence against the model's answer.
    return ctx.http.unavailable_count > before


def _note(res: CheckResult, sources: list[str]) -> str:
    """The provenance line stored on the entry. Says who confirmed what."""
    where = " and ".join(sources) if sources else "audit"
    line = f"{date.today().isoformat()}: corrected from {where} — " + "; ".join(
        res.changes)
    if res.evidence:
        line += f" — {res.evidence}"
    return line[:600]
