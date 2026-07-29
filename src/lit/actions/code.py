"""`lit code` — find the repository that implements a paper.

`lit add` records a code link only when the paper's own text prints one, which
is the strong claim and the only one a reader agent is allowed to make. Plenty
of papers release code without printing the URL, or release it after
publication; this goes and looks for it.

The whole design is about not writing down a URL that does not exist. A model
asked where a paper's code lives will produce a plausible one from memory, so
what comes back is checked twice before it is stored:

1. It must be a repository URL on a known code host — `normalize_code_url`.
2. The URL must actually resolve. A 404 means the model invented it, and the
   result is dropped rather than filed.

What survives is stored marked `code_source: web`, with the evidence the scout
quoted, so a searched-for repository is never mistaken for one the authors
printed in their paper.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMError
from ..models import CODE_FROM_WEB, Entry, normalize_code_url
from ..prompts import CODE_SCOUT_SYSTEM, find_code_prompt
from ..runner import run_parallel
from .context import Ctx

# The scout searches, opens pages and reads them. Enough turns to try a couple
# of routes and still check what it found; not so many that a paper with no
# code becomes an expensive answer.
MAX_TURNS = 18


@dataclass
class CodeResult:
    """What the search concluded for one entry."""

    key: str
    title: str = ""
    # found     — a repository was confirmed (and stored, unless --dry-run)
    # unchanged — the same URL the entry already had
    # none      — the scout looked and found nothing it could confirm
    # rejected  — it proposed something that did not survive checking
    # skipped   — the entry already has a code link and --force was not given
    # error     — the search itself failed
    status: str = "none"
    code_url: str | None = None
    official: bool = False
    confidence: str = ""
    evidence: str = ""
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("found", "unchanged")

    def to_dict(self) -> dict:
        return {
            "key": self.key, "title": self.title, "status": self.status,
            "code_url": self.code_url, "official": self.official,
            "confidence": self.confidence, "evidence": self.evidence,
            "message": self.message,
        }


def find_code(
    ctx: Ctx,
    entries: list[Entry],
    *,
    force: bool = False,
    unofficial: bool = False,
    dry_run: bool = False,
) -> list[CodeResult]:
    """Search the web for each entry's implementation and record what is found.

    One cheap agent call per entry, run in parallel. `force` re-searches entries
    that already carry a code link; `unofficial` also accepts a third-party
    reimplementation when no official release exists.
    """
    if not entries:
        return []

    todo: list[Entry] = []
    skipped: list[CodeResult] = []
    for e in entries:
        if e.code_url and not force:
            skipped.append(CodeResult(
                key=e.key, title=e.title, status="skipped", code_url=e.code_url,
                message=f"already has a code link ({e.code_provenance()}) — "
                        "re-search it with --force",
            ))
        else:
            todo.append(e)
    if not todo:
        return skipped

    ctx.log(
        f"[bold]Searching[/bold] the web for code implementing "
        f"{len(todo)} paper(s)…"
    )

    def work(entry: Entry) -> CodeResult:
        return _search_one(ctx, entry, unofficial=unofficial, dry_run=dry_run)

    def report(r) -> None:
        if not r.ok:
            entry: Entry = r.item  # type: ignore[assignment]
            ctx.log(f"  [red]![/red] {entry.key}: {r.error}")
            return
        out: CodeResult = r.value
        if out.status in ("found", "unchanged"):
            mark = "[green]✓[/green]" if out.official else "[yellow]~[/yellow]"
            ctx.log(f"  {mark} {out.key}: {out.code_url}")
        elif out.status == "rejected":
            ctx.log(f"  [yellow]-[/yellow] {out.key}: {out.message}")
        else:
            ctx.vlog(f"  [dim]·[/dim] {out.key}: {out.message or 'no code found'}")

    runs = run_parallel(todo, work, workers=ctx.workers, on_done=report)

    out: list[CodeResult] = []
    for r in runs:
        entry: Entry = r.item  # type: ignore[assignment]
        if r.ok and r.value is not None:
            out.append(r.value)
        else:
            out.append(CodeResult(key=entry.key, title=entry.title,
                                  status="error", message=str(r.error)))
    return out + skipped


# --------------------------------------------------------------------------
# One entry
# --------------------------------------------------------------------------

def _search_one(ctx: Ctx, entry: Entry, *, unofficial: bool,
                dry_run: bool) -> CodeResult:
    res = CodeResult(key=entry.key, title=entry.title)

    try:
        data = ctx.llm.json(
            find_code_prompt(
                title=entry.title,
                authors=entry.authors,
                year=entry.year,
                doi=entry.doi,
                arxiv_id=entry.arxiv_id,
                venue=entry.venue,
                abstract=entry.abstract,
            ),
            system=CODE_SCOUT_SYSTEM,
            role="code",
            tools=["WebSearch", "WebFetch"],
            max_turns=MAX_TURNS,
            required=("repo_url",),
        )
    except LLMError as exc:
        res.status = "error"
        res.message = str(exc)
        return res

    res.evidence = str(data.get("evidence") or "").strip()
    res.confidence = str(data.get("confidence") or "").strip().lower()
    # A scout that says nothing about who released it has not confirmed an
    # author release, but it has not called it third-party either. Both fail the
    # default bar; only one of them is worth naming a reimplementation.
    stated_official = data.get("official")
    res.official = bool(stated_official)

    # A model with nothing to report says so in several ways: a JSON null, an
    # absent key, or the word "null" as a string. All of them mean the same
    # thing, and none of them is a failure.
    raw = str(data.get("repo_url") or "").strip()
    if raw.lower() in ("", "null", "none", "n/a"):
        res.status = "none"
        res.message = "no public repository found for this paper"
        return res

    url = normalize_code_url(raw)
    if not url:
        res.status = "rejected"
        res.message = f"not a code repository URL: {raw[:120]}"
        return res

    if not res.official and not unofficial:
        res.status = "rejected"
        res.code_url = url
        what = ("a third-party reimplementation" if stated_official is False
                else "not confirmed as the authors' own")
        res.message = (
            f"{url} is {what} — re-run with --unofficial to record it"
        )
        return res

    reachable, note = _check_reachable(ctx, url)
    if not reachable:
        res.status = "rejected"
        res.code_url = url
        res.message = f"{url} {note}"
        return res

    res.code_url = url
    if entry.code_url == url:
        res.status = "unchanged"
        res.message = "already recorded"
        return res

    res.status = "found"
    res.message = f"found {url}"
    if dry_run:
        res.message += " (dry run — not stored)"
        return res

    entry.code_url = url
    entry.code_source = CODE_FROM_WEB
    entry.code_reason = _reason(res, note)
    ctx.library.save_entry(entry)
    return res


def _reason(res: CodeResult, note: str) -> str:
    """The provenance line stored on the entry. Short, and honest about doubt."""
    bits = ["author release" if res.official else "third-party reimplementation"]
    if res.confidence in ("high", "medium", "low"):
        bits.append(f"{res.confidence} confidence")
    if note:
        bits.append(note)
    line = ", ".join(bits)
    if res.evidence:
        line += f" — {res.evidence[:200]}"
    return line


def _check_reachable(ctx: Ctx, url: str) -> tuple[bool, str]:
    """Does this repository actually exist?

    The one check that catches an invented URL: a model's plausible
    `github.com/<lab>/<method>` 404s, and a real repository does not. Being
    unable to reach the host at all is a different thing from being told the
    repository is not there — the first is our network, the second is an answer
    — so an unreachable host keeps the link and records the doubt instead.
    """
    before = ctx.http.unavailable_count
    r = ctx.http.get(url, retries=2)
    if r is not None:
        return True, ""
    if ctx.http.unavailable_count > before:
        return True, "link not verified (host unreachable)"
    return False, "does not resolve — the scout invented it"
