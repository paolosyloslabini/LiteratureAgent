"""`lit` — the command line interface.

Every capability is a subcommand, and every read command takes `--json`, so the
tool is fully usable by a script or another agent without a terminal. The TUI
(`lit browse`) is a viewer layered on top; nothing depends on it.
"""

from __future__ import annotations

import json as jsonlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import bundle, entryfile
from .actions.add import (
    add_paper,
    link_references,
    read_entries,
    reread as reread_action,
)
from .actions.ask import ask as ask_action
from .actions.claim import trace_claim
from .actions.context import Ctx
from .actions.discover import add_candidates, discover
from .actions.inbox import process_inbox
from .actions.refresh import refresh_entries
from .actions.search import search as search_action
from .config import CONFIG_PATH, Config, load_config
from .library import Library, LibraryError, list_libraries, normalize_name, resolve_library
from .models import LEVELS, STATUS_UNREAD, STATUS_VERIFIED, Entry, level_rank
from .render import chain_tree, entries_table, entry_panel, print_answer
from .store import Store

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build and query a research library. Entries are Markdown files; "
         "every summary is written by an agent that read the whole paper.",
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)


class State:
    library_name: Optional[str] = None
    json_mode: bool = False
    verbose: bool = False
    yes: bool = False
    model: Optional[str] = None
    # Concurrency cap (`--workers`), not the `find --parallel` switch.
    workers: Optional[int] = None
    _cfg: Optional[Config] = None

    @property
    def cfg(self) -> Config:
        if self._cfg is None:
            self._cfg = load_config()
            if self.model:
                self._cfg.llm.override_model = self.model
            if self.workers:
                self._cfg.llm.max_parallel = self.workers
        return self._cfg


state = State()


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def _fail(message: str, code: int = 1):
    if state.json_mode:
        console.print_json(jsonlib.dumps({"error": message}))
    else:
        err_console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)


def _lib() -> Library:
    try:
        return resolve_library(state.cfg, state.library_name)
    except LibraryError as exc:
        _fail(str(exc))
        raise  # unreachable, keeps type checkers happy


def _ctx() -> Ctx:
    return Ctx(
        cfg=state.cfg,
        library=_lib(),
        console=console,
        json_mode=state.json_mode,
        verbose=state.verbose,
        yes=state.yes,
    )


def _emit(data: dict) -> None:
    console.print_json(jsonlib.dumps(data, default=str))


def _usage_footer(ctx: Ctx) -> str:
    return ctx.usage.summary()


@app.callback()
def main_callback(
    library: Optional[str] = typer.Option(
        None, "-L", "--library", help="Library name or path (default: the one in "
                                      "config, or the only one that exists)."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON on stdout instead of "
                              "formatted output. Use this from scripts and agents."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show progress detail."),
    yes: bool = typer.Option(
        False, "-y", "--yes", help="Approve everything without asking (autonomous)."),
    model: Optional[str] = typer.Option(
        None, "--model", help="Force one model for every agent role in this command "
                              "(e.g. haiku, sonnet, opus)."),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-j",
        help="Max concurrent agents (default: llm.max_parallel). This is how "
             "MANY run at once; `lit find --parallel` is what decides whether "
             "the scout agents run at all."),
):
    state.library_name = library
    state.json_mode = json_out
    state.verbose = verbose
    state.yes = yes
    state.model = model
    state.workers = workers


# --------------------------------------------------------------------------
# Library management
# --------------------------------------------------------------------------

@app.command("new")
def cmd_new(
    name: str = typer.Argument(..., help="Short name for the library, e.g. llm-benchmarks."),
    scope: str = typer.Option(..., "--scope", "-s",
                              help="The topic this library investigates."),
    min_level: str = typer.Option(
        "C", "--min-level",
        help=f"Reject entries below this level {LEVELS}, or 'any'. Permissive by "
             "default — filter at read time with `lit ls --level A` instead."),
    allow_non_published: bool = typer.Option(
        False, "--allow-non-published",
        help="Also admit videos, lecture notes and other unpublished material."),
    use: bool = typer.Option(True, "--use/--no-use",
                             help="Make this the default library."),
):
    """Create a new library."""
    cfg = state.cfg
    if min_level not in LEVELS and min_level != "any":
        _fail(f"--min-level must be one of {LEVELS} or 'any'")
    try:
        lib = Library.create(cfg.root_path, name, scope, min_level=min_level,
                             allow_non_published=allow_non_published)
    except LibraryError as exc:
        _fail(str(exc))
        return
    if use:
        cfg.default_library = lib.name
        cfg.save()

    if state.json_mode:
        _emit({"created": lib.name, "path": str(lib.path), "scope": scope})
        return
    console.print(Panel(
        f"[bold]{lib.name}[/bold]\n[dim]{scope}[/dim]\n\n{lib.path}\n\n"
        f"Add your first paper:\n  [cyan]lit add \"<title, DOI or arXiv id>\"[/cyan]\n"
        f"Or fill it automatically:\n  [cyan]lit find \"<topic>\" -n 10[/cyan]",
        title="Library created", border_style="green",
    ))


@app.command("libs")
def cmd_libs():
    """List all libraries."""
    cfg = state.cfg
    libs = list_libraries(cfg)
    if state.json_mode:
        _emit({"root": str(cfg.root_path), "default": cfg.default_library,
               "libraries": [
                   {"name": l.name, "scope": l.settings.scope,
                    "entries": len(l), "path": str(l.path)} for l in libs
               ]})
        return
    if not libs:
        console.print(f"[dim]No libraries in {cfg.root_path}.[/dim]\n"
                      'Create one:  [cyan]lit new <name> --scope "<topic>"[/cyan]')
        return
    t = Table(header_style="bold", expand=True)
    t.add_column("", no_wrap=True)
    t.add_column("name", style="cyan", no_wrap=True)
    t.add_column("scope", ratio=3)
    t.add_column("entries", justify="right", no_wrap=True)
    for l in libs:
        marker = "→" if l.name == cfg.default_library else " "
        t.add_row(marker, l.name, l.settings.scope or "[dim]—[/dim]", str(len(l)))
    console.print(t)
    console.print(f"[dim]{cfg.root_path}[/dim]")


@app.command("use")
def cmd_use(name: str = typer.Argument(..., help="Library to make the default.")):
    """Set the default library."""
    cfg = state.cfg
    target = cfg.root_path / normalize_name(name)
    if not (target / "library.toml").exists():
        _fail(f"no library named {name!r} in {cfg.root_path}")
    cfg.default_library = normalize_name(name)
    cfg.save()
    if state.json_mode:
        _emit({"default_library": cfg.default_library})
    else:
        console.print(f"Default library is now [cyan]{cfg.default_library}[/cyan]")


@app.command("info")
def cmd_info():
    """Show this library's scope, settings and statistics."""
    lib = _lib()
    with Store(lib) as store:
        stats = store.stats()
    if state.json_mode:
        _emit({"name": lib.name, "path": str(lib.path),
               "scope": lib.settings.scope,
               "min_level": lib.settings.min_level,
               "allow_non_published": lib.settings.allow_non_published,
               **stats})
        return

    levels = "  ".join(f"{k}: {v}" for k, v in stats["by_level"].items()) or "—"
    types = "  ".join(f"{k}: {v}" for k, v in stats["by_type"].items()) or "—"
    status = "  ".join(f"{k}: {v}" for k, v in stats["by_status"].items()) or "—"
    tags = ", ".join(f"{t} ({n})" for t, n in stats["top_tags"][:12]) or "—"
    yr = stats["year_range"]
    console.print(Panel(
        f"[dim]{lib.settings.scope}[/dim]\n\n"
        f"[bold]{stats['entries']}[/bold] entries"
        + (f", {yr[0]}–{yr[1]}" if yr else "")
        + f", {stats['references']} tracked references\n\n"
        f"Levels    {levels}\n"
        f"Status    {status}\n"
        f"Types     {types}\n"
        f"Tags      {tags}\n\n"
        f"[dim]{lib.path}\nmin level {lib.settings.min_level} · "
        f"non-published {'allowed' if lib.settings.allow_non_published else 'rejected'}[/dim]",
        title=f"[cyan]{lib.name}[/cyan]", border_style="cyan",
    ))


@app.command("config")
def cmd_config(
    action: str = typer.Argument("show", help="show | set | path"),
    key: Optional[str] = typer.Argument(None, help="e.g. llm.models.reader, fetch.email"),
    value: Optional[str] = typer.Argument(None),
    library_scope: bool = typer.Option(
        False, "--library", help="Set a library setting (scope, min_level, "
                                 "allow_non_published, auto_approve) instead."),
):
    """Show or change configuration."""
    cfg = state.cfg

    if action == "path":
        console.print(str(CONFIG_PATH))
        return

    if action == "show":
        if library_scope:
            lib = _lib()
            data = {"scope": lib.settings.scope, "min_level": lib.settings.min_level,
                    "allow_non_published": lib.settings.allow_non_published,
                    "auto_approve": lib.settings.auto_approve}
        else:
            data = {"root": cfg.root, "default_library": cfg.default_library,
                    "llm": {"model": cfg.llm.model, "models": cfg.llm.models,
                            "max_parallel": cfg.llm.max_parallel,
                            "timeout_s": cfg.llm.timeout_s},
                    "fetch": {"email": cfg.fetch.email,
                              "semantic_scholar_api_key":
                                  "set" if cfg.fetch.semantic_scholar_api_key else "",
                              "max_pdf_mb": cfg.fetch.max_pdf_mb,
                              "timeout_s": cfg.fetch.timeout_s,
                              "fallback_url_template":
                                  cfg.fetch.fallback_url_template,
                              "fallback_cmd": cfg.fetch.fallback_cmd}}
        if state.json_mode:
            _emit(data)
        else:
            console.print_json(jsonlib.dumps(data, indent=2))
        return

    if action != "set" or not key:
        _fail("usage: lit config [show|set|path] [key] [value]")

    if library_scope:
        lib = _lib()
        if not hasattr(lib.settings, key):
            _fail(f"unknown library setting {key!r}")
        cur = getattr(lib.settings, key)
        setattr(lib.settings, key, _coerce(value, cur))
        lib.save_settings()
        console.print(f"{lib.name}.{key} = {getattr(lib.settings, key)!r}")
        return

    obj, attr = cfg, key
    if key.startswith("llm.models."):
        role = key.split(".", 2)[2]
        cfg.llm.models[role] = str(value)
        cfg.save()
        console.print(f"llm.models.{role} = {value!r}")
        return
    for part in key.split(".")[:-1]:
        if not hasattr(obj, part):
            _fail(f"unknown config section {part!r}")
        obj = getattr(obj, part)
    attr = key.split(".")[-1]
    if not hasattr(obj, attr):
        _fail(f"unknown config key {key!r}")
    setattr(obj, attr, _coerce(value, getattr(obj, attr)))
    cfg.save()
    console.print(f"{key} = {getattr(obj, attr)!r}")


def _coerce(value, current):
    if value is None:
        return current
    if isinstance(current, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except ValueError:
            return current
    return value


# --------------------------------------------------------------------------
# Adding papers
# --------------------------------------------------------------------------

@app.command("add")
def cmd_add(
    query: str = typer.Argument(..., help="Title, DOI, arXiv id, or URL."),
    pdf: Optional[Path] = typer.Option(
        None, "--pdf", exists=True, dir_okay=False,
        help="Read this local PDF instead of trying to download one."),
    force: bool = typer.Option(False, "--force", "-f",
                               help="Add even if it fails the quality bar."),
    refresh: bool = typer.Option(False, "--refresh",
                                 help="Re-fetch and re-read an entry already present."),
    no_read: bool = typer.Option(
        False, "--no-read",
        help="File it from its metadata without reading it. `lit read <key>` "
             "summarizes it later."),
    tag: list[str] = typer.Option([], "--tag", "-t", help="Extra tag (repeatable)."),
):
    """Add one paper: fetch it, read the whole thing, and summarize it."""
    ctx = _ctx()
    try:
        res = add_paper(ctx, query, local_pdf=pdf, force=force, refresh=refresh,
                        extra_tags=list(tag), read=not no_read)
        if res.entry:
            link_references(ctx, res.entry)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({**res.to_dict(), "usage": ctx.usage.summary()})
        raise typer.Exit(0 if res.ok else 1)

    if not res.ok:
        style = "yellow" if res.status in ("duplicate", "rejected") else "red"
        console.print(f"[{style}]{res.status}:[/{style}] {res.message}")
        raise typer.Exit(0 if res.status == "duplicate" else 1)

    console.print(entry_panel(res.entry, full=False))
    for w in res.warnings:
        console.print(f"[yellow]note:[/yellow] {w}")
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


@app.command("find")
def cmd_find(
    query: str = typer.Argument(..., help="Topic or request, in plain language."),
    limit: int = typer.Option(10, "-n", "--limit", help="Max papers to add."),
    parallel: bool = typer.Option(
        False, "--parallel", "-P",
        help="Also run LLM scout agents, one per angle, alongside the free "
             "index search. Finds work keyword search cannot — and costs "
             "tokens for it."),
    read: bool = typer.Option(
        False, "--read",
        help="Read each paper in full and write its summaries now. Off by "
             "default: papers are filed from their metadata, and `lit read` "
             "summarizes the ones you decide are worth it."),
    angles: int = typer.Option(5, "--angles",
                               help="How many scout angles to run with --parallel (1-5)."),
    review: bool = typer.Option(
        False, "--review", help="Show the candidates and ask before adding any."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="List candidates and stop."),
    no_refs: bool = typer.Option(
        False, "--no-refs", help="Skip mining the references of existing entries."),
    no_web: bool = typer.Option(
        False, "--no-web",
        help="Skip the online search entirely; use only what your library cites."),
    no_plan: bool = typer.Option(
        False, "--no-plan",
        help="Search your query verbatim instead of reading it into a topic, "
             "year window and document type first."),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore the quality bar."),
    tag: list[str] = typer.Option([], "--tag", "-t", help="Tag everything added."),
):
    """Search a topic and add the papers that fit.

    Write the request the way you would say it — "surveys of retrieval-
    augmented generation since 2022" — and one cheap call turns it into what
    the indexes can answer: a topic, alternate phrasings, a year window, a
    document type. `--no-plan` searches the string verbatim instead.

    Beyond that the default path spends nothing on searching: Crossref,
    OpenAlex and arXiv are queried several ways, the references of your own
    entries are mined, and one cheap call ranks the pool against your query.
    Papers are filed from their metadata; `--read` (or `lit read` afterwards)
    buys the full section-by-section summaries.
    """
    ctx = _ctx()
    try:
        found = discover(ctx, query, limit=limit, angles=angles,
                         use_references=not no_refs, use_web=not no_web,
                         use_scouts=parallel, use_plan=not no_plan)

        if not found.candidates:
            msg = "No new candidates found."
            if state.json_mode:
                _emit({**found.to_dict(), "message": msg})
            else:
                console.print(f"[yellow]{msg}[/yellow]")
                for e in found.scout_errors:
                    console.print(f"[dim]scout error: {e}[/dim]")
            return

        if not state.json_mode:
            _print_candidates(found.candidates, found)

        if dry_run:
            if state.json_mode:
                _emit(found.to_dict())
            return

        chosen = found.candidates
        needs_ok = review or not (state.yes or ctx.library.settings.auto_approve)
        if needs_ok and not state.json_mode:
            chosen = _prompt_selection(found.candidates)
            if not chosen:
                console.print("[dim]Nothing selected.[/dim]")
                return

        # The plan's tags go on everything this find files. An unread entry has
        # no agent-written tags of its own — these are the only handle `lit
        # search -t` will have on it until someone pays to read it.
        tags = list(tag) + [t for t in (found.plan.tags if found.plan else [])
                            if t not in tag]
        found.unread = not read
        found.added = add_candidates(ctx, chosen, force=force, extra_tags=tags,
                                     read=read)
        for res in found.added:
            if res.entry:
                link_references(ctx, res.entry)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({**found.to_dict(), "usage": ctx.usage.summary()})
        return

    ok = [r for r in found.added if r.ok]
    sources = ", ".join(filter(None, [
        f"{found.from_references} from references" if found.from_references else "",
        f"{found.from_search} from indexed search" if found.from_search else "",
        f"{found.from_scouts} from web scouts" if found.from_scouts else "",
    ])) or "no sources reached"
    console.print(
        f"\n[bold]{len(ok)}[/bold] added of {len(found.added)} attempted "
        f"(pool of {found.pool_size}: {sources})"
    )

    unread = [r for r in ok if r.entry and r.entry.is_unread]
    if unread:
        console.print(
            f"[cyan]{len(unread)} filed unread[/cyan] — metadata and abstract "
            f"only, no summaries yet."
            f"\n[dim]Read the ones worth it: `lit read {unread[0].entry.key}`, "
            f"or `lit read --all` for every one of them.[/dim]"
        )
    unverified = [r for r in ok if r.entry and not r.entry.is_verified
                  and not r.entry.is_unread]
    if unverified:
        console.print(
            f"[yellow]{len(unverified)} added as UNVERIFIED[/yellow] (no full text): "
            + ", ".join(r.entry.key for r in unverified)
            + f"\n[dim]Drop their PDFs in {ctx.library.inbox_dir} and run "
              f"`lit inbox` to fill them in.[/dim]"
        )
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


def _print_candidates(candidates, found) -> None:
    t = Table(title=f"{len(candidates)} candidates", header_style="bold", expand=True)
    t.add_column("#", justify="right", no_wrap=True)
    t.add_column("title", ratio=3)
    t.add_column("yr", justify="right", no_wrap=True)
    t.add_column("cites", justify="right", no_wrap=True)
    t.add_column("match", justify="right", no_wrap=True)
    t.add_column("imp", justify="right", no_wrap=True)
    t.add_column("src", no_wrap=True)
    t.add_column("why", ratio=2)
    for i, c in enumerate(candidates, 1):
        src = {"references": "refs", "scout": "web", "both": "both"}.get(c.source, "?")
        if c.cocitations:
            src += f" ×{c.cocitations}"
        elif len(c.angles) > 1:
            src += f" ×{len(c.angles)}"
        t.add_row(
            str(i), c.title, str(c.year or "—"),
            f"{c.citation_count:,}" if c.citation_count is not None else "—",
            "—" if c.relevance is None else f"{c.relevance:.2f}",
            f"{c.importance:.2f}",
            src, c.why,
        )
    console.print(t)


def _prompt_selection(candidates):
    console.print(
        "\nAdd which? [cyan]all[/cyan] / [cyan]none[/cyan] / numbers like "
        "[cyan]1,3,5-7[/cyan]"
    )
    raw = typer.prompt("selection", default="all").strip().lower()
    if raw in ("none", "n", "q", ""):
        return []
    if raw in ("all", "a", "*"):
        return candidates
    picked: list = []
    for part in raw.replace(" ", "").split(","):
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
            except ValueError:
                continue
            picked += [c for i, c in enumerate(candidates, 1) if lo <= i <= hi]
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if 1 <= i <= len(candidates):
                picked.append(candidates[i - 1])
    seen, out = set(), []
    for c in picked:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


@app.command("read")
def cmd_read(
    keys: list[str] = typer.Argument(
        None, help="Entry keys to read. Omit and use --all for the whole backlog."),
    all_unread: bool = typer.Option(
        False, "--all", help="Read every entry that has no summary yet."),
    limit: int = typer.Option(
        0, "-n", "--limit", help="With --all, stop after this many (best levels first)."),
    level: Optional[str] = typer.Option(
        None, "--level", help="With --all, only entries at this level or better."),
    tag: Optional[str] = typer.Option(
        None, "--tag", help="With --all, only entries carrying this tag."),
):
    """Read papers in full and write their summaries.

    This is the expensive step, kept separate on purpose: `lit find` files
    papers from their metadata for almost nothing, and you spend a reader agent
    only on the ones that earned it.
    """
    ctx = _ctx()
    try:
        lib = ctx.library
        wanted: list[Entry] = []
        missing: list[str] = []
        if keys:
            for k in keys:
                entry = lib.get(k)
                if entry is None:
                    missing.append(k)
                else:
                    wanted.append(entry)
        elif all_unread:
            wanted = [e for e in lib.entries() if e.needs_read]
            if level:
                wanted = [e for e in wanted if level_rank(e.level) <= level_rank(level)]
            if tag:
                t = tag.strip().lower()
                wanted = [e for e in wanted if t in [x.lower() for x in e.tags]]
            # Best-ranked first, so a truncated run reads the papers that matter.
            wanted.sort(key=lambda e: (level_rank(e.level), -(e.citation_count or 0)))
            if limit > 0:
                wanted = wanted[:limit]
        else:
            _fail("give one or more entry keys, or --all to read the whole backlog.")

        if missing:
            _fail(f"no entry with key(s): {', '.join(missing)}")
        if not wanted:
            msg = "Nothing to read — every entry already has a summary."
            if state.json_mode:
                _emit({"read": [], "message": msg})
            else:
                console.print(f"[dim]{msg}[/dim]")
            return

        results = read_entries(ctx, wanted)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({"read": [r.to_dict() for r in results],
               "usage": ctx.usage.summary()})
        return

    ok = [r for r in results if r.ok]
    console.print(f"\n[bold]{len(ok)}[/bold] of {len(results)} read and summarized")
    for r in results:
        if not r.ok:
            console.print(f"[yellow]{r.status}:[/yellow] {r.message}")
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


@app.command("reread")
def cmd_reread(
    key: str = typer.Argument(..., help="Entry key to re-read."),
    pdf: Optional[Path] = typer.Option(None, "--pdf", exists=True, dir_okay=False),
):
    """Re-fetch and re-read an entry (e.g. after supplying a PDF)."""
    ctx = _ctx()
    try:
        res = reread_action(ctx, key, local_pdf=pdf)
    finally:
        ctx.close()
    if state.json_mode:
        _emit({**res.to_dict(), "usage": ctx.usage.summary()})
        raise typer.Exit(0 if res.ok else 1)
    # A failed re-read can still have paid for a fetch and a partial read, so the
    # footer goes out before the error exit too.
    if not res.ok:
        console.print(f"[red]{res.status}:[/red] {res.message}")
        if footer := _usage_footer(ctx):
            console.print(f"[dim]{footer}[/dim]")
        raise typer.Exit(1)
    console.print(entry_panel(res.entry))
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


@app.command("refresh")
def cmd_refresh(
    keys: list[str] = typer.Argument(None, help="Entry keys. Omit for all entries."),
    no_relevel: bool = typer.Option(
        False, "--no-relevel", help="Leave levels alone even if metrics changed."),
):
    """Re-fetch citation counts, references and venues. No LLM, no re-reading."""
    ctx = _ctx()
    try:
        # A variadic Argument with no value arrives as None, not [].
        results = refresh_entries(ctx, list(keys or []) or None, relevel=not no_relevel)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({"refreshed": [r.to_dict() for r in results]})
        return
    if not results:
        console.print("[dim]Nothing to refresh.[/dim]")
        return
    changed = [r for r in results if r.updated]
    failed = [r for r in results if r.error]
    console.print(
        f"\n[bold]{len(changed)}[/bold] of {len(results)} entries updated"
        + (f", {len(failed)} could not be resolved" if failed else "")
    )
    if changed:
        console.print("[dim]Summaries and notes were not touched. "
                      "To re-read a paper, use `lit reread <key>`.[/dim]")


@app.command("inbox")
def cmd_inbox(
    keep: bool = typer.Option(False, "--keep", help="Leave the PDFs in the inbox."),
    add: list[Path] = typer.Option(
        [], "--add", exists=True, dir_okay=False,
        help="Process these PDFs directly instead of scanning the inbox."),
):
    """Read PDFs you dropped in pdfs/inbox/ and fill in their entries."""
    ctx = _ctx()
    try:
        res = process_inbox(ctx, keep=keep, only=list(add) or None)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({**res.to_dict(), "usage": ctx.usage.summary()})
        return
    if not res.items:
        console.print(
            f"[dim]No PDFs in {ctx.library.inbox_dir}.[/dim]\n"
            "Drop paywalled PDFs there and re-run, or use "
            "[cyan]lit add \"<title>\" --pdf <file>[/cyan]."
        )
        return
    ok = sum(1 for i in res.items if i.result and i.result.ok)
    console.print(f"\n[bold]{ok}[/bold] of {len(res.items)} PDFs processed")
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


# --------------------------------------------------------------------------
# Browsing
# --------------------------------------------------------------------------

@app.command("ls")
def cmd_ls(
    level: Optional[str] = typer.Option(None, "--level", help="Minimum level, e.g. A."),
    tag: Optional[str] = typer.Option(None, "--tag", "-t"),
    status: Optional[str] = typer.Option(
        None, "--status", help="verified | unread | UNVERIFIED"),
    year_from: Optional[int] = typer.Option(None, "--from"),
    year_to: Optional[int] = typer.Option(None, "--to"),
    sort: str = typer.Option("level", "--sort",
                             help="level | year | title | cites | added"),
    limit: int = typer.Option(200, "-n", "--limit"),
):
    """List entries in the library."""
    lib = _lib()
    entries = lib.entries()

    if level:
        entries = [e for e in entries if level_rank(e.level) <= level_rank(level)]
    if tag:
        entries = [e for e in entries if tag.lower() in [t.lower() for t in e.tags]]
    if status:
        s = status.lower()
        want = (STATUS_VERIFIED if s.startswith(("v", "r"))
                else STATUS_UNREAD if s.startswith("un") and "read" in s
                else "UNVERIFIED")
        entries = [e for e in entries if e.status == want]
    if year_from:
        entries = [e for e in entries if (e.year or 0) >= year_from]
    if year_to:
        entries = [e for e in entries if (e.year or 9999) <= year_to]

    keyfn = {
        "level": lambda e: (level_rank(e.level), -(e.year or 0)),
        "year": lambda e: -(e.year or 0),
        "title": lambda e: e.title.lower(),
        "cites": lambda e: -(e.citation_count or 0),
        "added": lambda e: e.added,
    }.get(sort, lambda e: (level_rank(e.level), -(e.year or 0)))
    entries.sort(key=keyfn)
    entries = entries[:limit]

    if state.json_mode:
        _emit({"library": lib.name, "count": len(entries),
               "entries": [_entry_json(e) for e in entries]})
        return
    if not entries:
        console.print("[dim]No entries match.[/dim]")
        return
    console.print(entries_table(entries, title=f"{lib.name} — {len(entries)} entries"))


@app.command("show")
def cmd_show(
    key: str = typer.Argument(..., help="Entry key."),
    full: bool = typer.Option(True, "--full/--brief",
                              help="Include the section-by-section summary."),
    raw: bool = typer.Option(False, "--raw", help="Print the raw Markdown file."),
):
    """Show one entry."""
    lib = _lib()
    entry = lib.get(key)
    if entry is None:
        _fail(f"no entry with key {key!r}")
    if raw:
        console.print(lib.entry_path(key).read_text(encoding="utf-8"))
        return
    if state.json_mode:
        _emit(_entry_json(entry, full=True))
        return
    console.print(entry_panel(entry, full=full))


@app.command("abstract")
def cmd_abstract(
    keys: list[str] = typer.Argument(..., help="Entry keys."),
):
    """Print the publisher's abstract for one or more entries."""
    lib = _lib()
    entries = []
    for k in keys:
        e = lib.get(k)
        if e is None:
            _fail(f"no entry with key {k!r}")
        entries.append(e)

    if state.json_mode:
        _emit([{"key": e.key, "title": e.title, "abstract": e.abstract}
               for e in entries])
        return

    blocks = []
    for e in entries:
        body = e.abstract.strip() or f"(none on record — try `lit refresh {e.key}`)"
        # Only label the blocks when there is more than one to tell apart.
        blocks.append(f"{e.key} — {e.title}\n\n{body}" if len(entries) > 1 else body)
    # Plain print, not rich: this output is meant to be piped.
    print("\n\n".join(blocks))


@app.command("note")
def cmd_note(
    key: str = typer.Argument(..., help="Entry key."),
    text: Optional[str] = typer.Argument(None, help="Note text. Omit to open $EDITOR."),
    append: bool = typer.Option(True, "--append/--replace",
                                help="Append to your existing notes, or replace them."),
):
    """Edit your own notes on an entry. The agent never writes here."""
    lib = _lib()
    path = lib.entry_path(key)
    if not path.exists():
        _fail(f"no entry with key {key!r}")
    entry = entryfile.load(path)

    if text is None:
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        tmp = lib.path / f".note-{key}.md"
        tmp.write_text(entry.notes, encoding="utf-8")
        try:
            subprocess.call([editor, str(tmp)])
            new = tmp.read_text(encoding="utf-8").strip()
        finally:
            tmp.unlink(missing_ok=True)
    else:
        new = (f"{entry.notes}\n\n{text}".strip() if append and entry.notes.strip()
               else text.strip())

    entryfile.set_notes(path, new)
    if state.json_mode:
        _emit({"key": key, "notes": new})
    else:
        console.print(f"[green]saved[/green] notes on [cyan]{key}[/cyan]")


@app.command("delete")
@app.command("rm", hidden=True)
def cmd_delete(
    keys: list[str] = typer.Argument(..., help="Entry keys to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without asking."),
):
    """Delete entries: the record, its PDF and its cached text. Also `lit rm`.

    Nothing is deleted until every key resolves, so one typo in a batch cannot
    take the rest of it with it. Your notes are part of the record and go too,
    which is why an entry carrying them is called out before the prompt.
    """
    lib = _lib()
    entries: list[Entry] = []
    missing: list[str] = []
    for k in keys:
        entry = lib.get(k)
        if entry is None:
            missing.append(k)
        else:
            entries.append(entry)
    if missing:
        _fail(f"no entry with key(s): {', '.join(missing)}")

    noted = [e for e in entries if e.notes.strip()]
    if noted and not state.json_mode:
        console.print(
            "[yellow]Your notes will go too:[/yellow] "
            + ", ".join(e.key for e in noted)
        )
    # --json means nobody is at the terminal to answer, so it does not prompt.
    if not (yes or state.yes or state.json_mode):
        for e in entries:
            console.print(f"  [cyan]{e.key}[/cyan]  {e.title[:70]}")
        what = "1 entry" if len(entries) == 1 else f"{len(entries)} entries"
        if not typer.confirm(f"Delete {what}?"):
            raise typer.Exit(0)

    deleted = [e.key for e in entries if lib.delete_entry(e.key)]
    if state.json_mode:
        _emit({"deleted": deleted})
    else:
        console.print(f"[green]deleted[/green] {', '.join(deleted)}")


@app.command("reindex")
def cmd_reindex(
    rebuild: bool = typer.Option(False, "--rebuild",
                                 help="Discard the index and build it from scratch."),
):
    """Rebuild the search index from the entry files."""
    lib = _lib()
    if rebuild:
        lib.index_path.unlink(missing_ok=True)
    with Store(lib) as store:
        n = store.sync(force=rebuild)
    if state.json_mode:
        _emit({"reindexed": n, "entries": len(lib)})
    else:
        console.print(f"Indexed [bold]{n}[/bold] changed entries "
                      f"({len(lib)} total).")


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------

@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="What you're looking for."),
    limit: int = typer.Option(10, "-n", "--limit"),
    level: Optional[str] = typer.Option(None, "--level"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t"),
    year_from: Optional[int] = typer.Option(None, "--from"),
    year_to: Optional[int] = typer.Option(None, "--to"),
    raw: bool = typer.Option(False, "--raw",
                             help="Skip LLM ranking; return bm25 order only."),
):
    """Find in-library sources for a query, with a short answer from summaries."""
    ctx = _ctx()
    try:
        res = search_action(ctx, query, limit=limit, level=level, tag=tag,
                            year_from=year_from, year_to=year_to, rank=not raw)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({**res.to_dict(), "usage": ctx.usage.summary()})
        return
    if res.answer:
        console.print(Panel(res.answer, title="Short answer", border_style="cyan"))
    # Ranking can run and still return nothing relevant, so the empty case has
    # usually been paid for. Every path falls through to the footer.
    if not res.matches:
        console.print("[dim]No matching entries.[/dim]")
    else:
        console.print(entries_table(
            [m.entry for m in res.matches],
            show_why={m.entry.key: m.why for m in res.matches},
        ))
        if res.gaps:
            console.print("\n[bold]Gaps in the library[/bold]")
            for g in res.gaps:
                console.print(f"  · {g}")
            console.print(f'[dim]Fill them:  lit find "{query}"[/dim]')
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


@app.command("ask")
def cmd_ask(
    question: str = typer.Argument(..., help="The question to answer."),
    read: int = typer.Option(5, "-r", "--read",
                             help="How many papers to open and read in full."),
    expand: int = typer.Option(
        0, "-e", "--expand",
        help="Also pull in up to N works cited by those papers but not yet in "
             "the library. They are quoted from like any other source, and "
             "filed unread — `lit read <key>` summarizes one later."),
    level: Optional[str] = typer.Option(None, "--level"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t"),
    save: Optional[Path] = typer.Option(None, "--save",
                                        help="Write the answer to a Markdown file."),
):
    """Answer a question by reading the actual papers, with quotes and a bibliography."""
    ctx = _ctx()
    try:
        res = ask_action(ctx, question, read=read, expand=expand, level=level, tag=tag)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({**res.to_dict(), "usage": ctx.usage.summary()})
        return

    print_answer(console, question=question, answer=res.answer,
                 bibliography=res.bibliography(), footer=_usage_footer(ctx))
    if res.pulled:
        console.print(f"[dim]Pulled into the library: {', '.join(res.pulled)}[/dim]")
    if res.unreadable:
        console.print(f"[yellow]Could not read: {', '.join(res.unreadable)}[/yellow]")

    if save:
        save.write_text(_answer_markdown(res), encoding="utf-8")
        console.print(f"[green]saved[/green] {save}")


def _answer_markdown(res) -> str:
    lines = [f"# {res.question}", "", res.answer, "", "## Bibliography", ""]
    for b in res.bibliography():
        authors = ", ".join(b.get("authors") or [])
        ident = (f"https://doi.org/{b['doi']}" if b.get("doi")
                 else f"https://arxiv.org/abs/{b['arxiv_id']}" if b.get("arxiv_id")
                 else b.get("url") or "")
        lines.append(f"- **[{b['key']}]** {authors}. *{b['title']}*. "
                     f"{b.get('venue') or ''} {b.get('year') or ''}. {ident}".strip())
    return "\n".join(lines) + "\n"


@app.command("claim")
def cmd_claim(
    claim: str = typer.Argument(..., help="The claim to trace to its origin."),
    start: Optional[str] = typer.Option(
        None, "--start", help="Paper to start from (key, DOI, arXiv id or title)."),
    max_hops: int = typer.Option(5, "--max-hops"),
    add: bool = typer.Option(False, "--add",
                             help="Add each paper in the chain to the library."),
):
    """Trace a claim back through citations to the paper that originates it."""
    ctx = _ctx()
    try:
        if start:
            entry = ctx.library.get(start)
            if entry is not None:
                start = entry.doi or entry.arxiv_id or entry.title
        res = trace_claim(ctx, claim, start=start, max_hops=max_hops,
                          add_to_library=add)
    finally:
        ctx.close()

    if state.json_mode:
        _emit({**res.to_dict(), "usage": ctx.usage.summary()})
        return

    if not res.chain:
        console.print("[yellow]Could not trace this claim.[/yellow]")
        for n in res.notes:
            console.print(f"  [dim]{n}[/dim]")
        raise typer.Exit(1)

    console.print(chain_tree(res.chain, claim))
    console.print(f"\n[bold]Chain:[/bold] {res.chain_str()}")
    if res.origin:
        marker = "[green]origin confirmed[/green]" if res.complete else \
                 "[yellow]earliest source reached (not confirmed as origin)[/yellow]"
        console.print(f"{marker}: [bold]{res.origin.title}[/bold] "
                      f"({res.origin.year or 'n.d.'})")
    for n in res.notes:
        console.print(f"[dim]note: {n}[/dim]")
    if footer := _usage_footer(ctx):
        console.print(f"[dim]{footer}[/dim]")


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

@app.command("cite")
def cmd_cite(
    keys: list[str] = typer.Argument(None, help="Entry keys. Omit for the whole library."),
    fmt: str = typer.Option("bibtex", "--format", "-f",
                            help="bibtex | markdown | json | keys"),
    level: Optional[str] = typer.Option(None, "--level"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t"),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Write to a file."),
):
    """Emit citations for entries — BibTeX for a paper, Markdown for notes."""
    lib = _lib()
    entries = [e for e in (lib.get(k) for k in (keys or [])) if e] if keys \
        else lib.entries()
    if keys:
        missing = [k for k in keys if lib.get(k) is None]
        if missing:
            _fail(f"no entry with key(s): {', '.join(missing)}")
    if level:
        entries = [e for e in entries if level_rank(e.level) <= level_rank(level)]
    if tag:
        entries = [e for e in entries if tag.lower() in [t.lower() for t in e.tags]]
    entries.sort(key=lambda e: (e.authors[0].split()[-1].lower() if e.authors else "",
                                e.year or 0))

    if fmt == "bibtex":
        text = "\n\n".join(e.bibtex or f"% no bibtex for {e.key}" for e in entries)
    elif fmt == "markdown":
        text = "\n".join(
            f"- **[{e.key}]** {', '.join(e.authors)}. *{e.title}*. "
            f"{e.venue or ''} {e.year or ''}."
            + (f" https://doi.org/{e.doi}" if e.doi
               else f" https://arxiv.org/abs/{e.arxiv_id}" if e.arxiv_id else "")
            for e in entries
        )
    elif fmt == "keys":
        text = "\n".join(e.key for e in entries)
    elif fmt == "json":
        text = jsonlib.dumps([_entry_json(e) for e in entries], indent=2, default=str)
    else:
        _fail("--format must be one of: bibtex, markdown, json, keys")
        return

    if out:
        out.write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]wrote[/green] {len(entries)} entries to {out}")
    else:
        # Plain print, not rich: this output is meant to be piped.
        print(text)


@app.command("export")
def cmd_export(
    out: Optional[Path] = typer.Option(None, "-o", "--out",
                                       help="Destination .litlib file or directory."),
    with_pdfs: bool = typer.Option(False, "--with-pdfs",
                                   help="Include downloaded PDFs (much larger)."),
    key: list[str] = typer.Option([], "--key", help="Export only these entries."),
):
    """Export this library as a portable .litlib bundle to share with others."""
    lib = _lib()
    dest = out or Path.cwd() / f"{lib.name}.litlib"
    res = bundle.export_library(lib, dest, with_pdfs=with_pdfs, keys=list(key) or None)
    if state.json_mode:
        _emit(res.to_dict())
        return
    size = res.bytes / 1024
    unit = "KB" if size < 1024 else "MB"
    shown = size if size < 1024 else size / 1024
    console.print(
        f"[green]exported[/green] {res.entries} entries"
        + (f" and {res.pdfs} PDFs" if res.pdfs else "")
        + f" → {res.path} ({shown:.1f} {unit})"
    )


@app.command("import")
def cmd_import(
    path: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="A .litlib bundle."),
    into: Optional[str] = typer.Option(
        None, "--into", help="Merge into this existing library instead of "
                             "creating one."),
    name: Optional[str] = typer.Option(None, "--name",
                                       help="Name for the new library."),
    strategy: str = typer.Option(
        "newer", "--strategy",
        help="On conflict: skip | replace | newer. Your notes always survive."),
    no_pdfs: bool = typer.Option(False, "--no-pdfs"),
    show: bool = typer.Option(False, "--show", help="Describe the bundle and stop."),
):
    """Import a .litlib bundle from a colleague."""
    cfg = state.cfg
    try:
        manifest = bundle.peek(path)
        if show:
            if state.json_mode:
                _emit(manifest)
            else:
                console.print_json(jsonlib.dumps(manifest, indent=2))
            return
        target = resolve_library(cfg, into) if into else None
        res = bundle.import_bundle(
            path, root=cfg.root_path, into=target, name=name,
            strategy=strategy, with_pdfs=not no_pdfs,
        )
    except LibraryError as exc:
        _fail(str(exc))
        return

    if state.json_mode:
        _emit(res.to_dict())
        return
    console.print(
        f"[green]imported[/green] into [cyan]{res.library}[/cyan]: "
        f"{len(res.added)} added, {len(res.updated)} updated, "
        f"{len(res.skipped)} skipped"
        + (f", {res.pdfs} PDFs" if res.pdfs else "")
    )
    if res.notes_preserved:
        console.print(
            f"[yellow]Your notes were kept[/yellow] on "
            f"{len(res.notes_preserved)} entries; theirs were appended below yours."
        )
    console.print("[dim]Run `lit reindex` if search looks stale.[/dim]")


# --------------------------------------------------------------------------
# Agent integration + TUI
# --------------------------------------------------------------------------

@app.command("skill")
def cmd_skill(
    action: str = typer.Argument("install", help="install | path | show"),
    project: bool = typer.Option(
        False, "--project", help="Install into ./.claude/skills instead of "
                                 "~/.claude/skills."),
):
    """Install the Claude Code skill so agents can use your libraries."""
    from .skillfile import SKILL_BODY, install_skill, skill_source

    if action == "show":
        print(SKILL_BODY)
        return
    if action == "path":
        console.print(str(skill_source()))
        return
    if action != "install":
        _fail("usage: lit skill [install|path|show]")

    dest = install_skill(project=project)
    if state.json_mode:
        _emit({"installed": str(dest)})
        return
    console.print(Panel(
        f"Installed to [bold]{dest}[/bold]\n\n"
        "Claude Code can now use your libraries. Try asking it:\n"
        '  [cyan]"what does my library say about X?"[/cyan]\n'
        '  [cyan]"add the Transformer paper to my llm-benchmarks library"[/cyan]\n'
        '  [cyan]"find me a source for the claim that ..."[/cyan]',
        title="Skill installed", border_style="green",
    ))


@app.command("browse")
def cmd_browse():
    """Open the interactive library browser (optional; everything is also a subcommand)."""
    try:
        from .tui import run_browser
    except ImportError as exc:  # pragma: no cover
        _fail(f"the TUI needs the 'textual' package: {exc}")
        return
    run_browser(_lib(), state.cfg)


@app.command("path")
def cmd_path(
    key: Optional[str] = typer.Argument(None, help="Entry key. Omit for the library."),
):
    """Print a filesystem path — useful for piping into other tools."""
    lib = _lib()
    if key is None:
        print(lib.path)
        return
    if not lib.has(key):
        _fail(f"no entry with key {key!r}")
    print(lib.entry_path(key))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _entry_json(e: Entry, full: bool = False) -> dict:
    data = {
        "key": e.key, "title": e.title, "authors": e.authors, "year": e.year,
        "venue": e.venue, "type": e.type, "doi": e.doi, "arxiv_id": e.arxiv_id,
        "url": e.url, "code_url": e.code_url, "status": e.status, "level": e.level,
        "level_reason": e.level_reason, "one_liner": e.one_liner, "tags": e.tags,
        "citation_count": e.citation_count, "citation": e.citation(),
        "key_findings": e.key_findings, "notes": e.notes,
        "added": e.added, "updated": e.updated,
        "pages": e.pages, "pages_read": e.pages_read,
        "partial_read": e.is_partial_read, "coverage": e.coverage(),
    }
    if full:
        # Long prose, like the sections below it: kept out of list output.
        data["abstract"] = e.abstract
        data["sections"] = [{"name": s.name, "summary": s.summary} for s in e.sections]
        data["references"] = [
            {"title": r.title, "year": r.year, "doi": r.doi,
             "arxiv_id": r.arxiv_id, "key": r.key} for r in e.references
        ]
        data["bibtex"] = e.bibtex
        data["read_source"] = e.read_source
    return data


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        err_console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
