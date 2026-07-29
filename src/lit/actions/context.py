"""Shared per-command context: config, library, HTTP, LLM, console."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich.console import Console

from ..config import Config
from ..fetch.http import HttpClient
from ..library import Library
from ..llm import ClaudeCLI, Usage
from ..models import normalize_arxiv, normalize_doi

# How much paper text one reading prompt may carry when the config does not say.
# The live value is `llm.read_chars`; this is only the floor under a config that
# predates the setting.
DEFAULT_READ_CHARS = 400_000


@dataclass
class Ctx:
    cfg: Config
    library: Library
    console: Console
    json_mode: bool = False
    verbose: bool = False
    yes: bool = False  # pre-approve everything (autonomous mode)
    usage: Usage = field(default_factory=Usage)
    _http: HttpClient | None = field(default=None, repr=False)
    _llm: ClaudeCLI | None = field(default=None, repr=False)

    @property
    def http(self) -> HttpClient:
        if self._http is None:
            self._http = HttpClient(self.cfg.fetch)
        return self._http

    @property
    def llm(self) -> ClaudeCLI:
        if self._llm is None:
            self._llm = ClaudeCLI(
                self.cfg.llm,
                # Run the child agent inside the library so any CLAUDE.md it
                # picks up is the library's, not the user's current repo.
                cwd=str(self.library.path),
                usage=self.usage,
                verbose=self.verbose,
            )
        return self._llm

    @property
    def workers(self) -> int:
        return max(1, self.cfg.llm.max_parallel)

    @property
    def read_budget(self) -> int:
        """Chars of paper text one prompt may carry, from `llm.read_chars`.

        Every command that puts a full text in front of a model reads it from
        here. It lives on the context rather than in each action because the
        alternative — a module constant per action — is how `ask` and `claim`
        came to ignore the configured budget entirely.
        """
        return getattr(self.cfg.llm, "read_chars", DEFAULT_READ_CHARS)

    def log(self, message: str) -> None:
        """Progress output. Suppressed in --json mode so stdout stays parseable."""
        if not self.json_mode:
            self.console.print(message)

    def vlog(self, message: str) -> None:
        if self.verbose and not self.json_mode:
            self.console.print(f"[dim]{message}[/dim]")

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


@dataclass
class Target:
    """What the user asked for, after sniffing identifiers out of the string."""

    doi: str | None = None
    arxiv_id: str | None = None
    title: str | None = None
    url: str | None = None

    def describe(self) -> str:
        return self.title or self.doi or self.arxiv_id or self.url or "?"


def parse_target(raw: str) -> Target:
    """Work out whether the user gave a DOI, an arXiv id, a URL, or a title."""
    s = (raw or "").strip().strip("<>")
    if not s:
        return Target()

    if s.lower().startswith("arxiv:"):
        return Target(arxiv_id=normalize_arxiv(s[6:]))
    if s.lower().startswith("doi:"):
        return Target(doi=normalize_doi(s[4:]))

    doi = normalize_doi(s)
    if doi and (s.lower().startswith(("10.", "http")) or "doi.org" in s.lower()):
        return Target(doi=doi, url=s if s.startswith("http") else None)

    if s.startswith("http"):
        arx = normalize_arxiv(s) if "arxiv.org" in s.lower() else None
        if arx:
            return Target(arxiv_id=arx, url=s)
        return Target(url=s, doi=doi)

    # A bare arXiv id, but only if the whole string is one — otherwise a title
    # containing a number like "GPT-4 1706" would be misread as an id.
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s):
        return Target(arxiv_id=normalize_arxiv(s))
    if re.fullmatch(r"[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?", s):
        return Target(arxiv_id=normalize_arxiv(s))

    return Target(title=s)
