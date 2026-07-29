"""Global configuration (`~/.config/lit/config.toml`) and its defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import tomli_w
from platformdirs import user_config_dir

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 path
    import tomli as tomllib  # type: ignore[no-redef]


def config_dir() -> Path:
    """Where config lives. Resolved per call, not at import.

    LIT_CONFIG_DIR must be honoured whenever it is set — including when a test
    or a wrapper sets it after this module was first imported. Freezing this at
    import time meant a caller who set the variable later still wrote to the
    real user config.
    """
    return Path(os.environ.get("LIT_CONFIG_DIR") or user_config_dir("lit"))


def config_path() -> Path:
    return config_dir() / "config.toml"


class _LazyPath:
    """`CONFIG_PATH` kept as a name for display, resolved on each use."""

    def __fspath__(self) -> str:
        return str(config_path())

    def __str__(self) -> str:
        return str(config_path())

    def __repr__(self) -> str:
        return repr(config_path())


CONFIG_PATH = _LazyPath()


# Per-role model defaults. Work is split so the cheap, high-volume steps run on
# a cheap model and only the steps that need real comprehension pay for it.
#
#   plan      — reading a plain-language query into index search parameters
#   scout     — web search for candidate papers (many calls, shallow judgement)
#   filter    — filtering library hits by title/summary
#   rank      — scoring a pool of candidate papers for relevance to a query
#   reader    — reading a full paper and writing its summaries (the quality step)
#   analyst   — extracting quoted evidence, tracing a claim through a paper
#   synthesis — writing the final cited answer
DEFAULT_ROLE_MODELS: dict[str, str] = {
    "plan": "haiku",
    "scout": "haiku",
    "filter": "haiku",
    "rank": "haiku",
    "reader": "sonnet",
    "analyst": "sonnet",
    "synthesis": "sonnet",
}

# Per-role reasoning effort, and the reason `lit` always states one.
#
# The `claude` CLI inherits `effortLevel` from the user's own settings.json when
# no `--effort` is given. That setting is chosen for interactive work, and on a
# batch of paper reads it is silently expensive: at `xhigh` the reader spent
# 8,010 output tokens on a paper that `medium` recorded just as faithfully in
# 1,921 — the same headline finding, five top-level sections instead of ten
# sub-sections, for 36% less. None of these calls is a reasoning problem. They
# read a document and fill in fields, so `lit` names the effort it wants rather
# than inheriting one that was never chosen for this.
DEFAULT_ROLE_EFFORT: dict[str, str] = {
    "plan": "low",       # extracting search parameters from one sentence
    "scout": "low",      # many shallow calls
    "filter": "low",
    "rank": "low",
    "reader": "medium",  # the quality step, but still comprehension not reasoning
    "analyst": "medium",  # quotes must be exact, so not `low`
    "synthesis": "medium",
}

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class LLMConfig:
    # Fallback for any role not named in `models`. Accepts an alias ("haiku",
    # "sonnet", "opus") or a full model id.
    model: str = "sonnet"
    # Per-role overrides; see DEFAULT_ROLE_MODELS above.
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_MODELS))
    # Set by `--model` to force one model for every role in a single command.
    override_model: str = ""
    # Fallback effort for any role not named in `efforts`. Set to "" to pass no
    # --effort at all and go back to inheriting the CLI's own setting.
    effort: str = "medium"
    # Per-role overrides; see DEFAULT_ROLE_EFFORT above.
    efforts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_EFFORT))
    # Set by `--effort` to force one level for every role in a single command.
    override_effort: str = ""
    # Reading a full paper is the expensive step; keep a generous ceiling.
    timeout_s: int = 900
    # How many papers to read concurrently in `find` / `ask`.
    max_parallel: int = 4
    # How much paper text goes into one reading prompt. `lit add` and `lit read`
    # act on a paper you picked deliberately and get the generous budget;
    # `lit find --read` reads a batch at once, so it gets a tighter one.
    read_chars: int = 400_000
    find_read_chars: int = 150_000
    # Retries on malformed JSON or a non-zero exit.
    max_retries: int = 2
    # Extra args passed through to the claude CLI, e.g. ["--effort", "high"].
    extra_args: list[str] = field(default_factory=list)

    def model_for(self, role: str | None) -> str:
        if self.override_model:
            return self.override_model
        if role and self.models.get(role):
            return self.models[role]
        return self.model

    def effort_for(self, role: str | None) -> str:
        """The reasoning effort for a role, or "" to let the CLI decide."""
        if self.override_effort:
            return self.override_effort
        if role and self.efforts.get(role):
            return self.efforts[role]
        return self.effort


@dataclass
class FetchConfig:
    # Sent to Crossref and Unpaywall for polite-pool rate limits. Unpaywall
    # requires it; without one that resolver is skipped.
    email: str = ""
    # Optional Semantic Scholar key, and the most useful thing in this file to
    # set. S2 supplies the citation counts `find` ranks on and is the only free
    # index that counts citations to arXiv-only work properly. Unauthenticated
    # callers share one pool with the rest of the world and are throttled to
    # roughly a request a second, so on a busy day facets come back empty and a
    # search silently loses its sense of which papers are landmark. A key is
    # free: https://www.semanticscholar.org/product/api#api-key-form
    semantic_scholar_api_key: str = ""
    timeout_s: int = 60
    max_pdf_mb: int = 60

    # ---- read budget for long documents -------------------------------------
    # A paper is read end to end. A book, thesis or long report is not: feeding
    # 400 pages to a model is slow and expensive, and the result is no better
    # than a well-chosen sample.
    #
    # Anything longer than `long_document_pages` counts as a long source, and
    # only `max_read_pages` of it are extracted — front matter (table of
    # contents, preface, introduction), an even spread through the body, and the
    # closing pages. Entries read this way are marked as partial reads so a
    # sampled summary is never presented as a complete one.
    long_document_pages: int = 50
    max_read_pages: int = 10

    # ---- last-resort full-text resolver -------------------------------------
    # Tried only after every open-access route (arXiv, PMC, Unpaywall, indexed
    # OA links, publisher OA, local PDFs) has failed. Both ship empty; nothing is
    # contacted unless you configure it.
    #
    # `fallback_url_template` takes {doi}, and may point at either a PDF or a
    # page with the PDF embedded — the embedded link is extracted automatically:
    #     lit config set fetch.fallback_url_template "https://<host>/{doi}"
    #
    # `fallback_cmd` runs an arbitrary command you supply; {doi} and {out} are
    # substituted, and the command must leave a PDF at {out}:
    #     lit config set fetch.fallback_cmd "my-fetcher {doi} -o {out}"
    #
    # Whether a given source is lawful to use depends on where you are, and
    # that judgement is yours — see the README.
    fallback_url_template: str = ""
    fallback_cmd: str = ""


@dataclass
class Config:
    # Where libraries live. Each library is a directory beneath this root.
    root: str = str(Path.home() / "lit")
    default_library: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)

    @property
    def root_path(self) -> Path:
        return Path(self.root).expanduser()

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        path.write_bytes(tomli_w.dumps(_drop_none(data)).encode("utf-8"))
        return path


def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path else config_path()
    cfg = Config()
    if path.exists():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        cfg.root = str(raw.get("root") or cfg.root)
        cfg.default_library = str(raw.get("default_library") or "")
        llm_raw = dict(raw.get("llm") or {})
        # `models` and `efforts` are merged into the defaults, so naming one
        # role doesn't wipe the rest.
        role_raw = llm_raw.pop("models", None) or {}
        effort_raw = llm_raw.pop("efforts", None) or {}
        _apply(cfg.llm, llm_raw)
        cfg.llm.models = {**DEFAULT_ROLE_MODELS, **{
            str(k): str(v) for k, v in role_raw.items() if v
        }}
        cfg.llm.efforts = {**DEFAULT_ROLE_EFFORT, **{
            str(k): str(v) for k, v in effort_raw.items() if v
        }}
        _apply(cfg.fetch, raw.get("fetch") or {})

    # Environment overrides win; handy for CI and for agent-driven runs.
    if os.environ.get("LIT_ROOT"):
        cfg.root = os.environ["LIT_ROOT"]
    if os.environ.get("LIT_LIBRARY"):
        cfg.default_library = os.environ["LIT_LIBRARY"]
    if os.environ.get("LIT_MODEL"):
        cfg.llm.model = os.environ["LIT_MODEL"]
    if os.environ.get("LIT_EFFORT"):
        cfg.llm.effort = os.environ["LIT_EFFORT"]
    for role in DEFAULT_ROLE_MODELS:
        env_key = f"LIT_MODEL_{role.upper()}"
        if os.environ.get(env_key):
            cfg.llm.models[role] = os.environ[env_key]
        effort_key = f"LIT_EFFORT_{role.upper()}"
        if os.environ.get(effort_key):
            cfg.llm.efforts[role] = os.environ[effort_key]
    if os.environ.get("LIT_EMAIL"):
        cfg.fetch.email = os.environ["LIT_EMAIL"]
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        cfg.fetch.semantic_scholar_api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    return cfg


def _apply(obj: Any, raw: dict[str, Any]) -> None:
    for k, v in raw.items():
        if hasattr(obj, k) and v is not None:
            setattr(obj, k, v)


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = _drop_none(v) if isinstance(v, dict) else v
    return out
