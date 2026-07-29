"""LLM access via the `claude` CLI in headless mode.

Uses the user's existing Claude Code authentication — no API key needed. Two
call shapes:

* `run(..., tools=None)`  — a sealed single-turn text call. Every tool is
  disallowed, so the model can only transform the text it was given. This is
  what reads papers and writes summaries.
* `run(..., tools=[...])` — an agentic call with named tools pre-approved
  (WebSearch/WebFetch). Used for discovering candidate papers.

Long inputs (full paper text) go over stdin rather than argv, which has a hard
size limit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field

from .config import LLMConfig

# Everything the model must not touch during a sealed text-transformation call.
_ALL_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch",
    "Task", "TodoWrite", "NotebookEdit", "KillShell", "BashOutput",
]


class LLMError(RuntimeError):
    """Raised when the CLI fails or returns unusable output."""


@dataclass
class LLMResult:
    text: str
    cost_usd: float = 0.0
    model: str = ""
    role: str = ""


@dataclass
class Usage:
    """Running call/cost totals, shared across a command so we can report at the end."""

    calls: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: LLMResult) -> None:
        with self._lock:
            self.calls += 1
            self.cost_usd += r.cost_usd
            if r.model:
                self.by_model[r.model] = self.by_model.get(r.model, 0) + 1

    def summary(self) -> str:
        if not self.calls:
            return ""
        models = ", ".join(f"{n}× {m}" for m, n in sorted(self.by_model.items()))
        cost = f"${self.cost_usd:.2f}" if self.cost_usd else "cost not reported"
        return f"{self.calls} agent call(s) — {models} — {cost}"


class ClaudeCLI:
    """Thin wrapper around `claude -p --output-format json`."""

    def __init__(self, cfg: LLMConfig | None = None, *, cwd: str | None = None,
                 usage: Usage | None = None):
        self.cfg = cfg or LLMConfig()
        self.cwd = cwd
        self.usage = usage or Usage()
        self._exe = shutil.which("claude")
        # Set once the installed CLI turns out not to know `--effort`, so an
        # older Claude Code degrades to its own default instead of failing every
        # call. Checked by asking, not by parsing a version number.
        self._no_effort_flag = False

    @property
    def available(self) -> bool:
        return self._exe is not None

    def require(self) -> None:
        if not self.available:
            raise LLMError(
                "the `claude` CLI was not found on PATH. Install Claude Code "
                "(https://claude.com/claude-code) — lit uses it for all LLM calls."
            )

    # ---------------- core ----------------

    def run(
        self,
        prompt: str,
        *,
        stdin_text: str | None = None,
        system: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 2,
        model: str | None = None,
        role: str | None = None,
        timeout_s: int | None = None,
        effort: str | None = None,
    ) -> LLMResult:
        # Two turns even for sealed text calls: the model occasionally emits a
        # tool_use block anyway, and with a ceiling of 1 that ends the session
        # as an error and loses the paper. With headroom it gets the denial back
        # and answers in text on the next turn. No tool is reachable either way.
        self.require()
        chosen = model or self.cfg.model_for(role)
        level = effort if effort is not None else self.cfg.effort_for(role)
        cmd = [
            self._exe, "-p", prompt,
            "--output-format", "json",
            "--model", chosen,
            "--max-turns", str(max_turns),
            "--strict-mcp-config",
        ]
        # Stated, not inherited: without this the call picks up whatever
        # `effortLevel` the user set for their own interactive sessions, which
        # on a batch of paper reads costs several times the output tokens the
        # work needs. See DEFAULT_ROLE_EFFORT.
        if level and not self._no_effort_flag:
            cmd += ["--effort", level]
        if tools:
            cmd += ["--allowed-tools", " ".join(tools)]
        else:
            cmd += ["--disallowed-tools", " ".join(_ALL_TOOLS)]
        if system:
            cmd += ["--append-system-prompt", system]
        cmd += list(self.cfg.extra_args)

        env = dict(os.environ)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)  # don't inherit the parent session's identity

        try:
            proc = subprocess.run(
                cmd,
                input=(stdin_text or ""),
                capture_output=True,
                text=True,
                timeout=timeout_s or self.cfg.timeout_s,
                cwd=self.cwd,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(
                f"claude timed out after {timeout_s or self.cfg.timeout_s}s"
            ) from exc
        except OSError as exc:
            raise LLMError(f"could not run claude: {exc}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:600]
            # An older CLI has no --effort. Drop it for the rest of this run and
            # retry once, rather than failing every call over a tuning flag.
            if level and not self._no_effort_flag and _rejected_effort(detail):
                self._no_effort_flag = True
                return self.run(
                    prompt, stdin_text=stdin_text, system=system, tools=tools,
                    max_turns=max_turns, model=model, role=role,
                    timeout_s=timeout_s, effort=effort,
                )
            raise LLMError(f"claude exited {proc.returncode}: {detail}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"claude returned non-JSON output: {proc.stdout[:400]!r}"
            ) from exc

        if payload.get("is_error"):
            raise LLMError(
                f"claude reported an error ({payload.get('subtype')}): "
                f"{str(payload.get('result'))[:400]}"
            )

        result = LLMResult(
            text=str(payload.get("result") or ""),
            cost_usd=float(payload.get("total_cost_usd") or 0.0),
            model=chosen,
            role=role or "",
        )
        self.usage.add(result)
        return result

    def json(
        self,
        prompt: str,
        *,
        stdin_text: str | None = None,
        system: str | None = None,
        required: tuple[str, ...] = (),
        tools: list[str] | None = None,
        max_turns: int = 2,
        model: str | None = None,
        role: str | None = None,
        timeout_s: int | None = None,
        effort: str | None = None,
    ) -> dict:
        """Run a call that must produce a JSON object, retrying on bad output.

        Retries cover CLI-level failures as well as malformed replies. Reading a
        paper is minutes of work and a whole PDF fetch; a transient hiccup —
        an API blip, a session that ends on a stray tool_use — must not throw
        that away when a second attempt would succeed.

        A reply that arrived but would not parse gets a *repair* pass rather
        than a full retry: the model already read the paper and only formatted
        its answer wrong, so the next call carries the unusable reply instead of
        the document. Re-sending the paper would cost as much as the original
        read to be told the same thing in valid JSON. If the repair also fails,
        the attempt after it redoes the work in full.
        """
        attempt_prompt, attempt_stdin = prompt, stdin_text
        attempt_tools, repairing = tools, False
        attempt_effort = effort
        last_err = ""

        for attempt in range(self.cfg.max_retries + 1):
            try:
                res = self.run(
                    attempt_prompt,
                    stdin_text=attempt_stdin,
                    system=system,
                    tools=attempt_tools,
                    max_turns=max_turns,
                    model=model,
                    role=role,
                    timeout_s=timeout_s,
                    effort=attempt_effort,
                )
            except LLMError as exc:
                last_err = str(exc)
                if attempt == self.cfg.max_retries:
                    raise
                # Nothing came back to repair, so redo the whole call.
                attempt_prompt, attempt_stdin = prompt, stdin_text
                attempt_tools, repairing = tools, False
                attempt_effort = effort
                continue

            try:
                data = extract_json(res.text)
            except ValueError as exc:
                last_err = str(exc)
            else:
                missing = [k for k in required if k not in data]
                if not missing:
                    return data
                last_err = f"missing required key(s): {', '.join(missing)}"

            if not repairing and res.text.strip():
                # Repair the reply we have. Sealed, because reformatting text
                # the model already wrote needs no tools even when the original
                # call was an agentic one — and `low`, because restating an
                # answer as valid JSON is not a reasoning task.
                attempt_prompt = _repair_prompt(last_err, required)
                attempt_stdin = "=== REPLY TO REPAIR ===\n" + res.text
                attempt_tools, repairing = None, True
                attempt_effort = "low"
            else:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: your previous reply could not be used ({last_err}). "
                    "Reply with the raw JSON object only — no prose, no markdown "
                    "fences, no explanation before or after."
                )
                attempt_stdin, attempt_tools, repairing = stdin_text, tools, False
                attempt_effort = effort

        raise LLMError(f"claude did not return usable JSON after "
                       f"{self.cfg.max_retries + 1} attempts: {last_err}")


def _rejected_effort(stderr: str) -> bool:
    """Did the CLI fail because it does not know `--effort`?

    Matched on the message rather than on a version check, so a CLI that adds,
    renames or removes the flag is handled the same way: state the level, and
    fall back to the CLI's own default if it will not take one.
    """
    low = (stderr or "").lower()
    return "effort" in low and (
        "unknown option" in low or "unknown argument" in low
        or "unrecognized" in low or "invalid option" in low
    )


# --------------------------------------------------------------------------
# JSON salvage
# --------------------------------------------------------------------------

def _repair_prompt(err: str, required: tuple[str, ...]) -> str:
    """Ask for the same answer back as valid JSON, without resending the source.

    The reply being repaired arrives over stdin, so this stays small enough for
    argv however long that reply is.
    """
    keys = (f" It must contain the key(s): {', '.join(required)}."
            if required else "")
    return "\n".join([
        "The text after the marker below is a reply that was supposed to be a "
        f"single JSON object, but it could not be used ({err}).",
        "",
        "Convert it into that JSON object.",
        "",
        "- Keep the content exactly as written. Do not re-do the work, do not "
        "add findings, and do not drop any.",
        "- Reply with the raw JSON object only — no prose, no markdown fences, "
        f"no explanation before or after.{keys}",
        "- If the reply is cut off mid-way, emit a valid object holding "
        "whatever it does contain rather than inventing an ending.",
    ])


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply that may be wrapped in prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty reply")

    # Strip a markdown fence if the whole reply is one.
    if text.startswith("```"):
        body = text.split("\n", 1)[1] if "\n" in text else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        text = body.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}
    except json.JSONDecodeError:
        pass

    # Otherwise scan for the first balanced {...}, respecting strings/escapes.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise ValueError("no JSON object found in reply")
