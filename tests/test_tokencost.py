"""The token budget is honoured, and no call pays for work it discards.

Every test here pins a saving that is invisible at runtime: nothing breaks if
`ask` quietly starts sending 400k chars again, or reads a pulled paper twice —
the bill just goes up. So they assert on what actually reaches the model.
"""

from __future__ import annotations

import pytest
from factories import make_meta
from rich.console import Console

from lit.actions.context import Ctx
from lit.llm import LLMError
from lit.models import Entry, Section


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


class RecordingLLM:
    """Captures every call so a test can weigh what was sent."""

    available = True

    def __init__(self, replies=None):
        self.calls: list[dict] = []
        self.replies = list(replies or [])

    def _record(self, prompt, stdin_text, kw):
        self.calls.append({"prompt": prompt, "stdin": stdin_text or "", **kw})

    def json(self, prompt, *, stdin_text=None, **kw):
        self._record(prompt, stdin_text, kw)
        if self.replies:
            return self.replies.pop(0)
        return {"relevant": [], "one_liner": "x",
                "sections": [{"name": "S", "summary": "y"}]}

    def run(self, prompt, *, stdin_text=None, **kw):
        self._record(prompt, stdin_text, kw)

        class R:
            text = "an answer"
        return R()

    @property
    def text_sent(self) -> int:
        return sum(len(c["prompt"]) + len(c["stdin"]) for c in self.calls)


# --------------------------------------------------------------------------
# The configured read budget reaches every full-text path
# --------------------------------------------------------------------------

def test_ctx_read_budget_follows_the_config(ctx):
    ctx.cfg.llm.read_chars = 12_345
    assert ctx.read_budget == 12_345


@pytest.mark.parametrize("command", ["add", "ask", "claim"])
def test_every_full_text_path_honours_read_chars(monkeypatch, ctx, command):
    """`lit config set llm.read_chars` used to be ignored by ask and claim."""
    from lit.fetch.fulltext import FullText

    budget = 5_000
    ctx.cfg.llm.read_chars = budget
    huge = FullText("z" * 300_000, "arxiv")
    llm = RecordingLLM()
    ctx._llm = llm

    for mod in ("add", "ask", "claim"):
        monkeypatch.setattr(f"lit.actions.{mod}.fetch_fulltext",
                            lambda *a, **k: huge, raising=False)
    monkeypatch.setattr("lit.actions.add.resolve_metadata",
                        lambda *a, **k: make_meta())
    monkeypatch.setattr("lit.actions.claim.resolve_metadata",
                        lambda *a, **k: make_meta())

    if command == "add":
        from lit.actions.add import add_paper
        add_paper(ctx, "A Paper", force=True)
    elif command == "ask":
        from lit.actions.ask import ask
        ctx.library.save_entry(Entry(key="k1", title="A Paper",
                                     status="verified", one_liner="x"))
        llm.replies = [{"relevant": [{"key": "k1", "relevance": 0.9}]},
                       {"relevant": True, "quotes": []}]
        ask(ctx, "Paper", read=1)   # must match FTS, or `ask` never opens one
    else:
        from lit.actions.claim import trace_claim
        llm.replies = [{"is_origin": True, "states_claim": True}]
        trace_claim(ctx, "a claim", start="10.1/x", max_hops=1)

    full_text_calls = [c for c in llm.calls if len(c["stdin"]) > 1000]
    assert full_text_calls, f"{command} sent no full text"
    for call in full_text_calls:
        assert len(call["stdin"]) < budget + 2_000, (
            f"{command} sent {len(call['stdin']):,} chars on a {budget:,} budget"
        )


# --------------------------------------------------------------------------
# `ask` does not buy an answer it throws away
# --------------------------------------------------------------------------

def stock(lib, n: int) -> None:
    for i in range(n):
        lib.save_entry(Entry(
            key=f"paper{i}", title=f"Paper {i}", status="verified",
            one_liner="A one-line summary of the contribution.",
            tags=["causal", "agents"],
            key_findings=[f"Finding {j} with a number attached." for j in range(5)],
            sections=[Section(name=f"Section {j}", summary="s" * 300)
                      for j in range(12)],
            notes="user notes " * 40,
        ))


def test_ask_selection_does_not_request_a_prose_answer(monkeypatch, ctx):
    """The answer `search` writes from summaries is discarded by `ask`."""
    from lit.actions.ask import ask
    from lit.fetch.fulltext import FullText

    stock(ctx.library, 3)
    llm = RecordingLLM(replies=[
        {"relevant": [{"key": "paper0", "relevance": 0.9}]},
        {"relevant": True, "quotes": [], "summary": ""},
    ])
    ctx._llm = llm
    monkeypatch.setattr("lit.actions.ask.fetch_fulltext",
                        lambda *a, **k: FullText("t" * 9000, "arxiv"))

    ask(ctx, "causal", read=1)

    selection = llm.calls[0]["prompt"]
    assert '"relevant"' in selection
    assert '"answer"' not in selection
    assert '"gaps"' not in selection


def test_ask_selection_card_is_far_lighter_than_the_search_card(ctx):
    from lit.prompts import _entry_card

    stock(ctx.library, 1)
    entry = ctx.library.get("paper0")
    full = _entry_card(entry)
    brief = _entry_card(entry, brief=True)

    assert len(brief) * 4 < len(full), (
        f"brief card is {len(brief)} chars vs {len(full)} — not worth the branch"
    )
    # It still carries what a relevance judgement is made on.
    assert entry.one_liner in brief
    assert "causal" in brief
    assert "Finding 0" in brief
    # And drops what only an answer-from-summaries needs.
    assert "Sections:" not in brief
    assert "User notes" not in brief


def test_search_itself_still_answers(ctx):
    """`lit search` is the command whose whole point is the prose answer."""
    from lit.actions.search import search

    stock(ctx.library, 2)
    llm = RecordingLLM(replies=[{"answer": "An answer.", "relevant": [],
                                 "gaps": ["a gap"]}])
    ctx._llm = llm
    res = search(ctx, "causal")

    assert '"answer"' in llm.calls[0]["prompt"]
    assert "Sections:" in llm.calls[0]["prompt"]
    assert res.answer == "An answer."
    assert res.gaps == ["a gap"]


# --------------------------------------------------------------------------
# `ask --expand` reads a pulled paper once, not twice
# --------------------------------------------------------------------------

def test_expand_does_not_read_a_paper_it_is_about_to_read(monkeypatch, ctx):
    from lit.actions.ask import ask
    from lit.fetch.fulltext import FullText
    from lit.models import Reference

    ctx.library.save_entry(Entry(
        key="seed", title="Seed", status="verified", one_liner="x",
        references=[Reference(title="A Cited Work", doi="10.1/cited")],
    ))
    reads: list[str] = []

    def spy_add(ctx_, query, **kw):
        reads.append(f"read={kw.get('read', True)}")
        from lit.actions.add import add_paper
        return add_paper(ctx_, query, **kw)

    monkeypatch.setattr("lit.actions.ask.add_paper", spy_add)
    monkeypatch.setattr("lit.actions.add.resolve_metadata",
                        lambda *a, **k: make_meta(title="A Cited Work",
                                                  doi="10.1/cited"))
    monkeypatch.setattr("lit.actions.ask.fetch_fulltext",
                        lambda *a, **k: FullText("t" * 9000, "arxiv"))
    monkeypatch.setattr("lit.actions.add.fetch_fulltext",
                        lambda *a, **k: FullText("t" * 9000, "arxiv"))

    llm = RecordingLLM(replies=[
        {"relevant": [{"key": "seed", "relevance": 0.9}]},
        {"relevant": True, "quotes": [], "summary": ""},
        {"relevant": True, "quotes": [], "summary": ""},
    ])
    ctx._llm = llm
    res = ask(ctx, "Seed", read=1, expand=1)

    assert reads == ["read=False"], "the pulled paper was read before extraction"
    assert res.pulled, "expansion pulled nothing in"
    # It is still consulted — filed cheaply, then read once by `extract`.
    assert any(k != "seed" for k in res.consulted)


# --------------------------------------------------------------------------
# A malformed reply is repaired, not re-earned
# --------------------------------------------------------------------------

class ScriptedCLI:
    """A ClaudeCLI whose `run` replays a fixed list of replies."""

    def __init__(self, cfg, replies):
        from lit.llm import Usage
        self.cfg = cfg
        self.usage = Usage()
        self.replies = list(replies)
        self.calls: list[dict] = []

    def run(self, prompt, *, stdin_text=None, **kw):
        self.calls.append({"prompt": prompt, "stdin": stdin_text or "", **kw})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply

        class R:
            text = reply
        return R()

    json = None  # filled in below


def make_cli(replies):
    from lit.config import LLMConfig
    from lit.llm import ClaudeCLI

    cli = ScriptedCLI(LLMConfig(max_retries=2), replies)
    cli.json = ClaudeCLI.json.__get__(cli, ScriptedCLI)
    return cli


PAPER = "the full text of a long paper. " * 4000


def test_malformed_json_is_repaired_without_resending_the_paper():
    cli = make_cli(["I read it. Here are the findings, in prose.",
                    '{"one_liner": "x"}'])
    got = cli.json("read this", stdin_text=PAPER, required=("one_liner",))

    assert got == {"one_liner": "x"}
    assert len(cli.calls) == 2
    assert cli.calls[0]["stdin"] == PAPER
    repair = cli.calls[1]
    assert PAPER not in repair["stdin"], "the repair pass resent the paper"
    assert "REPLY TO REPAIR" in repair["stdin"]
    assert "in prose" in repair["stdin"], "the reply to repair was not carried"
    assert len(repair["stdin"]) < len(PAPER) / 100


def test_a_repair_pass_is_sealed_even_for_an_agentic_call():
    cli = make_cli(["prose, not json", '{"papers": []}'])
    cli.json("find papers", tools=["WebSearch"], required=("papers",))

    assert cli.calls[0]["tools"] == ["WebSearch"]
    assert cli.calls[1]["tools"] is None


def test_a_failed_repair_falls_back_to_redoing_the_work():
    cli = make_cli(["prose", "still not json", '{"one_liner": "x"}'])
    got = cli.json("read this", stdin_text=PAPER, required=("one_liner",))

    assert got == {"one_liner": "x"}
    assert len(cli.calls) == 3
    assert PAPER not in cli.calls[1]["stdin"]   # repair
    assert cli.calls[2]["stdin"] == PAPER       # then the full redo


def test_missing_required_keys_also_repair_rather_than_re_read():
    cli = make_cli(['{"sections": []}', '{"one_liner": "x", "sections": []}'])
    got = cli.json("read this", stdin_text=PAPER,
                   required=("one_liner", "sections"))

    assert got["one_liner"] == "x"
    assert PAPER not in cli.calls[1]["stdin"]
    assert "one_liner" in cli.calls[1]["prompt"]


def test_a_cli_failure_still_redoes_the_call_in_full():
    """Nothing came back, so there is nothing to repair."""
    cli = make_cli([LLMError("api blip"), '{"one_liner": "x"}'])
    got = cli.json("read this", stdin_text=PAPER, required=("one_liner",))

    assert got == {"one_liner": "x"}
    assert cli.calls[1]["stdin"] == PAPER


def test_an_empty_reply_is_not_treated_as_repairable():
    cli = make_cli(["", '{"one_liner": "x"}'])
    cli.json("read this", stdin_text=PAPER, required=("one_liner",))
    assert cli.calls[1]["stdin"] == PAPER


def test_giving_up_still_raises_after_the_configured_attempts():
    cli = make_cli(["prose", "more prose", "yet more prose"])
    with pytest.raises(LLMError, match="did not return usable JSON"):
        cli.json("read this", stdin_text=PAPER, required=("one_liner",))
    assert len(cli.calls) == 3


def test_the_repair_prompt_stays_small_enough_for_argv():
    from lit.llm import _repair_prompt

    assert len(_repair_prompt("no JSON object found", ("a", "b"))) < 2_000
