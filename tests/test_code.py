"""`lit code`: searching the web for a paper's implementation.

The LLM and the network are both stubbed, so what these test is the policy —
which of the scout's answers are believed, which are thrown away, and what ends
up written to the entry. The failure mode this command exists to avoid is a
plausible-looking repository URL that does not exist, so most of these are about
refusing one.
"""

from __future__ import annotations

import pytest
from factories import make_entry
from rich.console import Console

from lit.actions.code import find_code
from lit.actions.context import Ctx
from lit.llm import LLMError
from lit.models import CODE_FROM_PAPER, CODE_FROM_WEB

REPO = "https://github.com/tensorflow/tensor2tensor"

FOUND = {
    "repo_url": REPO,
    "official": True,
    "evidence": "the README says 'code for Attention Is All You Need'",
    "confidence": "high",
    "searched": ["attention is all you need github"],
}


class StubLLM:
    """Stands in for ClaudeCLI. Records the prompts and the tools it was given."""

    available = True

    def __init__(self, payload=None, fail=False):
        self.payload = payload
        self.fail = fail
        self.calls = 0
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def json(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        self.kwargs.append(kw)
        if self.fail:
            raise LLMError("model unavailable")
        return dict(self.payload if self.payload is not None else FOUND)

    def run(self, prompt, **kw):  # pragma: no cover - unused here
        raise NotImplementedError


class StubHttp:
    """A network that answers for a fixed set of URLs.

    `unavailable` marks the URLs whose host could not be reached at all — the
    case the action must tell apart from a 404.
    """

    def __init__(self, live=(REPO,), unavailable=()):
        self.live = set(live)
        self.unavailable = set(unavailable)
        self.asked: list[str] = []
        self._unavailable_count = 0

    @property
    def unavailable_count(self) -> int:
        return self._unavailable_count

    def get(self, url, **kw):
        self.asked.append(url)
        if url in self.unavailable:
            self._unavailable_count += 1
            return None
        return object() if url in self.live else None

    def close(self):
        pass


@pytest.fixture
def ctx(lib, cfg):
    return Ctx(cfg=cfg, library=lib, console=Console(quiet=True), json_mode=True)


def wire(ctx, payload=None, *, fail=False, live=(REPO,), unavailable=()):
    ctx._llm = StubLLM(payload, fail=fail)
    ctx._http = StubHttp(live=live, unavailable=unavailable)
    return ctx._llm


def stock(lib, **kw):
    return lib.save_entry(make_entry(code_url=None, **kw))


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_a_confirmed_repository_is_stored(ctx, lib):
    wire(ctx)
    entry = stock(lib)

    res = find_code(ctx, [entry])[0]

    assert res.status == "found"
    assert res.code_url == REPO
    assert lib.get(entry.key).code_url == REPO


def test_a_web_found_repository_is_marked_as_such(ctx, lib):
    wire(ctx)
    find_code(ctx, [stock(lib)])

    saved = lib.get("vaswani2017attention")
    assert saved.code_source == CODE_FROM_WEB
    assert saved.code_provenance() == "found on the web"
    # The evidence the scout quoted travels with it.
    assert "README" in saved.code_reason
    assert "author release" in saved.code_reason


def test_the_scout_gets_web_tools_and_the_cheap_model(ctx, lib):
    llm = wire(ctx)
    find_code(ctx, [stock(lib)])

    kw = llm.kwargs[0]
    assert kw["tools"] == ["WebSearch", "WebFetch"]
    assert kw["role"] == "code"
    assert cfg_model_for_code(ctx) == "haiku"


def cfg_model_for_code(ctx) -> str:
    return ctx.cfg.llm.model_for("code")


def test_the_prompt_carries_what_identifies_the_paper(ctx, lib):
    llm = wire(ctx)
    find_code(ctx, [stock(lib, doi="10.1000/xyz")])

    prompt = llm.prompts[0]
    assert "Attention Is All You Need" in prompt
    assert "1706.03762" in prompt
    assert "10.1000/xyz" in prompt
    assert "Vaswani" in prompt


# --------------------------------------------------------------------------
# Refusing what cannot be believed
# --------------------------------------------------------------------------

def test_an_invented_repository_is_not_stored(ctx, lib):
    """The one check that matters: a plausible URL that 404s is thrown away."""
    wire(ctx, {"repo_url": "https://github.com/google/attention-is-all-you-need",
               "official": True}, live=())
    entry = stock(lib)

    res = find_code(ctx, [entry])[0]

    assert res.status == "rejected"
    assert "does not resolve" in res.message
    assert lib.get(entry.key).code_url is None


def test_a_url_that_is_not_a_repository_is_refused(ctx, lib):
    wire(ctx, {"repo_url": "https://myproject.example.org/", "official": True})
    res = find_code(ctx, [stock(lib)])[0]

    assert res.status == "rejected"
    assert "not a code repository" in res.message
    assert lib.get("vaswani2017attention").code_url is None


def test_an_organization_page_is_not_a_repository(ctx, lib):
    wire(ctx, {"repo_url": "https://github.com/google-research", "official": True})
    assert find_code(ctx, [stock(lib)])[0].status == "rejected"


@pytest.mark.parametrize("answer", [None, "", "null", "none", "N/A"])
def test_finding_nothing_is_a_clean_answer(ctx, lib, answer):
    """A paper with no public code must not become a rejected-looking failure."""
    wire(ctx, {"repo_url": answer, "official": False})
    res = find_code(ctx, [stock(lib)])[0]

    assert res.status == "none"
    assert "no public repository" in res.message
    assert lib.get("vaswani2017attention").code_url is None


def test_a_failed_search_leaves_the_entry_alone(ctx, lib):
    wire(ctx, fail=True)
    entry = stock(lib)

    res = find_code(ctx, [entry])[0]

    assert res.status == "error"
    assert "model unavailable" in res.message
    assert lib.get(entry.key).code_url is None


def test_an_unreachable_host_keeps_the_link_but_records_the_doubt(ctx, lib):
    """Being unable to reach GitHub is not the same as GitHub saying 404."""
    wire(ctx, live=(), unavailable=(REPO,))

    res = find_code(ctx, [stock(lib)])[0]

    assert res.status == "found"
    saved = lib.get("vaswani2017attention")
    assert saved.code_url == REPO
    assert "not verified" in saved.code_reason


# --------------------------------------------------------------------------
# Official vs third-party
# --------------------------------------------------------------------------

THIRD_PARTY = {**FOUND, "official": False,
               "evidence": "an unaffiliated reimplementation citing the paper"}


def test_a_third_party_reimplementation_is_refused_by_default(ctx, lib):
    wire(ctx, THIRD_PARTY)
    res = find_code(ctx, [stock(lib)])[0]

    assert res.status == "rejected"
    assert "--unofficial" in res.message
    assert lib.get("vaswani2017attention").code_url is None


def test_a_repository_of_unstated_origin_is_not_called_third_party(ctx, lib):
    """Saying nothing about who released it is not the same as saying not them."""
    wire(ctx, {"repo_url": REPO})
    res = find_code(ctx, [stock(lib)])[0]

    assert res.status == "rejected"
    assert "not confirmed as the authors' own" in res.message
    assert lib.get("vaswani2017attention").code_url is None


def test_a_third_party_reimplementation_is_accepted_when_asked_for(ctx, lib):
    wire(ctx, THIRD_PARTY)
    res = find_code(ctx, [stock(lib)], unofficial=True)[0]

    assert res.status == "found"
    assert res.official is False
    assert "third-party" in lib.get("vaswani2017attention").code_reason


# --------------------------------------------------------------------------
# What gets searched at all
# --------------------------------------------------------------------------

def test_an_entry_that_already_has_a_link_is_skipped(ctx, lib):
    llm = wire(ctx)
    entry = lib.save_entry(make_entry(code_url="https://github.com/a/b"))

    res = find_code(ctx, [entry])[0]

    assert res.status == "skipped"
    assert llm.calls == 0  # no tokens spent re-finding what we have
    assert lib.get(entry.key).code_url == "https://github.com/a/b"


def test_force_re_searches_an_entry_that_has_a_link(ctx, lib):
    llm = wire(ctx)
    entry = lib.save_entry(make_entry(code_url="https://github.com/a/b"))

    res = find_code(ctx, [entry], force=True)[0]

    assert llm.calls == 1
    assert res.status == "found"
    assert lib.get(entry.key).code_url == REPO


def test_the_same_link_again_reports_unchanged(ctx, lib):
    wire(ctx)
    entry = lib.save_entry(make_entry(code_url=REPO))

    res = find_code(ctx, [entry], force=True)[0]

    assert res.status == "unchanged"
    assert res.ok


def test_dry_run_reports_without_writing(ctx, lib):
    wire(ctx)
    entry = stock(lib)

    res = find_code(ctx, [entry], dry_run=True)[0]

    assert res.status == "found"
    assert res.code_url == REPO
    assert "not stored" in res.message
    assert lib.get(entry.key).code_url is None


def test_nothing_to_do_costs_nothing(ctx, lib):
    llm = wire(ctx)
    assert find_code(ctx, []) == []
    assert llm.calls == 0


def test_several_papers_are_searched_in_one_run(ctx, lib):
    llm = wire(ctx)
    entries = [stock(lib), stock(lib, key="doe2010obscure", title="An Obscure Study")]

    results = find_code(ctx, entries)

    assert llm.calls == 2
    assert {r.key for r in results} == {"vaswani2017attention", "doe2010obscure"}


def test_one_failed_search_does_not_sink_the_batch(ctx, lib):
    class Flaky(StubLLM):
        def json(self, prompt, **kw):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("boom")
            return dict(FOUND)

    ctx._llm = Flaky()
    ctx._http = StubHttp()
    entries = [stock(lib), stock(lib, key="doe2010obscure", title="An Obscure Study")]

    statuses = {r.key: r.status for r in find_code(ctx, entries)}

    assert statuses["vaswani2017attention"] == "error"
    assert statuses["doe2010obscure"] == "found"


# --------------------------------------------------------------------------
# Provenance, on the entry itself
# --------------------------------------------------------------------------

def test_a_paper_printed_link_is_labelled_differently():
    entry = make_entry(code_url=REPO, code_source=CODE_FROM_PAPER)
    assert entry.code_provenance() == "printed in the paper"


def test_a_link_predating_the_provenance_field_reads_as_the_papers_own():
    """Only the reader ever wrote this field before `lit code` existed."""
    entry = make_entry(code_url=REPO, code_source=None)
    assert entry.code_provenance() == "printed in the paper"


def test_no_link_means_no_provenance():
    assert make_entry(code_url=None).code_provenance() == ""


def test_provenance_survives_a_round_trip_through_the_entry_file(lib):
    entry = make_entry(code_url=REPO, code_source=CODE_FROM_WEB,
                       code_reason="author release, high confidence")
    lib.save_entry(entry)

    saved = lib.get(entry.key)
    assert saved.code_source == CODE_FROM_WEB
    assert saved.code_reason == "author release, high confidence"
