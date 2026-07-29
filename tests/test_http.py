"""Telling "no such record" apart from "the index was unreachable".

`HttpClient.get` returns `None` for both, which is fine for callers that only
want the happy path — but `add` has to explain the failure to a user, and
"could not find a bibliographic record, try a DOI" is actively misleading when
the DOI was right and arXiv was simply rate-limiting. `unavailable_count` is
what lets the two be told apart.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from lit.config import FetchConfig
from lit.fetch import http as http_mod
from lit.fetch.http import HttpClient


def client_with(handler) -> HttpClient:
    """An HttpClient whose transport is driven by `handler`."""
    c = HttpClient(FetchConfig())
    c._client.close()
    c._client = httpx.Client(transport=httpx.MockTransport(handler),
                             follow_redirects=True)
    return c


@pytest.fixture
def no_sleep(monkeypatch):
    """Collapse retry backoff so the failure paths run instantly."""
    monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)


# --------------------------------------------------------------------------
# What counts as "unavailable"
# --------------------------------------------------------------------------

def test_a_clean_404_is_an_answer_not_an_outage(no_sleep):
    c = client_with(lambda req: httpx.Response(404))
    assert c.get("https://api.crossref.org/works/10.0/nope") is None
    assert c.unavailable_count == 0


def test_a_connection_error_is_an_outage(no_sleep):
    def boom(req):
        raise httpx.ConnectError("no route to host")

    c = client_with(boom)
    assert c.get("https://api.crossref.org/works/10.0/x") is None
    assert c.unavailable_count == 1


def test_exhausted_rate_limiting_is_an_outage(no_sleep):
    c = client_with(lambda req: httpx.Response(429))
    assert c.get("https://export.arxiv.org/api/query") is None
    assert c.unavailable_count == 1


def test_exhausted_server_errors_are_an_outage(no_sleep):
    c = client_with(lambda req: httpx.Response(503))
    assert c.get("https://api.openalex.org/works") is None
    assert c.unavailable_count == 1


def test_a_success_counts_nothing(no_sleep):
    c = client_with(lambda req: httpx.Response(200, json={"ok": True}))
    assert c.get_json("https://api.openalex.org/works") == {"ok": True}
    assert c.unavailable_count == 0


def test_recovering_within_the_retries_counts_nothing(no_sleep):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        return httpx.Response(200 if calls["n"] > 1 else 503, json={"ok": True})

    c = client_with(flaky)
    assert c.get_json("https://api.openalex.org/works") == {"ok": True}
    # The 503 was survived, so the caller got its answer and nothing is amiss.
    assert c.unavailable_count == 0


# --------------------------------------------------------------------------
# Per-host throttle
# --------------------------------------------------------------------------

def test_arxiv_requests_are_spaced_out(monkeypatch):
    monkeypatch.setitem(http_mod.HOST_MIN_INTERVAL, "export.arxiv.org", 0.05)
    c = client_with(lambda req: httpx.Response(200, json={}))

    start = time.monotonic()
    for _ in range(3):
        c.get("https://export.arxiv.org/api/query")
    elapsed = time.monotonic() - start

    # Three requests means two enforced gaps; the first is free.
    assert elapsed >= 0.10


def test_other_hosts_are_not_throttled(monkeypatch):
    monkeypatch.setitem(http_mod.HOST_MIN_INTERVAL, "export.arxiv.org", 5.0)
    c = client_with(lambda req: httpx.Response(200, json={}))

    start = time.monotonic()
    for _ in range(3):
        c.get("https://api.crossref.org/works")
    assert time.monotonic() - start < 1.0


def test_the_throttle_holds_across_threads(monkeypatch):
    """The client is shared by `find`'s workers, so the gap must be global.

    Without the lock, threads racing in `_throttle` all read the same clear
    deadline and fire at once — which is exactly the burst that gets a `find`
    run throttled by arXiv in the first place.
    """
    monkeypatch.setitem(http_mod.HOST_MIN_INTERVAL, "export.arxiv.org", 0.05)
    c = client_with(lambda req: httpx.Response(200, json={}))

    start = time.monotonic()
    threads = [threading.Thread(target=lambda: c.get("https://export.arxiv.org/api/query"))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    # Four concurrent callers still serialise into three gaps.
    assert elapsed >= 0.15
