"""Shared HTTP client with polite headers, retries and rate-limit backoff."""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..config import FetchConfig

USER_AGENT_BASE = "literature-agent/0.1 (https://github.com/paolosyloslabini/LiteratureAgent)"

# Minimum gap between two requests to the same host. arXiv asks callers for one
# request every three seconds and throttles hard when a burst ignores it: the
# API then stalls until the client's own timeout fires rather than answering
# 429. One `lit find` fans several workers out over a shared client, so without
# a floor here a single run can lose records that exist — which reads to the
# user as "this paper could not be found".
#
# Only the API carries arXiv's documented three-second guidance; the website is
# given a lighter floor because the HTML fallback path hits it once per paper
# and serialising that at 3s would cost a `find` run far more than it saves.
# Matching is on the exact hostname, so ar5iv and other subdomains are free.
HOST_MIN_INTERVAL: dict[str, float] = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 1.0,
}
DEFAULT_MIN_INTERVAL = 0.0


class HttpClient:
    """A small wrapper over httpx with retry/backoff for the metadata APIs."""

    def __init__(self, cfg: FetchConfig | None = None):
        self.cfg = cfg or FetchConfig()
        ua = USER_AGENT_BASE
        if self.cfg.email:
            # Crossref/OpenAlex give a faster "polite pool" when you identify.
            ua += f" mailto:{self.cfg.email}"
        self._client = httpx.Client(
            timeout=self.cfg.timeout_s,
            follow_redirects=True,
            headers={"User-Agent": ua},
        )
        # One client is shared by every worker thread in a `find` fan-out, so
        # both the throttle bookkeeping and the failure tally need guarding.
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}
        self._unavailable = 0

    @property
    def unavailable_count(self) -> int:
        """How many GETs gave up without ever reaching the upstream index.

        A clean 404 is an *answer* and is deliberately not counted here. Only
        timeouts, connection errors and exhausted 429/5xx retries are, because
        those leave us not knowing whether the record exists. Callers compare
        this before and after an operation to tell "no such paper" apart from
        "the index was down", which `get` returning `None` cannot express.
        """
        with self._lock:
            return self._unavailable

    def _note_unavailable(self) -> None:
        with self._lock:
            self._unavailable += 1

    def _throttle(self, url: str) -> None:
        """Block until this host's minimum interval has elapsed."""
        host = (urlsplit(url).hostname or "").lower()
        gap = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
        if not gap:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                ready = self._next_allowed.get(host, 0.0)
                if now >= ready:
                    # Claim the slot inside the lock, so two threads racing
                    # here cannot both decide they are clear to go.
                    self._next_allowed[host] = now + gap
                    return
                wait = ready - now
            time.sleep(wait)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        retries: int = 3,
        expect_json: bool = False,
    ) -> httpx.Response | None:
        """GET with backoff. Returns None instead of raising on failure.

        `None` on its own cannot say whether the work is missing or the index
        was merely unreachable, so the unreachable case also bumps
        `unavailable_count` for callers that need to report which happened.
        """
        delay = 1.0
        for attempt in range(retries):
            self._throttle(url)
            try:
                r = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError:
                if attempt == retries - 1:
                    self._note_unavailable()
                    return None
                time.sleep(delay)
                delay *= 2
                continue

            if r.status_code in (429, 500, 502, 503, 504):
                if attempt == retries - 1:
                    self._note_unavailable()
                    return None
                wait = delay
                if r.status_code == 429:
                    try:
                        wait = max(wait, float(r.headers.get("retry-after", 0)))
                    except ValueError:
                        pass
                time.sleep(min(wait, 30))
                delay *= 2
                continue

            if r.status_code >= 400:
                return None
            if expect_json:
                ctype = r.headers.get("content-type", "")
                if "json" not in ctype and not r.text.lstrip().startswith(("{", "[")):
                    return None
            return r
        return None

    def get_json(self, url: str, *, params: dict | None = None,
                 headers: dict | None = None, retries: int = 3) -> Any | None:
        r = self.get(url, params=params, headers=headers, retries=retries, expect_json=True)
        if r is None:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def download(self, url: str, dest, *, max_mb: int | None = None,
                 headers: dict | None = None) -> bool:
        """Stream a URL to `dest`. Returns False on any failure or size overrun."""
        max_bytes = (max_mb or self.cfg.max_pdf_mb) * 1024 * 1024
        try:
            with self._client.stream("GET", url, headers=headers) as r:
                if r.status_code >= 400:
                    return False
                total = 0
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(65536):
                        total += len(chunk)
                        if total > max_bytes:
                            fh.close()
                            tmp.unlink(missing_ok=True)
                            return False
                        fh.write(chunk)
                if total == 0:
                    tmp.unlink(missing_ok=True)
                    return False
                tmp.replace(dest)
                return True
        except (httpx.HTTPError, OSError):
            return False
