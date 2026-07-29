"""Shared HTTP client with polite headers, retries and rate-limit backoff."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import FetchConfig

USER_AGENT_BASE = "literature-agent/0.1 (https://github.com/paolosyloslabini/LiteratureAgent)"


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
        """GET with backoff. Returns None instead of raising on failure."""
        delay = 1.0
        for attempt in range(retries):
            try:
                r = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError:
                if attempt == retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
                continue

            if r.status_code in (429, 500, 502, 503, 504):
                if attempt == retries - 1:
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
