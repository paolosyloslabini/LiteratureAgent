"""The last-resort resolver: off by default, and genuinely last."""

from __future__ import annotations

from pathlib import Path

import pytest
from factories import make_meta

from lit.config import FetchConfig
from lit.fetch import fulltext as ft_mod
from lit.fetch.fulltext import _embedded_pdf_url, _fallback_resolver

USABLE = "x" * 9000


class FakeHttp:
    def __init__(self, *, downloads=None, pages=None):
        self.downloads = downloads or {}
        self.pages = pages or {}
        self.attempted: list[str] = []

    def download(self, url, dest, **kw):
        self.attempted.append(url)
        if self.downloads.get(url):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"%PDF-1.4")
            return True
        return False

    def get(self, url, **kw):
        self.attempted.append(url)
        body = self.pages.get(url)
        if body is None:
            return None
        return type("R", (), {"text": body, "headers": {"content-type": "text/html"}})()


@pytest.fixture
def usable_pdf(monkeypatch):
    monkeypatch.setattr(ft_mod, "extract_pdf",
                        lambda p, **kw: ft_mod.PdfText(USABLE, pages=8, pages_read=8))


# --------------------------------------------------------------------------
# Off by default
# --------------------------------------------------------------------------

def test_disabled_by_default(tmp_path):
    http = FakeHttp()
    assert _fallback_resolver(
        http, make_meta(doi="10.1234/abcd"),
        cfg=FetchConfig(), dest=tmp_path / "x.pdf",
    ) is None
    assert http.attempted == []  # nothing contacted at all


def test_default_config_ships_empty():
    cfg = FetchConfig()
    assert cfg.fallback_url_template == ""
    assert cfg.fallback_cmd == ""


def test_requires_a_doi(tmp_path):
    http = FakeHttp()
    cfg = FetchConfig(fallback_url_template="https://example.org/{doi}")
    assert _fallback_resolver(
        http, make_meta(doi=None), cfg=cfg, dest=tmp_path / "x.pdf"
    ) is None
    assert http.attempted == []


# --------------------------------------------------------------------------
# URL template
# --------------------------------------------------------------------------

def test_url_template_direct_pdf(tmp_path, usable_pdf):
    url = "https://example.org/10.1234/abcd"
    http = FakeHttp(downloads={url: True})
    cfg = FetchConfig(fallback_url_template="https://example.org/{doi}")
    res = _fallback_resolver(http, make_meta(doi="10.1234/abcd"), cfg=cfg,
                             dest=tmp_path / "x.pdf")
    assert res is not None
    assert res.source == "fallback-url"
    assert res.text == USABLE


def test_url_template_follows_an_embedded_viewer(tmp_path, usable_pdf):
    page = "https://example.org/10.1234/abcd"
    pdf = "https://example.org/files/paper.pdf"
    http = FakeHttp(
        downloads={pdf: True},
        pages={page: f'<html><iframe src="{pdf}"></iframe></html>'},
    )
    cfg = FetchConfig(fallback_url_template="https://example.org/{doi}")
    res = _fallback_resolver(http, make_meta(doi="10.1234/abcd"), cfg=cfg,
                             dest=tmp_path / "x.pdf")
    assert res is not None and res.source == "fallback-url"


def test_url_template_failure_is_not_fatal(tmp_path):
    http = FakeHttp()
    cfg = FetchConfig(fallback_url_template="https://example.org/{doi}")
    assert _fallback_resolver(http, make_meta(doi="10.1234/abcd"), cfg=cfg,
                              dest=tmp_path / "x.pdf") is None


def test_unusable_pdf_is_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr(ft_mod, "extract_pdf",
                        lambda p, **kw: ft_mod.PdfText("too short", 1, 1))
    url = "https://example.org/10.1234/abcd"
    http = FakeHttp(downloads={url: True})
    cfg = FetchConfig(fallback_url_template="https://example.org/{doi}")
    dest = tmp_path / "x.pdf"
    assert _fallback_resolver(http, make_meta(doi="10.1234/abcd"), cfg=cfg,
                              dest=dest) is None
    assert not dest.exists()  # a scanned/blocked page is not left behind


# --------------------------------------------------------------------------
# Command hook
# --------------------------------------------------------------------------

def test_command_hook(tmp_path, usable_pdf):
    dest = tmp_path / "out.pdf"
    cfg = FetchConfig(fallback_cmd=f"python3 -c \"open(r'{dest}','wb').write(b'%PDF')\"")
    res = _fallback_resolver(FakeHttp(), make_meta(doi="10.1234/abcd"),
                             cfg=cfg, dest=dest)
    assert res is not None
    assert res.source == "fallback-cmd"


def test_command_substitutes_doi_and_out(tmp_path, usable_pdf):
    marker = tmp_path / "args.txt"
    dest = tmp_path / "out.pdf"
    cfg = FetchConfig(fallback_cmd=(
        f"python3 -c \"import sys;open(r'{marker}','w').write(sys.argv[1]);"
        f"open(sys.argv[2],'wb').write(b'%PDF')\" {{doi}} {{out}}"
    ))
    _fallback_resolver(FakeHttp(), make_meta(doi="10.1234/abcd"), cfg=cfg, dest=dest)
    assert marker.read_text() == "10.1234/abcd"


def test_failing_command_is_not_fatal(tmp_path):
    cfg = FetchConfig(fallback_cmd="false")
    assert _fallback_resolver(FakeHttp(), make_meta(doi="10.1234/abcd"),
                              cfg=cfg, dest=tmp_path / "x.pdf") is None


def test_missing_command_is_not_fatal(tmp_path):
    cfg = FetchConfig(fallback_cmd="definitely-not-a-real-binary-xyz {doi}")
    assert _fallback_resolver(FakeHttp(), make_meta(doi="10.1234/abcd"),
                              cfg=cfg, dest=tmp_path / "x.pdf") is None


# --------------------------------------------------------------------------
# Embedded-PDF extraction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("html,expected", [
    ('<iframe src="https://x.org/a.pdf"></iframe>', "https://x.org/a.pdf"),
    ('<embed src="https://x.org/b.pdf" type="application/pdf">', "https://x.org/b.pdf"),
    ('<a href="https://x.org/c.pdf#page=1">download</a>', "https://x.org/c.pdf#page=1"),
])
def test_embedded_pdf_variants(html, expected):
    http = FakeHttp(pages={"https://host/page": html})
    assert _embedded_pdf_url(http, "https://host/page") == expected


def test_embedded_pdf_resolves_a_relative_url():
    http = FakeHttp(pages={"https://host/view/1": '<iframe src="/files/a.pdf">'})
    assert _embedded_pdf_url(http, "https://host/view/1") == "https://host/files/a.pdf"


def test_embedded_pdf_ignores_non_pdf_frames():
    http = FakeHttp(pages={"https://host/p": '<iframe src="https://ads.example/x.js">'})
    assert _embedded_pdf_url(http, "https://host/p") is None


def test_embedded_pdf_handles_an_unreachable_page():
    assert _embedded_pdf_url(FakeHttp(), "https://host/missing") is None


# --------------------------------------------------------------------------
# Ordering: open-access routes always win
# --------------------------------------------------------------------------

def test_open_access_route_is_used_before_the_fallback(tmp_path, monkeypatch, usable_pdf):
    """The fallback must never pre-empt arXiv or any other OA source."""
    from lit.fetch.fulltext import fetch_fulltext

    arxiv_pdf = "https://arxiv.org/pdf/1706.03762"
    http = FakeHttp(downloads={arxiv_pdf: True})
    cfg = FetchConfig(fallback_url_template="https://example.org/{doi}")

    called = {"fallback": False}
    monkeypatch.setattr(
        ft_mod, "_fallback_resolver",
        lambda *a, **k: called.__setitem__("fallback", True) or None,
    )
    res = fetch_fulltext(
        http, make_meta(doi="10.1234/abcd"), key="k",
        pdf_dir=tmp_path / "pdfs", text_dir=tmp_path / "text", cfg=cfg,
    )
    assert res is not None and res.source == "arxiv"
    assert not called["fallback"]
