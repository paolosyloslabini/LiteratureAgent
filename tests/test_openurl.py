"""Opening a link has to work where the user actually is.

Under WSL that is a Windows browser, reached through a Windows executable, so
these tests are about which command gets chosen and how the URL is quoted —
never about launching anything.
"""

from __future__ import annotations

import pytest

from lit import openurl


@pytest.fixture(autouse=True)
def no_browser_env(monkeypatch):
    monkeypatch.delenv("LIT_BROWSER", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)


# --------------------------------------------------------------------------
# Detecting WSL
# --------------------------------------------------------------------------

def test_wsl_is_detected_from_the_environment(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert openurl.is_wsl()


def test_wsl_is_detected_from_the_kernel_string(monkeypatch):
    monkeypatch.setattr(openurl, "_proc_version",
                        lambda: "Linux version 6.6.0-microsoft-standard-WSL2")
    assert openurl.is_wsl()


def test_plain_linux_is_not_wsl(monkeypatch):
    monkeypatch.setattr(openurl, "_proc_version", lambda: "Linux version 6.6.0-generic")
    assert not openurl.is_wsl()


# --------------------------------------------------------------------------
# Choosing a command
# --------------------------------------------------------------------------

def test_chrome_is_preferred_on_wsl(monkeypatch):
    monkeypatch.setattr(openurl, "find_chrome", lambda: "/mnt/c/chrome.exe")
    monkeypatch.setattr(openurl.shutil, "which", lambda _: None)
    what, argv = openurl.launch_commands("https://arxiv.org/abs/1706.03762", wsl=True)[0]
    assert what == "Chrome"
    assert argv == ["/mnt/c/chrome.exe", "https://arxiv.org/abs/1706.03762"]


def test_windows_default_browser_is_the_fallback_when_chrome_is_absent(monkeypatch):
    monkeypatch.setattr(openurl, "find_chrome", lambda: None)
    monkeypatch.setattr(openurl.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "wslview" else None)
    commands = openurl.launch_commands("https://example.org/p", wsl=True)
    assert [c[1][0] for c in commands] == ["/usr/bin/wslview"]


def test_powershell_quotes_the_url(monkeypatch):
    """An unquoted '&' in a query string ends the PowerShell statement."""
    monkeypatch.setattr(openurl, "find_chrome", lambda: None)
    monkeypatch.setattr(openurl.shutil, "which",
                        lambda name: "/mnt/c/ps.exe" if name == "powershell.exe" else None)
    _, argv = openurl.launch_commands("https://x.org/p?a=1&b=2", wsl=True)[0]
    assert argv[-1] == "Start-Process 'https://x.org/p?a=1&b=2'"
    assert "-NoProfile" in argv


def test_a_quote_in_the_url_cannot_break_out_of_the_powershell_string(monkeypatch):
    monkeypatch.setattr(openurl, "find_chrome", lambda: None)
    monkeypatch.setattr(openurl.shutil, "which",
                        lambda name: "/mnt/c/ps.exe" if name == "powershell.exe" else None)
    _, argv = openurl.launch_commands("https://x.org/it's", wsl=True)[0]
    assert argv[-1] == "Start-Process 'https://x.org/it''s'"


def test_nothing_windows_specific_is_offered_off_wsl(monkeypatch):
    monkeypatch.setattr(openurl, "find_chrome", lambda: "/mnt/c/chrome.exe")
    assert openurl.launch_commands("https://example.org", wsl=False) == []


def test_lit_browser_overrides_everything(monkeypatch):
    monkeypatch.setenv("LIT_BROWSER", "firefox --new-tab")
    monkeypatch.setattr(openurl, "find_chrome", lambda: "/mnt/c/chrome.exe")
    monkeypatch.setattr(openurl.shutil, "which", lambda _: None)
    what, argv = openurl.launch_commands("https://example.org", wsl=True)[0]
    assert what == "firefox"
    assert argv == ["firefox", "--new-tab", "https://example.org"]


# --------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------

def test_open_url_reports_what_opened_it(monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(openurl, "launch_commands",
                        lambda url, **_: [("Chrome", ["chrome.exe", url])])
    monkeypatch.setattr(openurl, "_spawn", lambda argv: spawned.append(argv) or True)
    assert openurl.open_url("https://example.org") == "Chrome"
    assert spawned == [["chrome.exe", "https://example.org"]]


def test_open_url_falls_through_to_the_next_command(monkeypatch):
    monkeypatch.setattr(openurl, "launch_commands", lambda url, **_: [
        ("Chrome", ["chrome.exe", url]),
        ("the Windows default browser", ["wslview", url]),
    ])
    monkeypatch.setattr(openurl, "_spawn", lambda argv: argv[0] != "chrome.exe")
    assert openurl.open_url("https://example.org") == "the Windows default browser"


def test_open_url_falls_back_to_the_stdlib(monkeypatch):
    monkeypatch.setattr(openurl, "launch_commands", lambda url, **_: [])
    monkeypatch.setattr(openurl.webbrowser, "open", lambda url: True)
    assert openurl.open_url("https://example.org") == "your browser"


def test_open_url_reports_failure(monkeypatch):
    monkeypatch.setattr(openurl, "launch_commands", lambda url, **_: [])
    monkeypatch.setattr(openurl.webbrowser, "open", lambda url: False)
    assert openurl.open_url("https://example.org") is None


@pytest.mark.parametrize("url", [None, "", "file:///etc/passwd", "javascript:alert(1)",
                                 "/mnt/c/Windows/System32/calc.exe"])
def test_only_web_urls_are_handed_to_a_windows_shell(url, monkeypatch):
    monkeypatch.setattr(openurl, "_spawn",
                        lambda argv: pytest.fail(f"spawned {argv} for {url!r}"))
    monkeypatch.setattr(openurl.webbrowser, "open",
                        lambda u: pytest.fail(f"opened {u!r}"))
    assert openurl.open_url(url) is None
