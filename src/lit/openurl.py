"""Opening a link in the browser the user is actually looking at.

`webbrowser.open` is right on a Linux desktop and useless under WSL: there is
usually no browser installed on the Linux side, and the window in front of the
user is a Windows one. So under WSL the URL is handed to Windows — Chrome
itself when we can find it, otherwise whatever Windows treats as the default
browser. `LIT_BROWSER` overrides the lot.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import webbrowser
from pathlib import Path

# Machine-wide Chrome installs, seen from WSL. A per-user install lands under a
# Windows profile instead, which is why the profiles are searched too.
_CHROME_PATHS = (
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
)
_CHROME_IN_PROFILE = "AppData/Local/Google/Chrome/Application/chrome.exe"
_WINDOWS_USERS = Path("/mnt/c/Users")

# A Windows executable started from a Linux working directory prints a warning
# about the path it cannot represent; start it somewhere it can.
_WINDOWS_CWD = Path("/mnt/c")

# Entry metadata is fetched from the network, so what it can hand to a Windows
# shell is limited to the two schemes a paper link is ever written in. `file:`
# and `javascript:` are not among them.
_SCHEMES = ("http://", "https://")


def is_wsl() -> bool:
    """True when this Linux is running inside Windows Subsystem for Linux."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    return "microsoft" in _proc_version().lower()


def _proc_version() -> str:
    try:
        return Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def find_chrome() -> str | None:
    """Path to Chrome on the Windows side, or None if it is not installed."""
    for path in _CHROME_PATHS:
        if Path(path).exists():
            return path
    try:
        profiles = sorted(_WINDOWS_USERS.iterdir()) if _WINDOWS_USERS.is_dir() else []
    except OSError:
        return None
    for profile in profiles:
        candidate = profile / _CHROME_IN_PROFILE
        try:
            if candidate.exists():
                return str(candidate)
        except OSError:
            continue
    return None


def launch_commands(url: str, *, wsl: bool | None = None) -> list[tuple[str, list[str]]]:
    """Ways to open `url`, best first, as (what it opens in, argv) pairs.

    Split out from `open_url` because the choice is the interesting part and
    the launching is a `Popen` call.
    """
    override = os.environ.get("LIT_BROWSER", "").strip()
    out: list[tuple[str, list[str]]] = []
    if override:
        argv = shlex.split(override)
        if argv:
            out.append((Path(argv[0]).name, [*argv, url]))

    if not (is_wsl() if wsl is None else wsl):
        return out

    chrome = find_chrome()
    if chrome:
        out.append(("Chrome", [chrome, url]))
    wslview = shutil.which("wslview")
    if wslview:
        out.append(("the Windows default browser", [wslview, url]))
    powershell = shutil.which("powershell.exe")
    if powershell:
        # `-Command` is PowerShell source, not an argument vector: an unquoted
        # `&` in a query string would end the statement, so the URL is quoted
        # and its own single quotes are doubled.
        quoted = url.replace("'", "''")
        out.append((
            "the Windows default browser",
            [powershell, "-NoProfile", "-NonInteractive", "-Command",
             f"Start-Process '{quoted}'"],
        ))
    return out


def open_url(url: str | None) -> str | None:
    """Open `url` in a browser. Returns what opened it, or None if nothing did.

    Never blocks: the browser is launched and left to get on with it.
    """
    if not url or not url.lower().startswith(_SCHEMES):
        return None

    for what, argv in launch_commands(url):
        if _spawn(argv):
            return what

    # Not WSL, or nothing Windows-side answered. The stdlib knows about
    # $BROWSER and the desktop's own handler.
    return "your browser" if webbrowser.open(url) else None


def _spawn(argv: list[str]) -> bool:
    try:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(_WINDOWS_CWD) if _WINDOWS_CWD.is_dir() else None,
        )
        return True
    except OSError:
        return False
