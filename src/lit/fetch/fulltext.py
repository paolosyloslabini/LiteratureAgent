"""Full-text acquisition: get the actual words of a paper, or fail honestly.

Resolvers are tried in order of reliability. The built-in ones are all sources
the publisher or author has made openly available (arXiv, PMC, Semantic Scholar and
Unpaywall OA links, publisher OA pages), plus PDFs you supply yourself.

After those, a single user-configured resolver of last resort can run — either
a URL template or a shell command. It ships empty and is never contacted unless
you set it; what you point it at, and whether that is lawful where you are, is
your call. See `FetchConfig` and the README.

If nothing works, the caller marks the entry UNVERIFIED and leaves the summaries
blank rather than inventing them from an abstract.
"""

from __future__ import annotations

import html as html_mod
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from ..config import FetchConfig
from .http import HttpClient
from .metadata import PaperMeta

# Below this many characters we assume extraction failed (scanned PDF with no
# text layer, a cover page, a captcha wall, an abstract-only landing page).
MIN_USABLE_CHARS = 4000

UNPAYWALL_API = "https://api.unpaywall.org/v2"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


@dataclass
class FullText:
    text: str
    source: str
    pdf_path: Path | None = None
    # Page accounting, when the source was a PDF. `pages_read < pages` means the
    # document was sampled rather than read end to end.
    pages: int | None = None
    pages_read: int | None = None

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def partial(self) -> bool:
        return bool(self.pages and self.pages_read and self.pages_read < self.pages)


@dataclass
class PdfText:
    text: str
    pages: int = 0
    pages_read: int = 0

    @property
    def partial(self) -> bool:
        return bool(self.pages and self.pages_read < self.pages)


def select_pages(total: int, budget: int) -> list[int]:
    """Choose which pages of a long document to read.

    Reading the first N pages of a 400-page book gets you chapter one and
    nothing else. Front matter carries the table of contents and preface — the
    map of the whole work — the closing pages carry the conclusions, and the
    body is sampled evenly in between so it is represented rather than skipped.
    """
    if budget <= 0 or total <= budget:
        return list(range(total))

    # The floors below must not push front+back past the budget, or a budget of
    # one or two pages would silently return more than asked for.
    front = min(max(1, budget // 3), budget)
    back = max(1, budget // 4) if budget - front >= 1 else 0
    middle = max(0, budget - front - back)

    pages = list(range(min(front, total)))
    lo, hi = front, total - back
    if middle > 0 and hi > lo:
        step = (hi - lo) / middle
        pages += [int(lo + i * step) for i in range(middle)]
    if back:
        pages += list(range(max(0, total - back), total))

    selected = sorted({p for p in pages if 0 <= p < total})
    return selected[:budget]


def fetch_fulltext(
    http: HttpClient,
    meta: PaperMeta,
    *,
    key: str,
    pdf_dir: Path,
    text_dir: Path,
    cfg: FetchConfig | None = None,
    local_pdf: Path | None = None,
    refresh: bool = False,
) -> FullText | None:
    """Try every available route to the full text. None if all fail."""
    cfg = cfg or FetchConfig()
    text_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    cache = text_dir / f"{key}.txt"
    dest = pdf_dir / f"{key}.pdf"

    if not refresh and cache.exists():
        cached = _read_cache(cache)
        # Scaled check, or a cached sample of a long document would be rejected
        # on reload as if extraction had failed.
        if cached and _usable(cached.text, cached.pages_read):
            cached.pdf_path = dest if dest.exists() else None
            return cached

    def read_pdf(path: Path) -> PdfText:
        return extract_pdf(
            path,
            max_pages=cfg.max_read_pages,
            long_document_pages=cfg.long_document_pages,
        )

    # 1. A PDF the user handed us wins over anything on the network.
    if local_pdf and Path(local_pdf).exists():
        got = read_pdf(Path(local_pdf))
        if _usable(got.text, got.pages_read):
            if Path(local_pdf).resolve() != dest.resolve():
                dest.write_bytes(Path(local_pdf).read_bytes())
            return _cache(cache, _full(got, f"local:{Path(local_pdf).name}", dest))

    # 2. A PDF we already downloaded for this key.
    if not refresh and dest.exists():
        got = read_pdf(dest)
        if _usable(got.text, got.pages_read):
            return _cache(cache, _full(got, "cache-pdf", dest))

    for url, label in _pdf_candidates(http, meta, cfg):
        if http.download(url, dest, max_mb=cfg.max_pdf_mb):
            got = read_pdf(dest)
            if _usable(got.text, got.pages_read):
                return _cache(cache, _full(got, label, dest))
            dest.unlink(missing_ok=True)

    for url, label in _html_candidates(meta):
        text = _html_to_text(http, url)
        if _usable(text):
            return _cache(cache, FullText(text, label, None))

    # Last resort, and only if the user configured one.
    ft = _fallback_resolver(http, meta, cfg=cfg, dest=dest)
    if ft is not None:
        return _cache(cache, ft)

    return None


def _fallback_resolver(http: HttpClient, meta: PaperMeta, *, cfg: FetchConfig,
                       dest: Path) -> FullText | None:
    """User-configured resolver of last resort. Disabled unless set.

    Runs only after every open-access route has failed, and only when the user
    has explicitly configured an endpoint or command. Nothing is shipped
    pre-configured, and nothing here is contacted by default.
    """
    if not meta.doi:
        return None  # both mechanisms are DOI-keyed

    def read_pdf(path: Path) -> PdfText:
        return extract_pdf(path, max_pages=cfg.max_read_pages,
                           long_document_pages=cfg.long_document_pages)

    if cfg.fallback_url_template:
        url = cfg.fallback_url_template.replace("{doi}", meta.doi)
        if http.download(url, dest, max_mb=cfg.max_pdf_mb):
            got = read_pdf(dest)
            if _usable(got.text, got.pages_read):
                return _full(got, "fallback-url", dest)
            dest.unlink(missing_ok=True)
        # Not a direct PDF: these endpoints usually serve a viewer page with the
        # document embedded, so follow the embedded link.
        embedded = _embedded_pdf_url(http, url)
        if embedded and http.download(embedded, dest, max_mb=cfg.max_pdf_mb):
            got = read_pdf(dest)
            if _usable(got.text, got.pages_read):
                return _full(got, "fallback-url", dest)
            dest.unlink(missing_ok=True)

    if cfg.fallback_cmd:
        cmd = cfg.fallback_cmd.replace("{doi}", meta.doi).replace("{out}", str(dest))
        try:
            proc = subprocess.run(
                shlex.split(cmd), capture_output=True, text=True,
                timeout=cfg.timeout_s * 4,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode == 0 and dest.exists():
            got = read_pdf(dest)
            if _usable(got.text, got.pages_read):
                return _full(got, "fallback-cmd", dest)
            dest.unlink(missing_ok=True)

    return None


def _embedded_pdf_url(http: HttpClient, page_url: str) -> str | None:
    """Find a PDF embedded in a viewer page (iframe/embed/direct link)."""
    r = http.get(page_url, retries=2)
    if r is None or "pdf" in r.headers.get("content-type", ""):
        return None
    html = r.text or ""
    patterns = [
        r'<iframe[^>]+src\s*=\s*["\']([^"\']+)["\']',
        r'<embed[^>]+src\s*=\s*["\']([^"\']+)["\']',
        r'href\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']',
    ]
    for pat in patterns:
        for m in re.finditer(pat, html, re.IGNORECASE):
            candidate = m.group(1).strip()
            if ".pdf" not in candidate.lower():
                continue
            return urljoin(page_url, candidate)
    return None


# --------------------------------------------------------------------------
# Candidate sources
# --------------------------------------------------------------------------

def _pdf_candidates(http: HttpClient, meta: PaperMeta,
                    cfg: FetchConfig) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(url: str | None, label: str) -> None:
        if url and url not in seen and url.startswith("http"):
            seen.add(url)
            out.append((url, label))

    if meta.arxiv_id:
        add(f"https://arxiv.org/pdf/{meta.arxiv_id}", "arxiv")
    add(meta.oa_pdf_url, "index-oa" if meta.is_oa else "oa-link")

    if meta.pmcid:
        add(f"{EUROPEPMC_API}/{meta.pmcid}/fullTextPDF", "europepmc")
        add(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{meta.pmcid}/pdf/", "pmc")

    if meta.doi and cfg.email:
        data = http.get_json(f"{UNPAYWALL_API}/{meta.doi}", params={"email": cfg.email})
        if data:
            best = data.get("best_oa_location") or {}
            add(best.get("url_for_pdf"), "unpaywall")
            for loc in (data.get("oa_locations") or [])[:4]:
                add(loc.get("url_for_pdf"), "unpaywall-alt")

    if meta.doi:
        # Some OA publishers serve a PDF straight off the DOI redirect.
        add(f"https://doi.org/{meta.doi}", "doi-direct")
    if meta.url and meta.url.lower().endswith(".pdf"):
        add(meta.url, "landing-pdf")
    return out


def _html_candidates(meta: PaperMeta) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if meta.arxiv_id:
        # arXiv's native HTML (2023+) then ar5iv's LaTeX-derived rendering.
        out.append((f"https://arxiv.org/abs/{meta.arxiv_id}v1", "arxiv-html"))
        out.append((f"https://ar5iv.labs.arxiv.org/html/{meta.arxiv_id}", "ar5iv"))
    if meta.pmcid:
        out.append((f"{EUROPEPMC_API}/{meta.pmcid}/fullTextXML", "europepmc-xml"))
    if meta.url and not meta.url.lower().endswith(".pdf"):
        out.append((meta.url, "landing-html"))
    return out


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_pdf(path: Path, *, max_pages: int | None = None,
                long_document_pages: int | None = None) -> PdfText:
    """Extract text from a PDF, optionally sampling a long document.

    `max_pages` only bites once the document exceeds `long_document_pages`, so
    ordinary papers are always read end to end and only books, theses and long
    reports are sampled. Omitted stretches are marked in the text so a reader
    knows what it is not seeing.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover
        return PdfText("")
    try:
        with fitz.open(path) as doc:
            if doc.is_encrypted and not doc.authenticate(""):
                return PdfText("")
            total = doc.page_count

            if max_pages and (long_document_pages is None
                              or total > long_document_pages):
                wanted = select_pages(total, max_pages)
            else:
                wanted = list(range(total))

            chunks: list[str] = []
            previous: int | None = None
            for i in wanted:
                if previous is not None and i > previous + 1:
                    skipped = i - previous - 1
                    chunks.append(
                        f"\n\n[... {skipped} page(s) omitted "
                        f"(pages {previous + 2}–{i}) ...]\n\n"
                    )
                chunks.append(doc[i].get_text("text"))
                previous = i
    except Exception:
        return PdfText("")

    return PdfText(text=_tidy("\n\n".join(chunks)), pages=total,
                   pages_read=len(wanted))


def pdf_to_text(path: Path, *, max_pages: int | None = None) -> str:
    """Text of a PDF, or an empty string if it has no usable text layer."""
    return extract_pdf(path, max_pages=max_pages, long_document_pages=0).text


def _html_to_text(http: HttpClient, url: str) -> str:
    r = http.get(url, retries=2)
    if r is None:
        return ""
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype:
        return ""
    return html_to_text(r.text)


def html_to_text(raw: str) -> str:
    """Crude but dependency-free HTML/XML to text."""
    if not raw:
        return ""
    # Drop non-content elements entirely, including their contents.
    raw = re.sub(
        r"<(script|style|nav|header|footer|noscript|svg|form)\b.*?</\1>",
        " ", raw, flags=re.IGNORECASE | re.DOTALL,
    )
    raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.DOTALL)
    # Preserve block boundaries so paragraphs don't run together.
    raw = re.sub(r"</(p|div|section|h[1-6]|li|tr|br|title)\s*>", "\n", raw,
                 flags=re.IGNORECASE)
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return _tidy(html_mod.unescape(raw))


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Rejoin words split across line breaks by PDF hyphenation.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _usable(text: str, pages_read: int | None = None) -> bool:
    """Did extraction actually produce a readable document?

    The flat threshold catches scanned PDFs and captcha walls. When only a
    sample of a long document was extracted there is legitimately less text, so
    the bar scales to the pages actually read — otherwise a real but sparsely
    typeset book would be mistaken for a failed extraction and filed UNVERIFIED.
    """
    n = len(text or "")
    if pages_read:
        return n >= min(MIN_USABLE_CHARS, max(600, pages_read * 250))
    return n >= MIN_USABLE_CHARS


def _full(got: PdfText, source: str, pdf_path: Path | None) -> FullText:
    return FullText(text=got.text, source=source, pdf_path=pdf_path,
                    pages=got.pages or None, pages_read=got.pages_read or None)


# The cache is a plain .txt file, so page accounting rides in a header line that
# is stripped on read. Without it a cache hit would silently turn a sampled read
# back into an apparently complete one.
_CACHE_HEADER = "<!--lit:pages="


def _cache(path: Path, ft: FullText) -> FullText:
    try:
        header = ""
        if ft.pages:
            header = f"{_CACHE_HEADER}{ft.pages},read={ft.pages_read or ft.pages}-->\n"
        path.write_text(header + ft.text, encoding="utf-8")
    except OSError:
        pass
    return ft


def _read_cache(path: Path) -> FullText | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pages = pages_read = None
    if raw.startswith(_CACHE_HEADER):
        line, _, rest = raw.partition("\n")
        m = re.match(r"<!--lit:pages=(\d+),read=(\d+)-->", line)
        if m:
            pages, pages_read = int(m.group(1)), int(m.group(2))
            raw = rest
    return FullText(text=raw, source="cache", pages=pages, pages_read=pages_read)


# Headings that begin a paper's own bibliography. Matched on a line of its own,
# optionally numbered, so a sentence mentioning "references" is not a hit.
_REF_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*|[IVXL]+\.?\s*)?"
    r"(references?|bibliography|works\s+cited|literature\s+cited)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Headings that begin the back matter, with whatever follows on the line so the
# caller can check it reads as a title. Anchoring to the start of a line is not
# enough on its own: extraction wraps prose freely, so "...is analysed in\n
# Appendix A.1, where we verify..." puts a cross-reference exactly where a
# heading would be. What separates them is what comes next — a title, or the
# rest of a sentence.
_APPENDIX_HEADING = re.compile(
    r"^[ \t]*(?:appendix|appendices"
    r"|supplement(?:ary|al)\s+(?:material|information|results|details))"
    r"(?:[ \t]+[A-Za-z0-9][\w.]*)?"     # an optional label: A, B.2, 1
    r"[ \t]*[:.\-–—]?[ \t]*(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE,
)


# A heading this early is front matter, not back matter — a table of contents
# listing "References", or a preface. Content still has to agree (see
# `_is_bibliography`); this only keeps the search away from the obvious traps.
_MIN_HEADING_POS = 0.05
_MIN_APPENDIX_POS = 0.20

# What a reference entry is made of: a year, a numbered marker, an "et al.", a
# DOI or arXiv id, a venue word. Prose uses these too, but not at this rate —
# a bibliography carries several per entry and an entry is only a line or two
# long. Counting them per 1000 characters is what tells a real reference list
# from the word "References" appearing on a line of its own somewhere it does
# not mean the bibliography.
_BIB_MARKER = re.compile(
    r"\b(?:19|20)\d{2}\b"                       # a publication year
    r"|^\s*\[\d{1,3}\]|^\s*\(\d{1,3}\)"         # [12] / (12) entry markers
    r"|\bet\s+al\b"
    r"|\barXiv\b|\bdoi\b|\bdoi\.org|https?://"
    r"|\bpp\.\s*\d|\bvol\.\s*\d"
    r"|\bIn\s+Proc|\bProceedings\b|\bJournal\b|\bConference\b|\bpreprint\b",
    re.IGNORECASE | re.MULTILINE,
)
_BIB_DENSITY = 2.0
_BIB_PROBE_CHARS = 6000
_BIB_WINDOW = 1500


def _heading_continues_a_sentence(rest: str) -> bool:
    """Is this line a back-matter heading, or a sentence that mentions one?

    A heading is followed by a title or by nothing. A cross-reference is
    followed by the rest of its sentence, which starts in lower case or on the
    punctuation that resumes it.
    """
    rest = rest.strip()
    if not rest:
        return False
    return not (rest[0].isupper() or rest[0].isdigit())


def _find_appendix(text: str, start: int = 0) -> int | None:
    """Offset of the real back-matter heading at or after `start`."""
    for m in _APPENDIX_HEADING.finditer(text, start):
        if not _heading_continues_a_sentence(m.group("rest")):
            return m.start()
    return None


def _marker_density(text: str) -> float:
    """Bibliographic markers per 1000 characters."""
    return (len(_BIB_MARKER.findall(text)) / len(text) * 1000) if text else 0.0


def _is_bibliography(text: str) -> bool:
    """Does the text right after a heading read as a list of references?"""
    return _marker_density(text[:_BIB_PROBE_CHARS]) >= _BIB_DENSITY


def _bibliography_end(text: str, start: int) -> int:
    """Where the reference list that begins at `start` stops.

    Ends at the back matter if the paper has any, otherwise where the column of
    years thins out — which is how a bibliography followed by unlabelled
    appendix content is found.
    """
    app = _find_appendix(text, start)
    limit = app if app is not None else len(text)
    pos = start
    while pos < limit:
        window = text[pos:min(pos + _BIB_WINDOW, limit)]
        if _marker_density(window) < _BIB_DENSITY / 2:
            return pos
        pos += _BIB_WINDOW
    return limit


def strip_reference_list(text: str) -> tuple[str, int]:
    """Drop the paper's own bibliography. Returns the text and chars removed.

    A reference list is routinely a fifth of an extracted paper and is pure
    weight in a reading prompt: the reader is explicitly told not to summarize
    it, and every command that needs the citations gets them as structured data
    from the metadata APIs instead. Cutting it before the text is sent is the
    single cheapest saving available on a read.

    Only the list itself goes. What follows it — an appendix, and on most recent
    papers that is the majority of the file — is left for the caller to decide
    about, because a summary does not need it but a quote hunt might. This used
    to cut to the end of the document, and refused to cut at all unless 30% of
    the file survived; on a paper whose bibliography starts at 25% and whose
    appendix fills the rest, that refusal sent the whole thing.
    """
    if not text:
        return text, 0

    floor = len(text) * _MIN_HEADING_POS
    spans: list[tuple[int, int]] = []
    for m in _REF_HEADING.finditer(text):
        if m.start() < floor:
            continue
        if spans and m.start() < spans[-1][1]:
            continue  # inside a list already claimed
        if not _is_bibliography(text[m.end():]):
            continue
        spans.append((m.start(), _bibliography_end(text, m.end())))

    if not spans:
        return text, 0

    out, prev, removed = [], 0, 0
    for start, end in spans:
        out.append(text[prev:start].rstrip())
        out.append("\n\n[... reference list omitted ...]\n\n")
        removed += end - start
        prev = end
    out.append(text[prev:])
    kept = "".join(out).rstrip()
    return kept, len(text) - len(kept)


def strip_appendix(text: str) -> tuple[str, int]:
    """Drop the back matter. Returns the text and chars removed.

    On a current machine-learning paper the appendix is routinely longer than
    the paper: prompts, hyperparameter tables, per-task breakdowns. A reader
    writing a library card does not need any of it — asked to work section by
    section it will faithfully summarize "Appendix F: In-Depth Analysis" at the
    same length as the method — so the summary paths drop it. The paths that go
    looking for evidence keep it, because a result table is exactly the kind of
    thing a question turns out to hang on.
    """
    if not text:
        return text, 0
    at = _find_appendix(text, int(len(text) * _MIN_APPENDIX_POS))
    if at is None:
        return text, 0
    kept = text[:at].rstrip()
    return kept + "\n\n[... appendix omitted ...]", len(text) - len(kept)


# A line repeated at least this many times, short enough to be furniture rather
# than prose, is a running header or footer stamped on every page.
_BOILER_MIN_REPEATS = 4
_BOILER_MAX_WORDS = 14
_BOILER_MAX_CHARS = 120
# Never let this eat a real document, however odd its typesetting. Across the
# 38 papers this was measured on, the most any one of them lost to running
# furniture was 8.5%, so this sits well clear of the honest cases while still
# refusing to act on a file that is mostly one repeated line.
_BOILER_MAX_SHARE = 0.25
_HAS_WORD = re.compile(r"[A-Za-z]{3}")


def dedupe_boilerplate(text: str) -> tuple[str, int]:
    """Drop per-page furniture. Returns the text and chars removed.

    PDF extraction stamps the running header, the footer and the arXiv spine
    onto every page: "Preprint. Under review.", "Published as a conference
    paper at ICLR 2024". On a 40-page document that is the same sentence forty
    times. Lines without letters are left alone, so a column of repeated numbers
    in a table survives.
    """
    if not text:
        return text, 0
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if (s and len(s) <= _BOILER_MAX_CHARS and len(s.split()) <= _BOILER_MAX_WORDS
                and _HAS_WORD.search(s)):
            counts[s] = counts.get(s, 0) + 1

    drop = {s for s, n in counts.items() if n >= _BOILER_MIN_REPEATS}
    if not drop:
        return text, 0
    would_remove = sum(len(s) * (n - 1) for s, n in counts.items() if s in drop)
    if would_remove > len(text) * _BOILER_MAX_SHARE:
        return text, 0

    # The first occurrence stays: on many papers the running header is the only
    # place the venue is printed, and the reader is asked to recover it.
    kept, seen = [], set()
    for line in lines:
        s = line.strip()
        if s in drop:
            if s in seen:
                continue
            seen.add(s)
        kept.append(line)
    out = "\n".join(kept)
    return out, len(text) - len(out)


def truncate_for_llm(text: str, max_chars: int = 400_000, *,
                     drop_references: bool = False,
                     drop_appendix: bool = False) -> tuple[str, bool]:
    """Fit a paper into a prompt, keeping the head and tail if it is huge.

    The head carries abstract/intro/method; the tail carries results and
    discussion. The middle is where a long appendix usually sits.

    `drop_references` removes the paper's own bibliography first, so the budget
    is spent on the parts a reader is actually asked about. It is off by default
    because claim tracing reads citation context and wants it left in place.

    `drop_appendix` additionally removes the back matter, for the callers
    writing a summary rather than hunting for a quote. Per-page furniture goes
    whenever either is set — nothing asks to be told the running header twice.
    """
    if drop_references:
        text, _ = strip_reference_list(text)
    if drop_appendix:
        text, _ = strip_appendix(text)
    if drop_references or drop_appendix:
        text, _ = dedupe_boilerplate(text)
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[... middle of document omitted to fit context ...]\n\n"
        + text[-tail:]
    ), True
