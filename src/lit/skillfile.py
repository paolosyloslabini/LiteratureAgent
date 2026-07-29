"""The Claude Code skill, and its installer.

The skill text lives here as a string so `lit skill install` works from an
installed wheel with no data files to locate, and so `lit skill show` can always
print exactly what would be installed.
"""

from __future__ import annotations

from pathlib import Path

SKILL_NAME = "literature"

SKILL_BODY = """---
name: literature
description: >-
  Search, cite and query the user's research libraries built with the `lit` CLI
  (literature-agent). Use whenever the user asks about their library or
  bibliography, asks you to find/add papers, asks a research question that
  should be answered from real papers with citations, asks for a BibTeX entry or
  a source for a claim, or is writing a paper and needs references. Triggers on:
  "my library", "my bibliography", "find papers on", "add this paper", "what do
  my papers say about", "cite this", "who first showed", "source for the claim
  that", "trace this claim".
---

# Literature libraries (`lit`)

The user keeps curated research libraries on disk. Each library has a **scope**
(a topic) and contains entries: real published papers that an agent has read in
full, with summaries, tags, a quality level, a BibTeX entry, and the paper's own
reference list.

Prefer these commands over generic web search when the question is about
literature the user already collects. The library is vetted; the open web is not.

## Orientation — run these first

```bash
lit libs                 # which libraries exist, and which is the default
lit info                 # scope, entry count, levels, top tags of the current one
lit --json info          # same, machine-readable
```

**`--json` and `-L <name>` go before the subcommand, not after it** —
`lit --json ls`, `lit -L <name> info`. Every command below takes `--json`;
always pass it when you are going to parse the output.

## Reading the library

```bash
lit --json ls --level A --tag benchmarks       # filter by level, tag, year, status
lit --json show <key>                          # one entry, with sections + references
lit abstract <key> [<key> ...]                 # just the abstract(s), to stdout
lit --json search "causal discovery in agents" # ranked entries + a short answer
```

Two fields worth knowing about on every entry:

- `abstract` — the **publisher's** abstract, verbatim from Crossref/arXiv/S2.
  It is metadata, so it is present even on UNVERIFIED entries. It is not a
  summary anybody wrote and never evidence that the paper was read; quote it as
  the authors' own framing, nothing more. `lit refresh` backfills it if blank.
- `code_url` — the code or artifact repository (GitHub, Hugging Face, Zenodo,
  …). Offer it when the user asks whether a paper has code or wants to reproduce
  it. `code_source` says where it came from and you must repeat that when you
  quote the link: `paper` (or null, on older entries) means the paper's own text
  printed it; `web` means `lit code` found it by searching, and `code_reason`
  holds the evidence. A null `code_url` means nobody has found a link yet — run
  `lit code <key>` rather than substituting a repository you happen to know of.

`lit search` works from stored summaries — it is fast and cheap. Use it to find
out **which** papers are relevant.

## Answering a question properly

```bash
lit --json ask "What are current agent capabilities in causal discovery?"
lit --json ask "..." --read 8 --expand 3
```

`lit ask` opens the actual PDFs and reads them in parallel, then answers with
**exact quotes** and a bibliography. Use it, not `lit search`, when the user
wants a real answer rather than a reading list. `--expand N` also pulls in up to
N works cited by those papers that aren't in the library yet. Those arrive with
`status: unread` — they are quoted from in this answer, but nobody has written
their stored summary, so `lit read <key>` is what fills it in.

It costs real time and tokens (one agent per paper), so pick `--read` sensibly:
3-5 for a focused question, 8+ for a survey.

## Adding papers

```bash
lit add "Attention Is All You Need"          # by title
lit add 10.1145/3292500.3330701              # by DOI
lit add arxiv:1706.03762                     # by arXiv id
lit add "Some Paywalled Paper" --pdf ~/Downloads/paper.pdf
lit --json find "LLM benchmarks for scientific discovery" -n 10
lit --json read <key>                        # read a filed paper in full
lit --json read --all --level A -n 5         # or work through the backlog
```

- `add` fetches one paper, reads the **whole text**, and writes the summaries.
- `find` searches Crossref, Semantic Scholar and arXiv several ways, mines the reference
  lists of papers already in the library, de-duplicates everything, and ranks it
  with one cheap call. It **files papers without reading them**: they get their
  metadata, abstract and level, and `status: unread`. This is cheap — prefer it,
  and do not apologize for the entries being unread.
- Give `find` the request in plain language, including any date or document-type
  condition — "surveys of retrieval-augmented generation since 2022". One cheap
  call reads it into a topic, alternate phrasings, a year window and a document
  type before the indexes are asked, and the `plan` key in `--json` reports what
  it understood. Do not pre-flatten the request into keywords yourself, and do
  not run several `find`s for one question. `--no-plan` searches the string
  verbatim.
- `read` is the expensive step, bought per paper. Run it on the entries that
  actually matter rather than on everything `find` returned. `find --read` does
  it inline instead, and costs accordingly.
- A paper already read in another library under the same root is not read again:
  its summaries are copied and the message says which library they came from.
  That is expected, not a failure — use `--refresh` if you want a fresh reading
  written against this library's scope.
- A reading covers the paper, not its back matter: the bibliography and the
  appendices are dropped before the reader sees them, and sections are recorded
  at top level. `lit ask` still reads the appendix, because evidence for a
  specific question often sits in a table there.
- `find --parallel` adds LLM scout agents to the free search. It finds work
  keyword search misses — adjacent fields, negative results — and costs real
  tokens. Use it when the free search came back thin, not by default.
- Use `--dry-run` to see candidates without adding any.
- Entries below the library's quality bar are rejected; `--force` overrides.
- If the full text can't be reached, the entry is stored **UNVERIFIED** with
  blank summaries. Never present an UNVERIFIED entry's content as if it were
  summarized — there is nothing there. Tell the user to drop the PDF in
  `pdfs/inbox/` and run `lit inbox`.
- Books and other long sources are **sampled**, not read whole. Such entries
  have `partial_read: true` and a `coverage` string like `10 of 412 pages
  (sampled)`. Say so when you cite one, and never imply its summary covers the
  whole work. `lit config set fetch.max_read_pages <n>` raises the budget.

## Finding a paper's code

```bash
lit --json code <key>                    # search the web for its repository
lit --json code --all --level A -n 10    # the backlog, best papers first
lit --json code <key> --dry-run          # report without storing
```

One cheap agent per paper, with web search. It only records a repository whose
page names *this* paper, and only after checking the URL resolves, so a `null`
result means it genuinely could not confirm one — do not fill that gap from your
own knowledge. A third-party reimplementation is refused unless `--unofficial`
is passed. Entries that already have a link are skipped; `--force` re-searches.

Use it when the user asks whether a paper has code, wants to reproduce a result,
or is looking for an implementation. Cite what it stores as web-found, not as
the paper's own — see `code_source` above.

## Keeping a library current

```bash
lit --json refresh          # re-fetch citation counts, references, venues
lit --json refresh <key>    # just one entry
```

Network-only and free — no LLM, nothing re-read. Run it before quoting citation
counts, or when a preprint in the library may have been published since. It
never touches summaries or notes. To re-read a paper itself, use `lit reread`.

## Citations and claims

```bash
lit cite vaswani2017attention                    # BibTeX to stdout
lit cite --level A --format markdown             # a reading list
lit cite --format bibtex -o refs.bib             # a .bib file for a paper
lit --json claim "transformers scale better than LSTMs"
```

`lit claim` traces a claim backwards through citations to the paper that
originates it, and reports the chain: `Paper A (2020) -> B (2015) -> C (2008,
original)`. Use it whenever the user asks "who first showed X" or needs a
primary source rather than a convenient recent citation.

## Sharing

```bash
lit export -o mylib.litlib          # a bundle to send someone
lit import theirs.litlib --into my-library
```

## Rules that matter

1. **Never write to `## Notes`.** That section of every entry file is the
   user's. Only `lit note <key> "..."` touches it, and only when the user asks
   you to record something. Do not edit entry files directly with Edit/Write —
   use the CLI, which regenerates them safely.
2. **Never invent a citation.** If `lit` doesn't have it, say so and offer to
   `lit add` it. Every key you cite must come from a command's output.
3. **Quote from `lit ask`, not from memory.** Its quotes are copied out of the
   real text; anything you recall about a paper is not.
4. **Check `status`.** `unread` means the paper is filed but nobody has read it
   yet — its abstract is real, its summaries are empty, and `lit read <key>`
   fills them in. `UNVERIFIED` means the full text was looked for and could not
   be found; only a PDF via `pdfs/inbox/` will fix that. Never present either
   one's summary as if it existed.
5. **Spend reads deliberately.** `lit read` and `lit find --read` are the only
   commands that cost serious tokens. Reading twenty papers to answer one
   question is the wrong shape: file cheaply with `find`, then read the few that
   the user's question actually turns on.
6. Long-running commands (`find --read`, `read`, `ask`, `claim`) run several
   agents at once. Run them in the background if the user is waiting, and tell
   them what's going.

## Setup, if `lit` isn't there

`lit` is not on PyPI — it installs from its repository:

```bash
lit libs || uv tool install git+https://github.com/paolosyloslabini/LiteratureAgent.git
lit new <name> --scope "<topic>"    # create the user's first library
```
"""


def skill_source() -> Path:
    """Where the skill would be installed by default."""
    return Path.home() / ".claude" / "skills" / SKILL_NAME / "SKILL.md"


def install_skill(*, project: bool = False, root: Path | None = None) -> Path:
    """Write SKILL.md into the user's (or project's) Claude Code skills dir."""
    if root is not None:
        base = Path(root)
    elif project:
        base = Path.cwd() / ".claude" / "skills"
    else:
        base = Path.home() / ".claude" / "skills"
    dest = base / SKILL_NAME / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(SKILL_BODY, encoding="utf-8")
    return dest
