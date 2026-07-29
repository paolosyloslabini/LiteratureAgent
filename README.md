# LiteratureAgent

A small, LLM-driven tool to collect, organize, and query research literature.

`lit` builds a curated library of papers you can search, cite, and actually ask
questions of. Every summary in it was written by an agent that read the whole
paper — never an abstract, never from memory. Papers it cannot reach are stored
flagged `UNVERIFIED` with blank summaries rather than plausibly filled in.

Entries are plain Markdown files, so the library is diffable, git-friendly, and
readable without this tool.

```
$ lit ask "What are current agent capabilities in causal discovery?"
Selecting sources from the library…
Reading 5 papers in full (4 at a time)…
  ✓ jin2023cladder: 3 quote(s)
  ✓ kiciman2023causal: 2 quote(s)
  · vashishtha2023causal: not relevant
Synthesizing the answer…
```

## Install

Needs Python 3.10+ and the [Claude Code](https://claude.com/claude-code) CLI —
`lit` shells out to `claude -p` for every LLM call, so it uses your existing
Claude authentication and needs no API key.

Not published to PyPI — install straight from the repository:

```bash
uv tool install git+https://github.com/paolosyloslabini/LiteratureAgent.git
# or: pipx install git+https://github.com/paolosyloslabini/LiteratureAgent.git

lit skill install     # let Claude Code drive your libraries
```

To work on it, clone and install editable so `lit` tracks your working tree:

```bash
git clone https://github.com/paolosyloslabini/LiteratureAgent.git
cd LiteratureAgent
uv tool install --editable .          # `lit` on PATH, follows your edits
```

Or in a plain virtualenv, without putting `lit` on your PATH:

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest            # 345 tests, no network, no API calls
.venv/bin/lit --help
```

## Quick start

```bash
lit new llm-benchmarks --scope "LLM benchmarks for scientific discovery"

lit add "Attention Is All You Need"          # by title
lit add 10.1145/3292500.3330701              # by DOI
lit add arxiv:1706.03762                     # by arXiv id
lit add "Paywalled Paper" --pdf ~/Downloads/paper.pdf

lit find "benchmarks for scientific discovery" -n 10

lit search "causal discovery"                # which of my papers cover this?
lit ask "What are the current capabilities?"  # read them and answer, with quotes
lit claim "transformers scale better than LSTMs"   # trace it to its origin

lit cite --level A --format bibtex -o refs.bib
```

## Commands

| | |
|---|---|
| `lit new` / `libs` / `use` / `info` | manage libraries |
| `lit add` | add one paper: fetch, read in full, summarize |
| `lit find` | search a topic and add papers, in parallel |
| `lit inbox` | adopt PDFs you dropped in `pdfs/inbox/` |
| `lit refresh` | re-fetch citations, references and venues (no LLM) |
| `lit ls` / `show` / `note` / `rm` | browse and annotate |
| `lit search` | find in-library sources (fast, from summaries) |
| `lit ask` | answer a question by reading the actual papers |
| `lit claim` | trace a claim back to the paper that originates it |
| `lit cite` | BibTeX, Markdown, JSON or bare keys |
| `lit export` / `import` | trade libraries with colleagues |
| `lit browse` | optional interactive browser |
| `lit skill install` | install the Claude Code skill |

Every command takes `--json`, so scripts and agents can use all of it.
`-L <library>` targets a specific library; `-v` shows progress detail.

## How it works

### Adding a paper is a read, not a lookup

`lit add` resolves the work through Crossref (authoritative, and usually
sufficient on its own), topping up from arXiv, OpenAlex and Semantic Scholar for
citation counts, open-access PDF links and reference lists. It then retrieves
the full text — arXiv, PMC/Europe PMC, Unpaywall and publisher OA links, or a
PDF you supply — and hands the entire document to a single agent that writes the
one-liner, the section-by-section summary, the key findings and the tags in one
pass.

If no full text can be reached, the entry is saved `UNVERIFIED` with **blank**
summaries. Drop the PDF into `pdfs/inbox/` and run `lit inbox` to fill it in.

**Books and long sources are sampled, not swallowed.** Feeding a 400-page book
to a model is slow and expensive and produces nothing a good sample wouldn't.
Anything longer than `fetch.long_document_pages` (50) is read as
`fetch.max_read_pages` (10) pages: the front matter and table of contents, an
even spread through the body, and the closing pages, with the omitted stretches
marked in the text. Ordinary papers — including 40-page ones with appendices —
are always read end to end.

Such an entry is recorded as a partial read and says so everywhere it appears:

```
$ lit show somebook
PARTIAL READ — 10 of 412 pages (sampled). The summary below covers that
sample, not the whole work.
```

The reader is told it is seeing a sample and instructed never to report findings
or numbers from pages it cannot see. Raise the budget with
`lit config set fetch.max_read_pages 40`, or set it to `0` for no limit.

Reference lists always come from the metadata APIs, never from the model.

A library outlives the day it was built, so `lit refresh` re-fetches citation
counts, reference lists and venues for entries you already have — and picks up
the DOI when a preprint you added has since been published. It is network-only:
no LLM call, nothing re-read, summaries and notes untouched.

### `find` is an orchestrator, not a swarm

One process builds the whole candidate pool and de-duplicates it *before* any
expensive work starts, so two agents never read the same paper:

1. **Reference mining** — free and hallucination-proof. Works cited by several
   papers already in your library, but missing from it, are exactly the gaps
   worth filling. One cheap call filters them for relevance.
2. **Scout agents** search the web in parallel, each on a different angle
   (foundational / recent / surveys & benchmarks / adjacent fields / critical
   work). Each is handed the titles you already have, plus what mining just
   surfaced, and told not to propose them.

Candidates found by more than one source rank highest. Only then does the read
fan-out begin, one agent per paper.

### Levels come from metrics, not vibes

Each entry gets `A*`/`A`/`B`/`C` from a curated venue table (CORE-style for CS,
plus the top general-science journals) and citation velocity, and records the
reason:

```yaml
level: A*
level_reason: 733 citations/year (preprint, landmark)
```

The model is only asked to judge when metrics are genuinely thin — a paper too
recent to be cited, or a venue not in the table — and then it gets the same
rubric plus the paper's full text. The default bar is deliberately permissive:
missing a paper that matters costs more than carrying one that doesn't, and you
can always filter at read time with `lit ls --level A`.

### Your notes are yours

Every entry file ends with a `## Notes` section below a marker. Nothing but
`lit note` ever writes there. This is enforced structurally: saving an entry
always re-reads the notes off disk and restores them, so an entry object built
from model output physically cannot overwrite them — including on re-reads and
on import.

## Library layout

```
~/lit/llm-benchmarks/
  library.toml                     scope, quality bar, settings
  entries/
    vaswani2017attention.md        YAML frontmatter + summaries + YOUR notes
  pdfs/
    vaswani2017attention.pdf
    inbox/                         drop PDFs here, then run `lit inbox`
  .text/                           extracted text cache (derived)
  .index.db                        SQLite/FTS5 search index (derived)
```

Both derived paths can be deleted at any time; `lit reindex` rebuilds them.

## Configuration

`lit config show`, `lit config set <key> <value>`, or edit the TOML directly.

```toml
root = "~/lit"
default_library = "llm-benchmarks"

[llm]
model = "sonnet"        # fallback for any role below
max_parallel = 4        # concurrent agents
timeout_s = 900

[llm.models]            # cheap steps run cheap
scout = "haiku"         # web search for candidates
filter = "haiku"        # ranking candidates and library hits
reader = "sonnet"       # reading a full paper — the quality step
analyst = "sonnet"      # extracting quoted evidence, tracing claims
synthesis = "sonnet"    # writing the final cited answer

[fetch]
email = ""              # enables Unpaywall + polite-pool rate limits
semantic_scholar_api_key = ""
fallback_url_template = ""   # last-resort resolver; see "Paywalls" below
fallback_cmd = ""
long_document_pages = 50     # longer than this counts as a book, not a paper
max_read_pages = 10          # pages sampled from such a document (0 = no limit)
```

Override per command with `--model <alias>` and `--parallel N`, or per role with
`LIT_MODEL_READER=opus`. Library-level settings live in `library.toml`:

```bash
lit config set --library min_level A
lit config set --library allow_non_published true    # admit videos, lectures
```

## Sharing libraries

```bash
lit export -o llm-benchmarks.litlib          # add --with-pdfs to include papers
lit import theirs.litlib --into my-library
```

A `.litlib` is a zip of `library.toml`, the entry files, and optionally the PDFs
— never the rebuildable index. On import, entries are matched by DOI/arXiv/title
rather than by key, a read entry is never replaced by an unread one, and **your
notes always survive**: theirs are appended below yours under an attributed
heading. Since entries are plain Markdown, sharing a library over git works too.

## Using it from Claude Code

```bash
lit skill install          # ~/.claude/skills/literature/
lit skill install --project
```

Claude Code can then run the subcommands itself — "what does my library say
about X?", "add the Transformer paper", "find me a source for this claim". The
skill tells it to prefer `lit ask` over its own recall, never to invent a
citation, and never to write to your notes.

## Paywalls

Out of the box `lit` uses open-access routes only: arXiv, PMC/Europe PMC,
Unpaywall, OpenAlex OA links and publisher OA pages. For anything else, download
the PDF yourself with whatever access you have and either:

```bash
lit add "Some Paper" --pdf ~/Downloads/paper.pdf
cp ~/Downloads/*.pdf ~/lit/mylib/pdfs/inbox/ && lit inbox
```

`lit inbox` identifies each PDF from its own first page, matches it to the
`UNVERIFIED` entry it belongs to, and fills in the summaries.

### Last-resort resolver

If open access and your own PDFs aren't enough, you can configure one fallback
that runs **after every other route has failed** — a URL template keyed on DOI,
or a shell command:

```bash
# A URL: may point at a PDF, or at a viewer page with the PDF embedded
# (the embedded link is extracted automatically).
lit config set fetch.fallback_url_template "https://<host>/{doi}"

# Or any command you supply; {doi} and {out} are substituted and it must
# leave a PDF at {out}.
lit config set fetch.fallback_cmd "my-fetcher {doi} -o {out}"
```

Both ship empty and nothing is contacted unless you set one. `lit` does not
bundle or endorse any particular endpoint: which source you point this at, and
whether using it is lawful where you live, is your decision — repository
legality varies by jurisdiction. Entries fetched this way record
`read_source: fallback-url` (or `fallback-cmd`) so the provenance stays visible.

## Development

```bash
uv pip install -e ".[dev]"
python -m pytest
```

## License

MIT
