# LiteratureAgent

`lit` builds and queries a curated research bibliography from the command line.

Every summary in a library is written by an agent that read the paper's full
text — never an abstract, never from memory. Papers whose text cannot be
retrieved are stored flagged `UNVERIFIED` with blank summaries rather than
plausibly filled in. Entries are plain Markdown files with YAML frontmatter, so
a library is diffable, git-friendly and readable without this tool.

```
$ lit ask "What are current agent capabilities in causal discovery?"
Selecting sources from the library…
Reading 5 papers in full (4 at a time)…
  ✓ jin2023cladder: 3 quote(s)
  ✓ kiciman2023causal: 2 quote(s)
  · vashishtha2023causal: not relevant
Synthesizing the answer…
```

## Requirements

- Python 3.10+
- The [Claude Code](https://claude.com/claude-code) CLI. `lit` shells out to
  `claude -p` for every LLM call, so it uses your existing Claude
  authentication and needs no API key.

## Install

Not published to PyPI; install from the repository:

```bash
uv tool install git+https://github.com/paolosyloslabini/LiteratureAgent.git
# or: pipx install git+https://github.com/paolosyloslabini/LiteratureAgent.git

lit skill install     # optional: let Claude Code drive your libraries
```

For development:

```bash
git clone https://github.com/paolosyloslabini/LiteratureAgent.git
cd LiteratureAgent
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest        # no network, no API calls
```

## Quick start

```bash
lit new llm-benchmarks --scope "LLM benchmarks for scientific discovery"

lit add "Attention Is All You Need"     # by title
lit add 10.1145/3292500.3330701         # by DOI
lit add arxiv:1706.03762                # by arXiv id
lit add "Paywalled Paper" --pdf ~/Downloads/paper.pdf

lit find "benchmarks for scientific discovery since 2023" -n 10
lit read --all --level A                # read the ones worth reading

lit search "causal discovery"           # which of my papers cover this?
lit ask "What are the current capabilities?"
lit claim "transformers scale better than LSTMs"

lit cite --level A --format bibtex -o refs.bib
```

## Commands

| | |
|---|---|
| `lit new` / `libs` / `use` / `info` | manage libraries |
| `lit add` | add one paper: fetch, read in full, summarize |
| `lit find` | search a topic and file the papers that fit |
| `lit read` / `reread` | read filed papers and write their summaries |
| `lit inbox` | adopt PDFs dropped in `pdfs/inbox/` |
| `lit refresh` | re-fetch citations, references and venues (no LLM) |
| `lit check` | verify suspicious metadata against authoritative sources |
| `lit ls` / `show` / `note` | browse and annotate |
| `lit delete` (`rm`) | remove entries, their PDFs and cached text |
| `lit abstract` | print an entry's abstract |
| `lit code` | find the repository that implements a paper |
| `lit search` | find in-library sources, from stored summaries |
| `lit ask` | answer a question by reading the actual papers |
| `lit claim` | trace a claim back to the paper that originates it |
| `lit cite` | BibTeX, Markdown, JSON or bare keys |
| `lit export` / `import` | trade libraries with colleagues |
| `lit browse` | interactive two-pane browser |
| `lit skill install` | install the Claude Code skill |

Every command accepts `--json`. `-L <library>` targets a specific library, `-v`
adds progress detail, and `--model` / `--effort` / `--workers` override the LLM
settings for one invocation.

## How it works

### Adding a paper

`lit add` resolves the work through Crossref, topping up from arXiv and Semantic
Scholar for citation counts, open-access PDF links and reference lists. It then
retrieves the full text — arXiv, PMC/Europe PMC, Unpaywall, publisher OA links,
or a PDF you supply — and hands the whole document to one agent that writes the
one-liner, section-by-section summary, key findings and tags.

Reference lists always come from the metadata APIs, never from the model.

Documents longer than `fetch.long_document_pages` (50) are sampled rather than
read end to end: `fetch.max_read_pages` (10) pages covering front matter, an
even spread of the body, and the closing pages. Such entries are recorded as
partial reads and labelled wherever they appear:

```
$ lit show somebook
PARTIAL READ — 10 of 412 pages (sampled). The summary below covers that
sample, not the whole work.
```

Ordinary papers, including long ones with appendices, are always read in full.

### Entry states

| state | meaning |
|---|---|
| `verified` | full text was read; summaries present |
| `unread` | filed from metadata; no read attempted yet |
| `UNVERIFIED` | a read was attempted and no full text could be found |

An unread entry is usable: its abstract is in the search index, so `lit search`
finds it and `lit ask` reads it from source when a question needs it.

### Abstracts and code links

Every entry keeps the publisher's abstract verbatim from
Crossref/arXiv/Semantic Scholar. It is metadata rather than agent output, so it
is present even on `UNVERIFIED` entries.

```bash
lit abstract vaswani2017attention
lit show vaswani2017attention --json | jq -r .abstract
```

A `code_url` is recorded only when the paper's own text prints one, checked
against the text the model was given. `lit code` searches the web for the rest:

```bash
lit code vaswani2017attention
lit code --all --level A -n 10
lit code <key> --dry-run
```

It opens candidate repositories and records one only if the repository names
*this* paper — title, arXiv id, DOI, or a BibTeX block citing it — quoting the
line that proves it; then `lit` confirms the URL resolves. Third-party
reimplementations are refused unless you pass `--unofficial`.

Web-found links are stored as `code_source: web` with the supporting evidence,
and every display distinguishes them from links the authors printed:

```
Code: https://github.com/tensorflow/tensor2tensor (printed in the paper)
Code: https://github.com/some/repo (found on the web)
author release, high confidence — the README cites the paper's arXiv id
```

### Finding papers

```bash
lit find "surveys of retrieval-augmented generation since 2022"
```

`find` spends one cheap call turning the request into index parameters — topic
in the field's vocabulary, alternate phrasings, a year window, a document type
— prints what it understood, and then searches. `--no-plan` searches your string
verbatim. The year window and document type bind every source.

Candidates come from reference mining (works cited by several of your entries
but missing from the library) and from indexed search across Crossref, Semantic
Scholar and arXiv, each query asked several ways: best keyword match,
most-cited, recent, reviews only, arXiv preprints. These are plain HTTP
requests. `--parallel` adds LLM scout agents, which cost real tokens and are
therefore opt-in.

The pool is de-duplicated, then ranked on relevance to your query (65%) and
computed importance (35%: citation velocity, venue rank, co-citation in your
library, agreement between sources), and cut to `-n`. Candidates scoring below
0.35 on relevance are dropped. Both numbers are printed per candidate.

`find` files entries as `unread` — it does not read them. Reading is the
expensive step and is bought separately:

```bash
lit find "benchmarks for scientific discovery" -n 20
lit ls --status unread
lit read vaswani2017attention
lit read --all --level A -n 5
lit find "..." -n 10 --read          # or read as you go
```

### Levels

Each entry gets `A*`/`A`/`B`/`C` from a curated venue table (CORE-style for CS
venues, plus major general-science journals) and citation velocity, and records
why:

```yaml
level: A*
level_reason: 733 citations/year (preprint, landmark)
```

Preprints and workshop papers carry no venue rank and are capped at `A` unless
their citation velocity is landmark. Papers from the last two years are judged
on venue alone. The model is asked to judge only when metrics are genuinely
thin, and is given the same rubric plus the paper's full text.

The default bar is permissive; filter at read time with `lit ls --level A`.

### Keeping metadata correct

`lit refresh` re-fetches citation counts, reference lists, venues and DOIs for
entries you already have. It is network-only: no LLM call, nothing re-read,
summaries and notes untouched.

Indexes also return metadata that is well-formed but wrong — a publisher name
where a venue belongs, a citation count of zero for a decade-old paper, a year
that disagrees with the arXiv id. `lit check` audits entries for these patterns
in code first, then spends one cheap agent per flagged entry to verify the
suspect fields against authoritative sources:

```bash
lit check                     # audit the library, report only
lit check --fix               # apply the corrections it can confirm
lit check <key> --json
```

Nothing is written without `--fix`, corrections are applied only to
machine-owned bibliographic fields, and each one records the source that
confirmed it. Summaries and notes are never touched.

### Notes

Every entry file ends with a `## Notes` section below a marker. Only `lit note`
writes there. Saving an entry always re-reads the notes from disk and restores
them, so an entry built from model output cannot overwrite them — including on
re-reads and on import.

## Browser

`lit browse` is a two-pane viewer: entries on top, the full record below. It
adds no capability over the subcommands.

| key | |
| --- | --- |
| `↑` `↓` | move through the list; the pane below follows |
| `^d` `^u` | scroll the record (or `tab` into it, or use the wheel) |
| `R` | read this paper, after a confirmation |
| `o` / `c` | open the paper / its code repository in a browser |
| `C` | find this paper's code, after a confirmation |
| `n` | edit your notes |
| `d` | delete this entry, after a confirmation |
| `/` `f` `s` `r` | search · filter · sort · reload |

`R` and `C` run in the background; the browser stays usable and the row shows
`reading…` or `code…`. Links open in your real browser — under WSL, the Windows
side. `LIT_BROWSER` overrides the choice.

## Library layout

```
~/lit/llm-benchmarks/
  library.toml                     scope, quality bar, settings
  entries/
    vaswani2017attention.md        frontmatter + summaries + your notes
  pdfs/
    vaswani2017attention.pdf
    inbox/                         drop PDFs here, then run `lit inbox`
  .text/                           extracted text cache (derived)
  .index.db                        SQLite/FTS5 search index (derived)
```

Both derived paths can be deleted at any time; `lit reindex` rebuilds them.

## Configuration

`lit config show`, `lit config set <key> <value>`, or edit
`~/.config/lit/config.toml` directly.

```toml
root = "~/lit"
default_library = "llm-benchmarks"

[llm]
model = "sonnet"            # fallback for any role below
max_parallel = 4            # concurrent agents
timeout_s = 900
read_chars = 400_000        # paper text per reading prompt
find_read_chars = 150_000   # tighter budget for `lit find --read`

[llm.models]                # cheap steps run on a cheap model
plan = "haiku"              # reading your query into search parameters
scout = "haiku"             # web search for candidates
code = "haiku"              # web search for a paper's implementation
check = "haiku"             # verifying suspicious metadata
filter = "haiku"
rank = "haiku"
reader = "sonnet"           # reading a full paper
analyst = "sonnet"          # extracting quotes, tracing claims
synthesis = "sonnet"        # writing the final cited answer

[llm.efforts]               # reasoning effort, stated rather than inherited
plan = "low"
scout = "low"
code = "low"
check = "low"
filter = "low"
rank = "low"
reader = "medium"
analyst = "medium"
synthesis = "medium"

[fetch]
email = ""                  # enables Unpaywall and polite-pool rate limits
semantic_scholar_api_key = ""
long_document_pages = 50
max_read_pages = 10         # 0 = no limit
fallback_url_template = ""  # last-resort resolver; see below
fallback_cmd = ""
```

Override per command with `--model`, `--effort` and `--workers` (`-j`), or per
role with `LIT_MODEL_READER=opus` / `LIT_EFFORT_READER=high`.

`lit` states an effort on every call rather than inheriting `effortLevel` from
your own `settings.json`, which is chosen for interactive work. Set
`effort = ""` to inherit it instead.

A Semantic Scholar API key is the most useful setting here: S2 supplies the
citation counts `find` ranks on, and unauthenticated callers share one throttled
pool. [Keys are free.](https://www.semanticscholar.org/product/api#api-key-form)

Library-level settings live in `library.toml`:

```bash
lit config set --library min_level A
lit config set --library allow_non_published true    # admit videos, lectures
```

## Sharing libraries

```bash
lit export -o llm-benchmarks.litlib      # --with-pdfs to include papers
lit import theirs.litlib --into my-library
```

A `.litlib` is a zip of `library.toml`, the entry files and optionally the PDFs
— never the rebuildable index. On import, entries are matched by
DOI/arXiv/title rather than by key, a read entry is never replaced by an unread
one, and your notes survive: theirs are appended below yours under an
attributed heading. Sharing over git works too.

## Using it from Claude Code

```bash
lit skill install              # ~/.claude/skills/literature/
lit skill install --project
```

Claude Code can then run the subcommands itself. The skill tells it to prefer
`lit ask` over its own recall, never to invent a citation, and never to write to
your notes.

## Full text and paywalls

By default `lit` uses open-access routes only: arXiv, PMC/Europe PMC, Unpaywall,
open-access links from the indexes, and publisher OA pages. For anything else,
supply the PDF yourself:

```bash
lit add "Some Paper" --pdf ~/Downloads/paper.pdf
cp ~/Downloads/*.pdf ~/lit/mylib/pdfs/inbox/ && lit inbox
```

`lit inbox` identifies each PDF from its first page, matches it to the entry it
belongs to, and fills in the summaries.

One last-resort resolver can be configured, tried only after every other route
has failed — a URL template keyed on DOI, or a shell command:

```bash
lit config set fetch.fallback_url_template "https://<host>/{doi}"
lit config set fetch.fallback_cmd "my-fetcher {doi} -o {out}"
```

Both ship empty and nothing is contacted unless you set one. `lit` does not
bundle or endorse any endpoint; which source you point this at, and whether
using it is lawful where you are, is your decision. Entries fetched this way
record `read_source: fallback-url` or `fallback-cmd`.

## Development

```bash
uv pip install -e ".[dev]"
python -m pytest
```

## License

MIT
