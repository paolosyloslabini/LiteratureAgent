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
.venv/bin/python -m pytest            # 577 tests, no network, no API calls
.venv/bin/lit --help
```

## Quick start

```bash
lit new llm-benchmarks --scope "LLM benchmarks for scientific discovery"

lit add "Attention Is All You Need"          # by title
lit add 10.1145/3292500.3330701              # by DOI
lit add arxiv:1706.03762                     # by arXiv id
lit add "Paywalled Paper" --pdf ~/Downloads/paper.pdf

lit find "benchmarks for scientific discovery since 2023" -n 10  # files them, reads nothing
lit read --all --level A         # read the good ones, when you want them read
lit find "..." --parallel        # add LLM scouts to the free index search

lit code --all -n 5                          # go find the code for them
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
| `lit find` | search a topic and file the papers that fit (no tokens by default) |
| `lit read` | read filed papers in full and write their summaries |
| `lit inbox` | adopt PDFs you dropped in `pdfs/inbox/` |
| `lit refresh` | re-fetch citations, references and venues (no LLM) |
| `lit ls` / `show` / `note` | browse and annotate |
| `lit delete` | remove entries, their PDFs and their cached text (`lit rm`) |
| `lit abstract` | print an entry's abstract, ready to pipe |
| `lit code` | search the web for the repository that implements a paper |
| `lit search` | find in-library sources (fast, from summaries) |
| `lit ask` | answer a question by reading the actual papers |
| `lit claim` | trace a claim back to the paper that originates it |
| `lit cite` | BibTeX, Markdown, JSON or bare keys |
| `lit export` / `import` | trade libraries with colleagues |
| `lit browse` | optional interactive browser |
| `lit skill install` | install the Claude Code skill |

Every command takes `--json`, so scripts and agents can use all of it. It is a
global flag, so it goes *before* the subcommand: `lit --json ls`. Same for
`-L <library>`, which targets a specific library, and `-v`, which shows
progress detail.

## How it works

### Adding a paper is a read, not a lookup

`lit add` resolves the work through Crossref (authoritative, and usually
sufficient on its own), topping up from arXiv and Semantic Scholar for
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

### Abstracts and code links

Every entry keeps the **publisher's abstract** verbatim, straight from
Crossref/arXiv/Semantic Scholar. It is metadata rather than anything an agent
wrote, so it is on record even for `UNVERIFIED` entries whose full text was
never reached, and it is searchable like the rest of the entry:

```bash
lit abstract vaswani2017attention        # plain text, pipeable
lit --json show vaswani2017attention | jq -r .abstract
```

Papers with code get a **`code_url`** — GitHub, GitLab, Hugging Face, Zenodo,
OSF, Code Ocean and the like — found by the agent while it reads the paper, in
the places authors actually put them: the abstract, a first-page footnote, or
the availability statement at the end. `lit show` prints it, and `--json`
carries it.

The link is only kept if the paper really printed it. A model asked for a
repository will otherwise supply a confident, well-formed URL from memory — the
right lab, a project that does not exist — so the returned URL is checked
against the very text the model was given, compared loosely enough to survive a
PDF snapping the URL across two lines. A bare `github.com/some-org` with no
project on it is rejected too. No link recorded means the paper printed none;
it does not mean there is no code.

Plenty of papers release code without printing the URL, or release it after
publication. `lit code` goes looking:

```bash
lit code vaswani2017attention        # one paper
lit code --all --level A -n 10       # the backlog, best papers first
lit code <key> --dry-run             # see what it found, store nothing
```

One cheap agent per paper (haiku by default), with web search. It works from
pages rather than recall: it opens the candidate repository, and records it only
if the repository names *this* paper — its title, arXiv id, DOI or a BibTeX
block citing it — quoting the line that proves it. Then `lit` checks the URL
resolves, because the invented ones 404. A repository the authors did not
release is refused unless you pass `--unofficial`; being unable to answer at all
is a normal outcome and costs one cheap call.

What it finds is stored marked `code_source: web`, alongside the evidence, and
every place that displays a link says which kind it is:

```
Code: https://github.com/tensorflow/tensor2tensor (printed in the paper)
Code: https://github.com/some/repo (found on the web)
author release, high confidence — the README cites the paper's arXiv id
```

A searched-for repository is a weaker claim than one the authors printed, and
the two are never shown as the same thing. Re-reading a paper keeps a
searched-for link, and replaces it if the text turns out to print one.

A library outlives the day it was built, so `lit refresh` re-fetches citation
counts, reference lists and venues for entries you already have — and picks up
the DOI when a preprint you added has since been published. It is network-only:
no LLM call, nothing re-read, summaries and notes untouched.

### `find` spends API calls, not tokens

Ask for what you actually want, in a sentence:

```bash
lit find "surveys of retrieval-augmented generation since 2022"
```

A keyword index cannot answer that sentence — handed it whole, Crossref matches
`surveys` and `since` as search terms. So the first thing `find` does is spend
one cheap call reading the request into parameters an index *does* have fields
for: the topic in the field's own vocabulary, a couple of alternate phrasings
for the communities that name it differently, a year window, a document type.
It prints what it understood before spending anything else, and `--no-plan`
searches your string verbatim instead.

The year window and document type then bind every source, not just the ones
with a filter field — a reference mined from your own library and a title a
scout came back with are held to "since 2022" too. The tags it derives go on
everything the run files, which is the only handle an unread entry has until
someone reads it.

One process then builds the whole candidate pool and de-duplicates it *before*
any expensive work starts, so two agents never handle the same paper. From here
everything is free:

1. **Reference mining** — hallucination-proof. Works cited by several papers
   already in your library, but missing from it, are exactly the gaps worth
   filling.
2. **Indexed search** across Crossref, Semantic Scholar and arXiv. The planned
   query is asked five ways — best keyword match, most-cited, published in the last two
   years, reviews only, arXiv preprints — because one query asked one way
   returns a monoculture. Alternate phrasings are asked on best-match only:
   re-running "most-cited" over a paraphrase returns what the first phrasing
   already found. Each facet is a plain HTTP request returning real works with
   real DOIs, citation counts and abstracts.

That is the whole search. No agent runs, and the only model calls in the entire
command are the one that reads your query and the one that ranks the pool —
both on the cheapest model configured.

**`--parallel` adds the scout agents** on top: LLM agents searching the web, one
per angle. They cost real tokens — an agentic loop re-sends its transcript every
turn, so fetched pages get paid for repeatedly — which is exactly why they are
opt-in. Their angles lead with what a keyword index cannot do: adjacent fields
that name the problem differently, and critical or negative-result work. The
facets above already cover most-cited, recent, surveys and preprints for nothing,
so the scouts are pointed at the rest.

The pool is then ranked on **relevance** to what you asked (65%) and
**importance** (35%), and cut to `-n`:

- *Relevance* is one cheap call scoring the whole pool at once, each candidate
  shown with the opening of its abstract. Every candidate is scored the same way
  whatever found it — a facet works from an angle, not from your query, so its
  hits need checking just as much as a mined reference does. Anything scoring
  below 0.35 is dropped outright, so a famous paper that does not answer the
  question cannot take a slot on reputation.
- *Importance* is computed, not judged: citation velocity and venue rank (the
  same metrics behind the levels below). Indexed hits arrive carrying those
  figures; anything else is filled in by one batched Semantic Scholar lookup
  **before** the cut.
  Papers from the last two years are scored on venue alone, so new work is not
  buried for having had no time to accumulate citations, and a paper nothing is
  known about is treated as middling rather than worthless. Co-citation in your
  library and agreement between sources contribute the remaining weight.

When the pool is larger than the scoring cap, corroboration decides who gets
scored — but not exclusively. A share of the cap is reserved for candidates
nothing corroborates, because on a topic you have not covered yet the paper that
opens a new direction is cited by none of your entries and was found by one
facet. Ranking on agreement alone would let your library's existing shape decide
what it is allowed to learn next.

`lit find` prints both numbers per candidate, so you can see why the cut fell
where it did.

### Reading is a separate, deliberate purchase

Filing a paper and reading it are two steps. `lit find` does the first: the
bibliographic record, the publisher's abstract, references and a level from
metrics, all from API calls, and the entry is marked `unread`.

Reading is the expensive half — a section-by-section summary of twenty papers is
by far the most costly thing this tool can do, and most of those twenty will not
turn out to be worth it. So you buy it per paper, once you can see what you got:

```bash
lit find "benchmarks for scientific discovery" -n 20   # cheap; nothing is read
lit ls --status unread                                 # see what turned up
lit read vaswani2017attention                          # read the ones that matter
lit read --all --level A -n 5                          # or the best of the backlog
lit find "..." -n 10 --read                            # or read as you go
```

An unread entry is not a broken one. Its abstract is in the search index, so
`lit search` finds it, and `lit ask` will read it from source when it needs to.
That is different from `UNVERIFIED`, which means the full text *was* looked for
and could not be found.

When a read does happen, the paper's own bibliography is stripped before the
text is sent. The reader is told not to summarize it, and every command that
needs the citations gets them as structured data from the metadata APIs — so it
was only ever paying for weight.

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

## The browser

`lit browse` is an optional two-pane viewer over a library: entries on top,
the full record below. It buys nothing the subcommands don't — it is a faster
way to work through a backlog.

| key | |
| --- | --- |
| `↑` `↓` | move through the list; the pane below follows |
| `^d` `^u` | scroll the record (or `tab` into it, or use the wheel) |
| `R` | read this paper — `lit read <key>`, after a confirmation |
| `o` / `c` | open the paper / its code repository in a browser |
| `C` | find this paper's code — `lit code <key>`, after a confirmation |
| `n` | edit your notes (saved exactly as `lit note` saves them) |
| `d` | delete this entry — `lit delete <key>`, after a confirmation |
| `/` `f` `s` `r` | search · filter · sort · reload |

A read started with `R` runs in the background: the browser stays usable, the
row shows `reading…`, and the cost lands in a notification when it finishes.
`C` works the same way, showing `code…`, and reports the repository it found —
or that this paper has none to find. When the entry already links one, the
confirmation says which link is about to be replaced.

The record below names the provenance of every code link it shows, so a
repository `C` found on the web never reads as one the authors printed.

`d` deletes the same things `lit delete` does, and cannot be undone, so it asks
first — saying so out loud when the entry carries notes you wrote. It declines
while an agent is still working on that entry, and leaves the cursor where the
row was so you can work down a backlog without scrolling back each time.

Links open in the browser you actually look at. Under WSL that means the
Windows side — Chrome if it is installed, otherwise the Windows default
browser. `LIT_BROWSER` overrides the choice with a command of your own.

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
read_chars = 400_000    # paper text per reading prompt (`lit add`, `lit read`)
find_read_chars = 150_000   # tighter budget for `lit find --read`, a whole batch

[llm.models]            # cheap steps run cheap
plan = "haiku"          # reading your query into index search parameters
scout = "haiku"         # web search for candidates
code = "haiku"          # web search for a paper's implementation
filter = "haiku"        # filtering library hits
rank = "haiku"          # scoring candidates for relevance to the query
reader = "sonnet"       # reading a full paper — the quality step
analyst = "sonnet"      # extracting quoted evidence, tracing claims
synthesis = "sonnet"    # writing the final cited answer

[llm.efforts]           # reasoning effort, stated rather than inherited
plan = "low"
scout = "low"
filter = "low"
rank = "low"
reader = "medium"       # reading a paper is comprehension, not reasoning
analyst = "medium"      # quotes must be exact, so not "low"
synthesis = "medium"

[fetch]
email = ""              # enables Unpaywall + polite-pool rate limits
semantic_scholar_api_key = ""
fallback_url_template = ""   # last-resort resolver; see "Paywalls" below
fallback_cmd = ""
long_document_pages = 50     # longer than this counts as a book, not a paper
max_read_pages = 10          # pages sampled from such a document (0 = no limit)
```

Override per command with `--model <alias>`, `--effort <level>` and `--workers N`
(`-j`) — global flags too, so before the subcommand — or per role with
`LIT_MODEL_READER=opus` / `LIT_EFFORT_READER=high`.
`--workers` sets how many agents run at once; `lit find --parallel` is a
different switch, deciding whether the scout agents run at all.

`lit` states an effort on every call rather than inheriting the `effortLevel`
from your own `settings.json` — that setting is chosen for interactive work, and
none of these calls is a reasoning problem. On a paper read, `xhigh` spent 8,010
output tokens where `medium` recorded the same paper just as faithfully in
1,921: five top-level sections instead of ten sub-sections, the same headline
finding, for 36% less. Set `effort = ""` to go back to inheriting it.

Library-level settings live in `library.toml`:

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
Unpaywall, open-access links from the indexes, and publisher OA pages. For anything else, download
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
