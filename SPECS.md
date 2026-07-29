# SPECS

This document contains the original specs for our tool. 

Our tool is designed to quickly create and mantain a bibliography, for learning or paper writing or curiosity.

## Libraries 
Each project creates a new, independent library. Libraries can be created, navigated and queried from the command line. 

A library has a scope: a specific topic the user is investigating (e.g. LLM Benchmarks)
A library contains entries: published papers, books, other documents

Each entry has (searchable) parameters:
- one-line summary (LLM generated)
- detailed summary: a section-by-section summary (much shorter than the whole paper, LLM generated)
- tags
- type: workshop paper, conference paper, book, etc
- link
- bib entry: venue, year, etc
- bibliography: papers referenced BY this paper
- level: A*, A, ... (an approximate classification of papers by quality, mimicking conference scores. LLM generated from clear metrics)
- user notes (NEVER WRITTEN BY the LLM)

Libraries only contain HIGH-QUALITY material: highly cited papers, or papers from A*, A conferences, or from established researchers, etc.
(OPTIONAL) For learning, libraries may contain videos or other non-published (but still high-quality) material. Tunable. 

## Actions/skills within a library:
- *Add a paper*: find a given paper (by name or description). Adding a paper *requires*
--the agent reading the whole content of the paper (not just skimming). 
--the same agent that read needs to fill the summaries and other properties.
--if a paper cannot be read (not reachable), this is flagged (UNVERIFIED) and the summaries are left blank
--use proper tools to read the paper and verify the bib entry.

- Search and add paper: automatically fill the library with papers that match a given topic and query, e.g. "Find papers in LLM benchmarks related to scientific discovery". Happens in parallel. 

- Search in library: provide in-library sources for a given query, e.g. "What are the current capabilities in causal discovery for AI agents?". This should return the relevant papers along with a short answer. 

- Answer: use the library to answer a question to the best of your abilities. Use the summaries to navigate and identify, but READ the actual papers to get conclusions, not just the summaries. Possibility to pull new papers (e.g. in mentioned in a bibliography). When answering, always provide exact quotes from the papers, and a bibliography. 

- Claim: find a source for a claim. 
1. Find the paper that makes the claim you want to cite
2. Fetch its references
3. Identify which reference the claim actually originates from
4. Fetch and read that earlier paper to confirm it is the true source
5. Repeat until you reach the paper that introduces the concept
6. Report the full chain: `Paper A (2020) -> Paper B (2015) -> Paper C (2008, original)`

## Library as an augment for agents
When writing a paper, the library is intended to augment both the user and the agent ability to support the user. Let's build commands that make it easy to navigate and extract biography. Let's make it easy to run tool autonomously (without the user having to supervise each step) but leave the option option for supervision (e.g. the user can choose to validate the new papers before adding, or can just automatically pre-approve them. )


## API GUIDE

Fetch papers with WebFetch? What alternatives? Make this more precise and useful. We want to be always able to read the papers, maybe try scihub when all else fail?

### CrossRef: DOI to BibTeX

Returns a ready-to-use BibTeX entry:
```
curl -sLH "Accept: application/x-bibtex" "https://doi.org/{DOI}"
```
For richer JSON metadata:
```
curl -sL "https://api.crossref.org/works/{DOI}" | jq '.message'
```

### arXiv API

Fetch metadata for an arXiv paper:
```
curl -s "https://export.arxiv.org/api/query?id_list={ARXIV_ID}"
```

Always check if a published version exists (look for `doi` in the response or search CrossRef). Prefer the published version.

### Semantic Scholar: Citation Chain Tracing

Fetch a paper's reference list:
```
curl -sL "https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}?fields=references.title,references.authors,references.year,references.externalIds" | jq '.references'
```