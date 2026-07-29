"""Prompt construction for every LLM-backed action.

Two rules run through all of these, both straight from the spec:

* A summary may only ever be written by an agent that has the paper's full text
  in front of it. There is no "summarize from the abstract" path anywhere.
* Quotes must be copied verbatim from the supplied text. If the text does not
  support a claim, the model says so instead of filling the gap.
"""

from __future__ import annotations

import json

from .models import ENTRY_TYPES, Entry
from .quality import rubric_text

READER_SYSTEM = (
    "You are a meticulous research librarian summarizing a paper you have just "
    "read in full. You are precise, you never speculate, and you never state "
    "anything the text in front of you does not support. If the provided text is "
    "truncated or a section is missing, say so rather than guessing at it."
)

ANALYST_SYSTEM = (
    "You are a research analyst answering questions strictly from the sources "
    "provided. Every substantive claim must be traceable to a specific source. "
    "You quote exactly, never paraphrase inside quotation marks, and you say "
    "plainly when the sources do not answer the question."
)

SCOUT_SYSTEM = (
    "You are a literature scout. You find real, high-quality, verifiable "
    "published work. You never invent titles, authors, venues or identifiers — "
    "an id you are not sure about must be left null rather than guessed."
)

PLANNER_SYSTEM = (
    "You are a reference librarian taking a request at the desk and writing "
    "down what to type into the catalogue. You restate what you were asked in "
    "the vocabulary the literature uses; you never widen the request, narrow "
    "it, or add a condition the person did not state."
)

# How much of each abstract the ranking call sees. Enough to tell what a paper
# actually does; short enough that a 40-candidate pool stays a cheap call.
ABSTRACT_SNIPPET = 260


# --------------------------------------------------------------------------
# Reading a paper -> entry fields
# --------------------------------------------------------------------------

def read_paper_prompt(
    *,
    title: str,
    scope: str,
    known_venue: str | None,
    known_year: int | None,
    truncated: bool,
    needs_level: bool,
    sampled_pages: tuple[int | None, int | None] | None = None,
) -> str:
    schema = {
        "one_liner": "string — ONE sentence, max 30 words, what this work "
                     "contributes. No 'This paper...' preamble.",
        "sections": [
            {
                "name": "string — the section heading as it appears in the paper",
                "summary": "string — 2-5 sentences on what THIS section actually "
                           "says: the specific methods, numbers, and findings, not "
                           "a description of what the section is about",
            }
        ],
        "key_findings": [
            "string — a concrete, checkable result or claim, with the number or "
            "condition attached where the paper gives one"
        ],
        "tags": ["string — 3-8 lowercase topical keywords, single words or "
                 "hyphenated-phrases"],
        "type": f"string — one of: {', '.join(ENTRY_TYPES)}",
        "venue_from_text": "string|null — the venue stated in the paper itself, "
                           "if any (look at the header/footer/first page)",
        "code_url": "string|null — the code or artifact repository this document "
                    "links to (GitHub, GitLab, Hugging Face, Zenodo, OSF, Code "
                    "Ocean and the like), copied character-for-character from "
                    "the text. null if it does not give one.",
    }
    if needs_level:
        schema["level"] = "string — one of A*, A, B, C"
        schema["level_reason"] = (
            "string — one sentence citing the specific evidence you used"
        )

    parts = [
        f"Below (after the marker) is the FULL TEXT of a document titled: {title!r}.",
        "",
        "Read all of it, then produce a structured record of it.",
        "",
        "Requirements:",
        "- Cover the document section by section, in the order the sections "
        "appear. Use the paper's own section names.",
        "- The detailed summary must be substantially shorter than the paper but "
        "must preserve its actual content: what was done, on what data, with what "
        "result. Include concrete numbers where the paper reports them.",
        "- Do not include the reference list as a section.",
        "- Write nothing you cannot point to in the text.",
        "- For code_url, look wherever authors put release links: the abstract, "
        "a footnote on the first page, the end of the introduction, and any "
        "availability or reproducibility statement near the end. Copy the URL "
        "exactly as printed. Never reconstruct one from the authors', project's "
        "or method's name, and never supply one you happen to know — if the "
        "document does not print a link, the answer is null.",
    ]
    if scope:
        parts += [
            "",
            f"This record is going into a library scoped to: {scope!r}. Choose "
            "tags and emphasis with that scope in mind, but summarize the whole "
            "document faithfully regardless.",
        ]
    if known_venue or known_year:
        known = ", ".join(
            filter(None, [known_venue, str(known_year) if known_year else None])
        )
        parts += ["", f"Bibliographic metadata already retrieved: {known}."]
    if truncated:
        parts += [
            "",
            "NOTE: the middle of this document was omitted to fit context. Summarize "
            "the sections you can actually see, and do not invent the rest.",
        ]
    if sampled_pages:
        read, total = sampled_pages
        parts += [
            "",
            f"IMPORTANT: this is a long document ({total} pages) and you are seeing "
            f"only {read} of them — the front matter, an even spread through the "
            "body, and the closing pages. Omitted stretches are marked in the text.",
            "Summarize what the sample actually shows. Where the table of contents "
            "or structure tells you a topic is covered in pages you cannot see, you "
            "may say the work covers it, but never describe findings, arguments or "
            "numbers from pages that were omitted. Prefer fewer, well-grounded "
            "sections over guessing at the whole book.",
        ]
    if needs_level:
        parts += ["", rubric_text(),
                  "", "Metrics were not decisive for this work, so assign the level "
                      "yourself using that rubric and the evidence in the text."]

    parts += [
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def plan_query_prompt(*, query: str, scope: str, this_year: int,
                      max_queries: int, kinds: list[str]) -> str:
    """Read a plain-language request into parameters an index can be given.

    The cheapest call in the tool, and the one that decides what every other
    source is asked. A person types "recent surveys on retrieval-augmented
    generation, nothing before 2022" — Crossref and Semantic Scholar have fields for
    two thirds of that, but only if someone splits the sentence into a topic, a
    date window and a document type first. Handed the sentence whole, they
    match "recent", "nothing" and "before" as search terms.

    It is deliberately an extraction job, not a research one: pull out what the
    request already says, phrase the topic the way the literature phrases it,
    and leave every field the request does not mention null.
    """
    schema = {
        "topic": "string — the subject alone, as a noun phrase, with every "
                 "instruction, filter and pleasantry stripped out",
        "queries": [
            f"string — up to {max_queries} keyword queries for a bibliographic "
            "index, most direct first. Vary the vocabulary, not the meaning: a "
            "second phrasing should use the terms a different community would "
            "use for the same thing. No boolean operators, no quotes, no dates."
        ],
        "tags": ["string — 3-6 lowercase topical tags for the papers this "
                 "finds; single words or hyphenated-phrases"],
        "year_from": "number|null — earliest publication year the request asks "
                     f"for. The current year is {this_year}.",
        "year_to": "number|null — latest publication year the request asks for",
        "kind": f"string|null — one of: {', '.join(kinds)}. Only when the "
                "request explicitly asks for that kind of document.",
        "intent": "string — one sentence on what this person is actually "
                  "looking for, for whoever ranks the results",
    }
    parts = [
        "Turn this literature search request into search parameters:",
        f"  {query!r}",
    ]
    if scope:
        parts.append(f"The library it is going into is scoped to: {scope!r}.")
    parts += [
        "",
        "Rules:",
        "- Extract, do not invent. Every field the request does not imply is "
        "null or empty. Do not add a date window, a document type or a "
        "sub-topic the person did not ask for.",
        "- Resolve relative dates against the current year: 'the last five "
        f"years' means year_from {this_year - 4}, 'since the transformer "
        "paper' means 2017, 'recent' on its own is not a date — leave it null "
        "and let ranking handle it.",
        "- Queries are for a keyword index, so use the field's own terms: "
        "spell out an acronym the first time, drop stop words, and keep each "
        "query to a handful of content words.",
        "- If the request names a specific paper, method, author or venue, "
        "keep that name in the queries verbatim — it is the strongest term "
        "you have.",
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ]
    return "\n".join(parts)


def discover_prompt(*, query: str, scope: str, angle: str, limit: int,
                    exclude_titles: list[str], intent: str = "",
                    constraints: tuple[str, ...] = ()) -> str:
    schema = {
        "papers": [
            {
                "title": "string — the exact published title",
                "authors": ["string"],
                "year": "number|null",
                "venue": "string|null — the venue acronym or journal name",
                "doi": "string|null — bare DOI, e.g. 10.1145/3292500.3330701",
                "arxiv_id": "string|null — e.g. 2301.00234",
                "why": "string — one sentence on why this is relevant to the query",
            }
        ]
    }
    parts = [
        f"Find up to {limit} high-quality published papers matching this request:",
        f"  {query!r}",
    ]
    if intent:
        parts.append(f"What that asks for: {intent}")
    if scope:
        parts.append(f"They must fit a library scoped to: {scope!r}.")
    if constraints:
        parts += ["", "The request sets these limits — a paper outside them is "
                      "not an answer, however good it is:"]
        parts += [f"  - {c}" for c in constraints]
    parts += [
        "",
        f"Search from this specific angle: {angle}",
        "",
        "Use WebSearch and WebFetch to find real papers. Rules:",
        "- Only propose work you have actually seen a source for in this session.",
        "- Prefer highly-cited papers, papers at top venues (NeurIPS, ICML, ICLR, "
        "ACL, CVPR, Nature, Science and peers), and work by established groups.",
        "- Give the DOI or arXiv id whenever you can find it; use null when you "
        "cannot. A wrong identifier is worse than a missing one.",
        "- Do not propose product announcements, marketing pages or vendor blog "
        "posts. Peer review is not the test: a lab's benchmark or dataset "
        "released as a preprint with a public leaderboard, artifact or code "
        "release is real work and belongs here.",
        "- Prefer what indexed keyword search would miss — work in a "
        "neighbouring field that names the problem differently, and work whose "
        "contribution is a limitation, a failed replication or a negative "
        "result. Papers that any citation-sorted search returns are already "
        "covered.",
    ]
    if exclude_titles:
        listed = "\n".join(f"  - {t}" for t in exclude_titles[:60])
        parts += ["", "These are already in the library — do not propose them again:",
                  listed]
    parts += [
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ]
    return "\n".join(parts)


def rank_candidates_prompt(*, query: str, scope: str, candidates: list,
                           intent: str = "",
                           constraints: tuple[str, ...] = ()) -> str:
    """Score the whole candidate pool for relevance to the query.

    Every candidate is scored by the same call regardless of where it came
    from, so a paper mined from a reference list and a paper proposed by a web
    scout compete on equal terms.

    Only *relevance* is asked for. Importance is computed in code from venue
    rank and citation velocity (see `Candidate.importance`) — those are real
    numbers we already hold by this point, and asking a model to guess at them
    would only add noise to figures we can measure.

    Each candidate gets the opening of its abstract where a source supplied one.
    This is the one place extra tokens are worth spending: candidates now come
    from keyword facets rather than from an agent that read around the topic
    and wrote a note, and a title alone is thin evidence for the cut that
    decides which papers exist in the library at all.
    """
    lines = []
    for i, c in enumerate(candidates):
        head = f"  [{i}] {c.title}"
        if c.year:
            head += f" ({c.year})"
        if c.venue:
            head += f" — {c.venue}"
        if c.citation_count is not None:
            head += f" — {c.citation_count} citations"
        lines.append(head)
        marks = []
        if len(c.angles) > 1:
            marks.append(f"found by {len(c.angles)} independent searches")
        if c.cocitations:
            marks.append(f"cited by {c.cocitations} paper(s) already in this library")
        if marks:
            lines.append(f"      {' · '.join(marks)}")
        if c.why:
            lines.append(f"      note: {c.why}")
        elif getattr(c, "abstract", ""):
            lines.append(f"      abstract: {_snippet(c.abstract, ABSTRACT_SNIPPET)}")
    listed = "\n".join(lines)

    schema = {
        "scores": [
            {
                "index": "number — the [i] index below",
                "relevance": "number 0-1 — how well this answers the query",
                "why": "string — one short sentence on what it contributes",
            }
        ]
    }
    return "\n".join([
        f"Query: {query!r}",
        f"What that asks for: {intent}" if intent else "",
        f"Library scope: {scope!r}" if scope else "",
        ("The request also sets these limits: " + "; ".join(constraints)
         + ". A paper that breaks one does not answer the query, however good "
           "it is on the topic — score it accordingly."
         if constraints else ""),
        "",
        "Below are candidate papers gathered for that query. Score how well each "
        "one actually answers it. Score every candidate — do not omit any, and do "
        "not re-order them.",
        "",
        "Calibrate like this:",
        "  1.0  directly addresses the query — a reader with this question wants it",
        "  0.7  solidly on topic: same problem, different method or setting",
        "  0.5  same research area, but answers a different question",
        "  0.2  background or prerequisite material from a neighbouring area",
        "  0.0  merely shares vocabulary with the query",
        "",
        "Judge relevance only. Do not mark a paper down for being old, obscure or "
        "unfamiliar, and do not mark one up for being famous — how important a "
        "paper is has already been measured separately. A landmark paper that "
        "does not answer this query is a low-relevance paper.",
        "",
        listed,
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ])


# Ordered by what a keyword index *cannot* do, because the free facets in
# `discover.SEARCH_FACETS` already cover most-cited, recent, surveys and
# preprints for nothing. Scout agents cost real tokens, so when only some of
# these run, the budget should go to the angles that need a reader who can
# follow an idea across vocabularies rather than match strings: work in a
# neighbouring field that never uses your terms, and work that reports the
# limitation or the failed replication.
DISCOVERY_ANGLES = [
    "adjacent subfields and applications that approach the same problem "
    "differently, including work that describes it in different vocabulary",
    "critical, negative-result, replication-failure or limitation-focused work "
    "on this topic",
    "the foundational and most-cited works that defined this area",
    "the most recent work (last two years), including strong preprints",
    "surveys, benchmarks, datasets and evaluation methodology in this area",
]


# --------------------------------------------------------------------------
# Search within the library
# --------------------------------------------------------------------------

def library_search_prompt(*, query: str, scope: str, entries: list[Entry],
                          brief: bool = False) -> str:
    """Rank library entries against a query, and answer from them.

    `brief` is for the caller that only needs the ranking — `lit ask` deciding
    which papers to open. It asks for no prose answer and shows each entry as a
    short card, because both the answer and the section summaries that support
    it are discarded by that caller.
    """
    cards = "\n\n".join(_entry_card(e, brief=brief) for e in entries)
    schema: dict = {}
    if not brief:
        schema["answer"] = (
            "string — 2-4 sentences answering the query from these summaries "
            "only. Say plainly if they are insufficient."
        )
    schema["relevant"] = [
        {
            "key": "string — the entry key exactly as given",
            "relevance": "number 0-1",
            "why": "string — one sentence tying this entry to the query",
        }
    ]
    if not brief:
        schema["gaps"] = [
            "string — what the library is missing to answer this properly"
        ]

    task = (
        "Decide which are genuinely relevant. These will be opened and read in "
        "full, so judge only whether each one bears on the query."
        if brief else
        "Decide which are genuinely relevant, and give a short answer based on "
        "them.\n\nThis answer is drawn from summaries, not full papers, so keep "
        "it at the level the summaries actually support and do not overstate."
    )
    return "\n".join([
        f"Library scope: {scope!r}" if scope else "",
        f"Query: {query!r}",
        "",
        "Below are candidate entries from the library, as their stored summaries.",
        task,
        "Rank by real relevance, and drop entries that merely share vocabulary.",
        "",
        cards,
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ])


# --------------------------------------------------------------------------
# Answering from full text
# --------------------------------------------------------------------------

def extract_evidence_prompt(*, question: str, entry: Entry, truncated: bool) -> str:
    schema = {
        "relevant": "boolean — does this paper actually bear on the question?",
        "summary": "string — 1-3 sentences on what this paper contributes to the "
                   "answer. Empty string if not relevant.",
        "quotes": [
            {
                "text": "string — copied VERBATIM from the document, 1-3 sentences",
                "section": "string|null — which section it came from",
                "supports": "string — what point this quote establishes",
            }
        ],
        "caveats": ["string — limitations of this evidence: scope, scale, "
                    "conditions under which it holds"],
    }
    parts = [
        f"Question: {question!r}",
        "",
        f"Below is the full text of {entry.title!r} ({entry.citation()}).",
        "Read it and extract only what genuinely bears on the question.",
        "",
        "Rules:",
        "- Every quote must be copied character-for-character from the text below. "
        "Never paraphrase inside a quote. If you cannot copy it exactly, omit it.",
        "- Give at most 4 quotes, and only ones that carry real evidential weight.",
        "- If the paper does not address the question, set relevant=false and "
        "return no quotes. That is a useful answer, not a failure.",
    ]
    if truncated:
        parts.append("- The middle of this document was omitted; work only from "
                     "what you can see.")
    parts += [
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ]
    return "\n".join(parts)


def synthesize_answer_prompt(*, question: str, scope: str, evidence: list[dict]) -> str:
    blocks = []
    for ev in evidence:
        quotes = "\n".join(
            f'    - "{q.get("text","")}"'
            + (f' [{q.get("section")}]' if q.get("section") else "")
            for q in ev.get("quotes", [])
        )
        blocks.append(
            f"[{ev['key']}] {ev['title']} ({ev['citation']})\n"
            f"  Contribution: {ev.get('summary','')}\n"
            f"  Quotes:\n{quotes or '    (none)'}\n"
            f"  Caveats: {'; '.join(ev.get('caveats', [])) or 'none noted'}"
        )
    return "\n".join([
        f"Question: {question!r}",
        f"Library scope: {scope!r}" if scope else "",
        "",
        "Below is evidence extracted from the full text of each source by an agent "
        "that read it. Write the answer.",
        "",
        "Requirements:",
        "- Answer the question directly in the first paragraph.",
        "- Support each substantive claim with a citation key in square brackets, "
        "e.g. [vaswani2017attention].",
        "- Include the strongest exact quotes, in quotation marks, attributed to "
        "their key. Reproduce them exactly as given below.",
        "- Note disagreement between sources where it exists, and state the "
        "limits of what these sources establish.",
        "- Do not add facts that are not in the evidence below.",
        "- Markdown. No top-level heading. Do not write a bibliography — one is "
        "appended automatically.",
        "",
        "EVIDENCE",
        "\n\n".join(blocks),
    ])


# --------------------------------------------------------------------------
# Claim tracing
# --------------------------------------------------------------------------

def claim_origin_prompt(*, claim: str, entry: Entry, references: list,
                        truncated: bool) -> str:
    ref_lines = "\n".join(
        f"  [{i}] {r.title}" + (f" ({r.year})" if r.year else "")
        + (f" doi:{r.doi}" if r.doi else "")
        for i, r in enumerate(references)
    )
    schema = {
        "states_claim": "boolean — does this document itself assert the claim?",
        "is_origin": "boolean — true if this document ORIGINATES the claim "
                     "(introduces the concept/result), false if it cites someone "
                     "else for it",
        "evidence_quote": "string — the verbatim sentence where the document "
                          "states or attributes the claim; empty if absent",
        "attributed_to": [
            {
                "index": "number — the [i] index from the reference list below",
                "confidence": "number 0-1",
                "why": "string — the citation context that points here",
            }
        ],
        "reasoning": "string — two sentences on how you decided",
    }
    parts = [
        f"Claim being traced: {claim!r}",
        "",
        f"Below is the full text of {entry.title!r} ({entry.citation()}).",
        "",
        "Determine whether this document originates the claim or inherits it from "
        "something it cites.",
        "",
        "- Find where the document states the claim, and look at what it cites there.",
        "- A document originates a claim when it introduces the concept or reports "
        "the result first-hand, not when it restates it with a citation.",
        "- If it attributes the claim to prior work, point at the specific entries "
        "in the reference list below. Rank by confidence; give at most 3.",
        "- Quote verbatim. If the document never states this claim, say so with "
        "states_claim=false.",
    ]
    if truncated:
        parts.append("- The middle of the document was omitted; work from what you see.")
    parts += [
        "",
        "REFERENCE LIST",
        ref_lines or "  (no reference list available)",
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ]
    return "\n".join(parts)


def find_claim_source_prompt(*, claim: str, scope: str) -> str:
    schema = {
        "papers": [
            {
                "title": "string",
                "authors": ["string"],
                "year": "number|null",
                "doi": "string|null",
                "arxiv_id": "string|null",
                "why": "string — why this paper is a good place to start tracing",
            }
        ]
    }
    return "\n".join([
        "I want to trace this claim back to the paper that originates it:",
        f"  {claim!r}",
        f"Field context: {scope!r}" if scope else "",
        "",
        "Find up to 3 real papers that state this claim and are good starting "
        "points for tracing it backwards — ideally recent papers that cite the "
        "original, or the original itself if you can identify it.",
        "",
        "Use WebSearch and WebFetch. Only propose papers you have seen a source "
        "for. Give DOI or arXiv id where you can, null where you cannot.",
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _snippet(text: str, limit: int) -> str:
    """First `limit` characters of a single-line rendering, cut on a word."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    head, sep, _ = cut.rpartition(" ")
    return (head if sep and len(head) > limit * 0.6 else cut) + "…"


def _entry_card(e: Entry, *, brief: bool = False) -> str:
    """One library entry, as the ranking call sees it.

    The full card carries the stored section summaries, which are most of its
    weight — they are what lets a caller answer a question from summaries
    alone. A `brief` card drops them, because deciding whether a paper bears on
    a query is a judgement the one-liner, tags and findings already support,
    and a pool of thirty full cards is the largest prompt this tool builds
    outside of reading a paper.
    """
    limit = 3 if brief else 5
    findings = "\n".join(f"    - {_snippet(f, 160) if brief else f}"
                         for f in e.key_findings[:limit])
    sections = "" if brief else "\n".join(
        f"    {s.name}: {s.summary[:300]}" for s in e.sections[:12]
    )
    # An entry filed but not yet read has no agent-written summary, only the
    # publisher's abstract. Showing that is what keeps it visible here: it was
    # added because it matched a query, and it should not be unfindable until
    # someone pays to read it.
    stand_in = ""
    if not e.one_liner and e.abstract.strip():
        stand_in = ("  Abstract (not yet read): "
                    f"{_snippet(e.abstract, 240 if brief else 600)}")
    return "\n".join(filter(None, [
        f"[{e.key}] {e.title} — {e.citation()} (level {e.level}, {e.type})",
        f"  One-liner: {e.one_liner or '(none — not read yet)'}",
        stand_in,
        f"  Tags: {', '.join(e.tags)}" if e.tags else "",
        f"  Key findings:\n{findings}" if findings else "",
        f"  Sections:\n{sections}" if sections else "",
        f"  User notes: {e.notes[:500]}" if not brief and e.notes.strip() else "",
    ]))
