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

def discover_prompt(*, query: str, scope: str, angle: str, limit: int,
                    exclude_titles: list[str]) -> str:
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
    if scope:
        parts.append(f"They must fit a library scoped to: {scope!r}.")
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
        "- Do not propose blog posts, marketing pages or unrefereed write-ups.",
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


def filter_candidates_prompt(*, query: str, scope: str, candidates: list) -> str:
    """Score a pool of candidate papers for relevance before we spend reads on them.

    Used on candidates mined from the reference lists of papers already in the
    library: that pool is large and only loosely related to the query, so it
    needs a cheap filter before the expensive per-paper read.
    """
    listed = "\n".join(
        f"  [{i}] {c.title}"
        + (f" ({c.year})" if c.year else "")
        + (f" — cited by {len(c.angles)} of your papers" if c.angles else "")
        for i, c in enumerate(candidates)
    )
    schema = {
        "keep": [
            {
                "index": "number — the [i] index below",
                "relevance": "number 0-1 — how well this fits the query and scope",
                "why": "string — one short sentence",
            }
        ]
    }
    return "\n".join([
        f"Query: {query!r}",
        f"Library scope: {scope!r}" if scope else "",
        "",
        "Below are candidate papers harvested from the reference lists of papers "
        "already in this library. Most will be tangential. Decide which are "
        "genuinely worth adding for the query above.",
        "",
        "Judge from the titles: keep work that plausibly addresses the query, drop "
        "work that merely shares vocabulary or is background material of a "
        "different area. Be selective — it is better to keep 5 good ones than 20 "
        "vague ones. Return an empty list if none fit.",
        "",
        listed,
        "",
        "Reply with a single JSON object, no markdown fence, matching this shape:",
        json.dumps(schema, indent=2),
    ])


DISCOVERY_ANGLES = [
    "the foundational and most-cited works that defined this area",
    "the most recent work (last two years), including strong preprints",
    "surveys, benchmarks, datasets and evaluation methodology in this area",
    "adjacent subfields and applications that approach the same problem differently",
    "critical, negative-result or limitation-focused work on this topic",
]


# --------------------------------------------------------------------------
# Search within the library
# --------------------------------------------------------------------------

def library_search_prompt(*, query: str, scope: str, entries: list[Entry]) -> str:
    cards = "\n\n".join(_entry_card(e) for e in entries)
    schema = {
        "answer": "string — 2-4 sentences answering the query from these summaries "
                  "only. Say plainly if they are insufficient.",
        "relevant": [
            {
                "key": "string — the entry key exactly as given",
                "relevance": "number 0-1",
                "why": "string — one sentence tying this entry to the query",
            }
        ],
        "gaps": ["string — what the library is missing to answer this properly"],
    }
    return "\n".join([
        f"Library scope: {scope!r}" if scope else "",
        f"Query: {query!r}",
        "",
        "Below are candidate entries from the library, as their stored summaries.",
        "Decide which are genuinely relevant, and give a short answer based on them.",
        "",
        "This answer is drawn from summaries, not full papers, so keep it at the "
        "level the summaries actually support and do not overstate.",
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

def _entry_card(e: Entry) -> str:
    findings = "\n".join(f"    - {f}" for f in e.key_findings[:5])
    sections = "\n".join(
        f"    {s.name}: {s.summary[:300]}" for s in e.sections[:12]
    )
    return "\n".join(filter(None, [
        f"[{e.key}] {e.title} — {e.citation()} (level {e.level}, {e.type})",
        f"  One-liner: {e.one_liner or '(none — unread)'}",
        f"  Tags: {', '.join(e.tags)}" if e.tags else "",
        f"  Key findings:\n{findings}" if findings else "",
        f"  Sections:\n{sections}" if sections else "",
        f"  User notes: {e.notes[:500]}" if e.notes.strip() else "",
    ]))
