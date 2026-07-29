"""Level assignment: A* / A / B / C from explicit, inspectable metrics.

The spec calls for a level "generated from clear metrics", so the rules live
here in code rather than in a model's judgement:

1. Venue rank, from the curated table below (CORE-style for CS venues, plus the
   top general-science journals).
2. Citation velocity — citations per year since publication — which is what
   actually separates a landmark paper from a forgettable one at the same venue.
3. A recency allowance: a paper from the last two years has not had time to
   accumulate citations, so it is judged on venue alone.

Every level carries a `reason` string naming the metric that produced it, so a
user can always see why something was ranked the way it was. The LLM is only
consulted when the venue is unknown *and* citation data is too thin to decide
(`needs_judgement`), and even then it is given this same rubric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .fetch.metadata import PaperMeta

# (level, canonical name, regex matched against the venue string)
_VENUE_TABLE: list[tuple[str, str, str]] = [
    # --- ML / AI ---
    ("A*", "NeurIPS", r"\bneurips\b|\bnips\b|neural information processing systems"),
    ("A*", "ICML", r"\bicml\b|international conference on machine learning"),
    ("A*", "ICLR", r"\biclr\b|international conference on learning representations"),
    ("A*", "AAAI", r"\baaai\b(?!.*workshop)"),
    ("A*", "IJCAI", r"\bijcai\b"),
    ("A*", "JMLR", r"journal of machine learning research|\bjmlr\b"),
    ("A*", "TPAMI", r"transactions on pattern analysis and machine intelligence|\btpami\b"),
    ("A", "AISTATS", r"\baistats\b|artificial intelligence and statistics"),
    ("A", "UAI", r"\buai\b|uncertainty in artificial intelligence"),
    ("A", "COLT", r"\bcolt\b|conference on learning theory"),
    ("A", "TMLR", r"transactions on machine learning research|\btmlr\b"),
    ("A", "AAMAS", r"\baamas\b|autonomous agents and multiagent"),
    ("A", "JAIR", r"journal of artificial intelligence research|\bjair\b"),
    # --- NLP ---
    ("A*", "ACL", r"\bacl\b|annual meeting of the association for computational linguistics"),
    ("A*", "EMNLP", r"\bemnlp\b|empirical methods in natural language processing"),
    ("A*", "NAACL", r"\bnaacl\b"),
    ("A", "COLING", r"\bcoling\b"),
    ("A", "EACL", r"\beacl\b"),
    ("A", "TACL", r"transactions of the association for computational linguistics|\btacl\b"),
    ("A", "CL", r"\bcomputational linguistics\b"),
    # --- Vision ---
    ("A*", "CVPR", r"\bcvpr\b|computer vision and pattern recognition"),
    ("A*", "ICCV", r"\biccv\b|international conference on computer vision"),
    ("A*", "ECCV", r"\beccv\b|european conference on computer vision"),
    ("A", "BMVC", r"\bbmvc\b"), ("A", "WACV", r"\bwacv\b"), ("A", "IJCV", r"\bijcv\b"),
    # --- Systems / architecture / networking ---
    ("A*", "OSDI", r"\bosdi\b"), ("A*", "SOSP", r"\bsosp\b"),
    ("A*", "NSDI", r"\bnsdi\b"), ("A*", "SIGCOMM", r"\bsigcomm\b"),
    ("A*", "ISCA", r"\bisca\b|international symposium on computer architecture"),
    ("A*", "ASPLOS", r"\basplos\b"), ("A*", "MICRO", r"\bmicro\b.*symposium|\bmicro-\d+"),
    ("A*", "SC", r"\bsupercomputing\b|\bsc'?\d{2}\b|international conference for high performance"),
    ("A", "EuroSys", r"\beurosys\b"), ("A", "USENIX ATC", r"usenix annual technical"),
    ("A", "PPoPP", r"\bppopp\b"), ("A", "IPDPS", r"\bipdps\b"), ("A", "HPCA", r"\bhpca\b"),
    ("A", "TPDS", r"transactions on parallel and distributed systems|\btpds\b"),
    # --- Security ---
    ("A*", "IEEE S&P", r"\boakland\b|ieee symposium on security and privacy|\bs&p\b"),
    ("A*", "USENIX Security", r"usenix security"),
    ("A*", "CCS", r"\bccs\b|conference on computer and communications security"),
    ("A*", "NDSS", r"\bndss\b"),
    # --- Data / DB / IR / web / mining ---
    ("A*", "SIGMOD", r"\bsigmod\b"), ("A*", "VLDB", r"\bvldb\b"),
    ("A*", "ICDE", r"\bicde\b"),
    ("A*", "KDD", r"sigkdd|\bkdd\b|knowledge discovery and data mining"),
    ("A*", "SIGIR", r"\bsigir\b"), ("A*", "WWW", r"\bwww\b.*conference|the web conference"),
    ("A", "WSDM", r"\bwsdm\b"), ("A", "CIKM", r"\bcikm\b"), ("A", "ICDM", r"\bicdm\b"),
    ("A", "RecSys", r"\brecsys\b"), ("A", "TKDE", r"\btkde\b"),
    # --- Theory / PL / SE ---
    ("A*", "STOC", r"\bstoc\b"), ("A*", "FOCS", r"\bfocs\b"), ("A*", "SODA", r"\bsoda\b"),
    ("A*", "POPL", r"\bpopl\b"), ("A*", "PLDI", r"\bpldi\b"), ("A*", "OOPSLA", r"\boopsla\b"),
    ("A*", "ICSE", r"\bicse\b"), ("A*", "FSE", r"\bfse\b|foundations of software engineering"),
    ("A", "ASE", r"\base\b.*automated software engineering"), ("A", "ISSTA", r"\bissta\b"),
    ("A", "ICFP", r"\bicfp\b"), ("A", "CAV", r"\bcav\b"), ("A", "TSE", r"\btse\b"),
    # --- HCI / graphics / robotics ---
    ("A*", "CHI", r"\bchi\b.*conference on human factors|\bchi\b conference"),
    ("A*", "SIGGRAPH", r"\bsiggraph\b"), ("A*", "UIST", r"\buist\b"),
    ("A*", "ICRA", r"\bicra\b"), ("A*", "IROS", r"\biros\b"), ("A*", "RSS", r"robotics: science and systems"),
    ("A", "CSCW", r"\bcscw\b"), ("A", "CoRL", r"\bcorl\b|conference on robot learning"),
    ("A", "IJRR", r"international journal of robotics research"),
    # --- General science ---
    ("A*", "Nature", r"^nature$|^nature\s"), ("A*", "Science", r"^science$"),
    ("A*", "PNAS", r"\bpnas\b|proceedings of the national academy"),
    ("A*", "Cell", r"^cell$"), ("A*", "NEJM", r"new england journal of medicine"),
    ("A*", "Lancet", r"^the lancet|^lancet"), ("A*", "JAMA", r"^jama\b"),
    ("A", "Nature Communications", r"nature communications"),
    # Megajournals: peer-reviewed and credible, but selectivity is low, so they
    # sit at B and let citation velocity promote individual papers.
    ("B", "PLOS ONE", r"plos one"), ("B", "Scientific Reports", r"scientific reports"),
]

_COMPILED = [(lvl, name, re.compile(pat, re.IGNORECASE)) for lvl, name, pat in _VENUE_TABLE]

# Citations per year needed for each level when the venue is unknown.
_VELOCITY_THRESHOLDS = [("A*", 100.0), ("A", 25.0), ("B", 5.0)]

# Preprints and workshop papers are normally capped at A, since they carry no
# venue rank to inherit. This is the "unless landmark" escape: work the field
# cites this heavily is A*-tier regardless of where it was published. Several
# genuinely foundational papers (the Transformer among them) were never given a
# Crossref DOI by their conference, so the venue path can't reach them at all.
LANDMARK_VELOCITY = 200.0

# A paper this new has not had time to be cited; judge it on venue alone.
RECENCY_GRACE_YEARS = 2


@dataclass
class LevelVerdict:
    level: str
    reason: str
    # True when neither venue nor citations gave a usable signal, so the caller
    # may ask the LLM to apply the same rubric using the paper's own content.
    needs_judgement: bool = False


def venue_level(venue: str | None) -> tuple[str, str] | None:
    """Look a venue up in the curated table. Returns (level, canonical name)."""
    if not venue:
        return None
    v = re.sub(r"\s+", " ", venue).strip()
    if _is_workshop(v):
        return None  # workshops are handled separately, never inherit venue rank
    for lvl, name, pat in _COMPILED:
        if pat.search(v):
            return lvl, name
    return None


def citations_per_year(citations: int | None, year: int | None) -> float | None:
    if citations is None or not year:
        return None
    age = max(date.today().year - year, 1)
    return citations / age


def assess(meta: PaperMeta, *, today_year: int | None = None) -> LevelVerdict:
    """Assign a level from venue rank and citation velocity."""
    today_year = today_year or date.today().year
    age = (today_year - meta.year) if meta.year else None
    vel = citations_per_year(meta.citation_count, meta.year)
    vhit = venue_level(meta.venue)

    # Workshop papers and preprints have no venue rank to inherit; they earn a
    # level purely through citation velocity, and are capped below A* unless
    # they are genuinely landmark.
    is_soft_venue = meta.type in ("workshop paper", "preprint") or _is_workshop(meta.venue)

    if vhit and not is_soft_venue:
        lvl, name = vhit
        reason = f"published at {name} ({lvl} venue)"
        # Citation velocity can promote a paper one step, never demote it: a
        # highly-cited A-venue paper is effectively A*-tier work.
        if lvl == "A" and vel is not None and vel >= _VELOCITY_THRESHOLDS[0][1]:
            return LevelVerdict("A*", f"{reason}; {vel:.0f} citations/year")
        return LevelVerdict(lvl, reason + (f"; {vel:.0f} citations/year" if vel else ""))

    if vel is not None and (age is None or age >= 1):
        for lvl, threshold in _VELOCITY_THRESHOLDS:
            if vel >= threshold:
                note = ""
                if is_soft_venue and lvl == "A*":
                    if vel >= LANDMARK_VELOCITY:
                        note = ", landmark"  # earns A* despite having no venue
                    else:
                        lvl = "A"  # preprint/workshop ceiling
                label = "preprint" if meta.type == "preprint" else (meta.venue or "unranked venue")
                return LevelVerdict(lvl, f"{vel:.0f} citations/year ({label}{note})")
        if age is not None and age > RECENCY_GRACE_YEARS:
            if vhit:
                # A ranked venue we already matched, just with weak uptake.
                return LevelVerdict("C", f"{vel:.1f} citations/year after {age} years")
            # Low velocity *and* no venue we recognise usually means the metadata
            # lookup was thin, not that the work is bad — plenty of good papers
            # sit at unindexed venues. Hand it to the reader agent, which has the
            # full text, rather than rejecting it on missing data.
            return LevelVerdict(
                "C",
                f"{vel:.1f} citations/year after {age} years, venue not in the "
                "ranked list",
                needs_judgement=True,
            )

    if age is not None and age <= RECENCY_GRACE_YEARS:
        return LevelVerdict(
            "unranked",
            f"published {meta.year} — too recent for citation metrics"
            + ("; venue not in the ranked list" if meta.venue else "; no venue"),
            needs_judgement=True,
        )

    return LevelVerdict(
        "unranked",
        "no ranked venue and no citation data",
        needs_judgement=True,
    )


def passes_quality_bar(verdict: LevelVerdict, meta: PaperMeta, min_level: str,
                       allow_non_published: bool) -> tuple[bool, str]:
    """Apply the library's admission policy. Returns (ok, explanation)."""
    from .models import meets_min_level

    if meta.type in ("video", "other") and not allow_non_published:
        if meta.type == "video":
            return False, (
                "non-published material (video); enable it with "
                "`lit config set allow_non_published true` in this library"
            )

    if verdict.needs_judgement:
        return True, "unranked but admitted (no metrics available yet)"

    if not meets_min_level(verdict.level, min_level):
        return False, f"level {verdict.level} is below this library's minimum of {min_level}"
    return True, "meets the quality bar"


def rubric_text() -> str:
    """The rubric handed to the LLM when metrics alone cannot decide."""
    return (
        "Level rubric (same one the metric-based scorer uses):\n"
        "- A*: top-tier venue (NeurIPS/ICML/ICLR/ACL/CVPR/SOSP/Nature/Science and "
        "peers), or >=100 citations/year.\n"
        "- A: strong venue (AISTATS, UAI, COLING, EuroSys, TMLR and peers), or "
        ">=25 citations/year.\n"
        "- B: credible peer-reviewed work with modest uptake, or >=5 citations/year.\n"
        "- C: little evidence of quality or uptake.\n"
        "Preprints and workshop papers are capped at A unless landmark.\n"
        "For work too recent to have citations, judge on: authors' track record and "
        "affiliation, methodological rigour, whether code/data are released, size and "
        "honesty of the evaluation, and whether the claims are supported."
    )


def _is_workshop(venue: str | None) -> bool:
    return bool(re.search(r"\bworkshop\b|\bwork-?shop\b", venue or "", re.IGNORECASE))
