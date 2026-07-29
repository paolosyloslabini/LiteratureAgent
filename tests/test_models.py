import pytest

from lit.models import (
    Reference,
    Section,
    level_rank,
    make_key,
    meets_min_level,
    normalize_arxiv,
    normalize_code_url,
    normalize_doi,
    slugify,
)


@pytest.mark.parametrize("raw,expected", [
    ("10.1145/3292500.3330701", "10.1145/3292500.3330701"),
    ("https://doi.org/10.1145/3292500.3330701", "10.1145/3292500.3330701"),
    ("doi:10.1038/nature14539", "10.1038/nature14539"),
    ("DOI: 10.1038/NATURE14539", "10.1038/nature14539"),
    ("see 10.1038/nature14539.", "10.1038/nature14539"),
    ("no identifier here", None),
    ("", None),
    (None, None),
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1706.03762", "1706.03762"),
    ("1706.03762v5", "1706.03762"),
    ("arXiv:1706.03762", "1706.03762"),
    ("https://arxiv.org/abs/1706.03762", "1706.03762"),
    ("https://arxiv.org/pdf/2301.00234v2", "2301.00234"),
    ("cs.CL/0501001", "cs.CL/0501001"),
    ("nothing", None),
])
def test_normalize_arxiv(raw, expected):
    assert normalize_arxiv(raw) == expected


def test_make_key_from_full_name():
    assert make_key(["Ashish Vaswani"], 2017, "Attention Is All You Need") == \
        "vaswani2017attention"


def test_make_key_from_surname_first():
    assert make_key(["Vaswani, Ashish"], 2017, "Attention Is All You Need") == \
        "vaswani2017attention"


def test_make_key_skips_stopwords_in_title():
    assert make_key(["Jane Doe"], 2020, "On the Measure of Intelligence") == \
        "doe2020measure"


def test_make_key_handles_missing_data():
    assert make_key([], None, "") == "anon"


def test_make_key_strips_accents():
    assert make_key(["Émile Borel"], 1921, "Théorie des jeux") == "borel1921theorie"


def test_level_ordering():
    assert level_rank("A*") < level_rank("A") < level_rank("B") < level_rank("C")
    assert level_rank("unranked") > level_rank("C")
    assert level_rank(None) == level_rank("unranked")


def test_meets_min_level():
    assert meets_min_level("A*", "B")
    assert meets_min_level("B", "B")
    assert not meets_min_level("C", "B")
    assert meets_min_level("C", "any")
    assert meets_min_level("unranked", None)


def test_section_coercion():
    assert Section.coerce({"name": "Intro", "summary": "x"}).name == "Intro"
    assert Section.coerce({"section": "Intro", "text": "x"}).summary == "x"
    assert Section.coerce({}) is None
    assert Section.coerce(None) is None


def test_reference_coercion():
    r = Reference.coerce(
        {"title": "T", "year": "2014", "doi": "https://doi.org/10.1234/abc"}
    )
    assert r.year == 2014
    assert r.doi == "10.1234/abc"
    assert Reference.coerce({"title": ""}) is None
    assert Reference.coerce("bare title").title == "bare title"


def test_reference_coercion_bad_year_is_none():
    assert Reference.coerce({"title": "T", "year": "n.d."}).year is None


def test_slugify():
    assert slugify("Attention Is All You Need!") == "attentionisallyouneed"
    assert slugify("") == ""


def test_entry_citation(entry):
    assert entry.citation() == "Vaswani et al., NeurIPS, 2017"


def test_entry_searchable_text_includes_notes(entry):
    entry.notes = "MYUNIQUETOKEN"
    assert "MYUNIQUETOKEN" in entry.searchable_text()


def test_entry_searchable_text_includes_the_abstract(entry):
    entry.abstract = "ABSTRACTTOKEN"
    assert "ABSTRACTTOKEN" in entry.searchable_text()


# --------------------------------------------------------------------------
# Code / artifact repository links
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("https://github.com/google/flax", "https://github.com/google/flax"),
    # Papers print these mid-sentence, so the punctuation has to come off.
    ("https://github.com/google/flax.", "https://github.com/google/flax"),
    ("(https://github.com/google/flax)", "https://github.com/google/flax"),
    ("(https://github.com/google/flax).", "https://github.com/google/flax"),
    ("<https://github.com/google/flax>", "https://github.com/google/flax"),
    ("https://github.com/google/flax/", "https://github.com/google/flax"),
    # A footnote link often loses its scheme in extraction.
    ("github.com/google/flax", "https://github.com/google/flax"),
    ("www.github.com/google/flax", "https://github.com/google/flax"),
    ("http://github.com/google/flax", "https://github.com/google/flax"),
    ("https://GitHub.com/google/Flax", "https://github.com/google/Flax"),
    # A deep link is a fine answer; the fragment is not part of the identity.
    ("https://github.com/google/flax/tree/main/examples#readme",
     "https://github.com/google/flax/tree/main/examples"),
    # Artifact hosts, where a single path segment is the whole record.
    ("https://zenodo.org/records/8329395", "https://zenodo.org/records/8329395"),
    ("https://huggingface.co/datasets/squad",
     "https://huggingface.co/datasets/squad"),
    ("https://osf.io/ab12c", "https://osf.io/ab12c"),
])
def test_normalize_code_url_accepts_repositories(raw, expected):
    assert normalize_code_url(raw) == expected


@pytest.mark.parametrize("raw", [
    None,
    "",
    "   ",
    # Not a code host: a project page, a DOI or the paper itself.
    "https://sites.google.com/view/our-project",
    "https://doi.org/10.1145/3292500.3330701",
    "https://arxiv.org/abs/1706.03762",
    # A host with no project on it is not a repository.
    "https://github.com",
    "https://github.com/",
    "https://github.com/deepmind",
    # Prose, not a URL.
    "code available on request",
])
def test_normalize_code_url_rejects_non_repositories(raw):
    assert normalize_code_url(raw) is None
