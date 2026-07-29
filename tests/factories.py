"""Test object factories."""

from __future__ import annotations

from lit.fetch.metadata import PaperMeta
from lit.models import Entry, Reference, Section


def make_entry(key="vaswani2017attention", **kw) -> Entry:
    data = dict(
        key=key,
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS",
        type="conference paper",
        arxiv_id="1706.03762",
        status="verified",
        level="A*",
        level_reason="published at NeurIPS (A* venue)",
        one_liner="Introduces the Transformer, an attention-only sequence model.",
        tags=["transformers", "attention"],
        key_findings=["28.4 BLEU on WMT 2014 EN-DE"],
        citation_count=100000,
        sections=[
            Section("Introduction", "Recurrence limits parallelism."),
            Section("Model Architecture", "Encoder-decoder with self-attention."),
        ],
        references=[
            Reference(title="Neural Machine Translation by Jointly Learning",
                      year=2014, arxiv_id="1409.0473"),
        ],
        bibtex="@inproceedings{vaswani2017attention,\n  title = {Attention Is All You Need}\n}",
    )
    data.update(kw)
    return Entry(**data)


def make_meta(**kw) -> PaperMeta:
    data = dict(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS",
        type="conference paper",
        arxiv_id="1706.03762",
        citation_count=100000,
    )
    data.update(kw)
    return PaperMeta(**data)
