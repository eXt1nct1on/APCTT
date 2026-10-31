"""Tests for evidence verification and extraction."""

from pathlib import Path

from trl_scout.index.retriever import DocIndex
from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import FakeLLM, LLMClient
from trl_scout.pipeline.evidence import assess_evidence
from trl_scout.pipeline.verify import explicit_trl_mentions, planned_items, quote_span
from trl_scout.schemas import Chunk, Document, Sentence, TechProfile


class TinyEncoder:
    """Deterministic encoder for evidence tests."""

    def encode(self, texts, **_kwargs):
        """Return simple keyword vectors."""
        return [
            [float("field" in text.lower()), float("battery" in text.lower())] for text in texts
        ]


def _document() -> Document:
    """Create a small document with one retrievable results chunk."""
    text = "The field trial tested the battery under load."
    chunk = Chunk(
        chunk_id="c001",
        page_start=2,
        page_end=2,
        section="results",
        text=text,
        sentences=[Sentence(sid="s0001", text=text)],
    )
    return Document(
        doc_id="evidence-test",
        filename="test.txt",
        doc_type="paper",
        n_pages=2,
        sha256="evidence-test",
        chunks=[chunk],
    )


def _tech() -> TechProfile:
    """Create a minimal technology profile."""
    return TechProfile(
        name="Battery",
        domain="energy",
        doc_type="research_paper",
        summary="Battery technology",
        intended_application="Storage",
        key_components=["cell"],
        claimed_results=[],
        author_stated_maturity=None,
        keywords=["battery"],
    )


def test_quote_verification_handles_hyphenation() -> None:
    """Fuzzy verification tolerates line-break hyphenation."""
    ok, score, start, end = quote_span("field trial tested", "field-\ntrial tested the battery")
    assert ok
    assert score >= 0.9
    assert "field-\ntrial tested" == "field-\ntrial tested the battery"[start:end]


def test_evidence_drops_fabricated_quote_and_downgrades(tmp_path: Path) -> None:
    """Only verified quotes reach criterion evidence IDs."""
    response = {
        "level": 1,
        "criteria": [
            {
                "criterion_id": "T1.1",
                "status": "met",
                "evidence": [
                    {
                        "chunk_id": "c001",
                        "quote": "The field trial tested the battery under load.",
                        "polarity": "achieved",
                        "attribution": "own",
                        "strength": 1,
                        "why": "Measured trial",
                    },
                    {
                        "chunk_id": "c001",
                        "quote": "A fabricated result not in source.",
                        "polarity": "achieved",
                        "attribution": "own",
                        "strength": 1,
                        "why": "Fabricated",
                    },
                ],
                "note": "Test",
            },
            {
                "criterion_id": "T1.2",
                "status": "met",
                "evidence": [
                    {
                        "chunk_id": "c001",
                        "quote": "A fabricated result not in source.",
                        "polarity": "achieved",
                        "attribution": "own",
                        "strength": 1,
                        "why": "Fabricated",
                    }
                ],
                "note": "Must downgrade",
            },
        ],
    }
    fake = FakeLLM({"RawLevel": response})
    document = _document()
    index = DocIndex(document, model=TinyEncoder(), cache_dir=tmp_path)
    criteria, evidence = assess_evidence(
        document,
        _tech(),
        LLMClient(completion=fake, cache=JsonCache(tmp_path)),
        index,
    )
    assert len([item for item in evidence if item.verified]) == 1
    assert criteria[0].evidence_ids == ["E001"]
    assert criteria[1].status == "not_met"
    assert criteria[1].evidence_ids == []


def test_signals_are_extracted() -> None:
    """TRL mentions and planned sentences are retained as source signals."""
    chunk = (
        _document()
        .chunks[0]
        .model_copy(
            update={
                "section": "conclusion",
                "text": "The authors report TRL 4 to 6. Future work will test the pilot plant.",
                "sentences": [
                    Sentence(sid="s0002", text="The authors report TRL 4 to 6."),
                    Sentence(sid="s0003", text="Future work will test the pilot plant."),
                ],
            }
        )
    )
    assert explicit_trl_mentions([chunk]) == ["TRL 4 to 6"]
    assert planned_items([chunk]) == ["Future work will test the pilot plant."]
