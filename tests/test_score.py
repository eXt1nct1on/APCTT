"""Tests for rule and judge TRL scoring."""

from trl_scout.config import ScoringConfig
from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import FakeLLM, LLMClient
from trl_scout.pipeline.score import calculate_level_scores, score_document
from trl_scout.schemas import CriterionResult, EvidenceItem, TechProfile


def _tech() -> TechProfile:
    """Build a minimal technology profile."""
    return TechProfile(
        name="Test technology",
        domain="test",
        doc_type="research_paper",
        summary="Test technology",
        intended_application="Testing",
        key_components=["component"],
        claimed_results=[],
        author_stated_maturity=None,
        keywords=["test"],
    )


def _evidence(
    criterion_id: str,
    *,
    polarity: str = "achieved",
    attribution: str = "own",
    evidence_id: str = "E001",
) -> EvidenceItem:
    """Build verified evidence."""
    return EvidenceItem(
        evidence_id=evidence_id,
        criterion_id=criterion_id,
        chunk_id="c001",
        page=1,
        quote="Measured result",
        polarity=polarity,
        attribution=attribution,
        strength=1,
        why="test",
        verified=True,
        match_score=1,
        span_start=0,
        span_end=14,
    )


def _result(
    criterion_id: str, status: str = "met", evidence_ids: list[str] | None = None
) -> CriterionResult:
    """Build a criterion result."""
    return CriterionResult(
        criterion_id=criterion_id,
        status=status,
        evidence_ids=evidence_ids or ["E001"],
        score=0,
        note="test",
    )


def _judge(fake_trl: int = 4) -> LLMClient:
    """Build a three-sample fake judge."""
    response = {
        "trl": fake_trl,
        "range_low": fake_trl,
        "range_high": fake_trl,
        "confidence": 0.8,
        "reasoning": "Supported by verified evidence.",
        "decisive_evidence_ids": ["E001"],
        "contradictions": [],
        "what_would_change_my_mind": "More testing.",
    }
    return LLMClient(
        completion=FakeLLM({"JudgeResponse": response}),
        cache=JsonCache("score-test-cache", enabled=False),
    )


def test_rule_levels_for_trl_2_4_and_7() -> None:
    """Hand-built criterion results produce expected rule TRLs."""
    for target in (2, 4, 7):
        ids = [
            f"T{level}.{criterion}"
            for level in range(1, target + 1)
            for criterion in range(1, 4 if level > 1 else 3)
        ]
        ids = [criterion for criterion in ids if criterion != "T1.3"]
        results = [
            _result(criterion_id, evidence_ids=[f"E{index:03d}"])
            for index, criterion_id in enumerate(ids, 1)
        ]
        evidence = [
            _evidence(criterion_id, evidence_id=f"E{index:03d}")
            for index, criterion_id in enumerate(ids, 1)
        ]
        scores = calculate_level_scores(results, evidence)
        assert max(level.level for level in scores if level.score >= 0.6) == target


def test_planned_only_gets_no_credit() -> None:
    """Planned evidence cannot produce a positive criterion score."""
    result = _result("T2.1")
    evidence = [_evidence("T2.1", polarity="planned")]
    assert calculate_level_scores([result], evidence)[1].score == 0


def test_disagreement_lowers_confidence_and_jump_flag() -> None:
    """A wide rule/judge disagreement lowers confidence and detects jumps."""
    results = [
        _result(f"T7.{criterion}", evidence_ids=[f"E{criterion:03d}"]) for criterion in range(1, 4)
    ]
    evidence = [
        _evidence(f"T7.{criterion}", evidence_id=f"E{criterion:03d}") for criterion in range(1, 4)
    ]
    verdict, _ = score_document(
        _tech(),
        "paper",
        {"n_pages": 1, "ocr_pages": []},
        results,
        evidence,
        _judge(4),
        ScoringConfig(),
    )
    assert "evidence_jump" in verdict.flags
    assert verdict.confidence < 0.45
