"""Tests for gap analysis and summary narration."""

from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import FakeLLM, LLMClient
from trl_scout.pipeline.gaps import build_gap_analysis
from trl_scout.pipeline.narrate import narrate_summary
from trl_scout.schemas import CriterionResult, LevelScore, TechProfile, TRLVerdict


def _tech() -> TechProfile:
    """Build a compact profile."""
    return TechProfile(
        name="Battery", domain="energy", doc_type="research_paper", summary="Battery",
        intended_application="Storage", key_components=["cell"], claimed_results=[],
        author_stated_maturity=None, keywords=["battery"],
    )


def _verdict(trl: int) -> TRLVerdict:
    """Build a verdict."""
    return TRLVerdict(
        trl=trl, range_low=trl, range_high=trl, confidence=0.5,
        votes={"rule": trl, "judge": []}, reasoning="test",
        decisive_evidence_ids=[], contradictions=[], flags=[],
    )


def _levels(status: str = "partial") -> list[LevelScore]:
    """Build current and target level scores."""
    return [
        LevelScore(
            level=4,
            score=0.5,
            criteria=[CriterionResult(
                criterion_id="T4.1", status=status, evidence_ids=[], score=0, note="weak"
            )],
        ),
        LevelScore(
            level=5,
            score=0,
            criteria=[CriterionResult(
                criterion_id="T5.1", status="not_met", evidence_ids=[], score=0, note="missing"
            )],
        ),
    ]


def _gap_llm() -> LLMClient:
    """Build a fake gap response with one invalid ID and sorted priorities."""
    return LLMClient(
        completion=FakeLLM({"RawGapAnalysis": {
            "target_trl": 5,
            "headline": "Scale-up evidence is missing.",
            "gaps": [
                {"criterion_id": "T5.1", "missing": "Relevant testing", "why_it_matters": "Needed",
                 "actions": [{"step": "Run tests", "done_when": "Pass", "effort": "high",
                              "typical_duration": "6 months", "owner": "Lab"}], "priority": 2},
                {"criterion_id": "T999.1", "missing": "Invalid", "why_it_matters": "Invalid",
                 "actions": [], "priority": 1},
            ],
            "risks": [], "assumptions": [],
        }}), cache=JsonCache("gap-test-cache", enabled=False)
    )


def test_invalid_ids_removed_and_sorting_works() -> None:
    """Only real criteria remain and priority ordering is deterministic."""
    result = build_gap_analysis(_verdict(4), _levels(), _tech(), [], _gap_llm())
    assert [item.criterion_id for item in result.items] == ["T5.1"]
    assert result.roadmap["now"] == ["Run tests"]


def test_trl_nine_uses_sustain_and_scale_branch() -> None:
    """TRL 9 targets sustain and scale without requiring a level 10."""
    fake = LLMClient(completion=FakeLLM({"RawGapAnalysis": {
        "target_trl": 9, "headline": "Sustain operations.", "gaps": [],
        "risks": [], "assumptions": [],
    }}), cache=JsonCache("gap-nine-cache", enabled=False))
    result = build_gap_analysis(_verdict(9), _levels(), _tech(), [], fake)
    assert result.target_trl == 9


def test_summary_uses_compact_assessment() -> None:
    """Summary output maps to the public Summary model."""
    fake = LLMClient(completion=FakeLLM({"RawSummary": {
        "summary": "A concise summary.", "key_takeaways": ["One"], "caveats": ["Caveat"]
    }}), cache=JsonCache("summary-test-cache", enabled=False))
    result = narrate_summary({"tech": {"summary": "Battery"}, "verdict": {"trl": 4}}, fake,
                             current_trl=4, target_trl=5)
    assert result.text == "A concise summary."
