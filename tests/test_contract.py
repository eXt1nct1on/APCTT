"""Contract tests for the rubric and assessment data model."""

import json
from pathlib import Path

from trl_scout.rubric import load_rubric
from trl_scout.schemas import Assessment

ROOT = Path(__file__).parents[1]


def test_rubric_has_nine_levels_and_unique_criteria() -> None:
    """The shipped rubric has exactly nine valid levels."""
    rubric = load_rubric()
    assert [level.level for level in rubric.levels] == list(range(1, 10))
    criterion_ids = [criterion.id for level in rubric.levels for criterion in level.criteria]
    assert len(criterion_ids) == len(set(criterion_ids))
    assert all(criterion.text for level in rubric.levels for criterion in level.criteria)


def test_example_assessment_validates() -> None:
    """The UI example conforms to the shared Assessment contract."""
    payload = json.loads((ROOT / "docs" / "example_assessment.json").read_text())
    assessment = Assessment.model_validate(payload)
    assert assessment.is_example is True
    assert assessment.verdict.trl == 4
    assert assessment.gaps.target_trl == 5


def test_models_round_trip() -> None:
    """An Assessment serializes and validates back without data loss."""
    payload = json.loads((ROOT / "docs" / "example_assessment.json").read_text())
    assessment = Assessment.model_validate(payload)
    assert Assessment.model_validate_json(assessment.model_dump_json()) == assessment
