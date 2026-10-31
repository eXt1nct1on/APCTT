"""TRL gap analysis and roadmap generation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from trl_scout.llm.client import FakeLLM, LLMClient, load_prompt
from trl_scout.rubric import load_rubric
from trl_scout.schemas import (
    CriterionResult,
    GapAnalysis,
    GapItem,
    LevelScore,
    TechProfile,
    TRLVerdict,
)

logger = logging.getLogger("trl_scout.pipeline.gaps")


class RawGapAnalysis(BaseModel):
    """Validated shape returned by the gap prompt."""

    target_trl: int = Field(..., ge=1, le=9, description="Target TRL.")
    headline: str = Field(..., description="Primary blocker.")
    gaps: list[GapItem] = Field(..., description="Gap items.")
    risks: list[str] = Field(..., description="Risks.")
    assumptions: list[str] = Field(..., description="Assumptions.")


_EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}


def _target_name(target: int) -> str:
    """Return a rubric name or the sustain-and-scale branch name."""
    if target == 9:
        return "Sustain and scale"
    return next(level.name for level in load_rubric().levels if level.level == target)


def _candidate_criteria(
    current_trl: int,
    target_trl: int,
    levels: Sequence[LevelScore],
) -> list[CriterionResult]:
    """Collect target gaps and weak spots at the current level."""
    by_level = {level.level: level for level in levels}
    target = by_level.get(target_trl)
    current = by_level.get(current_trl)
    candidates = []
    if target is not None:
        candidates.extend(
            criterion
            for criterion in target.criteria
            if criterion.status in {"not_met", "partial", "unknown"}
        )
    if current is not None:
        candidates.extend(
            criterion for criterion in current.criteria if criterion.status == "partial"
        )
    seen: set[str] = set()
    unique: list[CriterionResult] = []
    for criterion in candidates:
        if criterion.criterion_id not in seen:
            seen.add(criterion.criterion_id)
            unique.append(criterion)
    return unique


def _criteria_text(criteria: Sequence[CriterionResult]) -> str:
    """Format gap criteria with rubric text and current evidence status."""
    rubric = {
        criterion.id: criterion
        for level in load_rubric().levels
        for criterion in level.criteria
    }
    return "\n".join(
        f"{criterion.criterion_id} | {rubric[criterion.criterion_id].text} | "
        f"{criterion.status} | {criterion.note}"
        for criterion in criteria
        if criterion.criterion_id in rubric
    )


def _planned_text(planned: Sequence[str]) -> str:
    """Format planned source items for the gap prompt."""
    return "\n".join(f"- {item}" for item in planned) or "None stated."


def _roadmap(items: Sequence[GapItem]) -> dict[str, list[str]]:
    """Group actions into a simple now/next/later roadmap."""
    ordered = sorted(items, key=lambda item: (item.priority, min(
        (_EFFORT_ORDER[action.effort] for action in item.actions), default=1
    )))
    actions = [action for item in ordered for action in item.actions]
    roadmap = {"now": [], "next": [], "later": []}
    for index, action in enumerate(actions):
        bucket = "now" if index < 2 else "next" if index < 5 else "later"
        roadmap[bucket].append(action.step)
    return roadmap


def build_gap_analysis(
    verdict: TRLVerdict,
    levels: Sequence[LevelScore],
    tech: TechProfile,
    planned: Sequence[str],
    llm: LLMClient,
) -> GapAnalysis:
    """Generate, validate, filter, sort, and roadmap the next-TRL gaps."""
    target = min(verdict.trl + 1, 9)
    criteria = _candidate_criteria(verdict.trl, target, levels)
    system, user = load_prompt(
        "gaps.md",
        {
            "tech_summary": tech.summary,
            "trl": str(verdict.trl),
            "range_low": str(verdict.range_low),
            "range_high": str(verdict.range_high),
            "target": str(target),
            "target_name": _target_name(target),
            "gap_criteria": _criteria_text(criteria) or "No unmet criteria identified.",
            "planned_items": _planned_text(planned),
            "domain_hints": f"{tech.domain}; {tech.intended_application}",
        },
    )
    raw = llm.complete_json(system, user, RawGapAnalysis, max_tokens=3000)
    valid_ids = {criterion.criterion_id for criterion in criteria}
    items = [item for item in raw.gaps if item.criterion_id in valid_ids]
    invalid = [item.criterion_id for item in raw.gaps if item.criterion_id not in valid_ids]
    if invalid:
        logger.warning("Dropped gap recommendations for invalid criteria: %s", invalid)
    items.sort(
        key=lambda item: (
            item.priority,
            min((_EFFORT_ORDER[action.effort] for action in item.actions), default=1),
        )
    )
    return GapAnalysis(
        target_trl=target,
        headline=raw.headline,
        items=items,
        risks=raw.risks,
        assumptions=raw.assumptions,
        roadmap=_roadmap(items),
    )


def main() -> None:
    """Print a gap headline, items, and roadmap."""
    parser = argparse.ArgumentParser(description="Generate TRL gaps and roadmap.")
    parser.add_argument("source", help="Document path, URL, arXiv id, or raw text.")
    parser.add_argument("--mock", action="store_true", help="Use a canned fake response.")
    args = parser.parse_args()
    from trl_scout.index.retriever import DocIndex
    from trl_scout.ingest.loaders import load_document
    from trl_scout.pipeline.evidence import assess_evidence
    from trl_scout.pipeline.identify import identify_technology
    from trl_scout.pipeline.score import score_document
    from trl_scout.pipeline.verify import planned_items

    document = load_document(args.source)
    llm = LLMClient(completion=FakeLLM() if args.mock else None)
    tech = identify_technology(document, llm)
    criteria, evidence = assess_evidence(document, tech, llm, DocIndex(document))
    verdict, levels = score_document(
        tech, document.doc_type, document.meta, criteria, evidence, llm
    )
    gaps = build_gap_analysis(verdict, levels, tech, planned_items(document.chunks), llm)
    print(gaps.headline)
    for item in gaps.items:
        print(f"- {item.criterion_id} (priority {item.priority}): {item.missing}")
    for phase, actions in gaps.roadmap.items():
        print(f"{phase}:")
        for action in actions:
            print(f"  - {action}")


if __name__ == "__main__":
    main()
