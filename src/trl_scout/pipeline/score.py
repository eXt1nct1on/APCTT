"""Rule-based and judge-assisted TRL scoring."""

from __future__ import annotations

import argparse
import asyncio
import math
from collections.abc import Sequence
from statistics import median

from pydantic import BaseModel, Field

from trl_scout.config import ScoringConfig, scoring_config
from trl_scout.index.retriever import DocIndex
from trl_scout.ingest.loaders import load_document
from trl_scout.llm.client import FakeLLM, LLMClient, gather_limited, load_prompt
from trl_scout.pipeline.evidence import assess_evidence
from trl_scout.pipeline.identify import identify_technology
from trl_scout.rubric import load_rubric
from trl_scout.schemas import CriterionResult, EvidenceItem, LevelScore, TechProfile, TRLVerdict


class JudgeResponse(BaseModel):
    """Validated response from one TRL judge sample."""

    trl: int = Field(..., ge=1, le=9, description="Judge TRL.")
    range_low: int = Field(..., ge=1, le=9, description="Judge lower range.")
    range_high: int = Field(..., ge=1, le=9, description="Judge upper range.")
    confidence: float = Field(..., ge=0, le=1, description="Judge confidence.")
    reasoning: str = Field(..., description="Judge reasoning.")
    decisive_evidence_ids: list[str] = Field(..., description="Decisive evidence IDs.")
    contradictions: list[str] = Field(..., description="Contradictions.")
    what_would_change_my_mind: str = Field(..., description="Changing evidence.")


def _criterion_score(
    result: CriterionResult,
    evidence: Sequence[EvidenceItem],
    config: ScoringConfig,
) -> float:
    """Calculate one criterion score from verified evidence only."""
    verified = [
        item
        for item in evidence
        if item.verified
        and item.evidence_id in result.evidence_ids
        and item.criterion_id == result.criterion_id
    ]
    if not verified:
        return 0.0
    achieved = [item for item in verified if item.polarity == "achieved"]
    hypothetical = [item for item in verified if item.polarity == "hypothetical"]
    if not achieved and not hypothetical:
        return 0.0
    score = {"met": 1.0, "partial": 0.5}.get(result.status, 0.0)
    if not achieved and hypothetical and not result.criterion_id.startswith(("T1.", "T2.")):
        score = min(score, config.hypothetical_only_cap)
    if all(item.attribution == "cited" for item in verified):
        score = min(score, config.cited_only_cap)
    return score


def calculate_level_scores(
    criterion_results: Sequence[CriterionResult],
    evidence: Sequence[EvidenceItem],
    config: ScoringConfig = scoring_config,
) -> list[LevelScore]:
    """Calculate mean criterion scores for all nine rubric levels."""
    by_id = {result.criterion_id: result for result in criterion_results}
    by_level: dict[int, list[CriterionResult]] = {level: [] for level in range(1, 10)}
    for rubric_level in load_rubric().levels:
        for criterion in rubric_level.criteria:
            by_id.setdefault(
                criterion.id,
                CriterionResult(
                    criterion_id=criterion.id,
                    status="not_met",
                    evidence_ids=[],
                    score=0,
                    note="No assessed evidence.",
                ),
            )
    for result in by_id.values():
        by_level[int(result.criterion_id[1:].split(".", 1)[0])].append(result)
    levels: list[LevelScore] = []
    for level in range(1, 10):
        criteria = by_level[level]
        scores = [_criterion_score(result, evidence, config) for result in criteria]
        levels.append(
            LevelScore(
                level=level,
                score=sum(scores) / len(scores) if scores else 0.0,
                criteria=[
                    result.model_copy(update={"score": score})
                    for result, score in zip(criteria, scores, strict=False)
                ],
            )
        )
    return levels


def _rule_result(
    levels: Sequence[LevelScore],
    evidence: Sequence[EvidenceItem],
    config: ScoringConfig,
) -> tuple[int, int, int, list[str]]:
    """Compute preliminary TRL, range, and rule flags."""
    scores = {level.level: level.score for level in levels}
    eligible = [level for level, score in scores.items() if score >= config.rule_threshold]
    flags: list[str] = []
    rule_trl = max(eligible) if eligible else 1
    if not eligible and evidence:
        flags.append("insufficient_evidence")
    if rule_trl > 2 and scores.get(rule_trl - 1, 0.0) < config.evidence_jump_threshold:
        flags.append("evidence_jump")
    range_high = rule_trl
    next_score = scores.get(rule_trl + 1, 0.0)
    if config.range_partial_low <= next_score <= config.range_partial_high:
        range_high = min(9, rule_trl + 1)
    return rule_trl, rule_trl, range_high, flags


def _evidence_table(levels: Sequence[LevelScore], evidence: Sequence[EvidenceItem]) -> str:
    """Build the compact evidence table sent to judges."""
    by_id = {item.evidence_id: item for item in evidence if item.verified}
    rows: list[str] = []
    for level in levels:
        for criterion in level.criteria:
            ids = [
                f"{evidence_id}: {by_id[evidence_id].quote[:80]}"
                for evidence_id in criterion.evidence_ids
                if evidence_id in by_id
            ]
            rows.append(
                f"{level.level} | {criterion.criterion_id} | {criterion.status} | "
                f"{'; '.join(ids) or 'none'}"
            )
    return "\n".join(rows)


async def _judge_samples(
    tech: TechProfile,
    document_type: str,
    rule_trl: int,
    rule_low: int,
    rule_high: int,
    levels: Sequence[LevelScore],
    evidence: Sequence[EvidenceItem],
    llm: LLMClient,
) -> list[JudgeResponse]:
    """Run three judge samples and discard invalid evidence citations."""
    system, user = load_prompt(
        "trl_judge.md",
        {
            "tech_summary": tech.summary,
            "doc_type": document_type,
            "rule_trl": str(rule_trl),
            "rule_low": str(rule_low),
            "rule_high": str(rule_high),
            "evidence_table": _evidence_table(levels, evidence),
        },
    )
    valid_ids = {item.evidence_id for item in evidence if item.verified}

    async def one(sample_idx: int) -> JudgeResponse | None:
        try:
            result = await llm.acomplete_json(
                system,
                user,
                JudgeResponse,
                temperature=0.3,
                max_tokens=1200,
                sample_idx=sample_idx,
            )
        except RuntimeError:
            return None
        if not set(result.decisive_evidence_ids).issubset(valid_ids):
            return None
        return result

    results = await gather_limited([one(index) for index in range(3)])
    return [result for result in results if result is not None]


def _confidence(
    votes: Sequence[int],
    evidence: Sequence[EvidenceItem],
    document_type: str,
    meta: dict[str, object],
    config: ScoringConfig,
) -> tuple[float, str]:
    """Calculate confidence and its display label."""
    spread = max(votes) - min(votes) if votes else 0
    agreement = max(0.0, 1 - spread / config.confidence_spread_divisor)
    achieved_count = len(
        {item.evidence_id for item in evidence if item.verified and item.polarity == "achieved"}
    )
    factor = min(1.0, achieved_count / config.evidence_factor_cap)
    if document_type == "patent" and not any(
        item.verified and item.polarity == "achieved" for item in evidence
    ):
        factor *= config.patent_no_examples_factor
    pages = int(meta.get("n_pages", 0) or 0)
    ocr_pages = len(meta.get("ocr_pages", []) or [])
    if pages and ocr_pages / pages > config.ocr_page_fraction:
        factor *= config.ocr_factor
    ocr_confidence = meta.get("ocr_confidence")
    if ocr_confidence is not None and float(ocr_confidence) < 0.5:
        factor *= config.low_ocr_factor
    confidence = agreement * factor
    label = "Low" if confidence < 0.45 else "Medium" if confidence < 0.75 else "High"
    return confidence, label


def score_document(
    tech: TechProfile,
    document_type: str,
    meta: dict[str, object],
    criterion_results: Sequence[CriterionResult],
    evidence: Sequence[EvidenceItem],
    llm: LLMClient,
    config: ScoringConfig = scoring_config,
) -> tuple[TRLVerdict, list[LevelScore]]:
    """Score evidence and combine rule-based and judge votes."""
    levels = calculate_level_scores(criterion_results, evidence, config)
    rule_trl, rule_low, rule_high, flags = _rule_result(levels, evidence, config)
    judges = asyncio.run(
        _judge_samples(tech, document_type, rule_trl, rule_low, rule_high, levels, evidence, llm)
    )
    judge_votes = [judge.trl for judge in judges]
    votes = [rule_trl, *judge_votes]
    final_trl = math.floor(median(votes))
    range_low = min([rule_low, *[judge.range_low for judge in judges]])
    range_high = max([rule_high, *[judge.range_high for judge in judges]])
    confidence, label = _confidence(votes, evidence, document_type, meta, config)
    flags.append(f"confidence_{label.lower()}")
    decisive = sorted(
        {evidence_id for judge in judges for evidence_id in judge.decisive_evidence_ids}
    )
    verdict = TRLVerdict(
        trl=final_trl,
        range_low=max(1, range_low),
        range_high=min(9, range_high),
        confidence=confidence,
        votes={"rule": rule_trl, "judge": judge_votes, "confidence_label": label},
        reasoning=judges[0].reasoning
        if judges
        else "Rule-based score only; no valid judge response.",
        decisive_evidence_ids=decisive,
        contradictions=[item for judge in judges for item in judge.contradictions],
        flags=flags,
    )
    return verdict, list(levels)


def main() -> None:
    """Print level scores and the final verdict."""
    parser = argparse.ArgumentParser(description="Score TRL evidence.")
    parser.add_argument("source", help="Document path, URL, arXiv id, or raw text.")
    parser.add_argument("--mock", action="store_true", help="Use a canned fake response.")
    args = parser.parse_args()
    document = load_document(args.source)
    llm = LLMClient(completion=FakeLLM() if args.mock else None)
    tech = identify_technology(document, llm)
    criteria, evidence = assess_evidence(document, tech, llm, DocIndex(document))
    verdict, levels = score_document(
        tech,
        document.doc_type,
        {"n_pages": document.n_pages, **document.meta},
        criteria,
        evidence,
        llm,
    )
    print("level\tscore")
    for level in levels:
        print(f"{level.level}\t{level.score:.2f}")
    print(f"verdict\tTRL {verdict.trl}\trange {verdict.range_low}-{verdict.range_high}")
    print(f"confidence\t{verdict.confidence:.2f}\t{verdict.votes['confidence_label']}")


if __name__ == "__main__":
    main()
