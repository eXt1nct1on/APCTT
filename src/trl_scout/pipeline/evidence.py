"""Evidence assessment pipeline for all nine TRL levels."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from trl_scout.index.retriever import DocIndex
from trl_scout.ingest.loaders import load_document
from trl_scout.llm.client import FakeLLM, LLMClient, gather_limited, load_prompt
from trl_scout.pipeline.identify import identify_technology
from trl_scout.pipeline.verify import explicit_trl_mentions, planned_items, quote_span
from trl_scout.rubric import Level, load_rubric
from trl_scout.schemas import CriterionResult, Document, EvidenceItem, ProgressEvent, TechProfile

logger = logging.getLogger("trl_scout.pipeline.evidence")


class RawEvidence(BaseModel):
    """LLM-proposed evidence for one criterion."""

    chunk_id: str = Field(..., description="Chunk identifier.")
    quote: str = Field(..., description="Verbatim quote.")
    polarity: str = Field(..., description="Evidence polarity.")
    attribution: str = Field(..., description="Evidence attribution.")
    strength: float = Field(..., ge=0, le=1, description="Evidence strength.")
    why: str = Field(..., description="Evidence rationale.")


class RawCriterion(BaseModel):
    """LLM result for one rubric criterion."""

    criterion_id: str = Field(..., description="Criterion identifier.")
    status: str = Field(..., description="Criterion status.")
    evidence: list[RawEvidence] = Field(..., description="Proposed evidence.")
    note: str = Field(..., description="Criterion note.")


class RawLevel(BaseModel):
    """LLM result for one TRL level."""

    level: int = Field(..., ge=1, le=9, description="TRL level.")
    criteria: list[RawCriterion] = Field(..., description="Criterion assessments.")


def _criteria_text(level: Level) -> str:
    """Format rubric criteria for the evidence prompt."""
    return "\n".join(
        f"{criterion.id}: {criterion.text} (hints: {', '.join(criterion.hints)})"
        for criterion in level.criteria
    )


def _build_level_prompt(
    tech: TechProfile,
    document: Document,
    level: Level,
    passages: list[tuple[Any, float, list[str]]],
) -> tuple[str, str]:
    """Build the evidence prompt from trusted chunk metadata and text."""
    formatted = "\n".join(
        f"{chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | {chunk.section}\n"
        f"{chunk.text}"
        for chunk, _, _ in passages
    )
    return load_prompt(
        "evidence_level.md",
        {
            "tech_summary": tech.summary,
            "doc_type": document.doc_type,
            "level": str(level.level),
            "level_name": level.name,
            "criteria_list": _criteria_text(level),
            "passages": formatted,
        },
    )


def _evaluate_level(
    document: Document,
    tech: TechProfile,
    level: Level,
    index: DocIndex | None,
    llm: LLMClient,
) -> tuple[list[CriterionResult], list[EvidenceItem]]:
    """Evaluate one level and verify every proposed quote."""
    if index is None:
        raise RuntimeError("retrieval index is unavailable")
    passages = index.retrieve_for_level(level, k=8)
    system, user = _build_level_prompt(tech, document, level, passages)
    raw = llm.complete_json(system, user, RawLevel, max_tokens=3000)
    chunks = {chunk.chunk_id: chunk for chunk in document.chunks}
    expected_criteria = {criterion.id for criterion in level.criteria}
    criterion_results: list[CriterionResult] = []
    evidence: list[EvidenceItem] = []
    for criterion in raw.criteria:
        if criterion.criterion_id not in expected_criteria:
            continue
        verified_ids: list[str] = []
        criterion_evidence: list[EvidenceItem] = []
        for proposed in criterion.evidence:
            chunk = chunks.get(proposed.chunk_id)
            if chunk is None:
                continue
            ok, score, start, end = quote_span(proposed.quote, chunk.text)
            evidence_id = f"E{len(evidence) + len(criterion_evidence) + 1:03d}"
            item = EvidenceItem(
                evidence_id=evidence_id,
                criterion_id=criterion.criterion_id,
                chunk_id=chunk.chunk_id,
                page=chunk.page_start,
                quote=proposed.quote,
                polarity=proposed.polarity,
                attribution=proposed.attribution,
                strength=proposed.strength,
                why=proposed.why,
                verified=ok,
                match_score=score,
                span_start=start,
                span_end=end,
            )
            criterion_evidence.append(item)
            if ok:
                verified_ids.append(evidence_id)
            evidence.append(item)
        status = criterion.status
        if criterion_evidence and not verified_ids:
            status = "not_met"
        score = (
            sum(item.strength for item in criterion_evidence if item.verified)
            / len(criterion_evidence)
            if criterion_evidence
            else 0.0
        )
        criterion_results.append(
            CriterionResult(
                criterion_id=criterion.criterion_id,
                status=status,
                evidence_ids=verified_ids,
                score=score,
                note=criterion.note,
            )
        )
    return criterion_results, evidence


async def assess_evidence_async(
    document: Document,
    tech: TechProfile,
    llm: LLMClient,
    index: DocIndex | None,
    progress: Callable[[ProgressEvent], None] | None = None,
    failure_flags: list[str] | None = None,
) -> tuple[list[CriterionResult], list[EvidenceItem]]:
    """Assess all rubric levels concurrently with the configured LLM limit."""
    levels = load_rubric().levels

    async def evaluate(level: Level) -> tuple[list[CriterionResult], list[EvidenceItem]]:
        """Evaluate one level without allowing its failure to cancel siblings."""
        if progress is not None:
            progress(
                ProgressEvent(
                    step="evidence",
                    status="started",
                    detail=f"Assessing TRL {level.level}.",
                    pct=0,
                )
            )
        try:
            result = await asyncio.to_thread(_evaluate_level, document, tech, level, index, llm)
        except Exception as exc:
            if failure_flags is not None:
                failure_flags.append(f"level_{level.level}_failed")
            logger.exception("Evidence assessment failed for level %s", level.level)
            result = (
                [
                    CriterionResult(
                        criterion_id=criterion.id,
                        status="unknown",
                        evidence_ids=[],
                        score=0,
                        note=f"Level assessment failed: {exc}",
                    )
                    for criterion in level.criteria
                ],
                [],
            )
        if progress is not None:
            progress(
                ProgressEvent(
                    step="evidence",
                    status="completed",
                    detail=f"TRL {level.level} assessment complete.",
                    pct=0,
                )
            )
        return result

    results = await gather_limited(
        [evaluate(level) for level in levels]
    )
    criteria = [criterion for level_result, _ in results for criterion in level_result]
    evidence = [item for _, level_evidence in results for item in level_evidence]
    for number, item in enumerate(evidence, 1):
        item.evidence_id = f"E{number:03d}"
    evidence_ids_by_criterion: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        if item.verified:
            evidence_ids_by_criterion[item.criterion_id].append(item.evidence_id)
    criteria = [
        criterion.model_copy(
            update={"evidence_ids": evidence_ids_by_criterion[criterion.criterion_id]}
        )
        for criterion in criteria
    ]
    return criteria, evidence


def assess_evidence(
    document: Document,
    tech: TechProfile,
    llm: LLMClient,
    index: DocIndex | None,
    progress: Callable[[ProgressEvent], None] | None = None,
    failure_flags: list[str] | None = None,
) -> tuple[list[CriterionResult], list[EvidenceItem]]:
    """Synchronous entry point for evidence assessment."""
    return asyncio.run(
        assess_evidence_async(
            document,
            tech,
            llm,
            index,
            progress=progress,
            failure_flags=failure_flags,
        )
    )


def main() -> None:
    """Print a criterion table for a document."""
    parser = argparse.ArgumentParser(description="Assess evidence across TRL levels.")
    parser.add_argument("source", help="Document path, URL, arXiv id, or raw text.")
    parser.add_argument("--mock", action="store_true", help="Use a canned fake response.")
    args = parser.parse_args()
    document = load_document(args.source)
    llm = LLMClient(completion=FakeLLM() if args.mock else None)
    tech = identify_technology(document, llm)
    index = DocIndex(document)
    criteria, evidence = assess_evidence(document, tech, llm, index)
    verified = {item.evidence_id for item in evidence if item.verified}
    print("level\tcriterion\tstatus\tverified evidence")
    for criterion in criteria:
        level = criterion.criterion_id.split(".", 1)[0].removeprefix("T")
        count = len(set(criterion.evidence_ids) & verified)
        print(f"{level}\t{criterion.criterion_id}\t{criterion.status}\t{count}")
    print(f"author TRL mentions: {', '.join(explicit_trl_mentions(document.chunks)) or 'none'}")
    print(f"planned items: {len(planned_items(document.chunks))}")


if __name__ == "__main__":
    main()
