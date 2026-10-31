"""End-to-end TRL assessment orchestration."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from trl_scout.config import Settings
from trl_scout.config import settings as default_settings
from trl_scout.index.retriever import DocIndex
from trl_scout.ingest.loaders import load_document
from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import FakeLLM, LLMClient
from trl_scout.logging_utils import configure_logging
from trl_scout.pipeline.evidence import assess_evidence
from trl_scout.pipeline.gaps import build_gap_analysis
from trl_scout.pipeline.identify import identify_technology
from trl_scout.pipeline.narrate import narrate_summary
from trl_scout.pipeline.score import calculate_level_scores, score_document
from trl_scout.pipeline.verify import explicit_trl_mentions, planned_items
from trl_scout.schemas import (
    Assessment,
    CriterionResult,
    Document,
    EvidenceItem,
    GapAnalysis,
    ProgressEvent,
    Summary,
    TechProfile,
    TRLVerdict,
)

logger = logging.getLogger("trl_scout.pipeline.run")


def _jsonable(value: Any) -> Any:
    """Convert a Pydantic value or ordinary mapping to JSON-compatible data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _write_step(run_dir: Path, step: str, value: Any) -> None:
    """Persist one auditable pipeline output."""
    path = run_dir / f"{step}.json"
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _fallback_tech(document: Document) -> TechProfile:
    """Provide a usable profile when identification is degraded."""
    return TechProfile(
        name=document.meta.get("title") or Path(document.filename).stem or "Unknown technology",
        domain="unknown",
        doc_type=document.doc_type,
        summary="Technology identification was unavailable.",
        intended_application="Unknown",
        key_components=[],
        claimed_results=[],
        author_stated_maturity=None,
        keywords=[],
    )


def _fallback_verdict(levels: list[Any], flags: list[str]) -> TRLVerdict:
    """Create a rule-only verdict when scoring cannot complete."""
    eligible = [level.level for level in levels if level.score >= 0.6]
    trl = max(eligible, default=1)
    if not eligible and any(level.score > 0 for level in levels):
        flags = [*flags, "insufficient_evidence"]
    return TRLVerdict(
        trl=trl,
        range_low=trl,
        range_high=trl,
        confidence=0,
        votes={"rule": trl, "judge": [], "confidence_label": "Low"},
        reasoning="Scoring degraded; this is a rule-only fallback.",
        decisive_evidence_ids=[],
        contradictions=[],
        flags=flags,
    )


def _fallback_gaps(verdict: TRLVerdict) -> GapAnalysis:
    """Return an explicit empty gap result when the LLM gap step fails."""
    return GapAnalysis(
        target_trl=min(verdict.trl + 1, 9),
        headline="Gap analysis was unavailable.",
        items=[],
        risks=["Gap recommendations were not generated."],
        assumptions=[],
        roadmap={"now": [], "next": [], "later": []},
    )


def _fallback_summary(verdict: TRLVerdict) -> Summary:
    """Return a transparent summary when narration is unavailable."""
    return Summary(
        text=f"Assessment indicates TRL {verdict.trl} with confidence "
        f"{verdict.confidence:.2f}.",
        key_takeaways=[f"Assigned TRL: {verdict.trl}."],
        caveats=["Executive narration was unavailable."],
    )


class _MockCompletion(FakeLLM):
    """Example-shaped fake backend that remains valid for arbitrary source chunks."""

    def __init__(self) -> None:
        """Create a deterministic mock backend."""
        super().__init__()

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Return a valid response for each pipeline response model."""
        schema_name = kwargs.get("schema_name", "")
        user = str(kwargs.get("messages", [{}])[-1].get("content", ""))
        if schema_name == "IdentificationResponse":
            value: Any = {
                "doc_type": "research_paper",
                "doc_type_confidence": 0.9,
                "technology_name": "Example research technology",
                "domain": "Engineering",
                "one_line_summary": "An invented technology assessed from a research document.",
                "intended_application": "A practical engineering application.",
                "key_components": ["prototype", "test assembly"],
                "claimed_results": [],
                "author_stated_maturity": "Laboratory validation",
                "search_keywords": ["prototype", "validation", "test"],
            }
        elif schema_name == "RawLevel":
            match = re.search(r"Level:\s*(\d+)", user)
            level = int(match.group(1)) if match else 1
            value = {"level": level, "criteria": []}
        elif schema_name == "JudgeResponse":
            value = {
                "trl": 1,
                "range_low": 1,
                "range_high": 1,
                "confidence": 0.2,
                "reasoning": "The mock contains no generated evidence.",
                "decisive_evidence_ids": [],
                "contradictions": [],
                "what_would_change_my_mind": "Verified experimental evidence.",
            }
        elif schema_name == "RawGapAnalysis":
            match = re.search(r'"target_trl":\s*(\d+)', user)
            value = {
                "target_trl": int(match.group(1)) if match else 2,
                "headline": "More verified evidence is needed.",
                "gaps": [],
                "risks": [],
                "assumptions": [],
            }
        elif schema_name == "RawSummary":
            value = {
                "summary": "The document was assessed using the available evidence.",
                "key_takeaways": ["Evidence is limited in mock mode."],
                "caveats": ["This is a mock assessment."],
            }
        else:
            value = {"ok": True}
        return {
            "choices": [{"message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def _client_for(settings: Settings, llm: LLMClient | Callable[..., Any] | None) -> LLMClient:
    """Create or normalize the injectable LLM client."""
    if isinstance(llm, LLMClient):
        return llm
    return LLMClient(
        completion=llm,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        cache=JsonCache(settings.data_dir, enabled=not settings.no_cache),
    )


def assess(
    source: str | Path,
    *,
    settings: Settings = default_settings,
    llm: LLMClient | Callable[..., Any] | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> Assessment:
    """Run every assessment stage and return a persisted, failure-tolerant assessment."""
    run_id = uuid4().hex
    run_dir = Path(settings.data_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_id, settings.data_dir)
    client = _client_for(settings, llm)
    flags: list[str] = []
    started = time.perf_counter()
    total_steps = 10
    completed_steps = 0

    def emit(step: str, status: str, detail: str) -> None:
        """Emit progress and log elapsed time for a stage."""
        nonlocal completed_steps
        if status == "completed":
            completed_steps += 1
        event = ProgressEvent(
            step=step,
            status=status,
            detail=detail,
            pct=completed_steps / total_steps * 100,
        )
        if progress is not None:
            progress(event)
        logger.info("%s %s: %s (%.3fs)", step, status, detail, time.perf_counter() - started)

    emit("ingest", "started", "Loading source document.")
    try:
        document = load_document(source)
        _write_step(run_dir, "ingest", document)
        emit("ingest", "completed", f"Loaded {document.n_pages} page(s).")
    except Exception as exc:
        emit("ingest", "failed", str(exc))
        raise

    emit("index", "started", "Building document index.")
    try:
        index: DocIndex | None = DocIndex(document, cache_dir=settings.data_dir)
        _write_step(
            run_dir,
            "index",
            {"chunk_count": len(document.chunks), "cache_path": str(index.cache_path)},
        )
        emit("index", "completed", f"Indexed {len(document.chunks)} chunk(s).")
    except Exception as exc:
        flags.append("index_failed")
        index = None
        logger.exception("Document index failed")
        _write_step(run_dir, "index", {"status": "failed", "error": str(exc)})
        emit("index", "failed", "Continuing without retrieval.")

    emit("identify", "started", "Identifying technology.")
    try:
        tech = identify_technology(document, client)
        _write_step(run_dir, "identify", tech)
        emit("identify", "completed", tech.name)
    except Exception as exc:
        flags.append("identify_failed")
        tech = _fallback_tech(document)
        _write_step(run_dir, "identify", {"status": "failed", "error": str(exc), "fallback": tech})
        logger.exception("Technology identification failed")
        emit("identify", "failed", "Using degraded technology profile.")

    emit("evidence", "started", "Assessing evidence for all TRL levels.")
    criteria: list[CriterionResult]
    evidence: list[EvidenceItem]
    try:
        criteria, evidence = assess_evidence(
            document,
            tech,
            client,
            index,
            progress=progress,
            failure_flags=flags,
        )
        _write_step(run_dir, "evidence", {"criteria": criteria, "evidence": evidence})
        emit("evidence", "completed", f"Assessed {len(criteria)} criteria.")
    except Exception as exc:
        flags.append("evidence_failed")
        criteria, evidence = [], []
        _write_step(run_dir, "evidence", {"status": "failed", "error": str(exc)})
        logger.exception("Evidence assessment failed")
        emit("evidence", "failed", "Continuing with unknown evidence.")

    emit("verify", "started", "Extracting author signals and planned work.")
    mentions = explicit_trl_mentions(document.chunks)
    planned = planned_items(document.chunks)
    _write_step(
        run_dir,
        "verify",
        {"author_stated_trl_mentions": mentions, "planned_items": planned},
    )
    emit(
        "verify",
        "completed",
        f"Found {len(mentions)} TRL mention(s) and {len(planned)} plan item(s).",
    )

    emit("score", "started", "Scoring verified evidence.")
    try:
        verdict, levels = score_document(
            tech,
            document.doc_type,
            {"n_pages": document.n_pages, **document.meta},
            criteria,
            evidence,
            client,
        )
        _write_step(run_dir, "score", {"verdict": verdict, "levels": levels})
        emit("score", "completed", f"Assigned TRL {verdict.trl}.")
    except Exception as exc:
        flags.append("score_failed")
        levels = calculate_level_scores(criteria, evidence)
        verdict = _fallback_verdict(levels, flags)
        _write_step(
            run_dir,
            "score",
            {"status": "failed", "error": str(exc), "verdict": verdict, "levels": levels},
        )
        logger.exception("Scoring failed")
        emit("score", "failed", "Using rule-only fallback.")

    verdict = verdict.model_copy(update={"flags": sorted(set([*verdict.flags, *flags]))})

    emit("gaps", "started", "Generating gap analysis.")
    try:
        gaps = build_gap_analysis(verdict, levels, tech, planned, client)
        _write_step(run_dir, "gaps", gaps)
        emit("gaps", "completed", f"Generated {len(gaps.items)} gap(s).")
    except Exception as exc:
        flags.append("gaps_failed")
        gaps = _fallback_gaps(verdict)
        _write_step(run_dir, "gaps", {"status": "failed", "error": str(exc), "fallback": gaps})
        logger.exception("Gap analysis failed")
        emit("gaps", "failed", "Using empty gap fallback.")

    emit("summary", "started", "Writing executive summary.")
    try:
        summary = narrate_summary(
            {"tech": tech, "verdict": verdict, "levels": levels, "gaps": gaps},
            client,
            current_trl=verdict.trl,
            target_trl=gaps.target_trl,
        )
        _write_step(run_dir, "summary", summary)
        emit("summary", "completed", "Executive summary generated.")
    except Exception as exc:
        flags.append("summary_failed")
        summary = _fallback_summary(verdict)
        _write_step(
            run_dir,
            "summary",
            {"status": "failed", "error": str(exc), "fallback": summary},
        )
        logger.exception("Summary narration failed")
        emit("summary", "failed", "Using concise fallback summary.")

    emit("report", "started", "Checking report generation support.")
    report_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "Report builder is not available yet.",
    }
    try:
        from trl_scout.report.builder import build_report
    except ImportError:
        build_report = None
    if callable(build_report):
        try:
            report_result = {"status": "completed", "output": build_report}
        except Exception as exc:
            flags.append("report_failed")
            report_result = {"status": "failed", "error": str(exc)}
            logger.exception("Report generation failed")
    _write_step(run_dir, "report", report_result)
    emit("report", report_result["status"], str(report_result.get("reason", "Report complete.")))

    flags = sorted(set([*flags, *verdict.flags]))
    verdict = verdict.model_copy(update={"flags": flags})
    stats = client.stats.model_copy(
        update={"seconds": max(client.stats.seconds, time.perf_counter() - started)}
    )
    assessment = Assessment(
        run_id=run_id,
        created_at=datetime.now(UTC),
        is_example=False,
        source=str(source),
        tech=tech,
        verdict=verdict,
        levels=levels,
        evidence=evidence,
        gaps=gaps,
        summary=summary,
        author_stated_trl_mentions=mentions,
        planned_items=planned,
        flags=flags,
        stats=stats,
    )
    _write_step(run_dir, "assessment", assessment)
    return assessment


def main() -> None:
    """Run an assessment from the command line."""
    parser = argparse.ArgumentParser(description="Run a complete TRL assessment.")
    parser.add_argument("source", help="Document path, URL, arXiv id, or raw text.")
    parser.add_argument("--out", default=None, help="Output directory for assessment.json.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the LLM cache.")
    args = parser.parse_args()
    output = Path(args.out) if args.out else Path(default_settings.data_dir)
    run_settings = Settings(data_dir=str(output), no_cache=args.no_cache)
    assessment = assess(
        args.source,
        settings=run_settings,
        llm=LLMClient(
            completion=_MockCompletion() if args.mock else None,
            model=run_settings.llm_model,
            fallback_model=run_settings.llm_fallback_model,
            cache=JsonCache(run_settings.data_dir, enabled=not run_settings.no_cache),
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "assessment.json").write_text(
        assessment.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"Technology: {assessment.tech.name}")
    print(
        f"TRL: {assessment.verdict.trl} "
        f"(range {assessment.verdict.range_low}-{assessment.verdict.range_high}, "
        f"confidence {assessment.verdict.confidence:.2f})"
    )
    print("Top gaps:")
    for item in assessment.gaps.items[:3]:
        print(f"- {item.criterion_id}: {item.missing}")


if __name__ == "__main__":
    main()
