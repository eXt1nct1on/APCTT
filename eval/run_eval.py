"""Run labelled documents through TRL-Scout evaluation modes."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from trl_scout.config import Settings
from trl_scout.ingest.loaders import load_document
from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import LLMClient
from trl_scout.pipeline.run import _MockCompletion, assess
from trl_scout.pipeline.score import calculate_level_scores
from trl_scout.schemas import Assessment

LOGGER = logging.getLogger("trl_scout.eval")
_TRL_RE = re.compile(r"\b(?:TRL[-\s]?|technology readiness level\s*)[1-9]\b", re.I)


class ZeroShotResponse(BaseModel):
    """Baseline response containing only a TRL estimate."""

    trl: int = Field(..., ge=1, le=9, description="Estimated TRL.")
    confidence: float = Field(0.5, ge=0, le=1, description="Estimate confidence.")


class _EvaluationMockCompletion(_MockCompletion):
    """Mock backend with a response for the zero-shot baseline schema."""

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("schema_name") == "ZeroShotResponse":
            content = '{"trl": 1, "confidence": 0.2}'
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        return super().__call__(**kwargs)


def mask_trl_mentions(text: str) -> str:
    """Mask explicit TRL mentions without changing document structure."""
    return _TRL_RE.sub("[MASKED TRL]", text)


def _load_rows(labels_path: Path) -> list[dict[str, str]]:
    """Read labelled evaluation rows."""
    with labels_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _zero_shot(document: Any, llm: LLMClient, mask: bool) -> ZeroShotResponse:
    """Run exactly one abstract/introduction/conclusion baseline call."""
    selected = []
    for chunk in document.chunks:
        if chunk.section.lower() in {
            "abstract",
            "introduction",
            "conclusion",
            "conclusions",
            "unknown",
        }:
            selected.append(chunk.text)
    source = "\n\n".join(selected)
    if mask:
        source = mask_trl_mentions(source)
    system = (
        "Estimate technology readiness from the supplied document excerpt. "
        "Return JSON only with trl (1-9) and confidence (0-1). Do not use any rubric."
    )
    user = f"<DOCUMENT>\n{source[:12000]}\n</DOCUMENT>"
    return llm.complete_json(
        system, user, ZeroShotResponse, temperature=0.0, max_tokens=200, sample_idx=0
    )


def _assessment_row(
    label: dict[str, str],
    assessment: Assessment,
    *,
    mode: str,
    raw_evidence: list[Any] | None = None,
) -> dict[str, Any]:
    """Flatten an Assessment into one prediction row."""
    evidence = raw_evidence if raw_evidence is not None else assessment.evidence
    verification = [item.verified for item in evidence if item.quote]
    return {
        "doc_id": label["doc_id"],
        "filename": label["filename"],
        "doc_type": label["doc_type"],
        "gold_trl": label["gold_trl"],
        "predicted_trl": assessment.verdict.trl,
        "range_low": assessment.verdict.range_low,
        "range_high": assessment.verdict.range_high,
        "confidence": assessment.verdict.confidence,
        "confidence_label": assessment.verdict.votes.get("confidence_label", ""),
        "quote_verification_rate": sum(verification) / len(verification) if verification else 1.0,
        "llm_calls": assessment.stats.llm_calls,
        "seconds": assessment.stats.seconds,
        "mode": mode,
        "flags": "|".join(assessment.flags),
    }


def _rule_trl(assessment: Assessment) -> int:
    """Derive the highest threshold-passing level from ablated scores."""
    eligible = [level.level for level in assessment.levels if level.score >= 0.6]
    return max(eligible, default=1)


def _masked_source(document: Any) -> Path:
    """Write a masked plain-text copy used as the actual pipeline source."""
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="trl-scout-eval-", delete=False, encoding="utf-8"
    )
    try:
        handle.write("\n\n".join(mask_trl_mentions(chunk.text) for chunk in document.chunks))
    finally:
        handle.close()
    return Path(handle.name)


def run_evaluation(
    labels_path: str | Path = "eval/gold/labels.csv",
    *,
    mode: str = "full",
    output_root: str | Path = "eval/results",
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    mask: bool = True,
    mock: bool = False,
) -> Path:
    """Run one evaluation mode and write predictions plus metrics."""
    if mode not in {"full", "zero_shot", "no_polarity", "no_verify", "rule_only"}:
        raise ValueError(f"unknown evaluation mode: {mode}")
    labels = _load_rows(Path(labels_path))
    run_root = Path(output_root) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") / mode
    run_root.mkdir(parents=True, exist_ok=True)
    runtime = settings or Settings(data_dir=str(run_root / "data"), no_cache=False)
    client = llm or LLMClient(
        completion=_EvaluationMockCompletion() if mock else None,
        cache=JsonCache(runtime.data_dir, enabled=not runtime.no_cache),
    )
    predictions: list[dict[str, Any]] = []
    for label in labels:
        source = Path(label["filename"])
        if not source.is_absolute():
            source = Path(labels_path).parent / "docs" / source.name
        if not source.exists():
            LOGGER.warning("Skipping missing evaluation document: %s", source)
            continue
        document = load_document(source)
        pipeline_source = _masked_source(document) if mask else source
        calls_before = client.stats.llm_calls
        seconds_before = client.stats.seconds
        if mode == "zero_shot":
            response = _zero_shot(document, client, mask=mask)
            row = {
                "doc_id": label["doc_id"],
                "filename": label["filename"],
                "doc_type": label["doc_type"],
                "gold_trl": label["gold_trl"],
                "predicted_trl": response.trl,
                "range_low": response.trl,
                "range_high": response.trl,
                "confidence": response.confidence,
                "confidence_label": "",
                "quote_verification_rate": 0.0,
                "llm_calls": client.stats.llm_calls - calls_before,
                "seconds": client.stats.seconds - seconds_before,
                "mode": mode,
                "flags": "",
            }
        else:
            assessment = assess(pipeline_source, settings=runtime, llm=client)
            raw_evidence = assessment.evidence
            raw_verification_rate = (
                sum(item.verified for item in raw_evidence) / len(raw_evidence)
                if raw_evidence
                else 1.0
            )
            if mode == "no_verify":
                raw_evidence = [
                    item.model_copy(update={"verified": True}) for item in assessment.evidence
                ]
            if mode == "no_polarity":
                raw_evidence = [
                    item.model_copy(update={"polarity": "achieved"}) for item in raw_evidence
                ]
            if mode in {"no_verify", "no_polarity"}:
                levels = calculate_level_scores(
                    [criterion for level in assessment.levels for criterion in level.criteria],
                    raw_evidence,
                )
                assessment = assessment.model_copy(
                    update={
                        "levels": levels,
                        "verdict": assessment.verdict.model_copy(
                            update={
                                "trl": _rule_trl(assessment.model_copy(update={"levels": levels})),
                                "flags": [
                                    *assessment.verdict.flags,
                                    f"ablation_{mode}",
                                ],
                            }
                        ),
                    }
                )
            if mode == "rule_only":
                assessment = assessment.model_copy(
                    update={
                        "verdict": assessment.verdict.model_copy(
                            update={
                                "trl": assessment.verdict.votes.get("rule", assessment.verdict.trl)
                            }
                        )
                    }
                )
            row = _assessment_row(label, assessment, mode=mode, raw_evidence=assessment.evidence)
            row["quote_verification_rate"] = raw_verification_rate
            row["llm_calls"] = client.stats.llm_calls - calls_before
            row["seconds"] = client.stats.seconds - seconds_before
        predictions.append(row)
        if pipeline_source != source:
            try:
                pipeline_source.unlink()
            except OSError:
                LOGGER.warning("Could not remove temporary masked source: %s", pipeline_source)
    prediction_path = run_root / "predictions.csv"
    fields = (
        list(predictions[0])
        if predictions
        else [
            "doc_id",
            "filename",
            "doc_type",
            "gold_trl",
            "predicted_trl",
        ]
    )
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    from eval.metrics import write_metrics

    write_metrics(predictions, run_root)
    return run_root


def main() -> None:
    """Run evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate TRL-Scout.")
    parser.add_argument("--labels", default="eval/gold/labels.csv")
    parser.add_argument(
        "--mode",
        choices=["full", "zero_shot", "no_polarity", "no_verify", "rule_only"],
        default="full",
    )
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_evaluation(
        args.labels, mode=args.mode, output_root=args.out, mask=not args.no_mask, mock=args.mock
    )


if __name__ == "__main__":
    main()
