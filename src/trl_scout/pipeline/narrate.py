"""Executive summary narration from compact assessment data."""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from trl_scout.llm.client import FakeLLM, LLMClient, load_prompt
from trl_scout.schemas import Summary

logger = logging.getLogger("trl_scout.pipeline.narrate")


class RawSummary(BaseModel):
    """Validated shape returned by the summary prompt."""

    summary: str = Field(..., description="Executive summary.")
    key_takeaways: list[str] = Field(..., description="Key takeaways.")
    caveats: list[str] = Field(..., description="Caveats.")


def _compact_json(assessment: Any) -> str:
    """Serialize only compact structured assessment data, never source text."""
    if hasattr(assessment, "model_dump"):
        assessment = assessment.model_dump(mode="json")
    if not isinstance(assessment, dict):
        raise TypeError("assessment must be a mapping or Pydantic model")
    compact = {
        key: assessment[key]
        for key in ("tech", "verdict", "levels", "gaps")
        if key in assessment
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _recommendation_numbers(text: str) -> set[int]:
    """Extract TRL numbers mentioned in recommendations."""
    return {
        int(value)
        for value in re.findall(
            r"\b(?:TRL[-\s]?|technology readiness level\s*)([1-9])\b", text, re.I
        )
    }


def narrate_summary(
    assessment: Any,
    llm: LLMClient,
    *,
    current_trl: int | None = None,
    target_trl: int | None = None,
) -> Summary:
    """Generate a summary from compact assessment JSON and warn on TRL drift."""
    compact = _compact_json(assessment)
    system, user = load_prompt("summary.md", {"assessment_json": compact})
    raw = llm.complete_json(system, user, RawSummary, max_tokens=1000)
    allowed = {value for value in (current_trl, target_trl) if value is not None}
    mentioned = _recommendation_numbers(" ".join([raw.summary, *raw.key_takeaways, *raw.caveats]))
    unexpected = mentioned - allowed if allowed else set()
    if unexpected:
        logger.warning("Summary mentions TRL levels outside current/target: %s", sorted(unexpected))
    return Summary(
        text=raw.summary,
        key_takeaways=raw.key_takeaways,
        caveats=raw.caveats,
    )


def main() -> None:
    """Narrate a JSON assessment file."""
    parser = argparse.ArgumentParser(description="Write an executive summary.")
    parser.add_argument("assessment_json", help="Compact assessment JSON path.")
    parser.add_argument("--mock", action="store_true", help="Use a canned fake response.")
    args = parser.parse_args()
    from pathlib import Path

    payload = json.loads(Path(args.assessment_json).read_text(encoding="utf-8"))
    llm = LLMClient(completion=FakeLLM() if args.mock else None)
    summary = narrate_summary(payload, llm)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
