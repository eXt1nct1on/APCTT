"""Tests for the end-to-end assessment orchestrator."""

from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas

from trl_scout.config import Settings
from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import LLMClient
from trl_scout.pipeline.run import _MockCompletion, assess


class _Index:
    """Small retrieval stub for orchestration tests."""

    cache_path = Path("test-embeddings.npy")

    def __init__(self, document: Any, cache_dir: str | Path | None = None) -> None:
        """Accept the production index constructor arguments."""

    def retrieve_for_level(self, level: Any, k: int = 8) -> list[Any]:
        """Return no passages while exercising all nine level calls."""
        return []


class _FailingLevelCompletion(_MockCompletion):
    """Fail one level while keeping all other mock responses valid."""

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Raise for TRL 5 evidence requests."""
        if kwargs.get("schema_name") == "RawLevel":
            user = str(kwargs.get("messages", [{}])[-1].get("content", ""))
            if "TRL level under review: 5" in user:
                raise RuntimeError("simulated level failure")
        return super().__call__(**kwargs)


def _pdf(path: Path) -> None:
    """Create a small source PDF."""
    document = canvas.Canvas(str(path))
    document.drawString(72, 760, "Abstract")
    text = document.beginText(72, 740)
    for _ in range(18):
        text.textLine("A laboratory battery prototype is described with repeated battery testing.")
    document.drawText(text)
    document.drawString(72, 700, "Results")
    text = document.beginText(72, 680)
    for _ in range(18):
        text.textLine("Measured capacity reached 100 mAh after cycling and repeated measurement.")
    document.drawText(text)
    document.save()


def _client(tmp_path: Path, completion: Any) -> LLMClient:
    """Create a cache-isolated test client."""
    return LLMClient(
        completion=completion,
        cache=JsonCache(tmp_path / "cache", enabled=False),
    )


def test_assess_persists_outputs_and_progress(tmp_path: Path, monkeypatch: Any) -> None:
    """The mock pipeline returns an Assessment and persists every stage."""
    source = tmp_path / "source.pdf"
    _pdf(source)
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    events = []
    assessment = assess(
        source,
        settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
        llm=_client(tmp_path, _MockCompletion()),
        progress=events.append,
    )
    run_dir = tmp_path / "data" / "runs" / assessment.run_id
    assert assessment.tech.name == "Example research technology"
    assert (run_dir / "assessment.json").exists()
    assert all((run_dir / f"{step}.json").exists() for step in (
        "ingest", "index", "identify", "evidence", "verify", "score",
        "gaps", "summary", "report", "assessment",
    ))
    assert any(event.step == "evidence" and event.status == "completed" for event in events)


def test_assess_marks_one_failed_level_and_continues(tmp_path: Path, monkeypatch: Any) -> None:
    """A failed level is unknown while later pipeline stages still complete."""
    source = tmp_path / "source.pdf"
    _pdf(source)
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    assessment = assess(
        source,
        settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
        llm=_client(tmp_path, _FailingLevelCompletion()),
    )
    assert "level_5_failed" in assessment.flags
    assert any(
        criterion.criterion_id == "T5.1" and criterion.status == "unknown"
        for criterion in assessment.levels[4].criteria
    )
    assert (tmp_path / "data" / "runs" / assessment.run_id / "assessment.json").exists()
