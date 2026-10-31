"""Security, resilience, and determinism regression tests."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from api.main import _safe_filename
from reportlab.pdfgen import canvas

from trl_scout.config import Settings
from trl_scout.ingest import loaders
from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import LLMClient
from trl_scout.pipeline.run import _MockCompletion, assess
from trl_scout.report.builder import render_html
from trl_scout.schemas import Chunk, Document, Sentence


class _Index:
    """Retrieval stub for hardening tests."""

    cache_path = Path("test-embeddings.npy")

    def __init__(self, document: Any, cache_dir: str | Path | None = None) -> None:
        self.document = document

    def retrieve_for_level(self, level: Any, k: int = 8) -> list[Any]:
        return []


def _client(tmp_path: Path, completion: Any | None = None) -> LLMClient:
    return LLMClient(
        completion=completion or _MockCompletion(),
        cache=JsonCache(tmp_path / "cache", enabled=False),
    )


def _pdf(path: Path, text: str) -> None:
    document = canvas.Canvas(str(path))
    document.drawString(72, 760, text)
    document.save()


def _document(chunks: list[Chunk], *, meta: dict[str, Any] | None = None) -> Document:
    return Document(
        doc_id="hardening",
        filename="hardening.txt",
        doc_type="paper",
        n_pages=max((chunk.page_end for chunk in chunks), default=1),
        sha256="0" * 64,
        meta=meta or {},
        chunks=chunks,
    )


def _chunk(number: int, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{number:03d}",
        page_start=number,
        page_end=number,
        section=section,
        text=text,
        sentences=[Sentence(sid=f"s{number:04d}", text=text)],
    )


def test_prompt_injection_is_flagged_and_does_not_change_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt-like document text is treated as data and reported."""
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    clean = tmp_path / "clean.txt"
    clean.write_text(
        "Abstract\nA laboratory battery prototype was measured.\nResults\n"
        "Capacity was measured over repeated cycles.",
        encoding="utf-8",
    )
    injection = Path("tests/fixtures/injection.pdf")
    baseline = assess(
        clean,
        settings=Settings(data_dir=str(tmp_path / "clean-data"), no_cache=True),
        llm=_client(tmp_path / "clean"),
    )
    hardened = assess(
        injection,
        settings=Settings(data_dir=str(tmp_path / "injection-data"), no_cache=True),
        llm=_client(tmp_path / "injection"),
    )
    assert "possible_prompt_injection" in hardened.flags
    assert hardened.verdict.trl == baseline.verdict.trl
    assert hardened.verdict.confidence == baseline.verdict.confidence
    assert "possible_prompt_injection" in render_html(hardened)


def test_long_document_keeps_key_sections_and_sets_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long documents are bounded while retaining decision-critical sections."""
    chunks = [_chunk(index, "background", "background " * 120) for index in range(1, 221)]
    chunks.insert(0, _chunk(221, "abstract", "abstract " * 100))
    chunks.append(_chunk(222, "results", "results " * 100))
    chunks.append(_chunk(223, "conclusion", "conclusion " * 100))
    document = _document(chunks)
    monkeypatch.setattr("trl_scout.pipeline.run.load_document", lambda source: document)
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    assessment = assess(
        "ignored.txt",
        settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
        llm=_client(tmp_path),
    )
    assert "long_document" in assessment.flags
    persisted = json.loads(
        (tmp_path / "data" / "runs" / assessment.run_id / "ingest.json").read_text(
            encoding="utf-8"
        )
    )
    sections = {chunk["section"] for chunk in persisted["chunks"]}
    assert {"abstract", "results", "conclusion"} <= sections
    assert len(persisted["chunks"]) < len(chunks)


def test_low_ocr_confidence_lowers_verdict_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low OCR quality is flagged and reduces the confidence factor."""
    document = _document(
        [_chunk(1, "results", "Measured laboratory performance.")],
        meta={"ocr_pages": [1], "ocr_confidence": 0.2},
    )
    monkeypatch.setattr("trl_scout.pipeline.run.load_document", lambda source: document)
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    assessment = assess(
        "ignored.txt",
        settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
        llm=_client(tmp_path),
    )
    assert "low_ocr_confidence" in assessment.flags
    assert assessment.verdict.confidence < 0.9


def test_loader_records_ocr_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OCR extraction stores a confidence estimate in document metadata."""
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"not-used")
    monkeypatch.setattr(
        loaders,
        "_pdf_pages",
        lambda content: (
            [{"page": 1, "lines": [{"text": "OCR text", "font_size": 0}]}],
            [1],
            [0.2],
        ),
    )
    document = loaders.load_document(path)
    assert document.meta["ocr_confidence"] == 0.2


def test_non_english_documents_are_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-English source text produces an explicit reliability warning."""
    document = _document(
        [_chunk(1, "abstract", "Esta tecnologia fue probada en un laboratorio.")],
        meta={"language": "es"},
    )
    monkeypatch.setattr("trl_scout.pipeline.run.load_document", lambda source: document)
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    assessment = assess(
        "ignored.txt",
        settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
        llm=_client(tmp_path),
    )
    assert "non_english_document" in assessment.flags


def test_tiny_document_returns_graceful_result(tmp_path: Path) -> None:
    """Tiny inputs do not invoke the expensive pipeline or crash."""
    assessment = assess(
        "Only a few words.",
        settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
        llm=_client(tmp_path),
    )
    assert "insufficient_text" in assessment.flags
    assert assessment.verdict.confidence == 0


def test_upload_filename_is_safely_basenamed() -> None:
    """Upload names cannot escape the run directory."""
    assert _safe_filename("../../secret.pdf") == "secret.pdf"
    assert ".." not in _safe_filename("..\\secret.pdf")


def test_same_cached_run_has_same_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache-backed repeated assessments keep all substantive outputs stable."""
    monkeypatch.setattr("trl_scout.pipeline.run.DocIndex", _Index)
    source = tmp_path / "same.txt"
    source.write_text(
        "Abstract\nA laboratory prototype.\nResults\nMeasured performance.",
        encoding="utf-8",
    )
    settings = Settings(data_dir=str(tmp_path / "data"), no_cache=False)
    first = assess(source, settings=settings, llm=_client(tmp_path))
    second = assess(source, settings=settings, llm=_client(tmp_path))
    assert first.tech == second.tech
    assert first.verdict == second.verdict
    assert first.levels == second.levels
    assert first.evidence == second.evidence


def test_logging_does_not_contain_document_or_api_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pipeline logs contain metadata, not source contents or credentials."""
    secret = "sk-test-secret"
    source_text = f"Private document body {secret}"
    with caplog.at_level(logging.INFO, logger="trl_scout"):
        assessment = assess(
            source_text,
            settings=Settings(data_dir=str(tmp_path / "data"), no_cache=True),
            llm=_client(tmp_path),
        )
    run_log = (
        tmp_path
        / "data"
        / "runs"
        / assessment.run_id
        / "run.log"
    ).read_text(encoding="utf-8")
    combined = "\n".join([run_log, *(record.getMessage() for record in caplog.records)])
    assert source_text not in combined
    assert secret not in combined
