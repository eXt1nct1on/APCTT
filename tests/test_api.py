"""FastAPI endpoint tests."""

import asyncio
from pathlib import Path

from api.main import app
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from trl_scout.config import Settings
from trl_scout.pipeline.run import _MockCompletion


def _pdf_bytes(tmp_path: Path) -> bytes:
    """Create a small valid PDF upload."""
    path = tmp_path / "source.pdf"
    document = canvas.Canvas(str(path))
    document.drawString(72, 760, "Abstract")
    document.drawString(72, 740, "A laboratory prototype is described.")
    document.drawString(72, 700, "Results")
    document.drawString(72, 680, "Measured capacity reached 100 mAh.")
    document.save()
    return path.read_bytes()


def _wait(client: TestClient, run_id: str) -> dict:
    """Wait for a short mock assessment."""
    for _ in range(100):
        result = client.get(f"/v1/assessments/{run_id}").json()
        if result["status"] in {"done", "failed"}:
            return result
        asyncio.run(asyncio.sleep(0.01))
    raise AssertionError("assessment did not finish")


def test_upload_flow_and_events(tmp_path: Path) -> None:
    """A PDF upload is queued, completed, and exposes artifacts."""
    app.state.settings = Settings(data_dir=str(tmp_path / "data"), no_cache=True)
    app.state.llm = _MockCompletion()
    with TestClient(app) as client:
        response = client.post(
            "/v1/assessments",
            files={"file": ("../paper.pdf", _pdf_bytes(tmp_path), "application/pdf")},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        result = _wait(client, run_id)
        assert result["status"] == "done"
        events = client.get(f"/v1/assessments/{run_id}/events")
        assert events.status_code == 200
        assert "data:" in events.text
        assert client.get(f"/v1/assessments/{run_id}/assessment.json").status_code == 200


def test_upload_size_and_type_limits(tmp_path: Path) -> None:
    """Oversized and unsupported uploads return clear client errors."""
    app.state.settings = Settings(data_dir=str(tmp_path / "data"), max_upload_mb=1)
    with TestClient(app) as client:
        too_large = client.post(
            "/v1/assessments",
            files={"file": ("paper.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )
        assert too_large.status_code == 413
        bad_type = client.post(
            "/v1/assessments",
            files={"file": ("paper.exe", b"bad", "application/octet-stream")},
        )
        assert bad_type.status_code == 415
