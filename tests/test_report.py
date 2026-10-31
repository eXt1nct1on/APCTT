"""Tests for HTML and PDF report generation."""

import importlib.util
import json
from pathlib import Path

import pytest

from trl_scout.report.builder import build_report, render_html
from trl_scout.schemas import Assessment

EXAMPLE = Path(__file__).parents[1] / "docs" / "example_assessment.json"


def test_example_html_contains_verdict_and_verified_quotes() -> None:
    """The report contains the verdict and every verified source quote."""
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    html = render_html(payload)
    assert "TRL" in html and ">4<" in html
    for item in payload["evidence"]:
        if item["verified"]:
            assert item["quote"] in html
    Assessment.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))


@pytest.mark.skipif(importlib.util.find_spec("weasyprint") is None, reason="WeasyPrint unavailable")
def test_example_pdf_is_created(tmp_path: Path) -> None:
    """WeasyPrint produces a PDF when installed."""
    output = build_report(EXAMPLE, tmp_path / "report.pdf")
    assert output.suffix == ".pdf"
    assert output.exists()
    assert output.stat().st_size > 0
