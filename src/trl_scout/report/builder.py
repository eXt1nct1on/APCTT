"""Build self-contained HTML and optional PDF assessment reports."""

from __future__ import annotations

import argparse
import html
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from trl_scout.report.charts import trl_ladder
from trl_scout.schemas import Assessment

logger = logging.getLogger("trl_scout.report")
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _load_assessment(value: Assessment | str | Path | dict[str, Any]) -> Assessment:
    """Load and validate an assessment from supported input forms."""
    if isinstance(value, Assessment):
        return value
    if isinstance(value, (str, Path)):
        return Assessment.model_validate_json(Path(value).read_text(encoding="utf-8"))
    return Assessment.model_validate(value)


def _document_title(assessment: Assessment) -> str:
    """Return the source document title used in the report header."""
    title = assessment.source.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    return Path(title).stem or title


def render_html(value: Assessment | str | Path | dict[str, Any]) -> str:
    """Render a validated assessment into one self-contained HTML document."""
    assessment = _load_assessment(value)
    environment = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=True,
    )
    template = environment.get_template("report.html.j2")
    css = (_TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")
    evidence_by_level: dict[int, list[Any]] = {level.level: [] for level in assessment.levels}
    level_by_criterion = {
        criterion.criterion_id: level.level
        for level in assessment.levels
        for criterion in level.criteria
    }
    for item in assessment.evidence:
        evidence_by_level.setdefault(level_by_criterion.get(item.criterion_id, 0), []).append(item)
    evidence_by_level.pop(0, None)
    verified_quotes = {item.quote for item in assessment.evidence if item.verified}
    rendered_html = template.render(
        assessment=assessment,
        css=css,
        document_title=_document_title(assessment),
        confidence_percent=round(assessment.verdict.confidence * 100, 1),
        ladder_chart=trl_ladder(assessment.levels, assessment.verdict.trl),
        evidence_by_level=evidence_by_level,
    )
    missing = [
        quote for quote in verified_quotes if html.escape(quote) not in rendered_html
    ]
    if missing:
        raise ValueError(f"report omitted verified evidence quotes: {missing[:3]}")
    return rendered_html


def build_report(
    assessment: Assessment | str | Path | dict[str, Any],
    output: str | Path,
) -> Path:
    """Write HTML and, when available, a PDF report; return the requested path."""
    output_path = Path(output)
    html = render_html(assessment)
    html_path = (
        output_path.with_suffix(".html")
        if output_path.suffix.lower() == ".pdf"
        else output_path
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    if output_path.suffix.lower() != ".pdf":
        return html_path
    try:
        from weasyprint import HTML
    except ImportError:
        logger.warning(
            "WeasyPrint is not installed; saved the self-contained HTML report to %s",
            html_path,
        )
        return html_path
    try:
        HTML(string=html, base_url=str(html_path.parent)).write_pdf(str(output_path))
    except Exception:
        logger.exception("Could not create PDF; the HTML report remains at %s", html_path)
        return html_path
    return output_path


def main() -> None:
    """Build a report from an assessment JSON file."""
    parser = argparse.ArgumentParser(description="Build a TRL-Scout report.")
    parser.add_argument("assessment_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build_report(args.assessment_json, args.output)
    print(result)


if __name__ == "__main__":
    main()
