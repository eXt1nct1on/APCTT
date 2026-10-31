"""Document section extraction utilities."""

import re
from collections.abc import Iterable
from typing import Any

PAPER_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "experimental",
    "results",
    "discussion",
    "conclusion",
    "future work",
    "references",
}
PATENT_SECTIONS = {
    "abstract",
    "background",
    "summary",
    "brief description of drawings",
    "detailed description",
    "examples",
    "claims",
}
REPORT_SECTIONS = {
    "executive summary",
    "objectives",
    "milestones",
    "work packages",
    "deliverables",
    "test results",
    "references",
}


def detect_doc_type(text: str) -> str:
    """Classify a document using conservative patent, report, or paper heuristics."""
    lowered = text.lower()
    if (
        "what is claimed" in lowered
        or re.search(r"\bclaims?\s*:", lowered)
        or re.search(r"\b(?:patent|publication)\s*(?:no\.?|number)\s*[a-z0-9-]+", lowered)
    ):
        return "patent"
    if any(term in lowered for term in ("executive summary", "work packages", "deliverables")):
        return "report"
    return "paper"


def _normalise_heading(value: str) -> str:
    """Normalise a heading for matching while retaining readable section names."""
    value = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)[.)]\s*", "", value)
    value = re.sub(r"[^a-zA-Z ]", " ", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def _section_names(doc_type: str) -> set[str]:
    """Return recognised section names for a document type."""
    if doc_type == "patent":
        return PATENT_SECTIONS
    if doc_type == "report":
        return REPORT_SECTIONS
    return PAPER_SECTIONS


def detect_sections(
    pages: Iterable[dict[str, Any]], doc_type: str = "paper"
) -> list[dict[str, Any]]:
    """Split page lines into section-labelled text blocks."""
    names = _section_names(doc_type)
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] = {
        "section": "unknown",
        "page": 1,
        "page_end": 1,
        "lines": [],
    }

    def flush() -> None:
        if current["lines"]:
            blocks.append(
                {
                    "section": current["section"],
                    "page": current["page"],
                    "page_end": current["page_end"],
                    "text": "\n".join(line["text"] for line in current["lines"]).strip(),
                }
            )

    for page in pages:
        lines = page.get("lines", [])
        font_sizes = [float(line.get("font_size", 0)) for line in lines if line.get("text")]
        median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 0
        for line in lines:
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            normalised = _normalise_heading(text)
            is_heading = normalised in names or (
                len(text.split()) <= 12
                and float(line.get("font_size", 0)) >= median_size + 1.5
                and bool(re.search(r"[A-Za-z]", text))
            )
            if is_heading:
                flush()
                current = {
                    "section": normalised or "unknown",
                    "page": page["page"],
                    "page_end": page["page"],
                    "lines": [],
                }
            else:
                current["lines"].append({"text": text, "font_size": line.get("font_size", 0)})
            if current["lines"]:
                current["page_end"] = page["page"]
    flush()
    return blocks
