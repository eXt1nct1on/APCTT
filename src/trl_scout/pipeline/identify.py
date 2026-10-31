"""Technology identification pipeline step."""

from __future__ import annotations

import argparse

from pydantic import BaseModel, Field

from trl_scout.ingest.loaders import load_document
from trl_scout.llm.client import LLMClient, load_prompt
from trl_scout.schemas import Document, TechProfile


class IdentificationResponse(BaseModel):
    """Raw R1 response before mapping into the public technology profile."""

    doc_type: str = Field(..., description="LLM document type classification.")
    doc_type_confidence: float = Field(..., ge=0, le=1, description="Classification confidence.")
    technology_name: str = Field(..., description="Technology name.")
    domain: str = Field(..., description="Technology domain.")
    one_line_summary: str = Field(..., description="Short technology summary.")
    intended_application: str = Field(..., description="Intended application.")
    key_components: list[str] = Field(..., description="Key technology components.")
    claimed_results: list[dict[str, str]] = Field(..., description="Claimed results.")
    author_stated_maturity: str | None = Field(..., description="Author-stated maturity.")
    search_keywords: list[str] = Field(..., description="Search keywords.")


def _section_text(document: Document, labels: set[str], limit: int) -> str:
    """Collect and truncate chunks belonging to selected section labels."""
    words: list[str] = []
    for chunk in document.chunks:
        if chunk.section.lower() in labels:
            words.extend(chunk.text.split())
            if len(words) >= limit:
                break
    return " ".join(words[:limit])


def _source_excerpt(document: Document) -> str:
    """Build the bounded R1 excerpt from relevant document sections."""
    title = str(document.meta.get("title") or document.filename)
    abstract = _section_text(document, {"abstract", "executive summary", "summary"}, 450)
    introduction = _section_text(document, {"introduction", "background"}, 1200)
    ending = _section_text(
        document,
        {"conclusion", "conclusions", "claims", "claim", "discussion", "future work"},
        600,
    )
    parts = [f"Title: {title}"]
    if abstract:
        parts.append(f"Abstract or summary:\n{abstract}")
    if introduction:
        parts.append(f"Introduction or background:\n{introduction}")
    if ending:
        parts.append(f"Conclusion or claims summary:\n{ending}")
    if len(parts) == 1:
        parts.append(_section_text(document, {"unknown"}, 1800))
    return "\n\n".join(parts)


def _heuristic_doc_type(document: Document) -> str:
    """Map ingestion document types to the R1 vocabulary."""
    return {"paper": "research_paper", "report": "rd_report"}.get(
        document.doc_type, document.doc_type
    )


def identify_technology(document: Document, llm: LLMClient) -> TechProfile:
    """Identify a technology from bounded, section-selected document text."""
    system, user = load_prompt(
        "identify.md",
        {
            "doc_type_hint": _heuristic_doc_type(document),
            "title": str(document.meta.get("title") or document.filename),
            "text": _source_excerpt(document),
        },
    )
    response = llm.complete_json(system, user, IdentificationResponse, max_tokens=1200)
    doc_type = (
        response.doc_type if response.doc_type_confidence >= 0.5 else _heuristic_doc_type(document)
    )
    return TechProfile(
        name=response.technology_name,
        domain=response.domain,
        doc_type=doc_type,
        summary=response.one_line_summary,
        intended_application=response.intended_application,
        key_components=response.key_components,
        claimed_results=response.claimed_results,
        author_stated_maturity=response.author_stated_maturity,
        keywords=response.search_keywords,
    )


def main() -> None:
    """Load a document and print its technology profile as JSON."""
    parser = argparse.ArgumentParser(description="Identify technology in a document.")
    parser.add_argument("source", help="Document path, URL, arXiv id, or raw text.")
    args = parser.parse_args()
    profile = identify_technology(load_document(args.source), LLMClient())
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
