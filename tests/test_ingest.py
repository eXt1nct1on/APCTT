"""Tests for document ingestion."""

from pathlib import Path

import pytest
from docx import Document as DocxDocument
from reportlab.pdfgen import canvas

from trl_scout.ingest.loaders import load_document


def _make_pdf(path: Path) -> None:
    """Create a two-page PDF with headings and repeated headers."""
    output = canvas.Canvas(str(path))
    for body in [
        "Abstract\nThis is a battery abstract with enough text to avoid OCR.",
        "Results\nMeasured capacity was 100 mAh per gram in the laboratory.",
    ]:
        output.setFont("Helvetica", 10)
        output.drawString(72, 760, "Repeated Header")
        output.setFont("Helvetica-Bold", 14)
        y = 700
        for line in body.splitlines():
            output.drawString(72, y, line)
            y -= 24
        output.setFont("Helvetica", 10)
        output.drawString(72, 40, "Repeated Footer")
        output.showPage()
    output.save()


def test_pdf_preserves_pages_and_skips_ocr(tmp_path: Path) -> None:
    """Text PDFs retain page numbers and do not invoke OCR."""
    path = tmp_path / "paper.pdf"
    _make_pdf(path)
    document = load_document(path)
    assert document.n_pages == 2
    assert document.meta["ocr_pages"] == []
    assert {chunk.page_start for chunk in document.chunks} == {1, 2}
    assert {chunk.section for chunk in document.chunks} == {"abstract", "results"}


def test_txt_and_docx_load(tmp_path: Path) -> None:
    """Plain text and DOCX inputs produce searchable chunks."""
    txt = tmp_path / "note.txt"
    txt.write_text("Introduction\nA short text document about a test device.", encoding="utf-8")
    assert load_document(txt).chunks[0].section == "introduction"

    docx_path = tmp_path / "note.docx"
    docx = DocxDocument()
    docx.add_paragraph("Methods")
    docx.add_paragraph("The test device was measured in a laboratory.")
    docx.save(docx_path)
    assert load_document(docx_path).chunks[0].section == "methods"


def test_patent_heuristic() -> None:
    """Patent language is classified as a patent."""
    document = load_document("Claims: What is claimed is a battery separator.")
    assert document.doc_type == "patent"


def test_ssrf_guard_rejects_loopback() -> None:
    """Loopback URLs are rejected before any request is made."""
    with pytest.raises(ValueError, match="private or reserved"):
        load_document("http://127.0.0.1/test.pdf")
