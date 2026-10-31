"""Document loading utilities for local files, URLs, arXiv, and raw text."""

from __future__ import annotations

import hashlib
import io
import ipaddress
import logging
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fitz
import pdfplumber
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from trl_scout.config import settings
from trl_scout.ingest.chunker import chunk_sections, clean_pages
from trl_scout.ingest.sections import detect_doc_type, detect_sections
from trl_scout.pipeline.hardening import detect_language
from trl_scout.schemas import Document

logger = logging.getLogger("trl_scout.ingest")


def _safe_url(url: str) -> None:
    """Reject non-HTTP schemes and hosts resolving to private address ranges."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only HTTP(S) URLs are allowed")
    addresses = socket.getaddrinfo(parsed.hostname, None)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("URL host resolves to a private or reserved IP address")


def _normalise_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply shared cleaning to extracted pages."""
    return clean_pages(pages)


def _pdf_pages(content: bytes) -> tuple[list[dict[str, Any]], list[int], list[float]]:
    """Extract PDF pages, font sizes, OCR fallbacks, and table blocks."""
    pdf = fitz.open(stream=content, filetype="pdf")
    pages: list[dict[str, Any]] = []
    ocr_pages: list[int] = []
    ocr_confidences: list[float] = []
    for page_number, page in enumerate(pdf, 1):
        lines: list[dict[str, Any]] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if text:
                    lines.append(
                        {
                            "text": text,
                            "font_size": max(
                                (float(span.get("size", 0)) for span in spans), default=0
                            ),
                        }
                    )
        if len(" ".join(line["text"] for line in lines)) < 50:
            try:
                import pytesseract
                from PIL import Image

                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidence_values = [
                    float(value)
                    for value, word in zip(data.get("conf", []), data.get("text", []), strict=False)
                    if word.strip() and float(value) >= 0
                ]
                confidence = (
                    sum(confidence_values) / len(confidence_values) / 100
                    if confidence_values
                    else 0.0
                )
                text = pytesseract.image_to_string(image).strip()
                if text:
                    lines = [{"text": text, "font_size": 0}]
                    ocr_pages.append(page_number)
                    ocr_confidences.append(confidence)
            except (ImportError, OSError, RuntimeError) as exc:
                logger.warning("OCR unavailable for PDF page %s: %s", page_number, exc)
        pages.append({"page": page_number, "lines": lines})
    try:
        with pdfplumber.open(__import__("io").BytesIO(content)) as plumber:
            for page_number, page in enumerate(plumber.pages, 1):
                tables = page.extract_tables()
                for table in tables:
                    rows = [" | ".join(cell or "" for cell in row) for row in table]
                    if rows:
                        pages[page_number - 1]["lines"].append(
                            {
                                "text": f"[TABLE page {page_number}] " + " ".join(rows),
                                "font_size": 0,
                            }
                        )
    except (OSError, ValueError) as exc:
        logger.warning("Could not extract PDF tables: %s", exc)
    return pages, ocr_pages, ocr_confidences


def _docx_pages(content: bytes) -> list[dict[str, Any]]:
    """Extract paragraphs from a DOCX as one logical page."""
    import io

    document = DocxDocument(io.BytesIO(content))
    lines = [
        {"text": paragraph.text, "font_size": 0}
        for paragraph in document.paragraphs
        if paragraph.text
    ]
    return [{"page": 1, "lines": lines}]


def _text_pages(text: str) -> list[dict[str, Any]]:
    """Represent plain text as one logical page."""
    return [{"page": 1, "lines": [{"text": line, "font_size": 0} for line in text.splitlines()]}]


def _fetch_url(url: str) -> tuple[bytes, str]:
    """Fetch a remote document with SSRF and size protections."""
    _safe_url(url)
    response = requests.get(url, timeout=15, stream=True)
    response.raise_for_status()
    content_length = int(response.headers.get("content-length", 0))
    limit = settings.max_upload_mb * 1024 * 1024
    if content_length > limit:
        raise ValueError("remote document exceeds MAX_UPLOAD_MB")
    content = response.content
    if len(content) > limit:
        raise ValueError("remote document exceeds MAX_UPLOAD_MB")
    return content, response.headers.get("content-type", "").lower()


def _source_bytes(source: str | Path) -> tuple[bytes, str, str]:
    """Resolve a source to bytes, a type hint, and a display filename."""
    source_string = str(source)
    path = Path(source_string)
    if path.exists():
        content = path.read_bytes()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise ValueError("local document exceeds MAX_UPLOAD_MB")
        return content, path.suffix.lower(), path.name
    if re.match(r"^https?://", source_string, re.IGNORECASE):
        content, content_type = _fetch_url(source_string)
        suffix = (
            ".pdf" if "pdf" in content_type or source_string.lower().endswith(".pdf") else ".html"
        )
        return content, suffix, Path(urlparse(source_string).path).name or "remote-document"
    if (
        source_string.lower().startswith("arxiv:")
        or re.match(r"^[\w.-]+/\w[\w.-]+$", source_string)
        or re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", source_string)
    ):
        arxiv_id = source_string.removeprefix("arxiv:")
        return _fetch_url(f"https://export.arxiv.org/pdf/{arxiv_id}")[0], ".pdf", f"{arxiv_id}.pdf"
    return source_string.encode("utf-8"), ".txt", "raw-text"


def load_document(source: str | Path) -> Document:
    """Load and chunk a document from a path, URL, arXiv identifier, or raw text."""
    content, suffix, filename = _source_bytes(source)
    if suffix == ".pdf":
        pages, ocr_pages, ocr_confidences = _pdf_pages(content)
    elif suffix == ".docx":
        pages, ocr_pages, ocr_confidences = _docx_pages(content), [], []
    elif suffix in {".txt", ".md"}:
        pages, ocr_pages, ocr_confidences = (
            _text_pages(content.decode("utf-8", errors="replace")),
            [],
            [],
        )
    elif suffix == ".html":
        soup = BeautifulSoup(content, "html.parser")
        pages, ocr_pages, ocr_confidences = _text_pages(soup.get_text("\n")), [], []
    else:
        raise ValueError(f"unsupported document type: {suffix}")
    pages = _normalise_pages(pages)
    all_text = "\n".join(line["text"] for page in pages for line in page["lines"])
    doc_type = detect_doc_type(all_text)
    blocks = detect_sections(pages, doc_type)
    chunks = chunk_sections(blocks)
    meta = {
        "pages": pages,
        "ocr_pages": ocr_pages,
        "ocr_confidence": (
            sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
        ),
        "language": detect_language(all_text),
        "sections": [{"section": block["section"], "page": block["page"]} for block in blocks],
    }
    return Document(
        doc_id=hashlib.sha256(content).hexdigest()[:16],
        filename=filename,
        doc_type=doc_type,
        n_pages=len(pages),
        sha256=hashlib.sha256(content).hexdigest(),
        meta=meta,
        chunks=chunks,
    )
