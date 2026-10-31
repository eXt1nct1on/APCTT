"""Document cleaning, sentence splitting, and chunking utilities."""

import re
from collections import Counter
from typing import Any

import spacy

from trl_scout.schemas import Chunk, Sentence


def clean_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise page lines and remove repeated headers and footers."""
    page_count = len(pages)
    candidates: list[str] = []
    for page in pages:
        nonempty = [str(line.get("text", "")).strip() for line in page.get("lines", [])]
        candidates.extend([line for line in nonempty[:2] + nonempty[-2:] if line])
    repeated = (
        {value for value, count in Counter(candidates).items() if count > page_count / 2}
        if page_count > 1
        else set()
    )
    cleaned: list[dict[str, Any]] = []
    for page in pages:
        lines = []
        for line in page.get("lines", []):
            text = str(line.get("text", ""))
            if text.strip() in repeated:
                continue
            text = re.sub(r"-\s*\n\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                lines.append({**line, "text": text})
        cleaned.append({**page, "lines": lines})
    return cleaned


def chunk_sections(blocks: list[dict[str, Any]]) -> list[Chunk]:
    """Split section blocks into approximately 250-350 word chunks."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    chunks: list[Chunk] = []
    sentence_number = 1
    chunk_number = 1
    for block in blocks:
        text = str(block["text"]).strip()
        if not text:
            continue
        doc = nlp(text)
        sentences = []
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            if sentence_text:
                sentences.append(Sentence(sid=f"s{sentence_number:04d}", text=sentence_text))
                sentence_number += 1
        start = 0
        while start < len(sentences):
            end = start
            words = 0
            while end < len(sentences) and (
                words < 250 or words + len(sentences[end].text.split()) <= 350
            ):
                words += len(sentences[end].text.split())
                end += 1
            selected = sentences[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"c{chunk_number:03d}",
                    page_start=int(block["page"]),
                    page_end=int(block.get("page_end", block["page"])),
                    section=str(block["section"]),
                    text=" ".join(sentence.text for sentence in selected),
                    sentences=selected,
                )
            )
            chunk_number += 1
            if end == len(sentences):
                break
            overlap_words = 0
            overlap_start = end
            while overlap_start > start and overlap_words < 40:
                overlap_start -= 1
                overlap_words += len(sentences[overlap_start].text.split())
            start = max(start + 1, overlap_start)
    return chunks
