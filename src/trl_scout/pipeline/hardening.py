"""Input safety and bounded-analysis helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from trl_scout.schemas import Chunk, Document, Sentence

_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(?:all\s+)?prior\s+instructions\b",
    r"\boutput\s+trl\s+[1-9]\b",
    r"\bsystem\s+message\b",
    r"\bdeveloper\s+message\b",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_KEY_SECTIONS = {
    "abstract",
    "executive summary",
    "results",
    "test results",
    "experimental",
    "methods",
    "conclusion",
    "claims",
    "examples",
}
_LANGUAGE_MARKERS = {
    "es": (" el ", " la ", " una ", " tecnología ", " fue ", " para "),
    "fr": (" le ", " la ", " une ", " technologie ", " pour ", " avec "),
    "de": (" der ", " die ", " das ", " eine ", " technologie ", " für "),
}


def detect_prompt_injection(chunks: Iterable[Chunk]) -> bool:
    """Detect obvious instruction-like phrases in untrusted source text."""
    return any(_INJECTION_RE.search(chunk.text) for chunk in chunks)


def redact_prompt_injection(document: Document) -> Document:
    """Replace obvious instruction-like spans before sending text to an LLM."""
    chunks: list[Chunk] = []
    for chunk in document.chunks:
        text = _INJECTION_RE.sub("[REDACTED PROMPT-LIKE TEXT]", chunk.text)
        sentences = [
            Sentence(
                sid=sentence.sid,
                text=_INJECTION_RE.sub("[REDACTED PROMPT-LIKE TEXT]", sentence.text),
            )
            for sentence in chunk.sentences
        ]
        chunks.append(chunk.model_copy(update={"text": text, "sentences": sentences}))
    return document.model_copy(update={"chunks": chunks})


def detect_language(text: str) -> str:
    """Return a lightweight language code suitable for a reliability warning."""
    lowered = f" {text.lower()} "
    scores = {
        language: sum(lowered.count(marker) for marker in markers)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    language, score = max(scores.items(), key=lambda item: item[1], default=("en", 0))
    return language if score >= 2 else "en"


def bound_document(document: Document, max_words: int = 12000) -> tuple[Document, bool]:
    """Keep key sections and a representative sample within a word budget."""
    total_words = sum(len(chunk.text.split()) for chunk in document.chunks)
    if total_words <= max_words:
        return document, False

    key = [chunk for chunk in document.chunks if chunk.section.lower() in _KEY_SECTIONS]
    key_ids = {chunk.chunk_id for chunk in key}
    remaining = [chunk for chunk in document.chunks if chunk.chunk_id not in key_ids]
    selected: list[Chunk] = []
    words = 0
    for chunk in key:
        chunk_words = len(chunk.text.split())
        if words + chunk_words <= max_words:
            selected.append(chunk)
            words += chunk_words
    if remaining and words < max_words:
        slots = max(1, max_words - words)
        step = max(1, len(remaining) // max(1, slots // 250))
        for chunk in remaining[::step]:
            chunk_words = len(chunk.text.split())
            if words + chunk_words > max_words:
                break
            selected.append(chunk)
            words += chunk_words
    selected.sort(key=lambda chunk: document.chunks.index(chunk))
    meta = {
        **document.meta,
        "analysis_original_chunk_count": len(document.chunks),
        "analysis_word_count": words,
    }
    return document.model_copy(update={"chunks": selected, "meta": meta}), True
