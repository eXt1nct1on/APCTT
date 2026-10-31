"""Verification helpers for evidence quotes and author-stated signals."""

from __future__ import annotations

import re
from collections.abc import Iterable

from rapidfuzz import fuzz

from trl_scout.schemas import Chunk

_QUOTE_TRANSLATION = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _normalise_with_spans(text: str) -> tuple[str, list[int]]:
    """Normalise searchable text while retaining original character offsets."""
    text = text.translate(_QUOTE_TRANSLATION)
    output: list[str] = []
    offsets: list[int] = []
    pending_space = False
    index = 0
    while index < len(text):
        character = text[index]
        if character == "-" and index + 1 < len(text) and text[index + 1] == "\n":
            output.append(" ")
            offsets.append(index)
            index += 2
            continue
        if character.isspace():
            pending_space = bool(output)
            index += 1
            continue
        if pending_space:
            output.append(" ")
            offsets.append(index - 1)
            pending_space = False
        output.append(character.lower())
        offsets.append(index)
        index += 1
    return "".join(output).strip(), offsets


def _normalise(text: str) -> str:
    """Normalise text for quote matching."""
    return _normalise_with_spans(text)[0]


def verify_quote(quote: str, chunk_text: str) -> tuple[bool, float]:
    """Verify a quote with exact normalized and fuzzy matching."""
    candidate = _normalise(quote)
    source = _normalise(chunk_text)
    if not candidate or not source:
        return False, 0.0
    if candidate in source:
        return True, 1.0
    score = fuzz.partial_ratio(candidate, source) / 100
    return score >= 0.9, score


def quote_span(quote: str, chunk_text: str) -> tuple[bool, float, int, int]:
    """Return verification, score, and the exact original span for a quote."""
    candidate = _normalise(quote)
    source, offsets = _normalise_with_spans(chunk_text)
    if not candidate or not source:
        return False, 0.0, 0, 0
    start = source.find(candidate)
    if start >= 0:
        end = start + len(candidate)
        return True, 1.0, offsets[start], offsets[end - 1] + 1
    window_length = max(1, len(candidate))
    best_score = 0.0
    best_start = 0
    for window_start in range(max(1, len(source) - window_length + 1)):
        window = source[window_start : window_start + window_length]
        score = fuzz.partial_ratio(candidate, window) / 100
        if score > best_score:
            best_score = score
            best_start = window_start
    best_end = min(len(source), best_start + window_length)
    if best_score < 0.9:
        return False, best_score, 0, 0
    return True, best_score, offsets[best_start], offsets[best_end - 1] + 1


_TRL_MENTION = re.compile(
    r"\b(?:TRL[-\s]?([1-9])(?:\s*(?:to|-)\s*([1-9]))?|"
    r"technology readiness level\s*([1-9]))\b",
    re.IGNORECASE,
)


def explicit_trl_mentions(chunks: Iterable[Chunk]) -> list[str]:
    """Extract explicit TRL mentions and ranges as author-stated signals."""
    mentions: list[str] = []
    for chunk in chunks:
        for match in _TRL_MENTION.finditer(chunk.text):
            value = match.group(0)
            if value not in mentions:
                mentions.append(value)
    return mentions


def planned_items(chunks: Iterable[Chunk]) -> list[str]:
    """Extract future or planning sentences from relevant sections."""
    cues = re.compile(r"\b(?:will|plan(?:s|ned)?\s+to|future work|ongoing|next step)\b", re.I)
    sections = {"conclusion", "conclusions", "future work", "milestones", "milestone"}
    items: list[str] = []
    for chunk in chunks:
        if chunk.section.lower() not in sections:
            continue
        for sentence in chunk.sentences:
            text = sentence.text.strip()
            if cues.search(text) and text not in items:
                items.append(text)
    return items
