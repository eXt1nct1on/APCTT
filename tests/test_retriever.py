"""Tests for hybrid document retrieval."""

from pathlib import Path

import numpy as np

from trl_scout.index.retriever import DocIndex
from trl_scout.schemas import Chunk, Document, Sentence


class FakeEncoder:
    """Small deterministic encoder for offline retrieval tests."""

    calls = 0

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        """Encode based on terms relevant to the synthetic document."""
        type(self).calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("field trial" in lowered or "pilot plant" in lowered),
                    float("background" in lowered or "theory" in lowered),
                    float("references" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _document() -> Document:
    """Create a small document with operational and background passages."""
    texts = [
        ("background", "Background theory and prior battery research."),
        ("results", "The field trial operated a pilot plant with real users."),
        ("references", "References describe earlier field trial reports."),
    ]
    chunks = [
        Chunk(
            chunk_id=f"c{index:03d}",
            page_start=index,
            page_end=index,
            section=section,
            text=text,
            sentences=[Sentence(sid=f"s{index:04d}", text=text)],
        )
        for index, (section, text) in enumerate(texts, 1)
    ]
    return Document(
        doc_id="synthetic",
        filename="synthetic.txt",
        doc_type="paper",
        n_pages=3,
        sha256="synthetic-sha",
        chunks=chunks,
    )


def test_field_trial_ranks_first_and_deduplicates(tmp_path: Path) -> None:
    """Operational evidence is ranked first and level retrieval merges duplicates."""
    index = DocIndex(_document(), model=FakeEncoder(), cache_dir=tmp_path)
    results = index.search("TRL 7 field trial pilot plant operational")
    assert results[0][0].chunk_id == "c002"
    level_results = index.retrieve_for_level(7, k=8)
    assert len({chunk.chunk_id for chunk, _, _ in level_results}) == len(level_results)
    assert any("T7.1" in criterion_ids for _, _, criterion_ids in level_results)


def test_embedding_cache_is_reused(tmp_path: Path) -> None:
    """A second index build loads the document matrix without encoding it again."""
    FakeEncoder.calls = 0
    DocIndex(_document(), model=FakeEncoder(), cache_dir=tmp_path)
    first_calls = FakeEncoder.calls
    DocIndex(_document(), model=FakeEncoder(), cache_dir=tmp_path)
    assert first_calls == 1
    assert FakeEncoder.calls == 1
