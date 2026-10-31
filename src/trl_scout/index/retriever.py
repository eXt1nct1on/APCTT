"""Hybrid BM25 and embedding retrieval over ingested document chunks."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from trl_scout.config import settings
from trl_scout.rubric import Level, load_rubric
from trl_scout.schemas import Chunk, Document

logger = logging.getLogger("trl_scout.index")


class Encoder(Protocol):
    """Minimal sentence-transformers-compatible encoder."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """Encode texts into vectors."""


def _tokens(text: str) -> list[str]:
    """Tokenize text consistently for BM25 and query matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


class DocIndex:
    """Search index combining lexical BM25 and semantic cosine retrieval."""

    def __init__(
        self,
        document: Document,
        model: Encoder | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Build or restore indexes for a document."""
        self.document = document
        self.chunks = document.chunks
        self._model = model
        self._model_name = settings.embed_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.cache_dir = Path(cache_dir or settings.data_dir) / "embeddings"
        self.cache_path = self.cache_dir / (
            f"{document.sha256}-{hashlib.sha256(self._model_name.encode()).hexdigest()[:12]}.npy"
        )
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is required for retrieval; install project dependencies first"
            ) from exc
        self._bm25 = BM25Okapi([_tokens(chunk.text) for chunk in self.chunks])
        self._embeddings = self._load_embeddings()

    def _load_model(self) -> Encoder:
        """Load the configured sentence-transformers model on CPU."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic retrieval; "
                "install project dependencies or pass an encoder for tests"
            ) from exc
        try:
            self._model = SentenceTransformer(self._model_name, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"could not load embedding model '{self._model_name}' offline; "
                "download it first or configure EMBED_MODEL"
            ) from exc
        return self._model

    def _load_embeddings(self) -> np.ndarray:
        """Load document embeddings from cache or encode and save them."""
        if self.cache_path.exists():
            logger.info("Loading embedding cache for %s", self.document.sha256)
            return np.load(self.cache_path)
        model = self._load_model()
        try:
            values = model.encode(
                [chunk.text for chunk in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError("embedding model failed to encode document chunks") from exc
        embeddings = np.asarray(values, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.chunks):
            raise RuntimeError("embedding model returned an invalid matrix shape")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, embeddings)
        return embeddings

    @staticmethod
    def _cosine_scores(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine scores for normalized or unnormalized vectors."""
        matrix_norm = np.linalg.norm(embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        denominator = np.maximum(matrix_norm * query_norm, np.finfo(float).eps)
        return (embeddings @ query_embedding) / denominator

    @staticmethod
    def _is_achieved_query(query: str) -> bool:
        """Detect queries seeking demonstrated rather than planned work."""
        return bool(
            re.search(
                r"\b(achieved|demonstrat(?:ed|ion)|measur(?:ed|ement)|tested|trial|"
                r"operational|validated|results?|performance)\b",
                query.lower(),
            )
        )

    def search(
        self,
        query: str,
        k: int = 8,
        sections: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return top chunks using reciprocal rank fusion."""
        if not query.strip() or not self.chunks or k <= 0:
            return []
        allowed = {section.lower() for section in sections} if sections else None
        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            if allowed is None or chunk.section.lower() in allowed
        ]
        if not candidates:
            return []
        bm25_scores = np.asarray(self._bm25.get_scores(_tokens(query)))
        query_embedding = np.asarray(
            self._load_model().encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )[0]
        cosine_scores = self._cosine_scores(self._embeddings, query_embedding)
        bm25_rank = sorted(candidates, key=lambda index: bm25_scores[index], reverse=True)
        cosine_rank = sorted(candidates, key=lambda index: cosine_scores[index], reverse=True)
        bm25_positions = {index: rank for rank, index in enumerate(bm25_rank)}
        cosine_positions = {index: rank for rank, index in enumerate(cosine_rank)}
        achieved_query = self._is_achieved_query(query)
        scored: list[tuple[int, float]] = []
        for index in candidates:
            score = 1 / (60 + bm25_positions[index]) + 1 / (60 + cosine_positions[index])
            section = self.chunks[index].section.lower()
            if section in {"results", "experimental", "test", "validation"}:
                score += 0.002
            if achieved_query and section in {"background", "references"}:
                score -= 0.002
            scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[index], score) for index, score in scored[:k]]

    def retrieve_for_level(
        self,
        level: int | Level,
        k: int = 8,
    ) -> list[tuple[Chunk, float, list[str]]]:
        """Retrieve and merge top chunks for every criterion in a TRL level."""
        selected = (
            level
            if isinstance(level, Level)
            else next(item for item in load_rubric().levels if item.level == level)
        )
        merged: dict[str, tuple[Chunk, float, list[str]]] = {}
        for criterion in selected.criteria:
            query = f"{criterion.text} {' '.join(criterion.hints)}"
            for chunk, score in self.search(query, k=k):
                if chunk.chunk_id not in merged:
                    merged[chunk.chunk_id] = (chunk, score, [criterion.id])
                else:
                    previous_chunk, previous_score, criterion_ids = merged[chunk.chunk_id]
                    merged[chunk.chunk_id] = (
                        previous_chunk,
                        max(previous_score, score),
                        sorted(set(criterion_ids + [criterion.id])),
                    )
        return sorted(merged.values(), key=lambda item: item[1], reverse=True)[:k]
