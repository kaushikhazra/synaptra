"""Embedding service — lazy-loaded sentence-transformers. Vector search is DB-side."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings via sentence-transformers. Search is handled by SurrealDB."""

    DIMENSIONS = 384
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> None:
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)

    def warmup(self) -> None:
        """Eagerly load the embedding model. Call during service startup."""
        self._ensure_model()

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text. Returns float32 array of shape (384,)."""
        if self._model is None:
            log.warning(
                "EmbeddingService: model not pre-loaded — performance regression. "
                "Call warmup() at startup."
            )
        self._ensure_model()
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts. Returns (N, 384) array."""
        self._ensure_model()
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)
