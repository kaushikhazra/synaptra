"""Tests for EmbeddingService — warmup, idempotency, regression guard."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cognitive_memory.embeddings import EmbeddingService


class TestWarmupIdempotency:
    """warmup() must load the model exactly once regardless of how many times it's called."""

    def test_warmup_loads_model(self):
        svc = EmbeddingService()
        assert svc._model is None
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = MagicMock()
            svc.warmup()
        assert svc._model is not None

    def test_warmup_idempotent_no_reload(self):
        """Calling warmup() twice must not replace the already-loaded model."""
        svc = EmbeddingService()
        # Directly inject a sentinel model — bypasses the real ST import entirely
        sentinel = MagicMock(name="sentinel-model")
        svc._model = sentinel
        # Both calls must see the sentinel and leave it untouched
        svc.warmup()
        svc.warmup()
        assert svc._model is sentinel, "warmup() must not replace an already-loaded model"

    def test_model_loaded_property(self):
        svc = EmbeddingService()
        assert svc.model_loaded is False
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = MagicMock()
            svc.warmup()
        assert svc.model_loaded is True


class TestRegressionGuard:
    """embed() must emit a warning when model was not pre-loaded via warmup()."""

    def test_embed_warns_when_not_preloaded(self, caplog):
        svc = EmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            with caplog.at_level(logging.WARNING, logger="cognitive_memory.embeddings"):
                svc.embed("test text")

        assert any("not pre-loaded" in r.message for r in caplog.records), (
            "Expected regression-guard warning not emitted"
        )

    def test_embed_no_warn_when_preloaded(self, caplog):
        """After warmup(), embed() must not emit the regression-guard warning."""
        svc = EmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            svc.warmup()
            with caplog.at_level(logging.WARNING, logger="cognitive_memory.embeddings"):
                svc.embed("test text")

        regression_warnings = [
            r for r in caplog.records if "not pre-loaded" in r.message
        ]
        assert len(regression_warnings) == 0, (
            "Unexpected regression-guard warning emitted after warmup()"
        )
