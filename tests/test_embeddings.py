"""Tests for EmbeddingService — warmup, idempotency, regression guard, the fastembed backend's contract."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np

import importlib.util

import synaptra.embeddings as _embeddings

# tests/test_client.py replaces EmbeddingService._ensure_model for the whole pytest session when it
# is imported. These tests are about the real loader, so they use a private copy of the module.
_spec = importlib.util.spec_from_file_location(
    "cm_embeddings_pristine", _embeddings.__file__
)
_pristine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pristine)
EmbeddingService, PROVIDERS, model_dir = (
    _pristine.EmbeddingService,
    _pristine.PROVIDERS,
    _pristine.model_dir,
)


def fake_model(vectors=None):
    """A stand-in for fastembed's TextEmbedding: embed(list of texts) yields one vector per text."""
    model = MagicMock(name="TextEmbedding")
    model.embed.side_effect = lambda texts, **kw: iter(
        vectors
        if vectors is not None
        else [np.ones(384, dtype=np.float32) for _ in texts]
    )
    return model


class TestWarmupIdempotency:
    """warmup() must load the model exactly once regardless of how many times it's called."""

    def test_warmup_loads_model(self):
        svc = EmbeddingService()
        assert svc._model is None
        with patch("fastembed.TextEmbedding") as TE:
            TE.return_value = MagicMock()
            svc.warmup()
        assert svc._model is not None

    def test_warmup_idempotent_no_reload(self):
        """Calling warmup() twice must not replace the already-loaded model."""
        svc = EmbeddingService()
        # Directly inject a sentinel model — bypasses the real load entirely
        sentinel = MagicMock(name="sentinel-model")
        svc._model = sentinel
        # Both calls must see the sentinel and leave it untouched
        svc.warmup()
        svc.warmup()
        assert svc._model is sentinel, (
            "warmup() must not replace an already-loaded model"
        )

    def test_model_loaded_property(self):
        svc = EmbeddingService()
        assert svc.model_loaded is False
        with patch("fastembed.TextEmbedding") as TE:
            TE.return_value = MagicMock()
            svc.warmup()
        assert svc.model_loaded is True


class TestRegressionGuard:
    """embed() must emit a warning when model was not pre-loaded via warmup()."""

    def test_embed_warns_when_not_preloaded(self, caplog):
        svc = EmbeddingService()
        with patch("fastembed.TextEmbedding", return_value=fake_model()):
            with caplog.at_level(logging.WARNING, logger="synaptra.embeddings"):
                svc.embed("test text")

        assert any("not pre-loaded" in r.message for r in caplog.records), (
            "Expected regression-guard warning not emitted"
        )

    def test_embed_no_warn_when_preloaded(self, caplog):
        """After warmup(), embed() must not emit the regression-guard warning."""
        svc = EmbeddingService()
        with patch("fastembed.TextEmbedding", return_value=fake_model()):
            svc.warmup()
            with caplog.at_level(logging.WARNING, logger="synaptra.embeddings"):
                svc.embed("test text")

        regression_warnings = [
            r for r in caplog.records if "not pre-loaded" in r.message
        ]
        assert len(regression_warnings) == 0, (
            "Unexpected regression-guard warning emitted after warmup()"
        )


class TestBackendContract:
    """What the rest of the engine relies on, whatever runs underneath."""

    def test_the_model_is_loaded_on_the_cpu_from_the_pinned_cache_dir(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("SYNAPTRA_MODEL_DIR", str(tmp_path / "models"))
        svc = EmbeddingService()
        with patch("fastembed.TextEmbedding") as TE:
            TE.return_value = MagicMock()
            svc.warmup()
        args, kwargs = TE.call_args
        assert args == (EmbeddingService.MODEL_NAME,)
        assert kwargs["providers"] == PROVIDERS == ["CPUExecutionProvider"]
        assert kwargs["cache_dir"] == str(tmp_path / "models")
        assert kwargs["local_files_only"] is True  # cached: never waits on the network
        assert (tmp_path / "models").is_dir()

    def test_a_model_not_cached_yet_is_downloaded_in_the_load(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("SYNAPTRA_MODEL_DIR", str(tmp_path))
        calls = []

        def ctor(name, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise ValueError("not in the cache")
            return MagicMock()

        svc = EmbeddingService()
        with patch("fastembed.TextEmbedding", side_effect=ctor):
            svc.warmup()
        assert [bool(c.get("local_files_only")) for c in calls] == [True, False]
        assert svc.state == "ready"

    def test_embed_returns_one_unit_length_float32_vector(self):
        svc = EmbeddingService()
        raw = np.full(384, 3.0, dtype=np.float64)  # not unit length
        with patch("fastembed.TextEmbedding", return_value=fake_model([raw])):
            vec = svc.embed("anything")
        assert vec.shape == (384,) and vec.dtype == np.float32
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_embed_batch_returns_a_row_per_text_and_handles_none(self):
        svc = EmbeddingService()
        with patch("fastembed.TextEmbedding", return_value=fake_model()):
            out = svc.embed_batch(["a", "b", "c"])
            assert out.shape == (3, 384) and out.dtype == np.float32
            assert svc.embed_batch([]).shape == (0, 384)

    def test_the_model_dir_is_resolved_at_call_time(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SYNAPTRA_MODEL_DIR", raising=False)
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "one"))
        monkeypatch.setenv("HOME", str(tmp_path / "one"))
        first = model_dir()
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "two"))
        monkeypatch.setenv("HOME", str(tmp_path / "two"))
        second = model_dir()
        assert first == tmp_path / "one" / ".synaptra" / "models"
        assert second == tmp_path / "two" / ".synaptra" / "models"
        monkeypatch.setenv("SYNAPTRA_MODEL_DIR", str(tmp_path / "given"))
        assert model_dir() == tmp_path / "given"  # the environment wins
