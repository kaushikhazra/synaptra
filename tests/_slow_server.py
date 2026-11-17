"""Test helper — the real HTTP server, with an embedding model that is slow to load (or fails).

Spawned as a subprocess by test_boot_warmup.py. It runs the real ``server.main()``, so what
is tested is the real order of events: bind the port, then warm the model in a thread.

Environment:
    WARM_SECONDS  how long the fake model takes to "load" (default 5)
    WARM_FAIL     if set, loading raises RuntimeError with this text instead
    SYNAPTRA_PORT, SYNAPTRA_WARMUP_WAIT_S   as in production
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# In-memory, embedded: no external SurrealDB, never the live store.
os.environ["SYNAPTRA_DB"] = "mem://"
os.environ["SYNAPTRA_BACKEND"] = "surrealkv-file"

import numpy as np  # noqa: E402

import synaptra.embeddings as emb_module  # noqa: E402


class _MockModel:
    """Deterministic normalised 384-d vectors from the text."""

    def embed(self, texts, **kwargs):
        """The fastembed model's interface: one vector per text."""
        for text in texts:
            yield self._single(text)

    def _single(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(384).astype(np.float32)
        return vec / np.linalg.norm(vec)


def _slow_ensure(self):
    if self._model is None:
        time.sleep(float(os.environ.get("WARM_SECONDS", "5")))
        failure = os.environ.get("WARM_FAIL")
        if failure:
            raise RuntimeError(failure)
        self._model = _MockModel()


emb_module.EmbeddingService._ensure_model = _slow_ensure

from synaptra.server import main  # noqa: E402

main()
