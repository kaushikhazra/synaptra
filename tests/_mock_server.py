"""Test helper — patches the embedding model with a deterministic mock, then runs the HTTP MCP server.

Spawned as a subprocess by test_mcp_client.py. Listens on the port specified
by SYNAPTRA_PORT env var (default: 52199 for tests).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Force in-memory SurrealDB for tests (embedded, no external server needed)
os.environ["SYNAPTRA_DB"] = "mem://"
os.environ.setdefault("SYNAPTRA_PORT", "52199")
# Force the embedded surrealkv-file backend regardless of environment
os.environ["SYNAPTRA_BACKEND"] = "surrealkv-file"

import numpy as np

# Patch the embedding service BEFORE the server imports the engine
import synaptra.embeddings as emb_module


class _MockModel:
    """Deterministic mock that produces normalized 384-d vectors from text hash."""

    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.vstack([self._single(t) for t in text])
        return self._single(text)

    def _single(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec


def _patched_ensure(self):
    if self._model is None:
        self._model = _MockModel()


emb_module.EmbeddingService._ensure_model = _patched_ensure

# Now import and run the HTTP server
from synaptra.server import main

main()
