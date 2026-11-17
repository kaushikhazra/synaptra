"""Embedding service — fastembed (ONNX, CPU), lazy-loaded. Vector search is DB-side.

The model is all-MiniLM-L6-v2, the same weights the store's vectors were made with (a
sentence-transformers run of it; ``tests/data/golden_embeddings.json`` pins the match),
run through ONNX Runtime on the CPU. That avoids importing torch and transformers, which
took minutes on a cold start (issue #13) and held the GPU.

The model can still take a while to load, so the server binds its port first and warms the
model in a thread. The service therefore has a state, and callers that need a vector either
wait for it (``wait_ready``, bounded) or are told plainly that it is still warming up. They
are never handed a silent fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

WAIT_ENV = "SYNAPTRA_WARMUP_WAIT_S"
DEFAULT_WAIT_S = 60.0
POLL_S = 0.05

MODEL_DIR_ENV = "SYNAPTRA_MODEL_DIR"
PROVIDERS = ["CPUExecutionProvider"]  # never the GPU: the server must not hold it

COLD, WARMING, READY, FAILED = "cold", "warming", "ready", "failed"


class EmbeddingsNotReady(RuntimeError):
    """The embedding model cannot produce a vector right now."""


class EmbeddingsWarming(EmbeddingsNotReady):
    """The model is still loading; the caller can retry shortly."""


class EmbeddingsFailed(EmbeddingsNotReady):
    """The model could not be loaded; retrying will not help until that is fixed."""


def wait_bound() -> float:
    """How long an embedding call waits for the model: the environment, else 60 s."""
    raw = os.environ.get(WAIT_ENV)
    try:
        return DEFAULT_WAIT_S if raw is None else max(float(raw), 0.0)
    except ValueError:
        log.warning(
            "%s=%r is not a number; using %.0f s", WAIT_ENV, raw, DEFAULT_WAIT_S
        )
        return DEFAULT_WAIT_S


def model_dir() -> Path:
    """Where the model files live: the environment, else under the user's home.

    Resolved on every call, never frozen at import or install time, so moving or renaming
    the home or the install does not leave a stale path behind. fastembed's own default is
    the system temp dir, which the OS may clear and which would force a download at logon.
    """
    given = os.environ.get(MODEL_DIR_ENV)
    return Path(given) if given else Path.home() / ".synaptra" / "models"


class EmbeddingService:
    """Generates embeddings via fastembed. Search is handled by SurrealDB."""

    DIMENSIONS = 384
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()  # guards the state fields
        self._load_lock = threading.Lock()  # one model load at a time, whoever asks
        self._state = COLD
        self._cause: str | None = None
        self._started: float | None = None
        self._finished: float | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> str:
        # A model loaded lazily (a CLI call, a test) never passed through warm-up.
        if self._state == COLD and self._model is not None:
            return READY
        return self._state

    def status(self) -> dict:
        """What `memory_health` reports: state, seconds spent loading, and why it failed."""
        out: dict = {"state": self.state}
        if self._started is not None:
            end = self._finished if self._finished is not None else time.monotonic()
            out["elapsed_s"] = round(end - self._started, 1)
        if self._cause:
            out["cause"] = self._cause
        return out

    def _not_ready(self) -> EmbeddingsNotReady | None:
        if self._state == WARMING:
            elapsed = time.monotonic() - (self._started or time.monotonic())
            return EmbeddingsWarming(
                f"warming up: the embedding model is still loading ({elapsed:.0f} s so far); retry shortly"
            )
        if self._state == FAILED:
            return EmbeddingsFailed(f"embedding model unavailable: {self._cause}")
        return None

    def _load(self) -> None:
        """Load the model, recording the outcome in the state. Runs in a thread or inline."""
        try:
            self._ensure_model()
        except Exception as exc:  # noqa: BLE001 - any load failure is reported, not raised, in a thread
            with self._lock:
                self._state = FAILED
                self._cause = f"{type(exc).__name__}: {exc}"
                self._finished = time.monotonic()
            log.error("embedding model failed to load: %s", self._cause)
            raise
        with self._lock:
            self._state = READY
            self._cause = None
            if self._finished is None or self._finished < (self._started or 0):
                self._finished = time.monotonic()
        log.info(
            "embedding model ready after %.1f s",
            (self._finished or 0) - (self._started or self._finished or 0),
        )

    def start_warmup(self) -> None:
        """Begin loading in a background thread and return at once. Idempotent."""
        with self._lock:
            if self._state != COLD:
                return
            self._state = WARMING
            self._started = time.monotonic()

        def run() -> None:
            try:
                self._load()
            except Exception:  # noqa: BLE001 - already recorded in the state
                pass

        threading.Thread(target=run, name="embedding-warmup", daemon=True).start()

    async def wait_ready(self, timeout: float | None = None) -> None:
        """Wait, without blocking the event loop, until a vector can be produced.

        Returns at once when the model is ready, or when warm-up was never started
        (the lazy path of a CLI or a test). Raises ``EmbeddingsWarming`` when the bound
        passes first and ``EmbeddingsFailed`` when the load failed.
        """
        bound = wait_bound() if timeout is None else timeout
        deadline = time.monotonic() + bound
        while self._state == WARMING and time.monotonic() < deadline:
            await asyncio.sleep(POLL_S)
        problem = self._not_ready()
        if problem is not None:
            raise problem

    # ------------------------------------------------------------------ model
    def _ensure_model(self) -> None:
        """Lazy-load the embedding model on first use."""
        with self._load_lock:
            if self._model is None:
                from fastembed import TextEmbedding

                cache = model_dir()
                cache.mkdir(parents=True, exist_ok=True)
                kwargs = {"cache_dir": str(cache), "providers": PROVIDERS}
                try:
                    # Once the files are cached a boot never waits on the network.
                    self._model = TextEmbedding(
                        self.MODEL_NAME, local_files_only=True, **kwargs
                    )
                except Exception:  # noqa: BLE001 - not cached yet: download it, here in the warm-up thread
                    log.info("model not cached in %s; downloading it", cache)
                    self._model = TextEmbedding(self.MODEL_NAME, **kwargs)

    def warmup(self) -> None:
        """Eagerly load the embedding model, here and now. For callers that can afford to wait.

        Joins a background load already under way rather than starting a second one.
        """
        with self._lock:
            if self._state in (COLD, FAILED):
                self._state = WARMING
                self._started = time.monotonic()
                self._finished = None
        self._load()

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text. Returns float32 array of shape (384,)."""
        problem = self._not_ready()
        if problem is not None:
            raise problem
        if self._model is None:
            log.warning(
                "EmbeddingService: model not pre-loaded — performance regression. "
                "Call warmup() at startup."
            )
        self._ensure_model()
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts. Returns (N, 384) array."""
        problem = self._not_ready()
        if problem is not None:
            raise problem
        if not texts:
            return np.zeros((0, self.DIMENSIONS), dtype=np.float32)
        self._ensure_model()
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(
            norms == 0, 1.0, norms
        )  # unit length, as the store expects
