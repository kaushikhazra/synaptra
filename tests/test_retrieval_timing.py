"""Tests for Phase 8 — per-phase timing instrumentation in recall().

Verifies that when logging.recall_timing=true the six expected DEBUG log
messages are emitted, and that when it is false no timing messages appear.

asyncio_mode = auto (see pytest.ini)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from synaptra.models import Memory, MemoryState, MemoryType
from synaptra.retrieval import recall


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_NOW = datetime.now(timezone.utc)

_EXPECTED_PHASE_NAMES = [
    "recall.phase1_gather",
    "recall.phase2_neighbors_bulk",
    "recall.hydration_supersede_gather",
    "recall.scoring",
    "recall.contradictions_spreading_gather",
    "recall.reinforce_stability_gather",
]


def _make_memory(mid: str = "mem:abc") -> Memory:
    return Memory(
        id=mid,
        content="test content",
        memory_type=MemoryType.EPISODIC,
        state=MemoryState.ACTIVE,
        importance=0.5,
        stability=2.0,
        retrievability=0.8,
        access_count=1,
        created_at=_NOW,
        updated_at=_NOW,
        last_accessed=_NOW,
        tags=[],
    )


def _make_storage(memory: Memory) -> MagicMock:
    """Return a mock storage whose async methods return minimal valid data."""
    storage = MagicMock()
    mid = memory.id

    storage.vector_search = AsyncMock(return_value=[(mid, 0.9)])
    storage.fts_search = AsyncMock(return_value=[(mid, 1.0)])
    storage.get_recent_active_ids = AsyncMock(return_value=[(mid, _NOW)])
    storage.get_neighbors_bulk = AsyncMock(return_value={})
    storage.get_memories_by_ids = AsyncMock(return_value=[memory])
    storage.get_supersede_flags = AsyncMock(return_value=set())
    storage.get_contradictions_bulk = AsyncMock(return_value={})
    storage.spreading_activation_walk = AsyncMock(return_value=[])
    storage.bulk_reinforce = AsyncMock(return_value=None)
    storage.bulk_update_stability = AsyncMock(return_value=None)
    return storage


def _make_embeddings() -> MagicMock:
    embeddings = MagicMock()
    embeddings.embed = MagicMock(return_value=np.zeros(384, dtype=np.float32))
    return embeddings


def _make_config(recall_timing: bool) -> MagicMock:
    """Return a mock Config whose .get() delegates to a simple lookup table."""
    defaults = {
        "retrieval.default_limit": 10,
        "retrieval.phase1_candidate_multiplier": 3,
        "retrieval.phase1_candidate_cap": 30,
        "retrieval.phase2_seed_count": 5,
        "retrieval.rrf_k": 60,
        "retrieval.weights.semantic": 1.0,
        "retrieval.weights.keyword": 0.7,
        "retrieval.weights.temporal": 0.3,
        "retrieval.weights.graph": 0.5,
        "retrieval.supersede_penalty": 0.3,
        "decay.decay_influence": 0.5,
        "decay.growth_factor": 2.0,
        "spreading_activation.activation_strength": 0.3,
        "spreading_activation.spread_factor": 0.5,
        "spreading_activation.max_depth": 3,
        "spreading_activation.max_boost": 0.5,
        "logging.recall_timing": recall_timing,
    }
    cfg = MagicMock()
    cfg.get = MagicMock(side_effect=lambda key, default=None: defaults.get(key, default))
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRecallTimingLogs:
    """Per-phase timing logs fire when logging.recall_timing=true."""

    async def test_all_phase_logs_emitted_when_timing_enabled(self, caplog):
        memory = _make_memory()
        storage = _make_storage(memory)
        embeddings = _make_embeddings()
        config = _make_config(recall_timing=True)

        with caplog.at_level(logging.DEBUG, logger="synaptra.retrieval"):
            await recall("test query", storage, embeddings, config)

        logged_messages = [r.message for r in caplog.records]
        for phase_name in _EXPECTED_PHASE_NAMES:
            assert any(
                msg.startswith(phase_name) for msg in logged_messages
            ), f"Expected log starting with '{phase_name}' not found in: {logged_messages}"

    async def test_phase_logs_include_ms_suffix(self, caplog):
        """Each timing log message ends with 'ms'."""
        memory = _make_memory()
        storage = _make_storage(memory)
        embeddings = _make_embeddings()
        config = _make_config(recall_timing=True)

        with caplog.at_level(logging.DEBUG, logger="synaptra.retrieval"):
            await recall("test query", storage, embeddings, config)

        timing_messages = [
            r.message for r in caplog.records
            if any(r.message.startswith(p) for p in _EXPECTED_PHASE_NAMES)
        ]
        assert len(timing_messages) == len(_EXPECTED_PHASE_NAMES)
        for msg in timing_messages:
            assert msg.endswith("ms"), f"Expected message to end with 'ms': {msg!r}"

    async def test_no_timing_logs_when_disabled(self, caplog):
        """With logging.recall_timing=false, no recall.* timing logs are emitted."""
        memory = _make_memory()
        storage = _make_storage(memory)
        embeddings = _make_embeddings()
        config = _make_config(recall_timing=False)

        with caplog.at_level(logging.DEBUG, logger="synaptra.retrieval"):
            await recall("test query", storage, embeddings, config)

        timing_messages = [
            r.message for r in caplog.records
            if any(r.message.startswith(p) for p in _EXPECTED_PHASE_NAMES)
        ]
        assert timing_messages == [], (
            f"Expected no timing logs but got: {timing_messages}"
        )

    async def test_timing_logs_are_debug_level(self, caplog):
        """Timing log messages are emitted at DEBUG, not INFO or WARNING."""
        memory = _make_memory()
        storage = _make_storage(memory)
        embeddings = _make_embeddings()
        config = _make_config(recall_timing=True)

        with caplog.at_level(logging.DEBUG, logger="synaptra.retrieval"):
            await recall("test query", storage, embeddings, config)

        for record in caplog.records:
            if any(record.message.startswith(p) for p in _EXPECTED_PHASE_NAMES):
                assert record.levelno == logging.DEBUG, (
                    f"Expected DEBUG for '{record.message}' but got level {record.levelno}"
                )
