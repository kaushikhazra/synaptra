"""Regression test for GitHub issue #4.

memory_list must compute retrievability on the fly, matching memory_get.
A memory whose last_accessed is far enough in the past must report
retrievability well below 1.0 in both endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from synaptra.models import Memory, MemoryState, MemoryType
from synaptra.surreal_storage import SurrealStorage
from synaptra.engine import MemoryEngine


def _make_stale_memory(memory_id: str = "stale-1", days_ago: int = 60) -> Memory:
    """Build a memory whose last_accessed is `days_ago` days in the past."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=days_ago)
    return Memory(
        id=memory_id,
        content="something that should have decayed",
        memory_type=MemoryType.EPISODIC,
        state=MemoryState.ACTIVE,
        importance=0.5,
        stability=2.0,
        retrievability=1.0,  # stored value — the bug returns this unchanged
        access_count=1,
        created_at=past,
        updated_at=past,
        last_accessed=past,
        tags=[],
    )


@pytest.fixture
async def engine_with_stale():
    storage = SurrealStorage("mem://")
    engine = MemoryEngine(storage=storage)
    mem = _make_stale_memory()
    await storage.insert_memory(mem, embedding=None)
    yield engine
    engine.close()


async def test_list_retrievability_matches_get(engine_with_stale):
    """Issue #4: memory_list and memory_get must agree on retrievability."""
    engine = engine_with_stale

    get_result = await engine.get_memory("stale-1")
    assert get_result is not None
    r_get = get_result.memory.retrievability

    listed = await engine.list_memories()
    assert len(listed) == 1
    r_list = listed[0].retrievability

    # Both must be well below 1.0 (60 days, stability 2 → R ≈ 0.036)
    assert r_get < 0.1, f"get retrievability {r_get} not decayed"
    assert r_list < 0.1, f"list retrievability {r_list} not decayed"

    # They must agree within floating-point rounding
    assert abs(r_get - r_list) < 1e-6, (
        f"retrievability mismatch: get={r_get}, list={r_list}"
    )
