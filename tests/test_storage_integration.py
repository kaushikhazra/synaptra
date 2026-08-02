from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.metadata import version

import surrealdb

from synaptra.models import Memory, MemoryState, MemoryType
from synaptra.surreal_storage import SurrealStorage

SURREALDB_VERSION = getattr(surrealdb, "__version__", version("surrealdb"))

assert tuple(int(x) for x in SURREALDB_VERSION.split(".")) >= (1, 0, 8), (
    f"surrealdb SDK {SURREALDB_VERSION} below validated floor 1.0.8"
)


def _build_memory(idx: int) -> Memory:
    now = datetime.now(timezone.utc) + timedelta(seconds=idx)
    return Memory(
        id=f"mem-{idx}",
        content=f"memory {idx}",
        memory_type=MemoryType.SEMANTIC,
        state=MemoryState.ACTIVE,
        importance=0.5,
        stability=5.0,
        retrievability=1.0,
        access_count=idx,
        created_at=now,
        updated_at=now,
        last_accessed=now,
        tags=[f"tag-{idx}"],
    )


async def test_get_memories_by_ids_returns_requested_rows() -> None:
    storage = SurrealStorage("mem://")
    inserted = [_build_memory(idx) for idx in range(5)]
    for memory in inserted:
        await storage.insert_memory(memory, embedding=None)

    rows = await storage.get_memories_by_ids(["mem-3", "mem-1", "missing", "mem-4"])
    by_id = {memory.id: memory for memory in rows}

    assert set(by_id) == {"mem-1", "mem-3", "mem-4"}
    assert by_id["mem-1"].content == "memory 1"
    assert by_id["mem-3"].tags == ["tag-3"]
    assert by_id["mem-4"].access_count == 4

    storage.close()
