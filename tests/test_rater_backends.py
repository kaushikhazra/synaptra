"""Criterion 15 — `rater` must work on EVERY storage backend, proven per backend.

The three backends spell the null condition differently — `IS NULL` in SQLite,
`IS NONE` in SurrealQL — and each declares, persists and hydrates Memory fields
independently. A field added to one and missed in another passes every other
test in this suite and fails at runtime on whichever backend is deployed.

So each backend gets the same three assertions, run against it directly:
round-trip, null preservation, and the `rater_not` filter including nulls.

`surreal_server_storage` is exercised separately against the live server (see
the loop's cycle-4 and cycle-7 logs); it needs a running SurrealDB and so is not
reachable from the unit suite.
"""

from __future__ import annotations

from datetime import datetime, timezone

from synaptra.models import Memory, MemoryState, MemoryType
from synaptra.storage import Storage
from synaptra.surreal_storage import SurrealStorage

RATER_A = "model-a"
RATER_B = "model-b"
RATED_AT = datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc)


def _memory(idx: int, rater: str | None) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=f"rater-mem-{idx}",
        content=f"rater test memory {idx}",
        memory_type=MemoryType.SEMANTIC,
        state=MemoryState.ACTIVE,
        importance=0.5,
        stability=5.0,
        retrievability=1.0,
        access_count=0,
        created_at=now,
        updated_at=now,
        last_accessed=now,
        rater=rater,
        rated_at=RATED_AT if rater else None,
        tags=["rater-test"],
    )


# --- SQLite -----------------------------------------------------------------


def test_sqlite_rater_round_trips() -> None:
    storage = Storage(":memory:")
    try:
        storage.insert_memory(_memory(1, RATER_A))
        got = storage.get_memory("rater-mem-1")
        assert got is not None
        assert got.rater == RATER_A
        assert got.rated_at is not None
        assert got.rated_at.replace(tzinfo=timezone.utc) == RATED_AT
    finally:
        storage.close()


def test_sqlite_null_rater_stays_null() -> None:
    storage = Storage(":memory:")
    try:
        storage.insert_memory(_memory(2, None))
        got = storage.get_memory("rater-mem-2")
        assert got is not None
        assert got.rater is None
        assert got.rated_at is None
    finally:
        storage.close()


def test_sqlite_rater_not_includes_nulls() -> None:
    """The NULL branch: `rater != ?` alone drops nulls under SQL's 3-valued logic."""
    storage = Storage(":memory:")
    try:
        storage.insert_memory(_memory(1, RATER_A))
        storage.insert_memory(_memory(2, None))
        storage.insert_memory(_memory(3, RATER_B))

        ids = {m.id for m in storage.list_memories(rater_not=RATER_A, limit=50)}

        assert "rater-mem-2" in ids, "null-rater row must be a candidate"
        assert "rater-mem-3" in ids, "row rated by another model must be a candidate"
        assert "rater-mem-1" not in ids, "a model is not its own candidate"
    finally:
        storage.close()


# --- Embedded SurrealDB -----------------------------------------------------


async def test_surreal_embedded_rater_round_trips() -> None:
    storage = SurrealStorage("mem://")
    await storage.insert_memory(_memory(1, RATER_A), embedding=None)
    got = await storage.get_memory("rater-mem-1")
    assert got is not None
    assert got.rater == RATER_A
    assert got.rated_at is not None
    assert got.rated_at.replace(tzinfo=timezone.utc) == RATED_AT


async def test_surreal_embedded_null_rater_stays_null() -> None:
    storage = SurrealStorage("mem://")
    await storage.insert_memory(_memory(2, None), embedding=None)
    got = await storage.get_memory("rater-mem-2")
    assert got is not None
    assert got.rater is None
    assert got.rated_at is None


async def test_surreal_embedded_rater_not_includes_nulls() -> None:
    """Same assertion as SQLite, but the condition is spelled `IS NONE` here."""
    storage = SurrealStorage("mem://")
    await storage.insert_memory(_memory(1, RATER_A), embedding=None)
    await storage.insert_memory(_memory(2, None), embedding=None)
    await storage.insert_memory(_memory(3, RATER_B), embedding=None)

    rows = await storage.list_memories(rater_not=RATER_A, limit=50)
    ids = {m.id for m in rows}

    assert "rater-mem-2" in ids, "null-rater row must be a candidate"
    assert "rater-mem-3" in ids, "row rated by another model must be a candidate"
    assert "rater-mem-1" not in ids, "a model is not its own candidate"
