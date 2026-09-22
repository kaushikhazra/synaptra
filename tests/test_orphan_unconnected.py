"""get_orphan_unconnected must exclude memories that HAVE an edge.

f735976 replaced a correlated SurrealQL subquery (`id NOT IN <16 SELECTs>`)
with a Python set-membership filter in both backends.  The only coverage was
test_memory_health.py::test_orphan_no_relations_detected, which stores one
memory into an empty store and asserts `no_relations_count >= 1` — a filter
that wrongly reported EVERY active memory as an orphan passes that test
identically to a correct one.

The failure mode these tests exist to catch is an id-TEXT mismatch: the edge
endpoints come back from `SELECT VALUE in/out FROM <rel>` and are stringified,
while the candidates are stringified from `row["id"]`.  If those two renderings
ever disagree (RecordID on one side, something else on the other) the set
membership test silently never matches and every memory looks unconnected.

That is why these run against a REAL SurrealDB store and not a stand-in: a fake
with plain string keys cannot reproduce the bug and would pass while proving
nothing.  Rows are built through storage methods rather than the engine —
engine.store_memory auto-links, which would create edges the test did not ask
for.  Counts are asserted EXACTLY; `>=` is what made the original test vacuous.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from synaptra.models import Memory, MemoryState, MemoryType, RelType, Relationship
from synaptra.surreal_storage import SurrealStorage


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — shared by both backends
# ──────────────────────────────────────────────────────────────────────────────


async def _store(
    storage, content: str, *, offset: int = 0, state: MemoryState = MemoryState.ACTIVE
) -> str:
    """Insert one memory at the STORAGE layer and return its id.

    `offset` seconds are subtracted from created_at so ORDER BY created_at ASC
    is deterministic — the 50-item cap depends on that ordering.
    """
    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc) - timedelta(seconds=offset)
    await storage.insert_memory(
        Memory(
            id=mid,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            state=state,
            importance=0.5,
            stability=1.0,
            retrievability=0.9,
            access_count=0,
            tags=[],
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )
    )
    return mid


async def _relate(
    storage, src: str, tgt: str, rel_type: RelType = RelType.RELATES_TO
) -> None:
    await storage.insert_relationship(
        Relationship(
            id=str(uuid.uuid4()),
            source_id=src,
            target_id=tgt,
            rel_type=rel_type,
            strength=1.0,
            created_at=datetime.now(timezone.utc),
        )
    )


def _ids(items: list[dict]) -> set[str]:
    return {i["id"] for i in items}


# ──────────────────────────────────────────────────────────────────────────────
# The assertions themselves — written once, run against both backends
# ──────────────────────────────────────────────────────────────────────────────


async def _assert_edge_excludes_both_endpoints(storage, rel_type: RelType) -> None:
    """A->B leaves only C.  Covers a source-only and a target-only memory at once."""
    a = await _store(storage, "A is the edge source", offset=30)
    b = await _store(storage, "B is the edge target", offset=20)
    c = await _store(storage, "C has no edge at all", offset=10)
    await _relate(storage, a, b, rel_type)

    items, true_count = await storage.get_orphan_unconnected()

    assert true_count == 1, (
        f"{rel_type.value}: expected exactly 1 unconnected memory, got {true_count}. "
        f"A ({a}) is an edge SOURCE and B ({b}) an edge TARGET; both must be excluded."
    )
    assert _ids(items) == {c}
    assert a not in _ids(items), "edge source leaked into the orphan list"
    assert b not in _ids(items), "edge target leaked into the orphan list"


async def _assert_zero_edges_reports_everything(storage) -> None:
    a = await _store(storage, "lonely A", offset=30)
    b = await _store(storage, "lonely B", offset=20)
    c = await _store(storage, "lonely C", offset=10)

    items, true_count = await storage.get_orphan_unconnected()

    assert true_count == 3
    assert _ids(items) == {a, b, c}


async def _assert_empty_store(storage) -> None:
    items, true_count = await storage.get_orphan_unconnected()
    assert true_count == 0
    assert items == []


async def _assert_archived_excluded(storage) -> None:
    """Only active rows are candidates."""
    active = await _store(storage, "still active", offset=20)
    await _store(storage, "already archived", offset=10, state=MemoryState.ARCHIVED)

    items, true_count = await storage.get_orphan_unconnected()

    assert true_count == 1
    assert _ids(items) == {active}


async def _assert_cap_at_fifty(storage) -> None:
    """true_count is the real total; the returned list is capped at 50."""
    for n in range(51):
        await _store(storage, f"unconnected {n}", offset=100 - n)

    items, true_count = await storage.get_orphan_unconnected()

    assert true_count == 51, "true_count must report the real total, not the cap"
    assert len(items) == 50, "the returned list must be capped at 50"


# ──────────────────────────────────────────────────────────────────────────────
# Embedded backend — runs in the normal suite
# ──────────────────────────────────────────────────────────────────────────────


class TestOrphanUnconnectedEmbedded:
    @pytest.fixture
    def storage(self):
        return SurrealStorage("mem://")

    @pytest.mark.parametrize("rel_type", list(RelType))
    async def test_edge_excludes_both_endpoints(self, storage, rel_type):
        """One edge of EVERY relationship type must exclude both its endpoints.

        Parametrised over all 8 tables because the fix iterates REL_TABLES — a
        table missing from that loop would only show up on its own type.
        """
        await _assert_edge_excludes_both_endpoints(storage, rel_type)

    async def test_zero_edges_reports_everything(self, storage):
        await _assert_zero_edges_reports_everything(storage)

    async def test_empty_store(self, storage):
        await _assert_empty_store(storage)

    async def test_archived_excluded(self, storage):
        await _assert_archived_excluded(storage)

    async def test_cap_at_fifty(self, storage):
        await _assert_cap_at_fifty(storage)

    async def test_two_edges_share_an_endpoint(self, storage):
        """A->B and C->B: B is a target twice, only D survives."""
        a = await _store(storage, "A", offset=40)
        b = await _store(storage, "B", offset=30)
        c = await _store(storage, "C", offset=20)
        d = await _store(storage, "D", offset=10)
        await _relate(storage, a, b)
        await _relate(storage, c, b, RelType.SUPPORTS)

        items, true_count = await storage.get_orphan_unconnected()

        assert true_count == 1
        assert _ids(items) == {d}


# ──────────────────────────────────────────────────────────────────────────────
# Server backend — skipped unless a SurrealDB server is reachable
# ──────────────────────────────────────────────────────────────────────────────

SURREAL_URL = os.environ.get("SYNAPTRA_SURREAL_URL", "ws://127.0.0.1:8000/rpc")


@pytest.mark.integration
class TestOrphanUnconnectedServer:
    """Same assertions against SurrealServerStorage.

    This path has no other test and the 300 s hang that prompted the rewrite was
    measured on a server backend, so it is the genuinely unproven half.

    Each test runs in a THROWAWAY database (namespace `synaptra_test`) and drops
    it on teardown, so the live cognitive/memory store is never written to.
    The connection is pre-seeded onto the storage object; SurrealServerStorage
    only connects lazily when `_db` is None, so every query it issues lands in
    the scratch database.  The scratch database is schemaless — these tests use
    CREATE / RELATE / SELECT only, which need no DDL.
    """

    @pytest.fixture
    async def storage(self):
        from surrealdb import AsyncSurreal

        from synaptra.surreal_server_storage import SurrealServerStorage

        db_name = "orphan_" + uuid.uuid4().hex
        try:
            conn = AsyncSurreal(SURREAL_URL)
            await asyncio.wait_for(conn.connect(), timeout=3.0)
            await conn.use("synaptra_test", db_name)
        except Exception as exc:  # noqa: BLE001 — any failure means "no server"
            pytest.skip(f"No SurrealDB server at {SURREAL_URL}: {exc}")

        store = SurrealServerStorage(SURREAL_URL)
        store._db = conn  # pre-seed: keeps every query inside the scratch database
        try:
            yield store
        finally:
            try:
                await conn.query(f"REMOVE DATABASE {db_name}")
            finally:
                await store.close()

    async def test_edge_excludes_both_endpoints(self, storage):
        await _assert_edge_excludes_both_endpoints(storage, RelType.RELATES_TO)

    async def test_zero_edges_reports_everything(self, storage):
        await _assert_zero_edges_reports_everything(storage)

    async def test_empty_store(self, storage):
        await _assert_empty_store(storage)

    async def test_archived_excluded(self, storage):
        await _assert_archived_excluded(storage)

    async def test_cap_at_fifty(self, storage):
        await _assert_cap_at_fifty(storage)
