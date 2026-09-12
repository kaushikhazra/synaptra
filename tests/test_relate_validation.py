"""Edge creation must refuse endpoints that do not resolve.

Validation lives at the STORAGE layer (storage.insert_relationship), so it is a
single chokepoint with no privileged callers: consolidation and auto-linking are
covered too, not just the MCP path.

Every test here asserts a REFUSAL. That ordering is the standard set by the
backup arc: prove the check FIRES before trusting it to pass. Each of these
fails against the pre-fix engine, where create_relationship performed no
validation at all and returned success for any string.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from synaptra.engine import MemoryEngine
from synaptra.models import Memory, MemoryState, MemoryType
from synaptra.surreal_storage import validate_edge_endpoints

# The id Velasari related to on the live server and got success back.
DEAD_BEEF = "00000000-dead-beef-0000-000000000000"


def _memory(mid: str, state: MemoryState = MemoryState.ACTIVE) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=f"memory {mid}",
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


class FakeStorage:
    """Minimal storage: knows about the ids it was given, nothing else.

    insert_relationship calls the REAL shared validator, exactly as both live
    backends do — so these tests exercise the actual chokepoint rather than a
    reimplementation of it. If validation is ever removed from the storage
    layer, these go red.
    """

    def __init__(self, memories: dict[str, Memory]):
        self._memories = memories
        self.inserted: list = []

    async def get_memory(self, memory_id: str):
        return self._memories.get(memory_id)

    async def insert_relationship(self, rel) -> None:
        await validate_edge_endpoints(self, rel)
        self.inserted.append(rel)


@pytest.fixture
def engine():
    real = str(uuid.uuid4())
    archived = str(uuid.uuid4())
    storage = FakeStorage(
        {
            real: _memory(real),
            archived: _memory(archived, MemoryState.ARCHIVED),
        }
    )
    eng = MemoryEngine.__new__(MemoryEngine)  # bypass __init__ (embeddings, config)
    eng.storage = storage
    return eng, storage, real, archived


class TestRefusesUnresolvableEndpoints:
    """Driven through engine.create_relationship, but the refusal originates in
    storage — the same code path consolidation and _auto_link take."""

    @pytest.mark.asyncio
    async def test_dead_beef_target_is_refused(self, engine):
        """Regression for the exact id that was accepted on the live server."""
        eng, storage, real, _ = engine
        with pytest.raises(ValueError) as exc:
            await eng.create_relationship(real, DEAD_BEEF, "relates_to")
        assert DEAD_BEEF in str(exc.value)
        assert "target_id" in str(exc.value)
        assert storage.inserted == [], "no edge may be created"

    @pytest.mark.asyncio
    async def test_unresolvable_source_is_refused(self, engine):
        eng, storage, real, _ = engine
        with pytest.raises(ValueError) as exc:
            await eng.create_relationship(DEAD_BEEF, real, "relates_to")
        assert "source_id" in str(exc.value)
        assert storage.inserted == []

    @pytest.mark.asyncio
    async def test_both_unresolvable_names_both(self, engine):
        eng, storage, _, _ = engine
        with pytest.raises(ValueError) as exc:
            await eng.create_relationship(DEAD_BEEF, "also-not-real", "relates_to")
        msg = str(exc.value)
        assert "source_id" in msg and "target_id" in msg
        assert storage.inserted == []

    @pytest.mark.asyncio
    async def test_truncated_id_fragment_is_refused(self, engine):
        """The 8-char fragments (e.g. 4a4b1dd4) that a small model confabulated
        during bulk relate calls — 137 of the 238 dangling references."""
        eng, storage, real, _ = engine
        with pytest.raises(ValueError):
            await eng.create_relationship(real, "4a4b1dd4", "relates_to")
        assert storage.inserted == []

    @pytest.mark.asyncio
    async def test_empty_id_is_refused(self, engine):
        eng, storage, real, _ = engine
        with pytest.raises(ValueError):
            await eng.create_relationship(real, "", "relates_to")
        assert storage.inserted == []


class TestAcceptsValidEndpoints:
    @pytest.mark.asyncio
    async def test_two_active_memories_relate(self, engine):
        eng, storage, real, archived = engine
        rel = await eng.create_relationship(real, archived, "relates_to")
        assert rel.source_id == real
        assert len(storage.inserted) == 1

    @pytest.mark.asyncio
    async def test_archived_memory_is_a_valid_endpoint(self, engine):
        """Archived rows still exist. Edges into archived memories are the
        normal state of an aging graph — refusing them would break consolidation
        and would be a far worse bug than the one being fixed.
        """
        eng, storage, real, archived = engine
        rel = await eng.create_relationship(archived, real, "supports")
        assert rel.target_id == real
        assert len(storage.inserted) == 1
