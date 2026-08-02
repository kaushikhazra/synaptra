"""Tests for SurrealServerStorage bulk methods (Phase 3 — Tasks 3.1-3.4)
and spreading_activation_walk (Phase 4 — Task 4.1).

All tests mock _query so no live SurrealDB server is required.
asyncio_mode = auto (see pytest.ini)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from synaptra.models import ReinforceUpdate, RelType, SpreadingActivationRow
from synaptra.surreal_server_storage import (
    SurrealServerStorage,
    _build_walk_sql,
)
from surrealdb import RecordID


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_storage() -> SurrealServerStorage:
    """Return a SurrealServerStorage instance with _query mocked (not called)."""
    storage = SurrealServerStorage.__new__(SurrealServerStorage)
    # Minimal init so instance methods work without a real connection
    storage._url = "ws://127.0.0.1:8000/rpc"
    storage._db = None
    storage._lock = __import__("asyncio").Lock()
    storage._schema_applied = True
    return storage


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rel_row(src: str, tgt: str, table: str, strength: float = 1.0, neighbor_state: str = "active") -> dict:
    """Build a minimal relationship row as SurrealDB would return it."""
    return {
        "id": f"{table}:edge-{src}-{tgt}",
        "in": f"memory:{src}",
        "out": f"memory:{tgt}",
        "strength": strength,
        "created_at": _now(),
        "neighbor_state": neighbor_state,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task 3.1 — get_neighbors_bulk
# ──────────────────────────────────────────────────────────────────────────────

class TestGetNeighborsBulk:
    """Tests for SurrealServerStorage.get_neighbors_bulk"""

    async def test_returns_empty_dict_for_empty_input(self):
        storage = _make_storage()
        result = await storage.get_neighbors_bulk([])
        assert result == {}

    async def test_bidirectional_edges_both_captured(self):
        """Seeds with both outgoing and incoming edges are both returned."""
        storage = _make_storage()

        call_count = [0]

        async def _mock_query(sql, params=None):
            call_count[0] += 1
            # Only return rows for 'causes' table, first outgoing call
            if "causes" in sql and "WHERE in IN" in sql:
                return [_rel_row("seed-1", "neighbor-A", "causes")]
            if "causes" in sql and "WHERE out IN" in sql:
                return [_rel_row("neighbor-B", "seed-1", "causes")]
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-1"])

        assert "seed-1" in result
        neighbor_ids = [nid for nid, _ in result["seed-1"]]
        assert "neighbor-A" in neighbor_ids  # outgoing
        assert "neighbor-B" in neighbor_ids  # incoming

    async def test_outgoing_only_seed(self):
        """A seed with only outgoing edges returns only those neighbors."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "relates_to" in sql and "WHERE in IN" in sql:
                return [_rel_row("seed-X", "target-1", "relates_to", strength=0.8)]
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-X"])
        neighbor_ids = [nid for nid, _ in result["seed-X"]]
        assert "target-1" in neighbor_ids

    async def test_incoming_only_seed(self):
        """A seed with only incoming edges returns those neighbors."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "supports" in sql and "WHERE out IN" in sql:
                return [_rel_row("other-1", "seed-Y", "supports")]
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-Y"])
        neighbor_ids = [nid for nid, _ in result["seed-Y"]]
        assert "other-1" in neighbor_ids

    async def test_seeds_with_no_neighbors_return_empty_lists(self):
        """Seeds with no edges return empty lists, not missing keys."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-A", "seed-B"])
        assert "seed-A" in result
        assert "seed-B" in result
        assert result["seed-A"] == []
        assert result["seed-B"] == []

    async def test_inactive_neighbors_are_excluded(self):
        """Neighbors with state != 'active' are filtered out."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "causes" in sql and "WHERE in IN" in sql:
                return [_rel_row("seed-1", "archived-neighbor", "causes", neighbor_state="archived")]
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-1"])
        assert result["seed-1"] == []

    async def test_relationship_type_is_correct(self):
        """Returned Relationship has correct rel_type."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "contradicts" in sql and "WHERE in IN" in sql:
                return [_rel_row("seed-C", "target-C", "contradicts")]
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-C"])
        assert len(result["seed-C"]) == 1
        _, rel = result["seed-C"][0]
        assert rel.rel_type == RelType.CONTRADICTS

    async def test_multiple_seeds_edges_attributed_correctly(self):
        """Edges for seed-1 must not appear under seed-2."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "follows" in sql and "WHERE in IN" in sql:
                return [
                    _rel_row("seed-1", "neighbor-1", "follows"),
                    _rel_row("seed-2", "neighbor-2", "follows"),
                ]
            return []

        storage._query = _mock_query
        result = await storage.get_neighbors_bulk(["seed-1", "seed-2"])
        s1_ids = [nid for nid, _ in result["seed-1"]]
        s2_ids = [nid for nid, _ in result["seed-2"]]
        assert "neighbor-1" in s1_ids
        assert "neighbor-2" not in s1_ids
        assert "neighbor-2" in s2_ids
        assert "neighbor-1" not in s2_ids

    async def test_seeds_param_passed_as_record_ids(self):
        """$seeds must be a list of RecordID objects, not strings — type-mismatch regression guard."""
        storage = _make_storage()
        captured_params: list[dict] = []

        async def _mock_query(sql, params=None):
            if params and "seeds" in params:
                captured_params.append(params)
            return []

        storage._query = _mock_query
        await storage.get_neighbors_bulk(["abc-123", "def-456"])

        assert captured_params, "Expected at least one _query call with $seeds param"
        for params in captured_params:
            seeds = params["seeds"]
            assert isinstance(seeds, list), "$seeds must be a list"
            for seed in seeds:
                assert isinstance(seed, RecordID), (
                    f"$seeds elements must be RecordID, got {type(seed).__name__}: {seed!r}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Task 3.2 — get_supersede_flags
# ──────────────────────────────────────────────────────────────────────────────

class TestGetSupersedeFlags:
    """Tests for SurrealServerStorage.get_supersede_flags"""

    async def test_empty_input_returns_empty_set(self):
        storage = _make_storage()
        result = await storage.get_supersede_flags([])
        assert result == set()

    async def test_ids_with_incoming_supersedes_returned(self):
        """IDs that are the target (out) of a supersedes edge are returned."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return [{"out": "memory:old-mem-1"}, {"out": "memory:old-mem-2"}]

        storage._query = _mock_query
        result = await storage.get_supersede_flags(["old-mem-1", "old-mem-2", "fresh-mem"])
        assert "old-mem-1" in result
        assert "old-mem-2" in result

    async def test_ids_without_supersedes_not_returned(self):
        """IDs with no incoming supersedes edge are absent from the result."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return []

        storage._query = _mock_query
        result = await storage.get_supersede_flags(["fresh-1", "fresh-2"])
        assert result == set()

    async def test_mixed_input_returns_only_superseded(self):
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return [{"out": "memory:superseded-1"}]

        storage._query = _mock_query
        result = await storage.get_supersede_flags(["superseded-1", "not-superseded"])
        assert result == {"superseded-1"}

    async def test_query_targets_supersedes_table(self):
        """Verifies the query touches the supersedes table."""
        storage = _make_storage()
        captured = []

        async def _mock_query(sql, params=None):
            captured.append(sql)
            return []

        storage._query = _mock_query
        await storage.get_supersede_flags(["some-id"])
        assert any("supersedes" in sql for sql in captured)

    async def test_ids_param_passed_as_record_ids(self):
        """$ids must be a list of RecordID objects, not strings — type-mismatch regression guard."""
        storage = _make_storage()
        captured_params: list[dict] = []

        async def _mock_query(sql, params=None):
            if params:
                captured_params.append(params)
            return []

        storage._query = _mock_query
        await storage.get_supersede_flags(["mem-a", "mem-b", "mem-c"])

        assert captured_params, "Expected at least one _query call with params"
        for params in captured_params:
            ids = params.get("ids", [])
            assert isinstance(ids, list), "$ids must be a list"
            for item in ids:
                assert isinstance(item, RecordID), (
                    f"$ids elements must be RecordID, got {type(item).__name__}: {item!r}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Task 3.3 — get_contradictions_bulk
# ──────────────────────────────────────────────────────────────────────────────

class TestGetContradictionsBulk:
    """Tests for SurrealServerStorage.get_contradictions_bulk"""

    async def test_empty_input_returns_empty_dict(self):
        storage = _make_storage()
        result = await storage.get_contradictions_bulk([])
        assert result == {}

    async def test_outgoing_contradictions_captured(self):
        """Anchor as source (in) — other is the target (out)."""
        storage = _make_storage()
        call_count = [0]

        async def _mock_query(sql, params=None):
            call_count[0] += 1
            # First call: outgoing (in IN $ids)
            if "WHERE in IN" in sql:
                return [{
                    "anchor_id": "memory:anchor-1",
                    "other_id": "memory:contra-A",
                    "other_content": "contradicting content",
                    "other_state": "active",
                    "strength": 0.9,
                }]
            return []

        storage._query = _mock_query
        result = await storage.get_contradictions_bulk(["anchor-1"])
        assert "anchor-1" in result
        assert len(result["anchor-1"]) == 1
        other_id, content, strength = result["anchor-1"][0]
        assert other_id == "contra-A"
        assert strength == 0.9

    async def test_incoming_contradictions_captured(self):
        """Anchor as target (out) — other is the source (in)."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "WHERE out IN" in sql:
                return [{
                    "anchor_id": "memory:anchor-2",
                    "other_id": "memory:contra-B",
                    "other_content": "incoming contradiction",
                    "other_state": "active",
                    "strength": 0.7,
                }]
            return []

        storage._query = _mock_query
        result = await storage.get_contradictions_bulk(["anchor-2"])
        assert "anchor-2" in result
        assert len(result["anchor-2"]) == 1
        other_id, _, _ = result["anchor-2"][0]
        assert other_id == "contra-B"

    async def test_both_directions_combined(self):
        """Anchor appears both as source and target — both must be returned."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "WHERE in IN" in sql:
                return [{
                    "anchor_id": "memory:anchor-3",
                    "other_id": "memory:out-contra",
                    "other_content": "outgoing",
                    "other_state": "active",
                    "strength": 0.6,
                }]
            if "WHERE out IN" in sql:
                return [{
                    "anchor_id": "memory:anchor-3",
                    "other_id": "memory:in-contra",
                    "other_content": "incoming",
                    "other_state": "active",
                    "strength": 0.8,
                }]
            return []

        storage._query = _mock_query
        result = await storage.get_contradictions_bulk(["anchor-3"])
        other_ids = [t[0] for t in result["anchor-3"]]
        assert "out-contra" in other_ids
        assert "in-contra" in other_ids
        assert len(result["anchor-3"]) == 2

    async def test_deduplication_same_pair(self):
        """Same pair appearing in both directions should only appear once per anchor."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            if "WHERE in IN" in sql:
                return [{
                    "anchor_id": "memory:anchor-4",
                    "other_id": "memory:dup-contra",
                    "other_content": "dup content",
                    "other_state": "active",
                    "strength": 0.5,
                }]
            if "WHERE out IN" in sql:
                return [{
                    "anchor_id": "memory:anchor-4",
                    "other_id": "memory:dup-contra",
                    "other_content": "dup content",
                    "other_state": "active",
                    "strength": 0.5,
                }]
            return []

        storage._query = _mock_query
        result = await storage.get_contradictions_bulk(["anchor-4"])
        # Should be deduplicated — only one entry
        assert len(result["anchor-4"]) == 1

    async def test_no_contradictions_returns_empty_lists(self):
        """IDs with no contradictions have empty lists, not missing keys."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return []

        storage._query = _mock_query
        result = await storage.get_contradictions_bulk(["id-1", "id-2"])
        assert "id-1" in result
        assert "id-2" in result
        assert result["id-1"] == []
        assert result["id-2"] == []

    async def test_ids_param_passed_as_record_ids(self):
        """$ids must be a list of RecordID objects, not strings — type-mismatch regression guard."""
        storage = _make_storage()
        captured_params: list[dict] = []

        async def _mock_query(sql, params=None):
            if params:
                captured_params.append(params)
            return []

        storage._query = _mock_query
        await storage.get_contradictions_bulk(["anchor-x", "anchor-y"])

        assert captured_params, "Expected at least one _query call with params"
        for params in captured_params:
            ids = params.get("ids", [])
            assert isinstance(ids, list), "$ids must be a list"
            for item in ids:
                assert isinstance(item, RecordID), (
                    f"$ids elements must be RecordID, got {type(item).__name__}: {item!r}"
                )

    async def test_inactive_contradictions_excluded_python_side(self):
        """Rows with other_state != 'active' are dropped by Python filter, not SQL.
        The SQL has no AND predicate — state filter is purely Python-side."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            # Verify SQL has NO AND predicate (compound-predicate decouple)
            assert "AND" not in sql, (
                "get_contradictions_bulk SQL must not contain AND predicates after decouple fix"
            )
            if "WHERE in IN" in sql:
                return [
                    {
                        "anchor_id": "memory:anchor-5",
                        "other_id": "memory:active-contra",
                        "other_content": "active",
                        "other_state": "active",
                        "strength": 0.8,
                    },
                    {
                        "anchor_id": "memory:anchor-5",
                        "other_id": "memory:archived-contra",
                        "other_content": "archived",
                        "other_state": "archived",
                        "strength": 0.7,
                    },
                ]
            return []

        storage._query = _mock_query
        result = await storage.get_contradictions_bulk(["anchor-5"])
        other_ids = [t[0] for t in result["anchor-5"]]
        assert "active-contra" in other_ids, "Active contradiction must be included"
        assert "archived-contra" not in other_ids, "Archived contradiction must be excluded"


# ──────────────────────────────────────────────────────────────────────────────
# Task 3.4 — bulk_reinforce
# ──────────────────────────────────────────────────────────────────────────────

class TestBulkReinforce:
    """Tests for SurrealServerStorage.bulk_reinforce"""

    async def test_empty_input_does_not_call_query(self):
        storage = _make_storage()
        called = []

        async def _mock_query(sql, params=None):
            called.append(sql)
            return []

        storage._query = _mock_query
        await storage.bulk_reinforce([])
        assert called == []  # No query should be issued

    async def test_batch_k10_calls_query_once(self):
        """k=10 updates should result in a single query (FOR loop)."""
        storage = _make_storage()
        captured_params = []

        async def _mock_query(sql, params=None):
            captured_params.append(params)
            return []

        storage._query = _mock_query
        now = _now()
        updates = [
            ReinforceUpdate(
                memory_id=f"mem-{i}",
                stability=5.0 + i * 0.1,
                last_accessed=now,
                access_count=i + 1,
            )
            for i in range(10)
        ]
        await storage.bulk_reinforce(updates)
        assert len(captured_params) == 1
        assert len(captured_params[0]["updates"]) == 10

    async def test_batch_k50_all_updates_present(self):
        """k=50 updates — verify all 50 are included in the query params."""
        storage = _make_storage()
        captured_params = []

        async def _mock_query(sql, params=None):
            captured_params.append(params)
            return []

        storage._query = _mock_query
        now = _now()
        updates = [
            ReinforceUpdate(
                memory_id=f"mem-{i}",
                stability=1.0 + i * 0.01,
                last_accessed=now,
                access_count=i,
            )
            for i in range(50)
        ]
        await storage.bulk_reinforce(updates)
        assert len(captured_params[0]["updates"]) == 50
        ids_in_query = {u["memory_id"] for u in captured_params[0]["updates"]}
        expected_ids = {f"mem-{i}" for i in range(50)}
        assert ids_in_query == expected_ids

    async def test_update_fields_are_correct(self):
        """Verify stability, last_accessed, access_count are passed correctly."""
        storage = _make_storage()
        captured_params = []

        async def _mock_query(sql, params=None):
            captured_params.append(params)
            return []

        storage._query = _mock_query
        ts = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
        update = ReinforceUpdate(
            memory_id="test-mem",
            stability=7.5,
            last_accessed=ts,
            access_count=42,
        )
        await storage.bulk_reinforce([update])
        row = captured_params[0]["updates"][0]
        assert row["memory_id"] == "test-mem"
        assert row["stability"] == 7.5
        assert row["last_accessed"] == ts
        assert row["access_count"] == 42

    async def test_query_uses_for_loop_pattern(self):
        """Verify the generated SQL uses the FOR-loop pattern."""
        storage = _make_storage()
        captured_sqls = []

        async def _mock_query(sql, params=None):
            captured_sqls.append(sql)
            return []

        storage._query = _mock_query
        await storage.bulk_reinforce([
            ReinforceUpdate(memory_id="x", stability=1.0, last_accessed=_now(), access_count=1)
        ])
        assert len(captured_sqls) == 1
        sql = captured_sqls[0].lower()
        assert "for" in sql
        assert "stability" in sql
        assert "last_accessed" in sql
        assert "access_count" in sql


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — spreading_activation_walk
# ──────────────────────────────────────────────────────────────────────────────

def _walk_row(
    neighbor_id: str,
    depth: int,
    rel_strength: float = 0.8,
    current_stability: float = 5.0,
    state: str = "active",
) -> dict:
    """Build a row as ``RETURN $deduped`` would emit from SurrealDB."""
    return {
        "neighbor_id": f"memory:{neighbor_id}",
        "depth": depth,
        "rel_strength": rel_strength,
        "current_stability": current_stability,
        "state": state,
    }


class TestSpreadingActivationWalk:
    """Phase 4 tests for SurrealServerStorage.spreading_activation_walk.

    All tests mock ``_query`` so no live SurrealDB is required.
    ``result[-1]`` is what the method reads — the RETURN $deduped output.
    """

    # ── 1. Seed exclusion ────────────────────────────────────────────────────

    async def test_seed_ids_absent_from_results(self):
        """Seeds must not appear in returned rows even when the mock naively
        returns them (Python-side safety filter guards this invariant)."""
        storage = _make_storage()
        seed_id = "seed-abc"

        async def _mock_query(sql, params=None):
            # Simulate SQL accidentally leaking the seed into results
            return [
                None,  # placeholder for LET statements
                [
                    _walk_row(seed_id, depth=1),        # seed — must be excluded
                    _walk_row("neighbor-1", depth=1),   # legitimate neighbor
                ],
            ]

        storage._query = _mock_query
        rows = await storage.spreading_activation_walk([seed_id])

        returned_ids = {r.neighbor_id for r in rows}
        assert seed_id not in returned_ids, "Seed must be absent from results"
        assert "neighbor-1" in returned_ids

    # ── 2. state = 'active' filter ──────────────────────────────────────────

    async def test_archived_neighbors_excluded(self):
        """Rows with state != 'active' must be dropped by the Python filter."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return [
                None,
                [
                    _walk_row("archived-node", depth=1, state="archived"),
                    _walk_row("active-node", depth=1, state="active"),
                ],
            ]

        storage._query = _mock_query
        rows = await storage.spreading_activation_walk(["seed-1"])

        returned_ids = {r.neighbor_id for r in rows}
        assert "archived-node" not in returned_ids
        assert "active-node" in returned_ids

    # ── 3. Depth boundary — SQL structure ───────────────────────────────────

    async def test_depth_boundary_max_depth_2_no_d3_block(self):
        """SQL generated for max_depth=2 must not contain depth-3 artefacts."""
        sql = _build_walk_sql(max_depth=2)
        assert "$d3_raw" not in sql, "max_depth=2 must not generate $d3_raw block"
        assert "$d3_tagged" not in sql
        assert "$d2_raw" in sql
        assert "$d1_raw" in sql

    async def test_depth_boundary_max_depth_1_only_d1_block(self):
        """SQL for max_depth=1 contains only the d1 layer."""
        sql = _build_walk_sql(max_depth=1)
        assert "$d1_raw" in sql
        assert "$d2_raw" not in sql

    async def test_depth_boundary_depth3_nodes_absent_at_max_depth_2(self):
        """With max_depth=2, a mock that returns depth-3 rows is discarded
        via the shallowest-depth dedup (Python safety filter keeps the best)."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            # Verify SQL does not contain d3 block
            assert "$d3_raw" not in sql, "max_depth=2 must not send a d3 block"
            return [
                None,
                [
                    _walk_row("n1", depth=1),
                    _walk_row("n2", depth=2),
                ],
            ]

        storage._query = _mock_query
        rows = await storage.spreading_activation_walk(["seed-x"], max_depth=2)
        depths = {r.neighbor_id: r.depth for r in rows}
        assert depths.get("n1") == 1
        assert depths.get("n2") == 2

    # ── 4. Shallowest-depth dedup ────────────────────────────────────────────

    async def test_shallowest_depth_wins_same_neighbor_two_paths(self):
        """When a neighbor appears at depth 1 AND depth 2, result keeps depth=1."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return [
                None,
                [
                    _walk_row("n5", depth=1, rel_strength=0.95),
                    _walk_row("n5", depth=2, rel_strength=0.7),   # duplicate, deeper
                ],
            ]

        storage._query = _mock_query
        rows = await storage.spreading_activation_walk(["seed-1"])

        n5_rows = [r for r in rows if r.neighbor_id == "n5"]
        assert len(n5_rows) == 1, "n5 must appear exactly once"
        assert n5_rows[0].depth == 1, "Shallowest depth (1) must win"

    # ── 5. Multi-seed dedup ──────────────────────────────────────────────────

    async def test_multi_seed_dedup_shallowest_depth_wins(self):
        """A neighbor reachable from seed A at depth 1 and seed B at depth 2
        must appear once at depth=1 (seed A's shorter path wins)."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            # SQL processes both seeds in one pass; deduped result:
            return [
                None,
                [
                    _walk_row("n6", depth=1, rel_strength=0.5),   # via seed-B
                    _walk_row("n6", depth=2, rel_strength=0.85),  # via seed-A deeper path
                ],
            ]

        storage._query = _mock_query
        rows = await storage.spreading_activation_walk(["seed-a", "seed-b"])

        n6_rows = [r for r in rows if r.neighbor_id == "n6"]
        assert len(n6_rows) == 1, "n6 must appear exactly once across all seeds"
        assert n6_rows[0].depth == 1

    # ── 6. Bidirectional coverage — SQL structure ────────────────────────────

    async def test_sql_traverses_both_directions(self):
        """Generated SQL must probe both outgoing (WHERE in IN) and incoming
        (WHERE out IN) edges for every table, so nodes reachable only via
        incoming edges to a seed are included."""
        sql = _build_walk_sql(max_depth=1)
        # Each direction must appear at least once per depth layer
        assert "WHERE in IN $seeds" in sql, "Outgoing traversal (WHERE in IN) must be present"
        assert "WHERE out IN $seeds" in sql, "Incoming traversal (WHERE out IN) must be present"

    async def test_sql_covers_all_8_relation_tables(self):
        """SQL must reference all 8 relation table names."""
        sql = _build_walk_sql(max_depth=1)
        expected_tables = [
            "causes", "follows", "contradicts", "supports",
            "relates_to", "supersedes", "part_of", "describes",
        ]
        for table in expected_tables:
            assert table in sql, f"Table '{table}' missing from walk SQL"

    # ── Additional correctness / edge-case tests ─────────────────────────────

    async def test_empty_seed_ids_returns_empty(self):
        """No seeds → no query, empty result."""
        storage = _make_storage()
        called = []

        async def _mock_query(sql, params=None):
            called.append(sql)
            return []

        storage._query = _mock_query
        result = await storage.spreading_activation_walk([])
        assert result == []
        assert called == []  # _query must not be called

    async def test_max_depth_zero_returns_empty(self):
        """max_depth=0 → no query, empty result."""
        storage = _make_storage()
        called = []

        async def _mock_query(sql, params=None):
            called.append(sql)
            return []

        storage._query = _mock_query
        result = await storage.spreading_activation_walk(["seed-1"], max_depth=0)
        assert result == []
        assert called == []

    async def test_empty_db_result_returns_empty(self):
        """Empty/None _query result → empty list, no exception."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return []

        storage._query = _mock_query
        result = await storage.spreading_activation_walk(["seed-1"])
        assert result == []

    async def test_returns_spreading_activation_row_instances(self):
        """Returned items must be SpreadingActivationRow NamedTuples with correct fields."""
        storage = _make_storage()

        async def _mock_query(sql, params=None):
            return [
                None,
                [_walk_row("n-target", depth=2, rel_strength=0.6, current_stability=3.5)],
            ]

        storage._query = _mock_query
        rows = await storage.spreading_activation_walk(["seed-1"])

        assert len(rows) == 1
        r = rows[0]
        assert isinstance(r, SpreadingActivationRow)
        assert r.neighbor_id == "n-target"
        assert r.depth == 2
        assert abs(r.rel_strength - 0.6) < 1e-9
        assert abs(r.current_stability - 3.5) < 1e-9
        assert r.state == "active"

    async def test_single_query_called_for_multi_seed_walk(self):
        """spreading_activation_walk must issue exactly one _query call
        regardless of how many seeds are provided."""
        storage = _make_storage()
        call_count = [0]

        async def _mock_query(sql, params=None):
            call_count[0] += 1
            return [None, []]

        storage._query = _mock_query
        await storage.spreading_activation_walk(["s1", "s2", "s3"])
        assert call_count[0] == 1, "Must use a single round-trip for all seeds"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 — F1 filter push-down (Tasks 5.1, 5.2)
# ──────────────────────────────────────────────────────────────────────────────

def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


class TestFtsSearchFilters:
    """fts_search accepts type_filter, tags, time_range and inlines them into SQL."""

    async def test_no_filters_produces_plain_sql(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.fts_search("hello")
        sql, params = storage._query.call_args.args
        assert "memory_type" not in sql
        assert "CONTAINSALL" not in sql
        assert "created_at" not in sql
        assert "type_filter" not in params
        assert "tags" not in params
        assert "time_start" not in params

    async def test_type_filter_appended(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.fts_search("hello", type_filter="semantic")
        sql, params = storage._query.call_args.args
        assert "memory_type = $type_filter" in sql
        assert params["type_filter"] == "semantic"

    async def test_tags_filter_appended(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.fts_search("hello", tags=["python", "async"])
        sql, params = storage._query.call_args.args
        assert "tags CONTAINSALL $tags" in sql
        assert params["tags"] == ["python", "async"]

    async def test_time_range_appended(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        t0, t1 = _dt(2025, 1, 1), _dt(2025, 6, 1)
        await storage.fts_search("hello", time_range=(t0, t1))
        sql, params = storage._query.call_args.args
        assert "created_at >= $time_start" in sql
        assert "created_at <= $time_end" in sql
        assert "time_start" in params
        assert "time_end" in params

    async def test_type_and_tags_combined(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.fts_search("hello", type_filter="episodic", tags=["work"])
        sql, params = storage._query.call_args.args
        assert "memory_type = $type_filter" in sql
        assert "tags CONTAINSALL $tags" in sql
        assert params["type_filter"] == "episodic"
        assert params["tags"] == ["work"]

    async def test_all_filters_combined(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        t0, t1 = _dt(2025, 1, 1), _dt(2025, 12, 31)
        await storage.fts_search(
            "hello",
            type_filter="procedural",
            tags=["a", "b"],
            time_range=(t0, t1),
        )
        sql, params = storage._query.call_args.args
        assert "memory_type = $type_filter" in sql
        assert "tags CONTAINSALL $tags" in sql
        assert "created_at >= $time_start" in sql
        assert "created_at <= $time_end" in sql
        assert params["type_filter"] == "procedural"
        assert params["tags"] == ["a", "b"]
        assert "time_start" in params
        assert "time_end" in params

    async def test_none_tags_produces_no_containsall(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.fts_search("hello", tags=None)
        sql, _ = storage._query.call_args.args
        assert "CONTAINSALL" not in sql

    async def test_returns_empty_list_when_no_rows(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        result = await storage.fts_search("hello", type_filter="working")
        assert result == []


class TestVectorSearchFilters:
    """vector_search: two-step HNSW oversample → Python post-filter → truncate.

    SQL contract (new): pure HNSW, no AND predicates, only 'vec' param.
    Filter contract: state / type_filter / tags / time_range enforced in Python.
    """

    # ── SQL shape ────────────────────────────────────────────────────────────

    async def test_no_and_predicates_in_sql(self):
        """SQL must contain no AND predicates regardless of which filters are passed."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search(
            [0.1] * 384,
            type_filter="working",
            tags=["ml"],
            time_range=(_dt(2025, 1, 1), _dt(2025, 12, 31)),
        )
        sql, _ = storage._query.call_args.args
        assert " AND " not in sql, "SQL must contain no AND predicates after HNSW fix"

    async def test_hnsw_operator_always_present(self):
        """The HNSW <|N,40|> operator must always appear in the WHERE clause."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search([0.1] * 384, type_filter="semantic", tags=["t"])
        sql, _ = storage._query.call_args.args
        assert "embedding <|" in sql

    async def test_only_vec_param_passed(self):
        """Only 'vec' is passed as a query param; type_filter/tags/time_start go away."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search(
            [0.1] * 384,
            type_filter="semantic",
            tags=["ml"],
            time_range=(_dt(2025, 1, 1), _dt(2025, 6, 1)),
        )
        _, params = storage._query.call_args.args
        assert set(params.keys()) == {"vec"}, f"Expected only 'vec' param, got: {set(params.keys())}"

    async def test_no_filters_sql_has_no_filter_predicates(self):
        """With no filters, SQL must not contain filter predicates (= $type_filter,
        CONTAINSALL, created_at comparisons). 'memory_type' appears in SELECT (OK)."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search([0.1] * 384)
        sql, params = storage._query.call_args.args
        assert "= $type_filter" not in sql
        assert "CONTAINSALL" not in sql
        assert " AND " not in sql
        assert "time_start" not in params
        assert "type_filter" not in params
        assert "tags" not in params

    async def test_fetch_k_is_3x_top_k(self):
        """HNSW operator size and LIMIT in SQL must be 3 × top_k (oversample factor)."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search([0.1] * 384, top_k=10)
        sql, _ = storage._query.call_args.args
        assert "<|30,40|>" in sql, "fetch_k (30) must appear in HNSW operator"
        assert "LIMIT 30" in sql, "fetch_k (30) must appear in LIMIT"

    # ── Python post-filter: state ─────────────────────────────────────────

    async def test_python_filter_excludes_archived(self):
        """Archived rows returned by HNSW are dropped by Python post-filter."""
        storage = _make_storage()
        rows = [
            {"id": "memory:a1", "score": 0.9, "state": "active",   "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 6, 1)},
            {"id": "memory:a2", "score": 0.8, "state": "archived",  "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search([0.1] * 384, state="active")
        ids = [mid for mid, _ in result]
        assert "a1" in ids
        assert "a2" not in ids

    # ── Python post-filter: type_filter ──────────────────────────────────

    async def test_python_filter_type_filter(self):
        """Only rows matching type_filter pass; non-matching rows are dropped."""
        storage = _make_storage()
        rows = [
            {"id": "memory:t1", "score": 0.9, "state": "active", "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 6, 1)},
            {"id": "memory:t2", "score": 0.85, "state": "active", "memory_type": "episodic", "tags": [], "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search([0.1] * 384, type_filter="semantic")
        ids = [mid for mid, _ in result]
        assert "t1" in ids
        assert "t2" not in ids

    # ── Python post-filter: tags ──────────────────────────────────────────

    async def test_python_filter_tags_subset_match(self):
        """Row must contain ALL requested tags; partial match is excluded."""
        storage = _make_storage()
        rows = [
            {"id": "memory:g1", "score": 0.9, "state": "active", "memory_type": "semantic", "tags": ["ml", "nlp"], "created_at": _dt(2025, 6, 1)},
            {"id": "memory:g2", "score": 0.8, "state": "active", "memory_type": "semantic", "tags": ["ml"],        "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search([0.1] * 384, tags=["ml", "nlp"])
        ids = [mid for mid, _ in result]
        assert "g1" in ids
        assert "g2" not in ids

    async def test_empty_tags_no_filter(self):
        """Empty tags list → no tag filter; all state-passing rows are returned."""
        storage = _make_storage()
        rows = [
            {"id": "memory:e1", "score": 0.9, "state": "active", "memory_type": "semantic", "tags": [],            "created_at": _dt(2025, 6, 1)},
            {"id": "memory:e2", "score": 0.8, "state": "active", "memory_type": "semantic", "tags": ["unrelated"], "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search([0.1] * 384, tags=[])
        assert len(result) == 2

    async def test_none_tags_no_filter(self):
        """None tags → no tag filter; rows with null tags in DB still pass."""
        storage = _make_storage()
        rows = [
            {"id": "memory:n1", "score": 0.9, "state": "active", "memory_type": "semantic", "tags": None, "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search([0.1] * 384, tags=None)
        assert len(result) == 1

    # ── Python post-filter: time_range ────────────────────────────────────

    async def test_python_filter_time_range(self):
        """Rows outside time_range are excluded; tz-naive datetimes coerced safely."""
        storage = _make_storage()
        rows = [
            {"id": "memory:r1", "score": 0.9, "state": "active", "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 4, 15)},
            {"id": "memory:r2", "score": 0.8, "state": "active", "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 8, 1)},
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search(
            [0.1] * 384,
            time_range=(_dt(2025, 1, 1), _dt(2025, 6, 1)),
        )
        ids = [mid for mid, _ in result]
        assert "r1" in ids
        assert "r2" not in ids

    # ── Oversample + truncate ─────────────────────────────────────────────

    async def test_oversample_truncates_to_top_k(self):
        """When HNSW returns 3×top_k and all pass post-filter, result is exactly top_k."""
        storage = _make_storage()
        top_k = 5
        rows = [
            {"id": f"memory:m{i}", "score": 1.0 - i * 0.01, "state": "active",
             "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 6, 1)}
            for i in range(top_k * 3)
        ]
        storage._query = AsyncMock(return_value=rows)
        result = await storage.vector_search([0.1] * 384, top_k=top_k)
        assert len(result) <= top_k

    async def test_oversample_truncates_after_aggressive_filter(self):
        """When post-filter prunes most rows, result equals filtered count (≤ top_k)."""
        storage = _make_storage()
        top_k = 5
        active_rows = [
            {"id": f"memory:a{i}", "score": 0.9, "state": "active",
             "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 6, 1)}
            for i in range(3)
        ]
        archived_rows = [
            {"id": f"memory:x{i}", "score": 0.8, "state": "archived",
             "memory_type": "semantic", "tags": [], "created_at": _dt(2025, 6, 1)}
            for i in range(12)
        ]
        storage._query = AsyncMock(return_value=active_rows + archived_rows)
        result = await storage.vector_search([0.1] * 384, top_k=top_k)
        assert len(result) == 3   # only 3 active rows passed
        assert len(result) <= top_k


class TestVectorSearchForMemoryFilters:
    """vector_search_for_memory: pure HNSW → Python post-filter (active + self-exclude)."""

    async def test_excludes_self(self):
        """The query memory itself must never appear in results (RecordID vs str safe compare)."""
        storage = _make_storage()
        memory_id = "abc-123"
        inner_rows = [
            {"id": f"memory:{memory_id}", "score": 1.0,  "state": "active",   "tags": [], "created_at": _dt(2025, 6, 1)},
            {"id": "memory:neighbor-1",   "score": 0.85, "state": "active",   "tags": [], "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=[inner_rows])
        result = await storage.vector_search_for_memory(memory_id, top_k=10)
        ids = [mid for mid, _ in result]
        assert memory_id not in ids, "Self must be excluded from results"
        assert "neighbor-1" in ids

    async def test_excludes_archived(self):
        """Archived neighbors are dropped by Python post-filter."""
        storage = _make_storage()
        inner_rows = [
            {"id": "memory:active-1",   "score": 0.9, "state": "active",   "tags": [], "created_at": _dt(2025, 6, 1)},
            {"id": "memory:archived-1", "score": 0.8, "state": "archived", "tags": [], "created_at": _dt(2025, 6, 1)},
        ]
        storage._query = AsyncMock(return_value=[inner_rows])
        result = await storage.vector_search_for_memory("some-id", top_k=10)
        ids = [mid for mid, _ in result]
        assert "active-1" in ids
        assert "archived-1" not in ids

    async def test_truncates_to_top_k(self):
        """Result is truncated to top_k after Python post-filter."""
        storage = _make_storage()
        top_k = 3
        inner_rows = [
            {"id": f"memory:n{i}", "score": 0.9 - i * 0.01, "state": "active",
             "tags": [], "created_at": _dt(2025, 6, 1)}
            for i in range(top_k * 3)
        ]
        storage._query = AsyncMock(return_value=[inner_rows])
        result = await storage.vector_search_for_memory("self-id", top_k=top_k)
        assert len(result) <= top_k

    async def test_no_and_predicates_in_sql(self):
        """SQL sent to DB must contain no AND predicates (pure HNSW fetch)."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search_for_memory("some-id", top_k=5)
        sql, _ = storage._query.call_args.args
        assert " AND " not in sql, "vector_search_for_memory SQL must have no AND predicates"

    async def test_hnsw_operator_present(self):
        """HNSW <|N,40|> operator must be in the SQL."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search_for_memory("some-id", top_k=5)
        sql, _ = storage._query.call_args.args
        assert "embedding <|" in sql

    async def test_let_preamble_present(self):
        """Multi-statement query must preserve the LET $vec = ... preamble."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.vector_search_for_memory("some-id", top_k=5)
        sql, _ = storage._query.call_args.args
        assert "LET $vec" in sql


class TestGetRecentActiveIdsFilters:
    """get_recent_active_ids now also accepts tags for symmetry with fts/vector."""

    async def test_tags_filter_appended(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.get_recent_active_ids(10, tags=["project-x"])
        sql, params = storage._query.call_args.args
        assert "tags CONTAINSALL $tags" in sql
        assert params["tags"] == ["project-x"]

    async def test_no_tags_no_containsall(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.get_recent_active_ids(10)
        sql, _ = storage._query.call_args.args
        assert "CONTAINSALL" not in sql

    async def test_type_filter_still_works(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.get_recent_active_ids(5, type_filter="episodic")
        sql, params = storage._query.call_args.args
        assert "memory_type = $type_filter" in sql
        assert params["type_filter"] == "episodic"

    async def test_type_and_tags_combined(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.get_recent_active_ids(5, type_filter="semantic", tags=["ai"])
        sql, params = storage._query.call_args.args
        assert "memory_type = $type_filter" in sql
        assert "tags CONTAINSALL $tags" in sql
        assert params["type_filter"] == "semantic"
        assert params["tags"] == ["ai"]

    async def test_all_filters_combined(self):
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        t0, t1 = _dt(2025, 1, 1), _dt(2025, 7, 1)
        await storage.get_recent_active_ids(
            20,
            type_filter="working",
            tags=["urgent"],
            time_range=(t0, t1),
        )
        sql, params = storage._query.call_args.args
        assert "memory_type = $type_filter" in sql
        assert "tags CONTAINSALL $tags" in sql
        assert "created_at >= $time_start" in sql
        assert "created_at <= $time_end" in sql
        assert params["tags"] == ["urgent"]
        assert params["type_filter"] == "working"

    async def test_state_active_always_in_where(self):
        """The base state = 'active' condition must always be present."""
        storage = _make_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.get_recent_active_ids(10, tags=["foo"])
        sql, _ = storage._query.call_args.args
        assert "state = 'active'" in sql
