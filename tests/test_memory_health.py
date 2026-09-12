"""Tests for the memory-health feature.

Phase 4: unit tests for the three private helpers and an integration test
that calls await engine.get_health() directly against an in-memory SurrealDB store.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from unittest.mock import MagicMock, patch

import pytest

from synaptra.engine import (
    MemoryEngine,
    _build_decay_report,
    _build_gaps,
    _build_totals,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mem(memory_type: str, stability: float, last_accessed: datetime,
         importance: float = 0.5, mem_id: str | None = None,
         tags: list[str] | None = None) -> dict:
    """Build a minimal active-memory dict as returned by get_active_memories_for_decay."""
    return {
        "id": mem_id or str(uuid.uuid4()),
        "content_preview": "Test content preview",
        "memory_type": memory_type,
        "importance": importance,
        "stability": stability,
        "last_accessed": last_accessed,
        "tags": tags or [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# _build_totals
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildTotals:
    async def test_all_six_types_always_present(self):
        """All 6 memory types appear even when absent from query results (US-1.1)."""
        counts_raw = {
            ("episodic", "active"): 10,
            ("semantic", "active"): 5,
            ("semantic", "archived"): 2,
        }
        result = _build_totals(counts_raw)
        assert set(result["by_type"].keys()) == {
            "working", "episodic", "semantic", "procedural", "identity", "person"
        }

    async def test_active_archived_split_correct(self):
        counts_raw = {
            ("episodic", "active"): 10,
            ("episodic", "archived"): 3,
            ("semantic", "active"): 7,
        }
        result = _build_totals(counts_raw)
        assert result["by_type"]["episodic"] == {"active": 10, "archived": 3}
        assert result["by_type"]["semantic"] == {"active": 7, "archived": 0}
        assert result["by_type"]["working"] == {"active": 0, "archived": 0}

    async def test_grand_total_correct(self):
        counts_raw = {
            ("episodic", "active"): 10,
            ("episodic", "archived"): 3,
            ("semantic", "active"): 7,
            ("semantic", "archived"): 1,
        }
        result = _build_totals(counts_raw)
        assert result["by_state"] == {"active": 17, "archived": 4}
        assert result["total"] == 21

    async def test_empty_input_returns_all_zeros(self):
        result = _build_totals({})
        assert result["total"] == 0
        assert result["by_state"] == {"active": 0, "archived": 0}
        for t in ["working", "episodic", "semantic", "procedural", "identity", "person"]:
            assert result["by_type"][t] == {"active": 0, "archived": 0}


# ──────────────────────────────────────────────────────────────────────────────
# _build_decay_report
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildDecayReport:
    async def test_at_risk_below_threshold_included(self):
        """Memories with R < 0.3 appear in at_risk (US-2.1)."""
        now = _now()
        # Very old episodic memory — low retrievability
        old = _mem("episodic", stability=2.0, last_accessed=now - timedelta(days=90))
        result = _build_decay_report([old], now)
        assert result["at_risk_count"] >= 1
        assert any(e["memory_type"] == "episodic" for e in result["at_risk"])

    async def test_at_risk_above_threshold_excluded(self):
        """Memories with R >= 0.3 do not appear in at_risk."""
        now = _now()
        fresh = _mem("episodic", stability=14.0, last_accessed=now - timedelta(days=1))
        result = _build_decay_report([fresh], now)
        assert result["at_risk_count"] == 0
        assert result["at_risk"] == []

    async def test_identity_excluded_from_at_risk(self):
        """Identity memories never appear in at_risk even if R < 0.3 (D4)."""
        now = _now()
        # Artificially low stability to force R < 0.3
        old_identity = _mem("identity", stability=0.01, last_accessed=now - timedelta(days=90))
        result = _build_decay_report([old_identity], now)
        assert result["at_risk_count"] == 0
        assert all(e["memory_type"] != "identity" for e in result["at_risk"])

    async def test_at_risk_sorted_ascending_by_retrievability(self):
        """at_risk list is sorted lowest-R first."""
        now = _now()
        m1 = _mem("episodic", stability=2.0, last_accessed=now - timedelta(days=90))
        m2 = _mem("working", stability=0.5, last_accessed=now - timedelta(days=30))
        result = _build_decay_report([m1, m2], now)
        if len(result["at_risk"]) >= 2:
            rs = [e["retrievability"] for e in result["at_risk"]]
            assert rs == sorted(rs)

    async def test_at_risk_capped_at_50(self):
        """at_risk list is capped at 50 entries (D6); at_risk_count reflects true total."""
        now = _now()
        # Create 60 at-risk episodic memories
        mems = [_mem("episodic", stability=0.1, last_accessed=now - timedelta(days=100))
                for _ in range(60)]
        result = _build_decay_report(mems, now)
        assert len(result["at_risk"]) == 50
        assert result["at_risk_count"] == 60

    async def test_all_six_types_present_in_by_type(self):
        """All 6 types always present in by_type (US-2.2)."""
        now = _now()
        mems = [_mem("episodic", stability=2.0, last_accessed=now)]
        result = _build_decay_report(mems, now)
        for t in ["working", "episodic", "semantic", "procedural", "identity", "person"]:
            assert t in result["by_type"]

    async def test_types_with_no_active_have_null_averages(self):
        """Types absent from active memories get null averages (US-2.2)."""
        now = _now()
        mems = [_mem("episodic", stability=2.0, last_accessed=now)]
        result = _build_decay_report(mems, now)
        for t in ["working", "semantic", "procedural", "identity", "person"]:
            entry = result["by_type"][t]
            assert entry["count"] == 0
            assert entry["avg_retrievability"] is None
            assert entry["avg_importance"] is None
            assert entry["avg_stability"] is None

    async def test_per_type_averages_correct(self):
        """Per-type averages are computed correctly."""
        now = _now()
        m1 = _mem("semantic", stability=14.0, importance=0.6, last_accessed=now - timedelta(days=1))
        m2 = _mem("semantic", stability=20.0, importance=0.8, last_accessed=now - timedelta(days=2))
        result = _build_decay_report([m1, m2], now)
        entry = result["by_type"]["semantic"]
        assert entry["count"] == 2
        assert entry["avg_importance"] == round(mean([0.6, 0.8]), 4)
        assert entry["avg_stability"] == round(mean([14.0, 20.0]), 4)

    async def test_at_risk_entry_schema(self):
        """Each at-risk entry has all 7 fields required by US-2.1."""
        now = _now()
        old = _mem("episodic", stability=2.0, last_accessed=now - timedelta(days=90))
        result = _build_decay_report([old], now)
        assert result["at_risk_count"] >= 1
        entry = result["at_risk"][0]
        for field in ("id", "content_preview", "memory_type", "retrievability",
                      "last_accessed", "stability", "tags"):
            assert field in entry, f"Missing field: {field}"
        assert isinstance(entry["id"], str)
        assert isinstance(entry["content_preview"], str)
        assert isinstance(entry["memory_type"], str)
        assert isinstance(entry["retrievability"], float)
        assert isinstance(entry["last_accessed"], str)  # ISO 8601 string
        assert isinstance(entry["stability"], float)
        assert isinstance(entry["tags"], list)
        # retrievability is rounded to 4 decimal places
        assert entry["retrievability"] == round(entry["retrievability"], 4)


# ──────────────────────────────────────────────────────────────────────────────
# _build_gaps
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildGaps:
    def _make_totals(self, active_by_type: dict[str, int]) -> dict:
        """Build a minimal totals dict for testing _build_gaps."""
        by_type = {t: {"active": active_by_type.get(t, 0), "archived": 0}
                   for t in ["working", "episodic", "semantic", "procedural", "identity", "person"]}
        total_active = sum(v["active"] for v in by_type.values())
        return {
            "by_type": by_type,
            "by_state": {"active": total_active, "archived": 0},
            "total": total_active,
        }

    async def test_empty_types_detected(self):
        """Types with zero active memories are listed in empty_types (US-4.1)."""
        totals = self._make_totals({"episodic": 50, "semantic": 30})
        result = _build_gaps(totals, [], 0)
        assert "working" in result["empty_types"]
        assert "procedural" in result["empty_types"]
        assert "episodic" not in result["empty_types"]

    async def test_sparse_threshold_boundary_below(self):
        """Type at exactly 4.9% is sparse."""
        # 100 total active: semantic=50, episodic=46, working=4 → working = 4% < 5% → sparse
        totals = self._make_totals({"semantic": 50, "episodic": 46, "working": 4})
        result = _build_gaps(totals, [], 0)
        sparse_names = [e["type"] for e in result["sparse_types"]]
        assert "working" in sparse_names

    async def test_sparse_threshold_boundary_at_5_pct_not_sparse(self):
        """Type at exactly 5% is not sparse (pct < 5.0 is False at exactly 5%)."""
        # semantic=95, working=5, total=100 → working = 5.0% → NOT sparse
        totals = self._make_totals({"semantic": 95, "working": 5})
        result = _build_gaps(totals, [], 0)
        sparse_names = [e["type"] for e in result["sparse_types"]]
        assert "working" not in sparse_names

    async def test_identity_excluded_from_sparse(self):
        """identity is never in sparse_types regardless of count (D3)."""
        # identity = 1 out of 100 → would be 1% but identity is excluded
        totals = self._make_totals({"semantic": 99, "identity": 1})
        result = _build_gaps(totals, [], 0)
        sparse_names = [e["type"] for e in result["sparse_types"]]
        assert "identity" not in sparse_names

    async def test_top_tags_capped_at_20(self):
        """top_tags returns at most 20 entries (D7)."""
        tag_rows = [[f"tag{i}"] for i in range(30)]  # 30 distinct tags
        totals = self._make_totals({"semantic": 30})
        result = _build_gaps(totals, tag_rows, 0)
        assert len(result["tag_coverage"]["top_tags"]) == 20

    async def test_top_tags_sorted_descending(self):
        """top_tags is sorted by count descending."""
        tag_rows = [["a", "a", "b"], ["a", "c"], ["b"]]
        totals = self._make_totals({"semantic": 3})
        result = _build_gaps(totals, tag_rows, 0)
        counts = [e["count"] for e in result["tag_coverage"]["top_tags"]]
        assert counts == sorted(counts, reverse=True)

    async def test_total_unique_tags_count(self):
        """total_unique_tags reflects distinct tag count."""
        tag_rows = [["a", "b"], ["b", "c"], ["a"]]
        totals = self._make_totals({"semantic": 3})
        result = _build_gaps(totals, tag_rows, 0)
        assert result["tag_coverage"]["total_unique_tags"] == 3  # a, b, c

    async def test_untagged_count_passed_through(self):
        """untagged_count in tag_coverage comes from the untagged_count parameter."""
        totals = self._make_totals({"semantic": 10})
        result = _build_gaps(totals, [], 7)
        assert result["tag_coverage"]["untagged_count"] == 7


# ──────────────────────────────────────────────────────────────────────────────
# Integration test — await engine.get_health() against in-memory SurrealDB
# ──────────────────────────────────────────────────────────────────────────────

class TestGetHealthIntegration:
    """Calls await engine.get_health() against a real in-memory SurrealDB instance."""

    async def test_engine_accepts_injected_storage(self, tmp_path: Path):
        class FakeStorage:
            def __init__(self):
                self.closed = False

            def get_memories_by_ids(self, ids: list[str]):
                return []

            def get_recent_active_ids(
                self,
                limit: int,
                type_filter: str | None = None,
                time_range: tuple[datetime, datetime] | None = None,
            ):
                return []

            def get_config(self, key: str):
                return None

            def set_config(self, key: str, value):
                return None

            def get_all_config(self) -> dict[str, object]:
                return {}

            def close(self) -> None:
                self.closed = True

        fake_storage = FakeStorage()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("decay:\n  growth_factor: 4.0\n", encoding="utf-8")

        engine = MemoryEngine(storage=fake_storage, config_path=config_path)

        assert engine.storage is fake_storage
        assert engine.config._storage is fake_storage
        assert engine.config.get("decay.growth_factor") == 4.0

        engine.close()
        assert fake_storage.closed is True

    @pytest.fixture
    def engine(self):
        return MemoryEngine(db_path="mem://")

    async def test_health_returns_all_sections(self, engine):
        """Full health report has all required top-level keys (US-6.1)."""
        report = await engine.get_health()
        for key in ("generated_at", "totals", "decay", "orphans", "gaps", "consolidation"):
            assert key in report, f"Missing key: {key}"

    async def test_totals_all_types_present(self, engine):
        """by_type always contains all 6 types even on empty store (US-1.1)."""
        report = await engine.get_health()
        by_type = report["totals"]["by_type"]
        for t in ["working", "episodic", "semantic", "procedural", "identity", "person"]:
            assert t in by_type

    async def test_consolidation_never_run_on_empty_store(self, engine):
        """Consolidation section shows never_run=True when no consolidation has run (US-5.1)."""
        report = await engine.get_health()
        assert report["consolidation"]["never_run"] is True
        assert report["consolidation"]["last_run_at"] is None
        assert report["consolidation"]["last_run_summary"] is None

    async def test_orphan_no_tags_detected(self, engine):
        """Memory stored without tags appears in orphans.no_tags (US-3.1)."""
        # Store a memory without tags
        await engine.store_memory(
            "Untagged test memory for health check", tags=[], rater="test:pytest"
        )
        report = await engine.get_health()
        assert report["orphans"]["no_tags_count"] >= 1

    async def test_orphan_no_tags_count_correct(self, engine):
        """no_tags_count matches the actual number of untagged active memories (US-3.1)."""
        await engine.store_memory("Untagged A", tags=[], rater="test:pytest")
        await engine.store_memory("Untagged B", tags=[], rater="test:pytest")
        await engine.store_memory("Tagged C", tags=["mytag"], rater="test:pytest")
        report = await engine.get_health()
        assert report["orphans"]["no_tags_count"] == 2

    async def test_orphan_no_relations_detected(self, engine):
        """Isolated memory (no relationships) appears in orphans.no_relations (US-3.2)."""
        await engine.store_memory(
            "Isolated memory with unique content zzz999",
            tags=["isolated"],
            rater="test:pytest",
        )
        report = await engine.get_health()
        # The store is fresh — the stored memory is the only one, so it cannot
        # auto-link to anything.  At least 1 unconnected orphan must be reported.
        count = report["orphans"]["no_relations_count"]
        assert isinstance(count, int)
        assert count >= 1

    async def test_decay_by_type_present_for_all_types(self, engine):
        """decay.by_type always has all 6 types (US-2.2)."""
        report = await engine.get_health()
        for t in ["working", "episodic", "semantic", "procedural", "identity", "person"]:
            assert t in report["decay"]["by_type"]

    async def test_at_risk_is_list(self, engine):
        """decay.at_risk is always a list (US-2.1)."""
        report = await engine.get_health()
        assert isinstance(report["decay"]["at_risk"], list)
        assert isinstance(report["decay"]["at_risk_count"], int)

    async def test_gaps_section_structure(self, engine):
        """gaps section has the expected sub-keys (US-4.1, US-4.2)."""
        report = await engine.get_health()
        gaps = report["gaps"]
        assert "empty_types" in gaps
        assert "sparse_types" in gaps
        assert "tag_coverage" in gaps
        tc = gaps["tag_coverage"]
        assert "untagged_count" in tc
        assert "top_tags" in tc
        assert "total_unique_tags" in tc

    async def test_generated_at_is_iso_timestamp(self, engine):
        """generated_at is a valid ISO timestamp string (US-6.1)."""
        report = await engine.get_health()
        # Should parse without error
        dt = datetime.fromisoformat(report["generated_at"])
        assert dt.tzinfo is not None

    async def test_consolidation_grouping_by_date(self, engine):
        """Consolidation section reflects actual log entries grouped by date (US-5.1)."""
        from synaptra.models import ConsolidationLogEntry

        now = datetime.now(timezone.utc)
        # Insert 2 promote + 1 archive on today's date
        for action in ("promote", "promote", "archive"):
            await engine.storage.insert_consolidation_log(ConsolidationLogEntry(
                id=str(uuid.uuid4()),
                action=action,
                source_ids=[str(uuid.uuid4())],
                target_id=str(uuid.uuid4()),
                reason="test",
                created_at=now,
            ))

        report = await engine.get_health()
        consolidation = report["consolidation"]
        assert consolidation["never_run"] is False
        assert consolidation["last_run_at"] is not None
        # Verify it parses as a valid timestamp
        datetime.fromisoformat(consolidation["last_run_at"])
        summary = consolidation["last_run_summary"]
        assert summary["promote"] == 2
        assert summary["archive"] == 1
        assert summary["merge"] == 0
        assert summary["flag_contradiction"] == 0

    async def test_tag_coverage_untagged_matches_orphans_no_tags(self, engine):
        """gaps.tag_coverage.untagged_count equals orphans.no_tags_count (US-4.2)."""
        await engine.store_memory("Untagged first", tags=[], rater="test:pytest")
        await engine.store_memory("Untagged second", tags=[], rater="test:pytest")
        await engine.store_memory(
            "Tagged with sometag", tags=["sometag"], rater="test:pytest"
        )
        report = await engine.get_health()
        assert (
            report["gaps"]["tag_coverage"]["untagged_count"]
            == report["orphans"]["no_tags_count"]
        )
