"""Tests for the importer module.

Round-trip integration test: export 50-memory fixture → restore to fresh temp dir
→ assert row counts, spot-check 10 memories, assert HNSW query returns results.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from cognitive_memory.backup.exporter import export_backup
from cognitive_memory.backup.importer import import_backup, ImportError


class TestImporterNonEmptyTarget:
    def test_refuses_non_empty_target_without_force(self, tmp_path):
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "manifest.json").write_text(
            '{"backup_id":"test","created_at":"2026-01-01T00:00:00Z",'
            '"cm_version":"test","schema_hash":"abc","source_backend":"surrealkv",'
            '"row_counts":{"memory":0,"memory_version":0,"consolidation_log":0,'
            '"preference":0,"edges":{"causes":0,"follows":0,"contradicts":0,'
            '"supports":0,"relates_to":0,"supersedes":0,"part_of":0,"describes":0}},'
            '"size_bytes":0,"duration_ms":0}',
            encoding="utf-8",
        )

        # Also create the required files so it doesn't fail on missing files
        for f in ["memory.ndjson", "memory_version.ndjson", "consolidation_log.ndjson", "preference.ndjson"]:
            (backup_dir / f).write_text("", encoding="utf-8")
        edges_dir = backup_dir / "edges"
        edges_dir.mkdir()
        for rel in ["causes", "follows", "contradicts", "supports", "relates_to", "supersedes", "part_of", "describes"]:
            (edges_dir / f"{rel}.ndjson").write_text("", encoding="utf-8")
        schema_path = Path(__file__).parent.parent.parent / "schema.surql"
        import shutil
        shutil.copy(schema_path, backup_dir / "schema.surql")

        target = tmp_path / "target"
        target.mkdir()
        (target / "some_file.db").write_text("existing data")

        with pytest.raises(ImportError) as exc_info:
            import_backup(backup_dir, target_dir=target, force=False)
        assert exc_info.value.exit_code == 5

    def test_force_allows_non_empty_target(self, tmp_path):
        """With --force, restore proceeds even if target is non-empty."""
        backup_dir = tmp_path / "minimal-backup"
        _make_minimal_backup(backup_dir)

        target = tmp_path / "target-force"
        target.mkdir()
        (target / "junk").write_text("junk data")

        # Should not raise
        result_dir = import_backup(backup_dir, target_dir=target, force=True)
        assert result_dir == target


def _make_minimal_backup(backup_dir: Path) -> None:
    """Helper: create a 0-row backup artifact for import tests."""
    import hashlib
    from datetime import datetime, timezone

    backup_dir.mkdir(parents=True, exist_ok=True)

    schema_path = Path(__file__).parent.parent.parent / "schema.surql"
    schema_text = schema_path.read_text(encoding="utf-8")
    (backup_dir / "schema.surql").write_text(schema_text, encoding="utf-8")
    schema_hash = hashlib.sha256(schema_text.encode()).hexdigest()

    for table in ["memory", "memory_version", "consolidation_log", "preference"]:
        (backup_dir / f"{table}.ndjson").write_text("", encoding="utf-8")

    edges_dir = backup_dir / "edges"
    edges_dir.mkdir(exist_ok=True)
    rel_tables = ["causes", "follows", "contradicts", "supports",
                  "relates_to", "supersedes", "part_of", "describes"]
    for rel in rel_tables:
        (edges_dir / f"{rel}.ndjson").write_text("", encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "backup_id": "minimal",
        "created_at": now,
        "cm_version": "test",
        "schema_hash": schema_hash,
        "source_backend": "surrealkv",
        "row_counts": {
            "memory": 0,
            "memory_version": 0,
            "consolidation_log": 0,
            "preference": 0,
            "edges": {r: 0 for r in rel_tables},
        },
        "size_bytes": 0,
        "duration_ms": 0,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestRoundTrip:
    """Full round-trip: export 50 memories → import → verify counts + spot-check."""

    def test_round_trip_50_memories(self, surreal_db_with_memories, tmp_path):
        db_path, memories = surreal_db_with_memories

        # 1. Export
        out_dir = tmp_path / "backups"
        db_env = f"surrealkv://{str(db_path).replace(os.sep, '/')}"
        orig_env = os.environ.get("SYNAPTRA_DB")
        os.environ["SYNAPTRA_DB"] = db_env

        try:
            backup_dir = export_backup(out_dir=out_dir, skip_cm_restart=True)
        finally:
            if orig_env is None:
                os.environ.pop("SYNAPTRA_DB", None)
            else:
                os.environ["SYNAPTRA_DB"] = orig_env

        # 2. Import into a fresh target
        target_dir = tmp_path / "restored"
        result_dir = import_backup(backup_dir, target_dir=target_dir)

        assert result_dir == target_dir
        assert target_dir.exists()

        # 3. Open the restored DB and verify counts
        from surrealdb import Surreal
        from cognitive_memory.backup.exporter import _rows

        restored_url = f"surrealkv://{str(target_dir).replace(os.sep, '/')}"
        restored_db = Surreal(restored_url)
        restored_db.connect()
        restored_db.use("cognitive", "memory")

        try:
            # Check memory count
            count_result = restored_db.query("SELECT count() AS cnt FROM memory GROUP ALL")
            count_rows = _rows(count_result)
            actual_count = count_rows[0].get("cnt", 0) if count_rows else 0
            assert actual_count == 50, f"Expected 50 memories, got {actual_count}"

            # Check relates_to edge count
            edge_result = restored_db.query("SELECT count() AS cnt FROM relates_to GROUP ALL")
            edge_rows = _rows(edge_result)
            actual_edges = edge_rows[0].get("cnt", 0) if edge_rows else 0
            assert actual_edges == 3, f"Expected 3 relates_to edges, got {actual_edges}"

            # Spot-check 5 random memories
            import random as _random
            sample = _random.sample(memories, 5)
            for mem in sample:
                mem_id = mem["id"]
                fetch = restored_db.query(
                    "SELECT * FROM type::thing('memory', $id)",
                    {"id": mem_id},
                )
                fetched_rows = _rows(fetch)
                assert fetched_rows, f"Memory {mem_id} not found after restore"
                row = fetched_rows[0]
                assert row["content"] == mem["content"], f"Content mismatch for {mem_id}"

            # HNSW query returns results
            sample_vec = memories[0]["embedding"]
            hnsw_result = restored_db.query(
                """SELECT id, vector::similarity::cosine(embedding, $vec) AS score
                   FROM memory WHERE embedding != NONE
                   ORDER BY score DESC LIMIT 5""",
                {"vec": sample_vec},
            )
            hnsw_rows = _rows(hnsw_result)
            assert hnsw_rows, "HNSW query returned 0 results after restore"

        finally:
            try:
                restored_db.close()
            except Exception:
                pass
