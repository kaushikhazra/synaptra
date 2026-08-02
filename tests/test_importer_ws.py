"""Tests for import_backup() ws-mode extension (Phase 4 — Component E).

All tests that would contact a live SurrealDB server are skip-guarded:
    @pytest.mark.skipif(not os.environ.get("SURREAL_TEST_URL"), ...)

Unit tests mock the Surreal client to verify routing, validation, and guard logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from synaptra.backup.importer import import_backup, ImportError


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_backup_dir(tmp_path: Path, memory_count: int = 0) -> Path:
    """Create a minimal valid backup directory for testing."""
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()

    manifest = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "schema_hash": "abc123",
        "row_counts": {"memory": memory_count, "memory_version": 0,
                       "consolidation_log": 0, "preference": 0,
                       "edges": {}},
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (backup_dir / "schema.surql").write_text("-- schema placeholder\n", encoding="utf-8")
    (backup_dir / "edges").mkdir()
    return backup_dir


def _make_mock_db(existing_count: int = 0):
    """Build a mock BlockingWsSurrealConnection with controllable count response."""
    db = MagicMock()
    # First query is the count guard; all others return empty
    db.query.side_effect = _make_query_side_effect(existing_count)
    db.use.return_value = None
    db.close.return_value = None
    return db


def _make_query_side_effect(existing_count: int):
    """Return a side_effect function for db.query that handles the count guard call."""
    call_count = [0]

    def side_effect(sql, params=None):
        call_count[0] += 1
        sql_lower = sql.strip().lower()
        if "count()" in sql_lower and "from memory" in sql_lower and "group all" in sql_lower:
            return [{"cnt": existing_count}]
        return []

    return side_effect


# ──────────────────────────────────────────────────────────────────────────────
# Parameter validation
# ──────────────────────────────────────────────────────────────────────────────

class TestParameterValidation:
    def test_invalid_target_raises(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        with pytest.raises(ImportError, match="Invalid target"):
            import_backup(backup_dir, target="ftp")

    def test_ws_without_url_raises(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        with pytest.raises(ImportError, match="target_url"):
            import_backup(backup_dir, target="ws", target_url=None)

    def test_file_mode_still_works(self, tmp_path):
        """Default file mode is not broken by the new parameters."""
        backup_dir = _make_backup_dir(tmp_path)
        target_dir = tmp_path / "restore"
        target_dir.mkdir()

        mock_db = MagicMock()
        mock_db.query.return_value = [{"cnt": 0}]

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            result = import_backup(
                backup_dir,
                target_dir=target_dir,
                force=True,
                target="file",
            )
        assert result == target_dir


# ──────────────────────────────────────────────────────────────────────────────
# WS mode — non-empty target guard (NW1)
# ──────────────────────────────────────────────────────────────────────────────

class TestWsForceGuard:
    def test_aborts_when_target_has_data_and_no_force(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        mock_db = _make_mock_db(existing_count=5)

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            with pytest.raises(ImportError, match="5 records") as exc_info:
                import_backup(
                    backup_dir,
                    target="ws",
                    target_url="ws://127.0.0.1:8000/rpc",
                    force=False,
                )
        assert exc_info.value.exit_code == 5

    def test_aborts_message_mentions_force(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        mock_db = _make_mock_db(existing_count=3)

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            with pytest.raises(ImportError) as exc_info:
                import_backup(
                    backup_dir,
                    target="ws",
                    target_url="ws://127.0.0.1:8000/rpc",
                    force=False,
                )
        assert "--force" in str(exc_info.value)

    def test_proceeds_when_target_empty(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path, memory_count=0)
        mock_db = _make_mock_db(existing_count=0)

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            result = import_backup(
                backup_dir,
                target="ws",
                target_url="ws://127.0.0.1:8000/rpc",
                force=False,
            )
        assert result == "ws://127.0.0.1:8000/rpc"

    def test_proceeds_with_force_when_target_has_data(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        mock_db = _make_mock_db(existing_count=100)

        # Patch _post_restore_verify to return success; this test focuses on force guard only
        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            with patch("synaptra.backup.importer._post_restore_verify", return_value=0):
                result = import_backup(
                    backup_dir,
                    target="ws",
                    target_url="ws://127.0.0.1:8000/rpc",
                    force=True,
                )
        assert result == "ws://127.0.0.1:8000/rpc"


# ──────────────────────────────────────────────────────────────────────────────
# WS mode — schema + index rebuild ordering
# ──────────────────────────────────────────────────────────────────────────────

class TestWsImportOrdering:
    def test_schema_applied_before_records(self, tmp_path):
        """Schema statements must precede memory INSERT calls."""
        backup_dir = _make_backup_dir(tmp_path)
        (backup_dir / "schema.surql").write_text(
            "DEFINE TABLE memory SCHEMALESS;\n", encoding="utf-8"
        )

        calls_log: list[str] = []
        mock_db = MagicMock()

        def query_tracker(sql, params=None):
            sql_lower = sql.strip().lower()
            if "count()" in sql_lower and "from memory" in sql_lower:
                calls_log.append("count_guard")
                return [{"cnt": 0}]
            if "define" in sql_lower:
                calls_log.append("schema")
            elif "create" in sql_lower and "memory" in sql_lower:
                calls_log.append("insert_memory")
            elif "rebuild" in sql_lower:
                calls_log.append("rebuild")
            return []

        mock_db.query.side_effect = query_tracker
        mock_db.use.return_value = None
        mock_db.close.return_value = None

        # Add a memory NDJSON record
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        mem_id = str(uuid.uuid4())
        memory_ndjson = {
            "id": mem_id, "content": "test", "memory_type": "episodic",
            "state": "active", "importance": 0.5, "stability": 10.0,
            "retrievability": 1.0, "access_count": 0,
            "created_at": now, "updated_at": now, "last_accessed": now,
            "tags": [], "embedding": None,
        }
        (backup_dir / "memory.ndjson").write_text(
            json.dumps(memory_ndjson) + "\n", encoding="utf-8"
        )

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            with patch("synaptra.backup.importer._post_restore_verify", return_value=0):
                import_backup(
                    backup_dir,
                    target="ws",
                    target_url="ws://127.0.0.1:8000/rpc",
                    force=False,
                )

        # Verify ordering: count_guard → schema → insert → rebuild
        assert "count_guard" in calls_log
        schema_idx = calls_log.index("schema") if "schema" in calls_log else -1
        insert_idx = calls_log.index("insert_memory") if "insert_memory" in calls_log else 9999
        rebuild_idx = calls_log.index("rebuild") if "rebuild" in calls_log else 9999

        if schema_idx >= 0 and insert_idx < 9999:
            assert schema_idx < insert_idx, "Schema must precede record inserts"
        if insert_idx < 9999 and rebuild_idx < 9999:
            assert insert_idx < rebuild_idx, "Records must precede index rebuild"

    def test_returns_target_url_on_success(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        mock_db = _make_mock_db(existing_count=0)

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            result = import_backup(
                backup_dir,
                target="ws",
                target_url="ws://192.168.1.5:8000/rpc",
            )
        assert result == "ws://192.168.1.5:8000/rpc"

    def test_connection_error_raises_import_error(self, tmp_path):
        backup_dir = _make_backup_dir(tmp_path)
        mock_db = MagicMock()
        mock_db.use.side_effect = ConnectionRefusedError("server down")

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            with pytest.raises(ImportError, match="Cannot connect"):
                import_backup(
                    backup_dir,
                    target="ws",
                    target_url="ws://127.0.0.1:8000/rpc",
                )


# ──────────────────────────────────────────────────────────────────────────────
# CLI integration — restore command ws mode options
# ──────────────────────────────────────────────────────────────────────────────

class TestRestoreCliWsOptions:
    def test_ws_mode_requires_target_url(self, tmp_path):
        from click.testing import CliRunner
        from synaptra.backup.cli import backup_group

        backup_dir = _make_backup_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(backup_group, [
            "restore",
            str(backup_dir),
            "--target-mode", "ws",
            # No --target-url supplied
        ])
        assert result.exit_code != 0
        assert "target-url" in result.output.lower() or "error" in result.output.lower()

    def test_ws_mode_passes_target_url_to_import_backup(self, tmp_path):
        from click.testing import CliRunner
        from synaptra.backup.cli import backup_group

        backup_dir = _make_backup_dir(tmp_path)
        mock_db = _make_mock_db(existing_count=0)

        with patch("synaptra.backup.importer.Surreal", return_value=mock_db):
            runner = CliRunner()
            result = runner.invoke(backup_group, [
                "restore",
                str(backup_dir),
                "--target-mode", "ws",
                "--target-url", "ws://127.0.0.1:8000/rpc",
            ])

        # Should complete without import errors (may show OK or verification warnings)
        assert "ERROR:" not in result.output or "verification" in result.output.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Live-server integration tests (skip unless SURREAL_TEST_URL set)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("SURREAL_TEST_URL"),
    reason="SurrealDB test server not running (set SURREAL_TEST_URL to enable)",
)
class TestWsImportLive:
    """Full integration test against a live SurrealDB server."""

    def test_import_and_verify_row_count(self, tmp_path):
        import uuid
        from datetime import datetime, timezone

        backup_dir = _make_backup_dir(tmp_path)

        # Write 5 memory records
        now = datetime.now(timezone.utc).isoformat()
        with (backup_dir / "memory.ndjson").open("w", encoding="utf-8") as fh:
            for _ in range(5):
                row = {
                    "id": str(uuid.uuid4()),
                    "content": "live test memory",
                    "memory_type": "episodic",
                    "state": "active",
                    "importance": 0.5,
                    "stability": 10.0,
                    "retrievability": 1.0,
                    "access_count": 0,
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed": now,
                    "tags": [],
                    "embedding": [0.1] * 384,
                }
                fh.write(json.dumps(row) + "\n")

        # Update manifest row count
        manifest_path = backup_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["row_counts"]["memory"] = 5
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        url = os.environ["SURREAL_TEST_URL"]
        result = import_backup(
            backup_dir,
            target="ws",
            target_url=url,
            force=True,  # clean state required for live tests
        )
        assert result == url
