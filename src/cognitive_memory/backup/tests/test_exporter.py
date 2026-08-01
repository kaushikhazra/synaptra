"""Tests for the exporter module.

Round-trip: populate a SurrealKV fixture → export → verify manifest counts.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cognitive_memory.backup.exporter import (
    export_backup,
    _extract_id,
    _rows,
    _get_db_path,
)


class TestExtractId:
    def test_plain_string(self):
        assert _extract_id("abc123") == "abc123"

    def test_table_prefix(self):
        assert _extract_id("memory:abc123") == "abc123"

    def test_angle_bracket_uuid(self):
        uid = "3376390d-7ace-49e4-99ea-25bec044cc20"
        assert _extract_id(f"memory:\u27e8{uid}\u27e9") == uid

    def test_surrealkv_recordid_object(self):
        class FakeRecordId:
            def __str__(self):
                return "memory:abcdef"
        assert _extract_id(FakeRecordId()) == "abcdef"


class TestRows:
    def test_flat_list_of_dicts(self):
        data = [{"id": "a"}, {"id": "b"}]
        assert _rows(data) == data

    def test_nested_list(self):
        data = [[{"id": "a"}], [{"id": "b"}]]
        result = _rows(data)
        assert len(result) == 2

    def test_empty(self):
        assert _rows([]) == []
        assert _rows(None) == []


class TestExportBackup:
    """Round-trip test: export a live synthetic DB, verify manifest counts."""

    def test_export_creates_artifact(self, surreal_db_with_memories, tmp_path):
        db_path, memories = surreal_db_with_memories

        out_dir = tmp_path / "backups"
        db_env = f"surrealkv://{str(db_path).replace(os.sep, '/')}"

        # Patch env so exporter opens the same DB
        orig_env = os.environ.get("SYNAPTRA_DB")
        os.environ["SYNAPTRA_DB"] = db_env

        try:
            backup_dir = export_backup(
                out_dir=out_dir,
                name="test-backup",
                skip_cm_restart=True,
            )
        finally:
            if orig_env is None:
                os.environ.pop("SYNAPTRA_DB", None)
            else:
                os.environ["SYNAPTRA_DB"] = orig_env

        assert backup_dir.exists()
        assert (backup_dir / "manifest.json").exists()
        assert (backup_dir / "memory.ndjson").exists()
        assert (backup_dir / "schema.surql").exists()
        assert (backup_dir / "edges" / "relates_to.ndjson").exists()

        manifest = json.loads((backup_dir / "manifest.json").read_text())
        assert manifest["row_counts"]["memory"] == 50
        assert manifest["row_counts"]["edges"]["relates_to"] == 3

    def test_memory_ndjson_has_all_fields(self, surreal_db_with_memories, tmp_path):
        db_path, memories = surreal_db_with_memories

        out_dir = tmp_path / "backups2"
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

        required_fields = [
            "id", "content", "memory_type", "importance", "stability",
            "retrievability", "access_count", "source", "conversation_id",
            "tags", "embedding", "created_at", "updated_at", "last_accessed", "state",
        ]
        with (backup_dir / "memory.ndjson").open() as fh:
            first_line = fh.readline().strip()
        row = json.loads(first_line)
        for field in required_fields:
            assert field in row, f"Missing field: {field}"

    def test_memory_version_id_stripped(self, surreal_db_with_memories, tmp_path):
        """memory_version.memory_id must be a plain UUID, not 'memory:uuid'."""
        from surrealdb import Surreal
        import gc

        db_path, memories = surreal_db_with_memories

        # Open a connection to add a version entry
        db_url = f"surrealkv://{str(db_path).replace(os.sep, '/')}"
        db = Surreal(db_url)
        db.connect()
        db.use("cognitive", "memory")

        mem_id = memories[0]["id"]
        ver_id = "vv-test-001"
        db.query(
            """CREATE type::thing('memory_version', $id) SET
                memory_id = type::thing('memory', $mem_id),
                content = $content,
                metadata = $metadata,
                created_at = $ts
            """,
            {
                "id": ver_id,
                "mem_id": mem_id,
                "content": "test version content",
                "metadata": None,
                "ts": datetime.now(timezone.utc),
            },
        )
        try:
            db.close()
        except Exception:
            pass
        del db
        gc.collect()

        out_dir = tmp_path / "backups3"
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

        version_path = backup_dir / "memory_version.ndjson"
        assert version_path.exists()
        with version_path.open() as fh:
            for line in fh:
                row = json.loads(line.strip())
                mem_id_exported = row.get("memory_id", "")
                # Must NOT contain "memory:" prefix
                assert ":" not in mem_id_exported, (
                    f"memory_version.memory_id should be plain UUID, got: {mem_id_exported!r}"
                )
