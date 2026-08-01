"""Regression test — Bug 1: schema.surql CRLF mismatch.

The exporter must write schema.surql as a binary copy of the source file so that
on Windows the bytes are identical (no LF→CRLF conversion).  The manifest hash
must therefore match a sha256 computed over the raw bytes.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_db(rows: list[dict] | None = None) -> MagicMock:
    """Return a SurrealDB mock that returns empty rows for all queries."""
    db = MagicMock()
    db.connect.return_value = None
    db.use.return_value = None
    db.query.return_value = []
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaWriteBinary:
    """schema.surql must be written as binary copy — no CRLF conversion."""

    def test_schema_bytes_identical_to_source(self, tmp_path: Path) -> None:
        """Bytes written to backup dir must be identical to source schema bytes."""
        from cognitive_memory.backup.exporter import export_backup
        from cognitive_memory.backup import exporter as exporter_mod

        # Locate the real schema
        import cognitive_memory
        schema_src = Path(cognitive_memory.__file__).parent / "schema.surql"
        assert schema_src.exists(), "schema.surql not found in package"

        source_bytes = schema_src.read_bytes()

        # Run export with CM stop/restart skipped
        mock_db = _make_mock_db()
        with patch("surrealdb.Surreal", return_value=mock_db):
            backup_dir = export_backup(
                out_dir=tmp_path / "backups",
                name="test-schema-binary",
                skip_cm_restart=True,
            )

        written_bytes = (backup_dir / "schema.surql").read_bytes()
        assert written_bytes == source_bytes, (
            "schema.surql bytes in backup differ from source — "
            "CRLF conversion may have occurred"
        )

    def test_manifest_hash_uses_raw_bytes(self, tmp_path: Path) -> None:
        """manifest.json schema_hash must match sha256 of the raw bytes, not re-encoded text."""
        from cognitive_memory.backup.exporter import export_backup
        from cognitive_memory.backup import exporter as exporter_mod

        import cognitive_memory
        schema_src = Path(cognitive_memory.__file__).parent / "schema.surql"
        source_bytes = schema_src.read_bytes()
        expected_hash = hashlib.sha256(source_bytes).hexdigest()

        mock_db = _make_mock_db()
        with patch("surrealdb.Surreal", return_value=mock_db):
            backup_dir = export_backup(
                out_dir=tmp_path / "backups",
                name="test-schema-hash",
                skip_cm_restart=True,
            )

        manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_hash"] == expected_hash, (
            f"manifest.schema_hash={manifest['schema_hash'][:16]}... "
            f"expected={expected_hash[:16]}..."
        )

    def test_no_crlf_in_backup_when_source_has_lf_only(self, tmp_path: Path) -> None:
        """If the source schema has LF-only line endings, backup must also have LF-only."""
        from cognitive_memory.backup import exporter as exporter_mod
        from cognitive_memory.backup.exporter import export_backup

        import cognitive_memory
        schema_src = Path(cognitive_memory.__file__).parent / "schema.surql"
        source_bytes = schema_src.read_bytes()

        # Only run this assertion when source file has no CRLF
        if b"\r\n" in source_bytes:
            pytest.skip("Source schema already has CRLF — skipping LF-only assertion")

        mock_db = _make_mock_db()
        with patch("surrealdb.Surreal", return_value=mock_db):
            backup_dir = export_backup(
                out_dir=tmp_path / "backups",
                name="test-no-crlf",
                skip_cm_restart=True,
            )

        written = (backup_dir / "schema.surql").read_bytes()
        assert b"\r\n" not in written, (
            "CRLF found in backup schema.surql — binary write not being used"
        )
