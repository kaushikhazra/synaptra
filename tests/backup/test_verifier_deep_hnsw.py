"""Regression test — Bug 2: deep verify HNSW returns 0 results.

Root cause: _deep_verify passed raw JSON rows (with ISO string datetimes) directly
as CONTENT to a SCHEMAFULL table that declares TYPE datetime fields.  SurrealDB
silently rejects records whose datetime fields are ISO strings — no exception is
raised, but the record is not stored.  The HNSW index therefore contains 0 entries.

Fix: _coerce_datetimes() must be applied before the CREATE CONTENT call in
_deep_verify, the same way _import_memory() in importer.py does it.

This test verifies that:
1. _coerce_datetimes is called on each row inside _deep_verify.
2. When a SurrealDB instance actually stores the records (mocked to succeed),
   the deep verify HNSW check does not produce a "0 results" warning.
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384


def _make_embedding() -> list[float]:
    """Return a synthetic 384-float embedding."""
    return [math.sin(i * 0.01) for i in range(EMBEDDING_DIM)]


def _make_memory_row(mem_id: str, iso_datetimes: bool = True) -> dict:
    """Build a synthetic memory row as it appears in memory.ndjson."""
    ts = "2025-01-01T00:00:00+00:00" if iso_datetimes else datetime(2025, 1, 1, tzinfo=timezone.utc)
    return {
        "id": mem_id,
        "content": f"Test memory {mem_id}",
        "memory_type": "episodic",
        "state": "active",
        "importance": 0.5,
        "stability": 1.0,
        "retrievability": 0.9,
        "access_count": 1,
        "created_at": ts,
        "updated_at": ts,
        "last_accessed": ts,
        "source": None,
        "conversation_id": None,
        "tags": [],
        "embedding": _make_embedding(),
    }


def _write_ndjson_backup(backup_dir: Path, rows: list[dict]) -> None:
    """Write a minimal backup with the given memory rows."""
    (backup_dir / "edges").mkdir(exist_ok=True)

    # Write memory.ndjson
    with (backup_dir / "memory.ndjson").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    # Write empty edge files
    for rel in ["causes", "follows", "contradicts", "supports",
                "relates_to", "supersedes", "part_of", "describes"]:
        (backup_dir / "edges" / f"{rel}.ndjson").write_text("")

    # Write empty other tables
    for tbl in ["memory_version", "consolidation_log", "preference"]:
        (backup_dir / f"{tbl}.ndjson").write_text("")

    # Write a minimal schema stub (no HNSW — avoids needing real SurrealDB)
    (backup_dir / "schema.surql").write_text("-- stub schema\n", encoding="utf-8")

    # Write manifest
    import hashlib
    schema_bytes = (backup_dir / "schema.surql").read_bytes()
    manifest = {
        "backup_id": "test-backup",
        "created_at": "2025-01-01T00:00:00Z",
        "cm_version": "test",
        "schema_hash": hashlib.sha256(schema_bytes).hexdigest(),
        "source_backend": "surrealkv",
        "row_counts": {
            "memory": len(rows),
            "memory_version": 0,
            "consolidation_log": 0,
            "preference": 0,
            "edges": {r: 0 for r in ["causes", "follows", "contradicts", "supports",
                                      "relates_to", "supersedes", "part_of", "describes"]},
        },
        "size_bytes": 0,
        "duration_ms": 0,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeepVerifyDatetimeCoercion:
    """_deep_verify must coerce ISO datetime strings before CREATE CONTENT."""

    def test_coerce_datetimes_called_for_each_row(self, tmp_path: Path) -> None:
        """_coerce_datetimes must be invoked for every memory row in _deep_verify."""
        from cognitive_memory.backup import verifier as verifier_mod
        from cognitive_memory.backup.verifier import _deep_verify

        rows = [_make_memory_row(f"mem-{i:04d}") for i in range(3)]
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _write_ndjson_backup(backup_dir, rows)

        manifest = json.loads((backup_dir / "manifest.json").read_text())

        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.use.return_value = None
        mock_db.query.return_value = [{"id": "memory:mem-0000", "score": 1.0}]

        with (
            patch("surrealdb.Surreal", return_value=mock_db),
            patch("cognitive_memory.backup.importer._coerce_datetimes",
                  wraps=__import__(
                      "cognitive_memory.backup.importer", fromlist=["_coerce_datetimes"]
                  )._coerce_datetimes) as mock_coerce,
        ):
            result: dict = {"ok": True, "warnings": [], "errors": [], "exit_code": 0}
            _deep_verify(backup_dir, manifest, result)

        # Must be called once per memory row
        assert mock_coerce.call_count == len(rows), (
            f"_coerce_datetimes called {mock_coerce.call_count} times, "
            f"expected {len(rows)}"
        )

    def test_coerced_content_passed_to_create(self, tmp_path: Path) -> None:
        """The content dict passed to CREATE must have datetime objects, not ISO strings."""
        from cognitive_memory.backup import verifier as verifier_mod
        from cognitive_memory.backup.verifier import _deep_verify

        rows = [_make_memory_row("mem-coerce-check")]
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _write_ndjson_backup(backup_dir, rows)

        manifest = json.loads((backup_dir / "manifest.json").read_text())

        captured_contents: list[dict] = []

        def capture_query(sql: str, params: dict | None = None):
            if params and "content" in params:
                captured_contents.append(params["content"])
            return [{"id": "memory:mem-coerce-check", "score": 1.0}]

        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.use.return_value = None
        mock_db.query.side_effect = capture_query

        with patch("surrealdb.Surreal", return_value=mock_db):
            result: dict = {"ok": True, "warnings": [], "errors": [], "exit_code": 0}
            _deep_verify(backup_dir, manifest, result)

        assert captured_contents, "No CREATE CONTENT call was made"
        content = captured_contents[0]
        for dt_field in ("created_at", "updated_at", "last_accessed"):
            val = content.get(dt_field)
            assert isinstance(val, datetime), (
                f"Field '{dt_field}' should be a datetime object after coercion, "
                f"got {type(val).__name__}: {val!r}"
            )

    def test_hnsw_warning_not_emitted_when_records_created(self, tmp_path: Path) -> None:
        """When records are successfully created, no HNSW 0-results warning should appear."""
        from cognitive_memory.backup import verifier as verifier_mod
        from cognitive_memory.backup.verifier import _deep_verify

        rows = [_make_memory_row(f"mem-{i:04d}") for i in range(5)]
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        _write_ndjson_backup(backup_dir, rows)

        manifest = json.loads((backup_dir / "manifest.json").read_text())

        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.use.return_value = None
        # HNSW query returns results
        mock_db.query.return_value = [{"id": "memory:mem-0000", "score": 0.99}]

        with patch("surrealdb.Surreal", return_value=mock_db):
            result: dict = {"ok": True, "warnings": [], "errors": [], "exit_code": 0}
            _deep_verify(backup_dir, manifest, result)

        hnsw_warnings = [w for w in result["warnings"] if "HNSW" in w and "0 results" in w]
        assert not hnsw_warnings, (
            f"Unexpected HNSW 0-results warning(s): {hnsw_warnings}"
        )


class TestDeepVerifyHNSWZeroResultsRegression:
    """Regression: raw ISO string datetimes previously caused HNSW 0 results."""

    def test_raw_iso_datetimes_do_not_bypass_coercion(self, tmp_path: Path) -> None:
        """Verify that _coerce_datetimes transforms ISO string datetimes to datetime objects.

        This is the direct regression for Bug 2: before the fix, ISO strings were
        passed through as-is, causing SurrealDB to silently reject records.
        """
        from cognitive_memory.backup.importer import _coerce_datetimes

        row = _make_memory_row("regression-mem", iso_datetimes=True)
        # Confirm input has ISO strings
        assert isinstance(row["created_at"], str), "Test setup: created_at should be str"

        coerced = _coerce_datetimes(row)

        for field in ("created_at", "updated_at", "last_accessed"):
            val = coerced[field]
            assert isinstance(val, datetime), (
                f"After _coerce_datetimes, '{field}' must be a datetime, got {type(val).__name__}"
            )

    def test_embedding_preserved_after_coercion(self, tmp_path: Path) -> None:
        """_coerce_datetimes must not alter the embedding field."""
        from cognitive_memory.backup.importer import _coerce_datetimes

        emb = _make_embedding()
        row = _make_memory_row("emb-preserve")
        row["embedding"] = emb

        coerced = _coerce_datetimes(row)

        assert coerced["embedding"] == emb, "Embedding must be unchanged after datetime coercion"
        assert len(coerced["embedding"]) == EMBEDDING_DIM
