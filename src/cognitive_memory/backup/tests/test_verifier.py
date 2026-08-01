"""Tests for the verifier module.

Cases:
- Valid artifact passes.
- Truncated NDJSON fails.
- Wrong embedding dimension exits 6.
- Schema hash mismatch warns; exits 7 with --strict.
"""

from __future__ import annotations

import json
import os
import hashlib
import tempfile
from pathlib import Path

import pytest

from cognitive_memory.backup.exporter import export_backup
from cognitive_memory.backup.verifier import verify_backup, VerifyError


def _write_minimal_backup(backup_dir: Path, memory_count: int = 3, embed_dim: int = 384) -> None:
    """Write a minimal valid backup artifact to backup_dir."""
    import math, random, uuid
    from datetime import datetime, timezone

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Schema
    schema_path = Path(__file__).parent.parent.parent / "schema.surql"
    schema_text = schema_path.read_text(encoding="utf-8")
    (backup_dir / "schema.surql").write_text(schema_text, encoding="utf-8")
    schema_hash = hashlib.sha256(schema_text.encode()).hexdigest()

    # Memory NDJSON
    now = datetime.now(timezone.utc).isoformat()
    memories = []
    with (backup_dir / "memory.ndjson").open("w") as fh:
        for i in range(memory_count):
            embedding = [random.gauss(0, 1) for _ in range(embed_dim)]
            mem = {
                "id": str(uuid.uuid4()),
                "content": f"Memory {i}",
                "memory_type": "semantic",
                "state": "active",
                "importance": 0.5,
                "stability": 5.0,
                "retrievability": 1.0,
                "access_count": 0,
                "source": "test",
                "conversation_id": None,
                "tags": [],
                "embedding": embedding,
                "created_at": now,
                "updated_at": now,
                "last_accessed": now,
            }
            memories.append(mem)
            fh.write(json.dumps(mem) + "\n")

    # Empty tables
    for table in ["memory_version", "consolidation_log", "preference"]:
        (backup_dir / f"{table}.ndjson").write_text("", encoding="utf-8")

    # Empty edge files
    edges_dir = backup_dir / "edges"
    edges_dir.mkdir(exist_ok=True)
    rel_tables = ["causes", "follows", "contradicts", "supports",
                  "relates_to", "supersedes", "part_of", "describes"]
    for rel in rel_tables:
        (edges_dir / f"{rel}.ndjson").write_text("", encoding="utf-8")

    # Manifest
    manifest = {
        "backup_id": "test-backup",
        "created_at": now,
        "cm_version": "test",
        "schema_hash": schema_hash,
        "source_backend": "surrealkv",
        "row_counts": {
            "memory": memory_count,
            "memory_version": 0,
            "consolidation_log": 0,
            "preference": 0,
            "edges": {r: 0 for r in rel_tables},
        },
        "size_bytes": 0,
        "duration_ms": 0,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestVerifyValidArtifact:
    def test_valid_artifact_passes(self, tmp_path):
        backup_dir = tmp_path / "valid-backup"
        _write_minimal_backup(backup_dir, memory_count=5)
        result = verify_backup(backup_dir)
        assert result["ok"] is True
        assert result["exit_code"] == 0

    def test_empty_memory_passes(self, tmp_path):
        """W2 guard: 0-memory backup should pass verify."""
        backup_dir = tmp_path / "empty-backup"
        _write_minimal_backup(backup_dir, memory_count=0)
        result = verify_backup(backup_dir)
        assert result["ok"] is True


class TestVerifyMissingManifest:
    def test_missing_manifest_raises(self, tmp_path):
        backup_dir = tmp_path / "no-manifest"
        backup_dir.mkdir()
        with pytest.raises(VerifyError) as exc_info:
            verify_backup(backup_dir)
        assert exc_info.value.exit_code == 1


class TestVerifyTruncatedNDJSON:
    def test_truncated_ndjson_fails(self, tmp_path):
        backup_dir = tmp_path / "truncated-backup"
        _write_minimal_backup(backup_dir, memory_count=3)

        # Corrupt the memory NDJSON by truncating to partial line
        ndjson_path = backup_dir / "memory.ndjson"
        content = ndjson_path.read_text()
        ndjson_path.write_text(content[:50])  # cut mid-JSON

        # Fix manifest count to 3 but file only has partial line
        result = verify_backup(backup_dir)
        # Should have errors (count mismatch + invalid JSON)
        assert not result["ok"] or result["errors"]


class TestVerifyWrongEmbeddingDim:
    def test_wrong_embedding_dim_exits_6(self, tmp_path):
        backup_dir = tmp_path / "bad-emb"
        _write_minimal_backup(backup_dir, memory_count=3, embed_dim=128)  # wrong dim
        result = verify_backup(backup_dir)
        assert result["exit_code"] == 6
        assert not result["ok"]


class TestVerifySchemaHash:
    def test_schema_hash_mismatch_warns(self, tmp_path):
        backup_dir = tmp_path / "schema-mismatch"
        _write_minimal_backup(backup_dir, memory_count=2)

        # Corrupt the schema hash in manifest
        manifest_path = backup_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))

        result = verify_backup(backup_dir, strict=False)
        # Should warn but not fail
        assert any("hash" in w.lower() for w in result["warnings"]), (
            f"Expected schema hash warning, got: {result['warnings']}"
        )

    def test_schema_hash_mismatch_strict_exits_7(self, tmp_path):
        backup_dir = tmp_path / "schema-strict"
        _write_minimal_backup(backup_dir, memory_count=2)

        manifest_path = backup_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))

        result = verify_backup(backup_dir, strict=True)
        assert result["exit_code"] == 7
        assert not result["ok"]


class TestExporterVerifierRoundTrip:
    """Export a live synthetic DB → verify → assert manifest counts match."""

    def test_export_then_verify(self, surreal_db_with_memories, tmp_path):
        db_path, memories = surreal_db_with_memories

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

        result = verify_backup(backup_dir)
        assert result["ok"] is True, f"Verify failed: {result['errors']}"
        assert result["exit_code"] == 0
