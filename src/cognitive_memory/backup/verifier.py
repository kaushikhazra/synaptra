"""CM Backup Verifier.

Light verification: manifest + NDJSON integrity + schema hash + embedding spot-check.
Deep verification: load into temp SurrealKV, apply schema, import, REBUILD indexes, HNSW query.

Exit codes (returned as int, not sys.exit — callers decide when to exit):
  0   All checks passed
  6   Embedding dimension or value error
  7   Schema hash mismatch (only when strict=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REL_TABLES = [
    "causes", "follows", "contradicts", "supports",
    "relates_to", "supersedes", "part_of", "describes",
]

PLAIN_TABLES = ["memory", "memory_version", "consolidation_log", "preference"]

# Required fields per NDJSON file
REQUIRED_FIELDS: dict[str, list[str]] = {
    "memory": ["id", "content", "memory_type", "importance", "stability",
               "retrievability", "access_count", "created_at", "updated_at",
               "last_accessed", "state"],
    "memory_version": ["id", "memory_id", "content", "created_at"],
    "consolidation_log": ["id", "action", "source_ids", "reason", "created_at"],
    "preference": ["id", "val", "updated_at"],
}

EDGE_REQUIRED_FIELDS = ["in", "out", "strength", "created_at"]

EMBEDDING_DIM = 384


class VerifyError(Exception):
    """Raised when verification finds a critical problem."""
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def verify_backup(
    backup_dir: Path,
    strict: bool = False,
    deep: bool = False,
) -> dict[str, Any]:
    """Verify a CM backup artifact.

    Args:
        backup_dir: Path to the backup directory.
        strict:     If True, schema hash mismatch causes exit_code=7.
        deep:       If True, load into a temp SurrealKV instance and run HNSW query.

    Returns:
        Dict with keys: ok (bool), warnings (list[str]), errors (list[str]),
        exit_code (int — 0 on pass, 6/7 on specific failures).

    Raises:
        VerifyError: on critical failures (callers should exit with error.exit_code).
    """
    result: dict[str, Any] = {"ok": True, "warnings": [], "errors": [], "exit_code": 0}

    # 1. Check manifest
    manifest = _load_manifest(backup_dir)

    # 2. Check all referenced files exist
    _check_files_exist(backup_dir, manifest, result)

    # 3. Stream each NDJSON file — line count + field validation
    _check_ndjson_files(backup_dir, manifest, result)

    # 4. Schema hash check
    _check_schema_hash(backup_dir, manifest, strict, result)

    # 5. Spot-check 10 random memory rows for embedding integrity
    _check_embeddings(backup_dir, manifest, result)

    # 6. Deep mode
    if deep:
        _deep_verify(backup_dir, manifest, result)

    # If any errors accumulated, mark not ok
    if result["errors"]:
        result["ok"] = False

    return result


def _load_manifest(backup_dir: Path) -> dict:
    """Load and parse manifest.json; raise VerifyError if missing or corrupt."""
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        raise VerifyError(
            f"manifest.json not found in {backup_dir}", exit_code=1
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise VerifyError(f"manifest.json is not valid JSON: {e}", exit_code=1)
    return manifest


def _check_files_exist(backup_dir: Path, manifest: dict, result: dict) -> None:
    """Assert all referenced NDJSON files exist in the artifact directory."""
    missing = []
    for table in PLAIN_TABLES:
        if not (backup_dir / f"{table}.ndjson").exists():
            missing.append(f"{table}.ndjson")

    edges_dir = backup_dir / "edges"
    for rel in REL_TABLES:
        if not (edges_dir / f"{rel}.ndjson").exists():
            missing.append(f"edges/{rel}.ndjson")

    if not (backup_dir / "schema.surql").exists():
        missing.append("schema.surql")

    if missing:
        result["errors"].append(f"Missing files: {', '.join(missing)}")


def _check_ndjson_files(backup_dir: Path, manifest: dict, result: dict) -> None:
    """Stream each NDJSON file, validate each line, compare counts to manifest."""
    row_counts = manifest.get("row_counts", {})

    for table in PLAIN_TABLES:
        path = backup_dir / f"{table}.ndjson"
        if not path.exists():
            continue
        required = REQUIRED_FIELDS.get(table, [])
        count, errors = _validate_ndjson(path, required)
        result["errors"].extend(errors)

        expected = row_counts.get(table)
        if expected is not None and count != expected:
            result["errors"].append(
                f"{table}.ndjson: expected {expected} rows, found {count}"
            )

    edges_dir = backup_dir / "edges"
    edge_counts = row_counts.get("edges", {})
    for rel in REL_TABLES:
        path = edges_dir / f"{rel}.ndjson"
        if not path.exists():
            continue
        count, errors = _validate_ndjson(path, EDGE_REQUIRED_FIELDS)
        result["errors"].extend(errors)

        expected = edge_counts.get(rel)
        if expected is not None and count != expected:
            result["errors"].append(
                f"edges/{rel}.ndjson: expected {expected} rows, found {count}"
            )


def _validate_ndjson(path: Path, required_fields: list[str]) -> tuple[int, list[str]]:
    """Stream an NDJSON file. Returns (line_count, error_list)."""
    errors: list[str] = []
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path.name}:{lineno} invalid JSON: {e}")
                continue
            for field in required_fields:
                if field not in obj:
                    errors.append(f"{path.name}:{lineno} missing field '{field}'")
            count += 1
    return count, errors


def _check_schema_hash(backup_dir: Path, manifest: dict, strict: bool, result: dict) -> None:
    """Compare schema.surql sha256 against manifest. Warn or fail with exit_code=7."""
    schema_path = backup_dir / "schema.surql"
    if not schema_path.exists():
        return
    actual_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    manifest_hash = manifest.get("schema_hash", "")
    if actual_hash != manifest_hash:
        msg = (
            f"Schema hash mismatch: manifest={manifest_hash[:16]}... "
            f"actual={actual_hash[:16]}..."
        )
        if strict:
            result["errors"].append(msg)
            result["exit_code"] = 7
        else:
            result["warnings"].append(msg)

    # Also compare against current bundled CM schema (schema evolution check)
    current_schema_path = Path(__file__).parent.parent / "schema.surql"
    if current_schema_path.exists():
        current_hash = hashlib.sha256(current_schema_path.read_bytes()).hexdigest()
        if actual_hash != current_hash:
            msg2 = (
                "Backup schema differs from current CM schema "
                "(backup may be from an older CM version — apply schema migrations after restore)."
            )
            result["warnings"].append(msg2)


def _check_embeddings(backup_dir: Path, manifest: dict, result: dict) -> None:
    """Spot-check 10 random memory rows: embedding length == 384 and all floats finite.

    W2 guard: skip HNSW check when memory count is 0.
    """
    memory_ndjson = backup_dir / "memory.ndjson"
    if not memory_ndjson.exists():
        return

    rows = _sample_ndjson_rows(memory_ndjson, n=10)
    if not rows:
        return

    for row in rows:
        embedding = row.get("embedding")
        if embedding is None:
            continue  # embedding is optional<array<float>> — None is valid
        if not isinstance(embedding, list):
            result["errors"].append(
                f"memory {row.get('id', '?')}: embedding is not a list"
            )
            result["exit_code"] = 6
            continue
        if len(embedding) != EMBEDDING_DIM:
            result["errors"].append(
                f"memory {row.get('id', '?')}: embedding length {len(embedding)} != {EMBEDDING_DIM}"
            )
            result["exit_code"] = 6
            continue
        bad = [i for i, v in enumerate(embedding) if not isinstance(v, (int, float)) or not math.isfinite(v)]
        if bad:
            result["errors"].append(
                f"memory {row.get('id', '?')}: embedding has non-finite floats at positions {bad[:5]}"
            )
            result["exit_code"] = 6


def _sample_ndjson_rows(path: Path, n: int) -> list[dict]:
    """Read all rows from an NDJSON file and return up to n random samples."""
    all_rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if len(all_rows) <= n:
        return all_rows
    return random.sample(all_rows, n)


def _deep_verify(backup_dir: Path, manifest: dict, result: dict) -> None:
    """Load artifact into a temp SurrealKV instance and run a sample HNSW query.

    W2 guard: HNSW assertion is skipped when memory count == 0.
    """
    try:
        from surrealdb import Surreal
    except ImportError:
        result["warnings"].append("surrealdb package not available; deep verify skipped.")
        return

    import os
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="cm-backup-deepverify-")
    try:
        db_url = f"surrealkv://{tmpdir.replace(os.sep, '/')}/deep"
        db = Surreal(db_url)
        db.connect()
        db.use("cognitive", "memory")

        # Apply schema
        schema_text = (backup_dir / "schema.surql").read_text(encoding="utf-8")
        for stmt in schema_text.split(";"):
            lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                try:
                    db.query(clean)
                except Exception as e:
                    logger.debug("Deep verify schema stmt failed: %s", e)

        # Import memories (for HNSW query)
        # IMPORTANT: datetime fields (created_at, updated_at, last_accessed) must be
        # datetime objects — SurrealDB SCHEMAFULL TYPE datetime silently rejects ISO
        # strings, causing the record not to be created (no exception raised).
        from .importer import _coerce_datetimes, _parse_dt_for_surreal
        memory_ndjson = backup_dir / "memory.ndjson"
        mem_count = 0
        if memory_ndjson.exists():
            with memory_ndjson.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    mem_id = row.get("id", "")
                    coerced = _coerce_datetimes(row)
                    try:
                        db.query(
                            "CREATE type::thing('memory', $id) CONTENT $content",
                            {"id": mem_id, "content": coerced},
                        )
                        mem_count += 1
                    except Exception:
                        pass

        # Import edges
        edges_dir = backup_dir / "edges"
        if edges_dir.exists():
            for rel in REL_TABLES:
                edge_path = edges_dir / f"{rel}.ndjson"
                if not edge_path.exists():
                    continue
                with edge_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        try:
                            # Use LET variables — type::thing() inline in RELATE
                            # is not supported by the embedded SurrealDB Python SDK.
                            # IMPORTANT: created_at must be a datetime object — ISO
                            # strings silently fail SCHEMAFULL TYPE datetime validation.
                            db.query(
                                f"""LET $from = type::thing('memory', $src);
                                    LET $to = type::thing('memory', $tgt);
                                    RELATE $from->{rel}->$to
                                    SET strength = $strength, created_at = $created_at""",
                                {
                                    "src": row.get("in", ""),
                                    "tgt": row.get("out", ""),
                                    "strength": row.get("strength", 1.0),
                                    "created_at": _parse_dt_for_surreal(row.get("created_at")),
                                },
                            )
                        except Exception:
                            pass

        # REBUILD indexes
        try:
            db.query("REBUILD INDEX idx_memory_embedding ON memory")
        except Exception as e:
            logger.debug("REBUILD INDEX idx_memory_embedding failed: %s", e)
        try:
            db.query("REBUILD INDEX idx_memory_fts ON memory")
        except Exception as e:
            logger.debug("REBUILD INDEX idx_memory_fts failed: %s", e)

        # W2 guard: only run HNSW query if there are memories
        manifest_memory_count = manifest.get("row_counts", {}).get("memory", 0)
        if manifest_memory_count > 0 and mem_count > 0:
            # Sample a random memory's embedding and query for similar
            rows = _sample_ndjson_rows(memory_ndjson, n=1)
            if rows and rows[0].get("embedding"):
                sample_vec = rows[0]["embedding"]
                try:
                    query_result = db.query(
                        """SELECT id, vector::similarity::cosine(embedding, $vec) AS score
                           FROM memory WHERE embedding != NONE
                           ORDER BY score DESC LIMIT 5""",
                        {"vec": sample_vec},
                    )
                    from .exporter import _rows as _exporter_rows
                    hnsw_rows = _exporter_rows(query_result)
                    if not hnsw_rows:
                        result["warnings"].append(
                            "Deep verify: HNSW similarity query returned 0 results."
                        )
                    else:
                        logger.info(
                            "Deep verify HNSW query returned %d result(s).", len(hnsw_rows)
                        )
                except Exception as e:
                    result["warnings"].append(f"Deep verify: HNSW query failed: {e}")

        try:
            db.close()
        except Exception:
            pass

    except Exception as e:
        result["warnings"].append(f"Deep verify failed with exception: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
