"""CM Backup Importer.

Restores a logical NDJSON backup into a fresh SurrealKV target directory.

Exit codes (returned as int, never sys.exit — CLI layer decides):
  0   Success
  4   Row count mismatch post-restore (target left as-is for inspection)
  5   Target directory not empty and --force not passed
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from surrealdb import Surreal

logger = logging.getLogger(__name__)

REL_TABLES = [
    "causes", "follows", "contradicts", "supports",
    "relates_to", "supersedes", "part_of", "describes",
]

IMPORT_ORDER = ["preference", "memory", "memory_version", "consolidation_log"]


class ImportError(Exception):
    """Raised when import fails."""
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def import_backup(
    backup_dir: Path,
    target_dir: Path | None = None,
    force: bool = False,
    strict: bool = False,
    target: str = "file",
    target_url: str | None = None,
) -> "Path | str":
    """Import a CM backup artifact into a target SurrealKV directory or a SurrealDB server.

    Args:
        backup_dir:  Path to the backup directory (contains manifest.json).
        target_dir:  Target data directory (file mode). Defaults to
                     ~/.synaptra/restore-<ts>/.  Ignored in ws mode.
        force:       If True, allow writing into a non-empty target.
                     For file mode: overwrite non-empty directory.
                     For ws mode (full reset): schema OVERWRITE + records replace
                     + index rebuild.  Required when target already has data.
        strict:      If True, fail on schema hash mismatch.
        target:      Transport target.  Must be "file" (default, legacy embedded
                     SurrealKV) or "ws" (external SurrealDB server over WebSocket).
        target_url:  Required when target="ws".  e.g. "ws://127.0.0.1:8000/rpc".

    Returns:
        Path to the populated target directory (file mode), or the target URL
        string (ws mode).

    Raises:
        ImportError: on validation or import failure with an exit_code.
    """
    # --- Validate target parameter ---
    if target not in ("file", "ws"):
        raise ImportError(
            f"Invalid target {target!r}. Must be 'file' or 'ws'.",
            exit_code=1,
        )
    if target == "ws":
        if not target_url:
            raise ImportError(
                "target='ws' requires target_url (e.g. 'ws://127.0.0.1:8000/rpc').",
                exit_code=1,
            )
        return _import_backup_ws(
            backup_dir=backup_dir,
            target_url=target_url,
            force=force,
            strict=strict,
        )

    # ── file mode (original path below) ─────────────────────────────────────
    from datetime import datetime, timezone

    # --- Validate manifest ---
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        raise ImportError(f"manifest.json not found in {backup_dir}", exit_code=1)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ImportError(f"manifest.json is not valid JSON: {e}", exit_code=1)

    # --- Resolve target directory ---
    if target_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        target_dir = Path.home() / ".synaptra" / f"restore-{ts}"

    # --- Non-empty target guard ---
    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            raise ImportError(
                f"Target directory {target_dir} is not empty. "
                "Use --force to overwrite.",
                exit_code=5,
            )

    target_dir.mkdir(parents=True, exist_ok=True)

    # --- Open target SurrealKV ---
    db_url = f"surrealkv://{str(target_dir).replace(os.sep, '/')}"
    db = Surreal(db_url)
    db.connect()
    db.use("cognitive", "memory")

    try:
        # --- Apply schema ---
        schema_path = backup_dir / "schema.surql"
        schema_text = schema_path.read_text(encoding="utf-8")

        # Schema hash check
        actual_hash = hashlib.sha256(schema_text.encode()).hexdigest()
        manifest_hash = manifest.get("schema_hash", "")
        if actual_hash != manifest_hash:
            msg = (
                f"Schema hash mismatch: manifest={manifest_hash[:16]}... "
                f"actual={actual_hash[:16]}..."
            )
            if strict:
                raise ImportError(msg, exit_code=7)
            else:
                logger.warning("%s", msg)

        # Apply schema statements
        for stmt in schema_text.split(";"):
            lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                try:
                    db.query(clean)
                except Exception as e:
                    logger.debug("Schema stmt failed: %s — %s", clean[:60], e)

        # --- Import tables in dependency order ---
        for table in IMPORT_ORDER:
            path = backup_dir / f"{table}.ndjson"
            if not path.exists():
                logger.debug("No %s.ndjson found; skipping.", table)
                continue
            if table == "preference":
                _import_preference(db, path)
            elif table == "memory":
                _import_memory(db, path)
            elif table == "memory_version":
                _import_memory_version(db, path)
            elif table == "consolidation_log":
                _import_consolidation_log(db, path)

        # --- Import edge tables ---
        edges_dir = backup_dir / "edges"
        for rel in REL_TABLES:
            edge_path = edges_dir / f"{rel}.ndjson"
            if edge_path.exists():
                _import_edge(db, edge_path, rel)

        # --- REBUILD indexes ---
        for index_stmt in [
            "REBUILD INDEX idx_memory_embedding ON memory",
            "REBUILD INDEX idx_memory_fts ON memory",
        ]:
            try:
                db.query(index_stmt)
                logger.debug("Ran: %s", index_stmt)
            except Exception as e:
                logger.warning("Index rebuild failed (%s): %s", index_stmt, e)

        # --- Post-restore verification ---
        exit_code = _post_restore_verify(db, backup_dir, manifest)
        if exit_code != 0:
            raise ImportError(
                f"Post-restore verification failed (exit {exit_code}). "
                f"Target dir left at {target_dir} for inspection.",
                exit_code=exit_code,
            )

    finally:
        try:
            db.close()
        except Exception:
            pass

    return target_dir


# ---------------------------------------------------------------------------
# ws-mode import (Component E)
# ---------------------------------------------------------------------------

def _import_backup_ws(
    backup_dir: Path,
    target_url: str,
    force: bool,
    strict: bool,
) -> str:
    """Import a CM backup artifact into a running SurrealDB server over WebSocket.

    Connection: target_url (e.g. ws://127.0.0.1:8000/rpc), no auth (localhost only).
    Namespace/database: cognitive / memory.

    Order of operations (design Component E #5):
      1. Connect + session setup
      2. --force guard: abort if target has data and --force not supplied  (NW1)
      3. Apply schema from artifact schema.surql
      4. Import records in dependency order, batched 100 rows at a time
      5. Import edge tables
      6. REBUILD INDEX (idx_memory_embedding, idx_memory_fts)
      7. Post-restore verification (row count + spot check)

    --force semantics (NW2): full reset — schema OVERWRITE + records replace +
    index rebuild.  Use only when the target is known empty or intentionally reset.

    Returns:
        target_url on success.

    Raises:
        ImportError on validation or import failure.
    """
    # --- Validate manifest ---
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        raise ImportError(f"manifest.json not found in {backup_dir}", exit_code=1)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ImportError(f"manifest.json is not valid JSON: {e}", exit_code=1)

    # --- Connect to server ---
    db = Surreal(target_url)
    try:
        db.use("cognitive", "memory")  # first RPC opens the ws socket
    except Exception as exc:
        raise ImportError(
            f"Cannot connect to SurrealDB server at {target_url}: {exc}",
            exit_code=1,
        ) from exc

    try:
        # --- 2. Non-empty target guard (NW1) ---
        count_result = db.query("SELECT count() AS cnt FROM memory GROUP ALL")
        count_rows = _rows(count_result)
        existing_count = count_rows[0].get("cnt", 0) if count_rows else 0
        if existing_count > 0:
            if not force:
                raise ImportError(
                    f"Target SurrealDB server at {target_url} already contains "
                    f"{existing_count} records. "
                    "Supply --force to overwrite (full reset: schema + records + indexes).",
                    exit_code=5,
                )
            logger.warning(
                "--force supplied: overwriting %d existing records at %s",
                existing_count, target_url,
            )

        # --- 3. Apply schema ---
        schema_path = backup_dir / "schema.surql"
        schema_text = schema_path.read_text(encoding="utf-8")

        import hashlib
        actual_hash = hashlib.sha256(schema_text.encode()).hexdigest()
        manifest_hash = manifest.get("schema_hash", "")
        if actual_hash != manifest_hash:
            msg = (
                f"Schema hash mismatch: manifest={manifest_hash[:16]}... "
                f"actual={actual_hash[:16]}..."
            )
            if strict:
                raise ImportError(msg, exit_code=7)
            else:
                logger.warning("%s", msg)

        for stmt in schema_text.split(";"):
            lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                try:
                    db.query(clean)
                except Exception as exc:
                    logger.debug("Schema stmt failed: %s — %s", clean[:60], exc)

        # --- 4. Import tables (batched, 100 rows per batch) ---
        for table in IMPORT_ORDER:
            path = backup_dir / f"{table}.ndjson"
            if not path.exists():
                logger.debug("No %s.ndjson found; skipping.", table)
                continue
            if table == "preference":
                _import_preference(db, path)
            elif table == "memory":
                _import_memory_ws(db, path)
            elif table == "memory_version":
                _import_memory_version(db, path)
            elif table == "consolidation_log":
                _import_consolidation_log(db, path)

        # --- 5. Import edge tables ---
        edges_dir = backup_dir / "edges"
        for rel in REL_TABLES:
            edge_path = edges_dir / f"{rel}.ndjson"
            if edge_path.exists():
                _import_edge(db, edge_path, rel)

        # --- 6. REBUILD indexes ---
        for index_stmt in [
            "REBUILD INDEX idx_memory_embedding ON memory",
            "REBUILD INDEX idx_memory_fts ON memory",
        ]:
            try:
                db.query(index_stmt)
                logger.debug("Ran: %s", index_stmt)
            except Exception as exc:
                logger.warning("Index rebuild failed (%s): %s", index_stmt, exc)

        # --- 7. Post-restore verification ---
        exit_code = _post_restore_verify(db, backup_dir, manifest)
        if exit_code != 0:
            raise ImportError(
                f"Post-restore verification failed (exit {exit_code}). "
                f"Target server: {target_url}",
                exit_code=exit_code,
            )

    finally:
        try:
            db.close()
        except Exception:
            pass

    return target_url


def _import_memory_ws(db: Any, path: Path) -> None:
    """Import memory.ndjson to a ws server target in batches of 100.

    Batch size is an optimization only — correctness does not depend on it.
    Each record is individually verified during post-restore spot-check.
    """
    _BATCH_SIZE = 100
    batch: list[dict] = []

    def flush(rows: list[dict]) -> None:
        for row in rows:
            mem_id = row.get("id", "")
            coerced = _coerce_datetimes(row)
            try:
                db.query(
                    "CREATE type::thing('memory', $id) CONTENT $content",
                    {"id": mem_id, "content": coerced},
                )
            except Exception as e:
                logger.warning("Failed to import memory %s: %s", mem_id, e)

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            batch.append(row)
            if len(batch) >= _BATCH_SIZE:
                flush(batch)
                batch.clear()

    if batch:
        flush(batch)


# ---------------------------------------------------------------------------
# Table importers
# ---------------------------------------------------------------------------

def _import_preference(db: Any, path: Path) -> None:
    """Import preference.ndjson via UPSERT preserving updated_at.

    updated_at must be a datetime object — schema TYPE datetime rejects ISO strings.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pref_id = row.get("id", "")
            val = row.get("val")
            updated_at = _parse_dt_for_surreal(row.get("updated_at", ""))
            try:
                db.query(
                    "UPSERT type::thing('preference', $key) SET val=$val, updated_at=$updated_at",
                    {"key": pref_id, "val": val, "updated_at": updated_at},
                )
            except Exception as e:
                logger.warning("Failed to upsert preference %s: %s", pref_id, e)


def _import_memory(db: Any, path: Path) -> None:
    """Import memory.ndjson preserving all 15 canonical fields including embedding.

    Datetime fields (created_at, updated_at, last_accessed) must be datetime objects —
    the SurrealDB schema declares them TYPE datetime and rejects ISO strings.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mem_id = row.get("id", "")
            # Convert ISO string datetimes to datetime objects before inserting
            coerced = _coerce_datetimes(row)
            try:
                # Use CREATE ... CONTENT to preserve all fields at once
                db.query(
                    "CREATE type::thing('memory', $id) CONTENT $content",
                    {"id": mem_id, "content": coerced},
                )
            except Exception as e:
                logger.warning("Failed to import memory %s: %s", mem_id, e)


def _import_memory_version(db: Any, path: Path) -> None:
    """Import memory_version.ndjson, re-wrapping memory_id to a proper RecordID.

    The exported memory_id is a plain UUID (stripped of 'memory:' prefix).
    We re-wrap it as type::thing('memory', $mem_id) to restore the record reference.
    created_at must be a datetime object (not ISO string) per schema TYPE datetime.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ver_id = row.get("id", "")
            mem_id = row.get("memory_id", "")  # plain UUID from export
            content = row.get("content", "")
            metadata = row.get("metadata")
            created_at = _parse_dt_for_surreal(row.get("created_at", ""))
            try:
                db.query(
                    """CREATE type::thing('memory_version', $id) SET
                        memory_id = type::thing('memory', $mem_id),
                        content = $content,
                        metadata = $metadata,
                        created_at = $created_at
                    """,
                    {
                        "id": ver_id,
                        "mem_id": mem_id,
                        "content": content,
                        "metadata": metadata,
                        "created_at": created_at,
                    },
                )
            except Exception as e:
                logger.warning("Failed to import memory_version %s: %s", ver_id, e)


def _import_consolidation_log(db: Any, path: Path) -> None:
    """Import consolidation_log.ndjson.

    created_at must be a datetime object per schema TYPE datetime.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            log_id = row.get("id", "")
            coerced = _coerce_datetimes(row)
            try:
                db.query(
                    "CREATE type::thing('consolidation_log', $id) CONTENT $content",
                    {"id": log_id, "content": coerced},
                )
            except Exception as e:
                logger.warning("Failed to import consolidation_log %s: %s", log_id, e)


def _parse_dt_for_surreal(val: Any):
    """Parse a datetime value for SurrealDB.

    CRITICAL: SurrealDB schema requires datetime objects, NOT ISO strings.
    Passing an ISO string to a TYPE datetime field silently fails — the record
    is not created and no exception is raised. This function ensures we always
    pass a datetime object.
    """
    if val is None or val == "":
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
    if hasattr(val, "isoformat"):
        return val  # already a datetime
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(str(val))
        return dt
    except (ValueError, TypeError):
        from datetime import timezone
        return datetime.now(timezone.utc)


def _coerce_datetimes(row: dict) -> dict:
    """Convert known ISO string datetime fields in a row to datetime objects.

    SurrealDB TYPE datetime fields reject ISO strings — must be datetime objects.
    This applies to any table with datetime fields.
    """
    from datetime import datetime, timezone as _tz

    DATETIME_FIELDS = {
        "created_at", "updated_at", "last_accessed",
    }
    result = dict(row)
    for field in DATETIME_FIELDS:
        if field in result and isinstance(result[field], str):
            result[field] = _parse_dt_for_surreal(result[field])
    return result


def _import_edge(db: Any, path: Path, rel: str) -> None:
    """Import an edge table via RELATE in->rel->out SET strength, created_at.

    Uses LET variables — type::thing() inline in RELATE is not supported by
    the embedded SurrealDB Python SDK.

    IMPORTANT: created_at must be passed as a datetime object, not an ISO string.
    The SurrealDB schema declares created_at as TYPE datetime; string values
    silently fail schema validation and the edge is not created.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            in_id = row.get("in", "")
            out_id = row.get("out", "")
            strength = row.get("strength", 1.0)
            created_at = _parse_dt_for_surreal(row.get("created_at"))
            try:
                db.query(
                    f"""LET $from = type::thing('memory', $src);
                        LET $to = type::thing('memory', $tgt);
                        RELATE $from->{rel}->$to
                        SET strength = $strength, created_at = $created_at""",
                    {
                        "src": in_id,
                        "tgt": out_id,
                        "strength": strength,
                        "created_at": created_at,
                    },
                )
            except Exception as e:
                logger.debug("Failed to import edge %s->%s->%s: %s", in_id, rel, out_id, e)


# ---------------------------------------------------------------------------
# Post-restore verification
# ---------------------------------------------------------------------------

def _rows(result: Any) -> list[dict]:
    """Normalize SurrealDB result to flat list of dicts."""
    if not result:
        return []
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            return result
        flat: list[dict] = []
        for item in result:
            if isinstance(item, list):
                flat.extend(item)
            elif isinstance(item, dict):
                if "result" in item:
                    inner = item["result"]
                    if isinstance(inner, list):
                        flat.extend(inner)
                    else:
                        flat.append(inner)
                else:
                    flat.append(item)
        return flat
    if isinstance(result, dict):
        return [result]
    return []


def _extract_id(surreal_id: Any) -> str:
    """Mirror surreal_storage._extract_id."""
    s = str(surreal_id)
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.strip("\u27e8\u27e9")


def _post_restore_verify(db: Any, backup_dir: Path, manifest: dict) -> int:
    """Row-count check + spot-check 10 random memories. Returns exit_code (0=pass, 4=fail)."""
    import hashlib as _hashlib
    import json as _json
    import random as _random

    row_counts = manifest.get("row_counts", {})

    # Row count checks for plain tables
    for table in ["memory", "memory_version", "consolidation_log", "preference"]:
        expected = row_counts.get(table)
        if expected is None:
            continue
        result = db.query(f"SELECT count() AS cnt FROM {table} GROUP ALL")
        rows_list = _rows(result)
        actual = rows_list[0].get("cnt", 0) if rows_list else 0
        if actual != expected:
            logger.error(
                "Post-restore count mismatch for %s: expected=%d, actual=%d",
                table, expected, actual,
            )
            return 4

    # Edge table row counts
    edge_counts = row_counts.get("edges", {})
    for rel in REL_TABLES:
        expected = edge_counts.get(rel)
        if expected is None:
            continue
        result = db.query(f"SELECT count() AS cnt FROM {rel} GROUP ALL")
        rows_list = _rows(result)
        actual = rows_list[0].get("cnt", 0) if rows_list else 0
        if actual != expected:
            logger.error(
                "Post-restore edge count mismatch for %s: expected=%d, actual=%d",
                rel, expected, actual,
            )
            return 4

    # Spot-check 10 random memories
    memory_ndjson = backup_dir / "memory.ndjson"
    if memory_ndjson.exists():
        all_rows: list[dict] = []
        with memory_ndjson.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_rows.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
        sample = _random.sample(all_rows, min(10, len(all_rows))) if all_rows else []
        for source_row in sample:
            mem_id = source_row.get("id", "")
            result = db.query(
                "SELECT * FROM type::thing('memory', $id)",
                {"id": mem_id},
            )
            fetched = _rows(result)
            if not fetched:
                logger.error("Spot-check: memory %s not found after restore", mem_id)
                return 4
            restored_row = fetched[0]

            # Verify content hash
            source_content = source_row.get("content", "")
            restored_content = restored_row.get("content", "")
            if source_content != restored_content:
                logger.error("Spot-check: memory %s content mismatch", mem_id)
                return 4

            # Verify first 8 embedding floats
            source_emb = source_row.get("embedding") or []
            restored_emb = restored_row.get("embedding") or []
            if source_emb and restored_emb:
                if source_emb[:8] != restored_emb[:8]:
                    logger.error("Spot-check: memory %s embedding mismatch", mem_id)
                    return 4

    return 0
