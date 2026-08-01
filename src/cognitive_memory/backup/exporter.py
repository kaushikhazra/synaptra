"""CM Backup Exporter.

Stop-CM ritual → open SurrealKV directly → stream NDJSON → manifest → restart CM.

Exit codes:
  0   Success
  2   SurrealKV open failure
  3   Disk / write error during export
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical CM package version
try:
    from importlib.metadata import version as _pkg_version
    _CM_VERSION = _pkg_version("synaptra")
except Exception:
    _CM_VERSION = "unknown"

# Relation tables to export
REL_TABLES = [
    "causes", "follows", "contradicts", "supports",
    "relates_to", "supersedes", "part_of", "describes",
]

# Tables (non-edge) to export
PLAIN_TABLES = ["memory", "memory_version", "consolidation_log", "preference"]

# Default backups root
DEFAULT_BACKUPS_ROOT = Path.home() / ".synaptra" / "backups"

# How long to wait for the SurrealKV lock to release (seconds)
LOCK_POLL_INTERVAL = 2
LOCK_POLL_TIMEOUT = 30

# How long to wait for CM restart to become healthy (seconds)
RESTART_POLL_INTERVAL = 5
RESTART_POLL_MAX = 180


# ---------------------------------------------------------------------------
# OS-specific stop/start ritual
# ---------------------------------------------------------------------------

def _cm_task_exists() -> bool:
    """Return True if the CognitiveMemory scheduled task exists on Windows."""
    if platform.system() != "Windows":
        return False
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-ScheduledTask -TaskName CognitiveMemory -ErrorAction SilentlyContinue"],
            capture_output=True, text=True, timeout=15,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _stop_cm() -> None:
    """Stop the CM Windows scheduled task.

    W3 guard: distinguishes between "task doesn't exist / already stopped"
    (safe to proceed) and "task exists but stop errored" (unsafe — abort).

    Raises:
        NotImplementedError: on non-Windows platforms.
        RuntimeError: if the task exists but Stop-ScheduledTask failed.
    """
    if platform.system() != "Windows":
        raise NotImplementedError(
            "Stop-CM ritual is only implemented on Windows. "
            "On other platforms, ensure CM is not running before calling export_backup()."
        )

    task_exists = _cm_task_exists()
    if not task_exists:
        # Task doesn't exist — CM is presumably not running; SurrealKV lock is free.
        logger.info("CognitiveMemory scheduled task not found; assuming CM is not running, proceeding.")
        return

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Stop-ScheduledTask -TaskName CognitiveMemory"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # PowerShell returns non-zero for "task is already stopped" — that's fine.
        if "already" in stderr.lower() or "not running" in stderr.lower():
            logger.info("CM service was already stopped.")
        else:
            raise RuntimeError(
                f"Stop-ScheduledTask failed (exit {result.returncode}): {stderr}"
            )
    else:
        logger.info("CM service stopped via Stop-ScheduledTask.")


def _start_cm() -> bool:
    """Restart the CM Windows scheduled task.

    Returns True on success.  On failure, logs a warning (W1 guard) and
    returns False — caller should emit a stderr warning but still exit 0
    (backup artifact is valid).

    Raises:
        NotImplementedError: on non-Windows platforms.
    """
    if platform.system() != "Windows":
        raise NotImplementedError(
            "Start-CM ritual is only implemented on Windows."
        )

    if not _cm_task_exists():
        logger.warning("CognitiveMemory scheduled task not found; cannot restart CM.")
        return False

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Start-ScheduledTask -TaskName CognitiveMemory"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        logger.warning(
            "Start-ScheduledTask failed (exit %d): %s",
            result.returncode, result.stderr.strip(),
        )
        return False
    logger.info("CM service restart triggered.")
    return True


def _wait_for_cm_restart(cm_url: str = "http://127.0.0.1:8050/mcp") -> bool:
    """Poll CM's MCP endpoint until it responds or timeout.

    Returns True if CM responded within RESTART_POLL_MAX seconds.
    Logs progress every 30 s.
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        return False

    start = time.time()
    last_log = 0.0
    while True:
        elapsed = time.time() - start
        if elapsed > RESTART_POLL_MAX:
            return False
        if elapsed - last_log >= 30:
            logger.info("Waiting for CM restart... %.0f s elapsed", elapsed)
            last_log = elapsed
        try:
            req = urllib.request.Request(
                cm_url,
                data=b'{"jsonrpc":"2.0","method":"ping","id":1}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                if resp.status in (200, 405, 400):
                    return True
        except Exception:
            pass
        time.sleep(RESTART_POLL_INTERVAL)


def _wait_for_lock_release(db_path: Path) -> None:
    """Poll until SurrealKV lock file is gone (or timeout)."""
    lock_file = db_path / "lock"
    if not db_path.exists():
        return
    deadline = time.time() + LOCK_POLL_TIMEOUT
    while lock_file.exists() and time.time() < deadline:
        logger.debug("Waiting for SurrealKV lock to release...")
        time.sleep(LOCK_POLL_INTERVAL)
    if lock_file.exists():
        logger.warning("SurrealKV lock file still present after %d s; proceeding anyway.", LOCK_POLL_TIMEOUT)


# ---------------------------------------------------------------------------
# ID extraction (mirror surreal_storage._extract_id)
# ---------------------------------------------------------------------------

def _extract_id(surreal_id: Any) -> str:
    """Extract plain ID from a SurrealDB RecordID or string.

    Mirrors the implementation in surreal_storage.py:55.
    Strips table prefix and SurrealDB angle brackets (U+27E8 / U+27E9).
    """
    s = str(surreal_id)
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.strip("\u27e8\u27e9")


# ---------------------------------------------------------------------------
# SurrealDB result normalizer (lightweight, no import of surreal_storage)
# ---------------------------------------------------------------------------

def _rows(result: Any) -> list[dict]:
    """Normalize SurrealDB query result to a flat list of dicts."""
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


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON serializer for types not natively handled."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _serialize_row(row: dict) -> dict:
    """Convert a SurrealDB result row to a plain JSON-serializable dict."""
    out: dict = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, (list, tuple)):
            out[k] = [(_json_default(i) if not isinstance(i, (int, float, str, bool, type(None))) else i) for i in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    """Resolve the SurrealKV data directory path."""
    db_path_env = os.environ.get("SYNAPTRA_DB", "")
    if db_path_env:
        # Strip any protocol prefix
        raw = db_path_env
        for prefix in ("surrealkv://", "file://", "mem://"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        return Path(raw)
    return Path.home() / ".synaptra" / "data"


def export_backup(
    out_dir: Path | None = None,
    name: str | None = None,
    skip_cm_restart: bool = False,
) -> Path:
    """Export a full logical backup of the CM SurrealKV store.

    Args:
        out_dir: Parent directory for the backup artifact.
                 Defaults to ~/.synaptra/backups/.
        name:    Backup directory name. Defaults to cm-<timestamp>Z.
        skip_cm_restart: If True, don't stop/restart CM (for tests where CM is not running).

    Returns:
        Path to the created backup directory.

    Raises:
        SystemExit(2): SurrealKV open failure (no partial artifact left behind).
        SystemExit(3): Disk / write error (partial artifact cleaned up).
    """
    from surrealdb import Surreal

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup_name = name or f"cm-{ts}Z"
    backups_root = out_dir or DEFAULT_BACKUPS_ROOT
    backup_dir = backups_root / backup_name

    # Clean up orphaned partial backups (no manifest.json, older than 1 hour) — O6 guard
    _cleanup_orphaned_partials(backups_root)

    backup_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    db_path = _get_db_path()
    db_url = f"surrealkv://{str(db_path).replace(os.sep, '/')}"

    logger.info("Starting backup to %s", backup_dir)

    try:
        # --- Stop CM ritual ---
        if not skip_cm_restart:
            try:
                _stop_cm()
            except NotImplementedError:
                logger.warning("Non-Windows platform: stop-CM ritual skipped. Ensure CM is not running.")
            except RuntimeError as e:
                # Task exists but stop failed — unsafe to open SurrealKV
                _cleanup(backup_dir)
                logger.error("Stop-CM failed: %s", e)
                sys.exit(2)

            # Wait for lock release
            _wait_for_lock_release(db_path)

        # --- Open SurrealKV directly ---
        try:
            db = Surreal(db_url)
            db.connect()
            db.use("cognitive", "memory")
        except Exception as e:
            _cleanup(backup_dir)
            logger.error("SurrealKV open failed (%s): %s", db_url, e)
            if not skip_cm_restart:
                _try_restart_cm()
            sys.exit(2)

        try:
            # --- Stream tables inside a read transaction ---
            # BEGIN TRANSACTION provides snapshot consistency.
            # With CM stopped, this is belt-and-suspenders — good practice.
            # Note: some SurrealDB embedded versions may not support BEGIN TRANSACTION
            # over all query types; if it fails, we proceed without it — CM is stopped
            # so no concurrent writers exist.
            try:
                db.query("BEGIN TRANSACTION")
                in_transaction = True
            except Exception:
                in_transaction = False
                logger.debug("BEGIN TRANSACTION not supported; proceeding without (CM is stopped, safe).")

            row_counts: dict[str, Any] = {}

            # 1. Write schema.surql
            # Use binary copy to preserve exact bytes (avoids CRLF conversion on Windows).
            schema_path = Path(__file__).parent.parent / "schema.surql"
            schema_bytes = schema_path.read_bytes()
            (backup_dir / "schema.surql").write_bytes(schema_bytes)
            schema_hash = hashlib.sha256(schema_bytes).hexdigest()

            # 2. Stream plain tables
            for table in PLAIN_TABLES:
                count = _export_table(db, table, backup_dir)
                row_counts[table] = count

            # 3. Stream edge tables
            edges_dir = backup_dir / "edges"
            edges_dir.mkdir(exist_ok=True)
            edge_counts: dict[str, int] = {}
            for rel in REL_TABLES:
                count = _export_edge_table(db, rel, edges_dir)
                edge_counts[rel] = count
            row_counts["edges"] = edge_counts

            if in_transaction:
                try:
                    db.query("COMMIT TRANSACTION")
                except Exception:
                    pass

        finally:
            try:
                db.close()
            except Exception:
                pass

    except SystemExit:
        raise
    except Exception as e:
        _cleanup(backup_dir)
        logger.error("Disk/write error during backup: %s", e)
        if not skip_cm_restart:
            _try_restart_cm()
        sys.exit(3)

    # --- Restart CM ---
    cm_restarted = True
    if not skip_cm_restart:
        cm_restarted = _try_restart_cm()

    # --- Write manifest ---
    duration_ms = int((time.time() - start_time) * 1000)
    size_bytes = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())

    manifest = {
        "backup_id": backup_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cm_version": _CM_VERSION,
        "schema_hash": schema_hash,
        "source_backend": "surrealkv",
        "row_counts": row_counts,
        "size_bytes": size_bytes,
        "duration_ms": duration_ms,
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    logger.info(
        "Backup complete: %s (%d memories, %d ms, %.1f KB)",
        backup_dir,
        row_counts.get("memory", 0),
        duration_ms,
        size_bytes / 1024,
    )

    # W1 guard: warn if CM didn't restart, but still exit 0 (backup is valid)
    if not skip_cm_restart and not cm_restarted:
        print(
            "WARNING: Backup succeeded but CM service restart failed. "
            "Verify manually: Start-ScheduledTask -TaskName CognitiveMemory",
            file=sys.stderr,
        )

    return backup_dir


def _export_table(db: Any, table: str, backup_dir: Path) -> int:
    """Export a plain table to <backup_dir>/<table>.ndjson. Returns row count."""
    result = db.query(f"SELECT * FROM {table}")
    rows = _rows(result)

    out_path = backup_dir / f"{table}.ndjson"
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            row_dict = _serialize_row(row)

            # Normalize the record id to plain UUID/string
            if "id" in row_dict:
                row_dict["id"] = _extract_id(row_dict["id"])

            # memory_version: strip 'memory:' prefix from memory_id RecordID
            if table == "memory_version" and "memory_id" in row_dict:
                row_dict["memory_id"] = _extract_id(row_dict["memory_id"])

            fh.write(json.dumps(row_dict, default=_json_default) + "\n")
            count += 1

    return count


def _export_edge_table(db: Any, rel_table: str, edges_dir: Path) -> int:
    """Export an edge table to <edges_dir>/<rel_table>.ndjson.

    Emits {in, out, strength, created_at} per line — no edge id.
    """
    result = db.query(f"SELECT *, in, out FROM {rel_table}")
    rows = _rows(result)

    out_path = edges_dir / f"{rel_table}.ndjson"
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            row_dict = _serialize_row(row)
            edge_record = {
                "in": _extract_id(row_dict.get("in", "")),
                "out": _extract_id(row_dict.get("out", "")),
                "strength": row_dict.get("strength", 1.0),
                "created_at": row_dict.get("created_at", ""),
            }
            fh.write(json.dumps(edge_record, default=_json_default) + "\n")
            count += 1

    return count


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _cleanup(backup_dir: Path) -> None:
    """Remove a partial backup directory."""
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception as e:
        logger.warning("Could not clean up partial backup dir %s: %s", backup_dir, e)


def _cleanup_orphaned_partials(backups_root: Path) -> None:
    """Remove backup directories that have no manifest.json and are older than 1 hour."""
    if not backups_root.exists():
        return
    cutoff = time.time() - 3600
    for d in backups_root.iterdir():
        if d.is_dir() and not (d / "manifest.json").exists():
            try:
                age = d.stat().st_mtime
                if age < cutoff:
                    shutil.rmtree(d)
                    logger.info("Cleaned up orphaned partial backup: %s", d)
            except Exception:
                pass


def _try_restart_cm() -> bool:
    """Best-effort CM restart; returns True on success."""
    try:
        restarted = _start_cm()
        if restarted:
            # Optionally wait for CM to become healthy
            cm_url = os.environ.get(
                "SYNAPTRA_URL", "http://127.0.0.1:8050/mcp"
            )
            healthy = _wait_for_cm_restart(cm_url)
            if not healthy:
                logger.warning(
                    "CM service started but did not respond within %d s. "
                    "It may still be replaying the SurrealKV clog (~2 min is normal).",
                    RESTART_POLL_MAX,
                )
            return True
        return False
    except NotImplementedError:
        return True  # Non-Windows: nothing to restart
    except Exception as e:
        logger.warning("CM restart attempt failed: %s", e)
        return False
