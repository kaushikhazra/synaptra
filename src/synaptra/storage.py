"""SQLite storage layer — CRUD for all tables, FTS5, migrations."""

from __future__ import annotations

import importlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import (
    ConsolidationLogEntry,
    Memory,
    MemoryState,
    MemoryType,
    MemoryVersion,
    RelType,
    Relationship,
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Storage:
    """SQLite storage backend for synaptra system."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.row_factory = sqlite3.Row
        self._in_transaction = False
        self._run_migrations()

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    # --- Transaction Management ---

    @contextmanager
    def transaction(self):
        """Context manager for atomic operation groups. Nestable (inner is no-op)."""
        if self._in_transaction:
            yield  # Already in a transaction, let the outer one handle commit/rollback
            return
        self._in_transaction = True
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._in_transaction = False

    def _auto_commit(self) -> None:
        """Commit only if not inside a transaction() block."""
        if not self._in_transaction:
            self._conn.commit()

    # --- Migration Runner ---

    def _run_migrations(self) -> None:
        current_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        migration_files = sorted(MIGRATIONS_DIR.glob("[0-9]*.py"))

        for mig_file in migration_files:
            mig_number = int(mig_file.stem.split("_")[0])
            if mig_number <= current_version:
                continue
            spec = importlib.util.spec_from_file_location(mig_file.stem, mig_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                module.migrate(self._conn)
                self._conn.execute(f"PRAGMA user_version = {mig_number}")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # --- Memory CRUD ---

    def insert_memory(self, memory: Memory, embedding_bytes: bytes | None = None) -> None:
        tags_json = json.dumps(memory.tags)
        self._conn.execute(
            """INSERT INTO memory (id, content, memory_type, state, importance,
               stability, retrievability, access_count, created_at, updated_at,
               last_accessed, source, conversation_id, rater, rated_at, tags,
               embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id, memory.content, memory.memory_type.value,
                memory.state.value, memory.importance, memory.stability,
                memory.retrievability, memory.access_count,
                memory.created_at.isoformat(), memory.updated_at.isoformat(),
                memory.last_accessed.isoformat(), memory.source,
                memory.conversation_id, memory.rater,
                memory.rated_at.isoformat() if memory.rated_at else None,
                tags_json, embedding_bytes,
            ),
        )
        # FTS5 sync
        self._conn.execute(
            "INSERT INTO memory_fts(rowid, content) VALUES ((SELECT rowid FROM memory WHERE id = ?), ?)",
            (memory.id, memory.content),
        )
        self._auto_commit()

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def update_memory_fields(self, memory_id: str, **fields) -> None:
        if not fields:
            return
        set_clauses = []
        values = []
        for key, value in fields.items():
            if key == "tags":
                value = json.dumps(value)
            elif key == "memory_type" and isinstance(value, MemoryType):
                value = value.value
            elif key == "state" and isinstance(value, MemoryState):
                value = value.value
            elif isinstance(value, datetime):
                value = value.isoformat()
            set_clauses.append(f"{key} = ?")
            values.append(value)
        values.append(memory_id)
        self._conn.execute(
            f"UPDATE memory SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        # If content changed, update FTS5
        if "content" in fields:
            self._conn.execute(
                "DELETE FROM memory_fts WHERE rowid = (SELECT rowid FROM memory WHERE id = ?)",
                (memory_id,),
            )
            self._conn.execute(
                "INSERT INTO memory_fts(rowid, content) VALUES ((SELECT rowid FROM memory WHERE id = ?), ?)",
                (memory_id, fields["content"]),
            )
        self._auto_commit()

    def update_embedding(self, memory_id: str, embedding_bytes: bytes) -> None:
        self._conn.execute(
            "UPDATE memory SET embedding = ? WHERE id = ?",
            (embedding_bytes, memory_id),
        )
        self._auto_commit()

    def delete_memory(self, memory_id: str) -> None:
        # Cascade: relationships, versions, FTS, then memory
        self._conn.execute(
            "DELETE FROM relationship WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        self._conn.execute("DELETE FROM memory_version WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            "DELETE FROM memory_fts WHERE rowid = (SELECT rowid FROM memory WHERE id = ?)",
            (memory_id,),
        )
        self._conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        self._auto_commit()

    def list_memories(
        self,
        search: str | None = None,
        memory_type: str | None = None,
        state: str | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        importance_min: float | None = None,
        importance_max: float | None = None,
        rater_not: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        conditions = []
        params: list[Any] = []

        if search:
            # Use FTS5 to get matching rowids
            conditions.append(
                "memory.rowid IN (SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?)"
            )
            params.append(search)

        if memory_type:
            conditions.append("memory.memory_type = ?")
            params.append(memory_type)

        if state:
            conditions.append("memory.state = ?")
            params.append(state)

        if tags:
            for tag in tags:
                conditions.append("memory.tags LIKE ?")
                params.append(f'%"{tag}"%')

        if time_range:
            conditions.append("memory.created_at >= ? AND memory.created_at <= ?")
            params.extend([time_range[0].isoformat(), time_range[1].isoformat()])

        if importance_min is not None:
            conditions.append("memory.importance >= ?")
            params.append(importance_min)

        if importance_max is not None:
            conditions.append("memory.importance <= ?")
            params.append(importance_max)

        if rater_not is not None:
            # IS NULL is not optional here.  A pre-#8 row has no rater, so it
            # differs from every model — and `rater != ?` alone would drop all
            # of them under SQL's three-valued logic, returning an empty list
            # against a store where every row is a candidate.
            conditions.append("(memory.rater IS NULL OR memory.rater != ?)")
            params.append(rater_not)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"SELECT * FROM memory WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_all_active_memories(self) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT * FROM memory WHERE state = 'active'"
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_active_embeddings(self) -> dict[str, np.ndarray]:
        """Return {memory_id: embedding_array} for all active memories with embeddings."""
        rows = self._conn.execute(
            "SELECT id, embedding FROM memory WHERE state = 'active' AND embedding IS NOT NULL"
        ).fetchall()
        result = {}
        for row in rows:
            if row["embedding"]:
                result[row["id"]] = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return result

    def fts_search(self, query: str, state: str = "active", limit: int = 30) -> list[tuple[str, float]]:
        """FTS5 BM25 search. Returns [(memory_id, rank_score)]."""
        rows = self._conn.execute(
            """SELECT memory.id, rank FROM memory_fts
               JOIN memory ON memory.rowid = memory_fts.rowid
               WHERE memory_fts MATCH ? AND memory.state = ?
               ORDER BY rank
               LIMIT ?""",
            (query, state, limit),
        ).fetchall()
        return [(row["id"], row["rank"]) for row in rows]

    # --- Memory Version ---

    def insert_version(self, version: MemoryVersion) -> None:
        metadata_json = json.dumps(version.metadata) if version.metadata else None
        self._conn.execute(
            "INSERT INTO memory_version (id, memory_id, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (version.id, version.memory_id, version.content, metadata_json, version.created_at.isoformat()),
        )
        self._auto_commit()

    def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        rows = self._conn.execute(
            "SELECT * FROM memory_version WHERE memory_id = ? ORDER BY created_at DESC",
            (memory_id,),
        ).fetchall()
        return [
            MemoryVersion(
                id=r["id"], memory_id=r["memory_id"], content=r["content"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # --- Relationships ---

    def insert_relationship(self, rel: Relationship) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO relationship (id, source_id, target_id, rel_type, strength, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (rel.id, rel.source_id, rel.target_id, rel.rel_type.value, rel.strength, rel.created_at.isoformat()),
        )
        self._auto_commit()

    def delete_relationship(self, source_id: str, target_id: str, rel_type: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM relationship WHERE source_id = ? AND target_id = ? AND rel_type = ?",
            (source_id, target_id, rel_type),
        )
        self._auto_commit()
        return cursor.rowcount > 0

    def get_relationships_for(self, memory_id: str, rel_types: list[str] | None = None) -> list[Relationship]:
        """Get all relationships where memory is source or target."""
        type_filter = ""
        params: list[Any] = [memory_id, memory_id]
        if rel_types:
            placeholders = ", ".join(["?"] * len(rel_types))
            type_filter = f" AND rel_type IN ({placeholders})"
            params.extend(rel_types)

        rows = self._conn.execute(
            f"SELECT * FROM relationship WHERE (source_id = ? OR target_id = ?){type_filter}",
            params,
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def get_outgoing_relationships(self, memory_id: str) -> list[Relationship]:
        rows = self._conn.execute(
            "SELECT * FROM relationship WHERE source_id = ?", (memory_id,)
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def get_incoming_relationships(self, memory_id: str, rel_type: str | None = None) -> list[Relationship]:
        if rel_type:
            rows = self._conn.execute(
                "SELECT * FROM relationship WHERE target_id = ? AND rel_type = ?",
                (memory_id, rel_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM relationship WHERE target_id = ?", (memory_id,)
            ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def get_neighbors(self, memory_id: str, active_only: bool = True) -> list[tuple[str, Relationship]]:
        """Get neighbor memory IDs + relationships for graph traversal."""
        query = """
            SELECT r.*, m.state FROM relationship r
            JOIN memory m ON (
                CASE WHEN r.source_id = ? THEN r.target_id ELSE r.source_id END
            ) = m.id
            WHERE (r.source_id = ? OR r.target_id = ?)
        """
        params: list[Any] = [memory_id, memory_id, memory_id]
        if active_only:
            query += " AND m.state = 'active'"
        rows = self._conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            rel = self._row_to_relationship(r)
            neighbor_id = rel.target_id if rel.source_id == memory_id else rel.source_id
            result.append((neighbor_id, rel))
        return result

    def delete_auto_links(self, memory_id: str) -> None:
        """Delete auto-created relates_to links (strength < 1.0) from a memory."""
        self._conn.execute(
            "DELETE FROM relationship WHERE source_id = ? AND rel_type = 'relates_to' AND strength < 1.0",
            (memory_id,),
        )
        self._auto_commit()

    def bulk_update_stability(self, updates: list[tuple[float, str]]) -> None:
        """Batch update stability for multiple memories. updates = [(new_stability, memory_id), ...]"""
        self._conn.executemany(
            "UPDATE memory SET stability = ? WHERE id = ?",
            updates,
        )
        self._auto_commit()

    def has_incoming_supersedes(self, memory_id: str) -> bool:
        """Check if memory has an active incoming supersedes relationship."""
        row = self._conn.execute(
            """SELECT 1 FROM relationship r
               JOIN memory m ON r.source_id = m.id
               WHERE r.target_id = ? AND r.rel_type = 'supersedes' AND m.state = 'active'
               LIMIT 1""",
            (memory_id,),
        ).fetchone()
        return row is not None

    def get_contradictions_for(self, memory_id: str) -> list[tuple[str, str, float]]:
        """Get unresolved contradictions: [(other_memory_id, content_preview, strength)]."""
        rows = self._conn.execute(
            """SELECT m.id, substr(m.content, 1, 100) as preview, r.strength
               FROM relationship r
               JOIN memory m ON (
                   CASE WHEN r.source_id = ? THEN r.target_id ELSE r.source_id END
               ) = m.id
               WHERE (r.source_id = ? OR r.target_id = ?) AND r.rel_type = 'contradicts'
               AND m.state = 'active'""",
            (memory_id, memory_id, memory_id),
        ).fetchall()
        return [(r["id"], r["preview"], r["strength"]) for r in rows]

    # --- Consolidation Log ---

    def insert_consolidation_log(self, entry: ConsolidationLogEntry) -> None:
        self._conn.execute(
            "INSERT INTO consolidation_log (id, action, source_ids, target_id, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.id, entry.action, json.dumps(entry.source_ids), entry.target_id, entry.reason, entry.created_at.isoformat()),
        )
        self._auto_commit()

    def get_last_consolidation(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM consolidation_log ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_consolidation_summary(self) -> dict:
        """Get summary of last consolidation run."""
        last = self.get_last_consolidation()
        if last is None:
            return {"last_run": None, "promoted": 0, "merged": 0, "archived": 0}

        last_run = last["created_at"]
        counts = self._conn.execute(
            """SELECT action, COUNT(*) as cnt FROM consolidation_log
               WHERE created_at >= ? GROUP BY action""",
            (last_run,),
        ).fetchall()
        summary = {"last_run": last_run, "promoted": 0, "merged": 0, "archived": 0}
        for row in counts:
            if row["action"] == "promote":
                summary["promoted"] = row["cnt"]
            elif row["action"] == "merge":
                summary["merged"] = row["cnt"]
            elif row["action"] == "archive":
                summary["archived"] = row["cnt"]
        return summary

    # --- Config ---

    def get_config(self, key: str) -> Any | None:
        row = self._conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def set_config(self, key: str, value: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), now),
        )
        self._auto_commit()

    def get_all_config(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM config").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # --- Stats helpers ---

    def get_counts_by_type(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memory GROUP BY memory_type"
        ).fetchall()
        return {r["memory_type"]: r["cnt"] for r in rows}

    def get_counts_by_state(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT state, COUNT(*) as cnt FROM memory GROUP BY state"
        ).fetchall()
        return {r["state"]: r["cnt"] for r in rows}

    def get_total_memory_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM memory").fetchone()
        return row["cnt"]

    def get_db_size(self) -> int:
        if self._db_path == ":memory:":
            return 0
        return Path(self._db_path).stat().st_size

    # --- Internal helpers ---

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            state=MemoryState(row["state"]),
            importance=row["importance"],
            stability=row["stability"],
            retrievability=row["retrievability"],
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            source=row["source"],
            conversation_id=row["conversation_id"],
            rater=row["rater"],
            rated_at=(
                datetime.fromisoformat(row["rated_at"]) if row["rated_at"] else None
            ),
            tags=json.loads(row["tags"]) if row["tags"] else [],
        )

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            rel_type=RelType(row["rel_type"]),
            strength=row["strength"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
