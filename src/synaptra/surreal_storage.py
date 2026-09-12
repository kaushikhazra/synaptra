"""SurrealDB storage layer — replaces SQLite with embedded SurrealDB."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from surrealdb import RecordID, Surreal

from .models import (
    ConsolidationLogEntry,
    Memory,
    MemoryState,
    MemoryType,
    MemoryVersion,
    RelType,
    Relationship,
    ReinforceUpdate,
    SpreadingActivationRow,
)

SCHEMA_PATH = Path(__file__).parent / "schema.surql"

# Valid SurrealDB edge table names (one per RelType)
REL_TABLES = {
    "causes": "causes",
    "follows": "follows",
    "contradicts": "contradicts",
    "supports": "supports",
    "relates_to": "relates_to",
    "supersedes": "supersedes",
    "part_of": "part_of",
    "describes": "describes",
}


def _rid(table: str, record_id: str) -> str:
    """Build a SurrealDB record ID string like 'memory:abc123'."""
    return f"{table}:{record_id}"


def _extract_id(surreal_id) -> str:
    """Extract the plain ID string from a SurrealDB RecordID or string.

    SurrealDB wraps complex IDs (e.g., UUIDs) in angle brackets: memory:⟨uuid⟩
    This function strips both the table prefix and angle brackets.
    """
    s = str(surreal_id)
    if ":" in s:
        s = s.split(":", 1)[1]
    # Strip SurrealDB angle brackets (U+27E8 / U+27E9) used for complex record IDs
    return s.strip("\u27e8\u27e9")


def _to_iso(dt: datetime) -> str:
    """Convert datetime to ISO string for SurrealDB."""
    return dt.isoformat()


async def validate_edge_endpoints(storage, rel) -> None:
    """Refuse an edge whose endpoints do not resolve to existing memories.

    Lives at the STORAGE layer, not the engine, so there is exactly one
    chokepoint and no privileged callers.  An invariant enforced only on the
    path a caller happens to take is not an invariant — and the internal
    writers (consolidation, _auto_link, _contradiction_check) are precisely the
    ones nobody would think to audit later.

    Shared by both backends rather than duplicated: two copies of a rule are
    two chances to fix only one of them.

    Archived memories are VALID endpoints — the row still exists, and edges into
    archived memories are the normal state of an aging graph.  Refusing them
    would break consolidation, a worse bug than the one being fixed.

    Raises:
        ValueError: naming exactly which endpoint(s) did not resolve.
    """
    unresolved: list[str] = []
    for label, mid in (("source_id", rel.source_id), ("target_id", rel.target_id)):
        if not mid or await storage.get_memory(mid) is None:
            unresolved.append(f"{label}={mid!r}")

    if unresolved:
        raise ValueError(
            "Cannot create relationship: "
            + " and ".join(unresolved)
            + " does not resolve to an existing memory. "
            "Both endpoints must exist (archived memories are valid endpoints). "
            "No edge was created."
        )


class SurrealStorage:
    """SurrealDB embedded storage backend for synaptra memory system."""

    def __init__(self, db_path: str = "mem://"):
        self._db_path = db_path
        self._db = Surreal(db_path)
        self._db.connect()
        self._db.use("cognitive", "memory")
        self._ensure_schema()

    def _rows(self, result) -> list[dict]:
        """Normalize SurrealDB query result to a flat list of dicts."""
        if not result:
            return []
        if isinstance(result, list):
            if result and isinstance(result[0], dict):
                return result  # Already a flat list of dicts
            # Nested list (multi-statement query)
            flat = []
            for item in result:
                if isinstance(item, list):
                    flat.extend(item)
                elif isinstance(item, dict):
                    if "result" in item:
                        flat.extend(item["result"] if isinstance(item["result"], list) else [item["result"]])
                    else:
                        flat.append(item)
            return flat
        if isinstance(result, dict):
            return [result]
        return []

    def _ensure_schema(self) -> None:
        """Apply schema (DEFINE statements are idempotent)."""
        schema_sql = SCHEMA_PATH.read_text()
        for stmt in schema_sql.split(";"):
            # Strip comment lines from each statement block
            lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                try:
                    self._db.query(clean)
                except Exception as e:
                    logger.warning("Schema statement failed: %s — %s", clean[:80], e)

    def _check_result(self, result, operation: str) -> None:
        """Raise ValueError if a SurrealDB query result signals an error.

        SurrealDB does not always raise Python exceptions on write failures —
        it returns error strings or dicts with status='ERR' instead.  Callers
        must explicitly inspect the result after every write.
        """
        if isinstance(result, str):
            raise ValueError(f"SurrealDB {operation} failed: {result}")
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            if isinstance(first, dict) and first.get("status") == "ERR":
                raise ValueError(
                    f"SurrealDB {operation} failed: {first.get('result', 'unknown error')}"
                )

    def close(self) -> None:
        pass  # Embedded SurrealDB cleans up on GC

    # --- Memory CRUD ---

    async def insert_memory(self, memory: Memory, embedding: list[float] | None = None) -> None:
        result = self._db.query(
            """CREATE type::thing('memory', $id) SET
                content = $content,
                memory_type = $memory_type,
                state = $state,
                importance = $importance,
                stability = $stability,
                retrievability = $retrievability,
                access_count = $access_count,
                created_at = $created_at,
                updated_at = $updated_at,
                last_accessed = $last_accessed,
                source = $source,
                conversation_id = $conversation_id,
                tags = $tags,
                embedding = $embedding
            """,
            {
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "state": memory.state.value,
                "importance": memory.importance,
                "stability": memory.stability,
                "retrievability": memory.retrievability,
                "access_count": memory.access_count,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "last_accessed": memory.last_accessed,
                "source": memory.source,
                "conversation_id": memory.conversation_id,
                "tags": memory.tags,
                "embedding": embedding,
            },
        )
        self._check_result(result, "insert_memory")

    async def get_memory(self, memory_id: str) -> Memory | None:
        result = self._db.query(
            "SELECT * FROM type::thing('memory', $id)",
            {"id": memory_id},
        )
        rows = self._rows(result)
        if not rows:
            return None
        return self._row_to_memory(rows[0])

    async def update_memory_fields(self, memory_id: str, **fields) -> None:
        if not fields:
            return
        set_parts = []
        params = {"id": memory_id}
        for key, value in fields.items():
            param_name = f"f_{key}"
            if key == "memory_type" and isinstance(value, MemoryType):
                value = value.value
            elif key == "state" and isinstance(value, MemoryState):
                value = value.value
            # datetime objects are passed directly to SurrealDB SDK
            set_parts.append(f"{key} = ${param_name}")
            params[param_name] = value
        set_clause = ", ".join(set_parts)
        self._db.query(
            f"UPDATE type::thing('memory', $id) SET {set_clause}",
            params,
        )

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None:
        self._db.query(
            "UPDATE type::thing('memory', $id) SET embedding = $embedding",
            {"id": memory_id, "embedding": embedding},
        )

    async def delete_memory(self, memory_id: str) -> None:
        rid = _rid("memory", memory_id)
        # Delete all edges (each relationship type)
        for table in REL_TABLES.values():
            self._db.query(
                f"DELETE {table} WHERE in = type::thing('memory', $id) OR out = type::thing('memory', $id)",
                {"id": memory_id},
            )
        # Delete versions
        self._db.query(
            "DELETE memory_version WHERE memory_id = type::thing('memory', $id)",
            {"id": memory_id},
        )
        # Delete memory
        self._db.query(
            "DELETE type::thing('memory', $id)",
            {"id": memory_id},
        )

    async def list_memories(
        self,
        search: str | None = None,
        memory_type: str | None = None,
        state: str | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        importance_min: float | None = None,
        importance_max: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        conditions = []
        params: dict[str, Any] = {}

        if search:
            conditions.append("content @@ $search")
            params["search"] = search

        if memory_type:
            conditions.append("memory_type = $mtype")
            params["mtype"] = memory_type

        if state:
            conditions.append("state = $state")
            params["state"] = state

        if tags:
            for i, tag in enumerate(tags):
                conditions.append(f"$tag_{i} IN tags")
                params[f"tag_{i}"] = tag

        if time_range:
            conditions.append("created_at >= $t_start AND created_at <= $t_end")
            params["t_start"] = _to_iso(time_range[0])
            params["t_end"] = _to_iso(time_range[1])

        if importance_min is not None:
            conditions.append("importance >= $imp_min")
            params["imp_min"] = importance_min

        if importance_max is not None:
            conditions.append("importance <= $imp_max")
            params["imp_max"] = importance_max

        where = " AND ".join(conditions) if conditions else "true"
        params["lim"] = limit
        params["off"] = offset

        result = self._db.query(
            f"SELECT * FROM memory WHERE {where} ORDER BY created_at DESC LIMIT $lim START $off",
            params,
        )
        rows = self._rows(result)
        return [self._row_to_memory(r) for r in rows]

    async def get_all_active_memories(self) -> list[Memory]:
        result = self._db.query("SELECT * FROM memory WHERE state = 'active'")
        rows = self._rows(result)
        return [self._row_to_memory(r) for r in rows]

    async def get_memories_by_ids(self, ids: list[str]) -> list[Memory]:
        if not ids:
            return []
        result = self._db.query(
            "SELECT * FROM $records",
            {"records": [RecordID("memory", memory_id) for memory_id in ids]},
        )
        rows = self._rows(result)
        return [self._row_to_memory(r) for r in rows]

    async def get_recent_active_ids(
        self,
        limit: int,
        type_filter: str | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[tuple[str, datetime]]:
        conditions = ["state = 'active'"]
        params: dict[str, Any] = {"limit": limit}

        if type_filter:
            conditions.append("memory_type = $type_filter")
            params["type_filter"] = type_filter

        if time_range:
            conditions.append("created_at >= $time_start")
            conditions.append("created_at <= $time_end")
            params["time_start"] = _to_iso(time_range[0])
            params["time_end"] = _to_iso(time_range[1])

        result = self._db.query(
            f"""SELECT id, last_accessed FROM memory
                WHERE {' AND '.join(conditions)}
                ORDER BY last_accessed DESC
                LIMIT $limit""",
            params,
        )
        rows = self._rows(result)
        out: list[tuple[str, datetime]] = []
        for row in rows:
            last_accessed = self._parse_dt(row["last_accessed"])
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            out.append((_extract_id(row["id"]), last_accessed))
        return out

    async def fts_search(
        self,
        query: str,
        state: str = "active",
        limit: int = 30,
        type_filter: str | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[tuple[str, float]]:
        result = self._db.query(
            """SELECT id, search::score(1) AS score
               FROM memory
               WHERE content @1@ $query AND state = $state
               ORDER BY score DESC
               LIMIT $lim""",
            {"query": query, "state": state, "lim": limit},
        )
        rows = self._rows(result)
        return [(_extract_id(r["id"]), r["score"]) for r in rows]

    async def vector_search(
        self,
        embedding: list[float],
        state: str = "active",
        top_k: int = 30,
        type_filter: str | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[tuple[str, float]]:
        """Vector similarity search using SurrealDB HNSW index."""
        result = self._db.query(
            """SELECT id, vector::similarity::cosine(embedding, $vec) AS score
               FROM memory
               WHERE state = $state AND embedding != NONE
               ORDER BY score DESC
               LIMIT $top_k""",
            {"vec": embedding, "state": state, "top_k": top_k},
        )
        rows = self._rows(result)
        return [(_extract_id(r["id"]), r["score"]) for r in rows]

    async def vector_search_for_memory(self, memory_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find similar memories to an existing memory by its stored embedding."""
        result = self._db.query(
            """LET $vec = (SELECT embedding FROM type::thing('memory', $id))[0].embedding;
               SELECT id, vector::similarity::cosine(embedding, $vec) AS score
               FROM memory
               WHERE state = 'active' AND embedding != NONE AND id != type::thing('memory', $id)
               ORDER BY score DESC
               LIMIT $top_k""",
            {"id": memory_id, "top_k": top_k},
        )
        # Result may be nested due to LET
        rows = result[-1] if result else []
        if isinstance(rows, dict):
            rows = [rows]
        return [(_extract_id(r["id"]), r["score"]) for r in rows if "score" in r]

    # --- Memory Versions ---

    async def insert_version(self, version: MemoryVersion) -> None:
        result = self._db.query(
            """CREATE type::thing('memory_version', $id) SET
                memory_id = type::thing('memory', $mem_id),
                content = $content,
                metadata = $metadata,
                created_at = $created_at
            """,
            {
                "id": version.id,
                "mem_id": version.memory_id,
                "content": version.content,
                "metadata": version.metadata,
                "created_at": version.created_at,
            },
        )
        self._check_result(result, "insert_version")

    async def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        result = self._db.query(
            "SELECT * FROM memory_version WHERE memory_id = type::thing('memory', $id) ORDER BY created_at DESC",
            {"id": memory_id},
        )
        rows = self._rows(result)
        return [
            MemoryVersion(
                id=_extract_id(r["id"]),
                memory_id=memory_id,
                content=r["content"],
                metadata=r.get("metadata"),
                created_at=self._parse_dt(r["created_at"]),
            )
            for r in rows
        ]

    # --- Relationships ---

    async def insert_relationship(self, rel: Relationship) -> None:
        # Single chokepoint — see validate_edge_endpoints.
        await validate_edge_endpoints(self, rel)

        table = REL_TABLES[rel.rel_type.value]
        result = self._db.query(
            f"""LET $from = type::thing('memory', $src);
                LET $to = type::thing('memory', $tgt);
                RELATE $from->{table}->$to
                SET strength = $strength, created_at = $created_at""",
            {
                "src": rel.source_id,
                "tgt": rel.target_id,
                "strength": rel.strength,
                "created_at": rel.created_at,
            },
        )
        self._check_result(result, f"insert_relationship({table})")

    async def delete_relationship(self, source_id: str, target_id: str, rel_type: str) -> bool:
        table = REL_TABLES[rel_type]
        result = self._db.query(
            f"DELETE {table} WHERE in = type::thing('memory', $src) AND out = type::thing('memory', $tgt)",
            {"src": source_id, "tgt": target_id},
        )
        return True  # SurrealDB DELETE doesn't return rowcount easily

    async def get_relationships_for(self, memory_id: str, rel_types: list[str] | None = None) -> list[Relationship]:
        tables = [REL_TABLES[rt] for rt in rel_types] if rel_types else list(REL_TABLES.values())
        all_rels = []
        for table in tables:
            result = self._db.query(
                f"SELECT * FROM {table} WHERE in = type::thing('memory', $id) OR out = type::thing('memory', $id)",
                {"id": memory_id},
            )
            rows = self._rows(result)
            for r in rows:
                all_rels.append(self._row_to_relationship(r, table))
        return all_rels

    async def get_outgoing_relationships(self, memory_id: str) -> list[Relationship]:
        all_rels = []
        for table in REL_TABLES.values():
            result = self._db.query(
                f"SELECT * FROM {table} WHERE in = type::thing('memory', $id)",
                {"id": memory_id},
            )
            rows = self._rows(result)
            for r in rows:
                all_rels.append(self._row_to_relationship(r, table))
        return all_rels

    async def get_incoming_relationships(self, memory_id: str, rel_type: str | None = None) -> list[Relationship]:
        tables = [REL_TABLES[rel_type]] if rel_type else list(REL_TABLES.values())
        all_rels = []
        for table in tables:
            result = self._db.query(
                f"SELECT * FROM {table} WHERE out = type::thing('memory', $id)",
                {"id": memory_id},
            )
            rows = self._rows(result)
            for r in rows:
                all_rels.append(self._row_to_relationship(r, table))
        return all_rels

    async def get_neighbors(self, memory_id: str, active_only: bool = True) -> list[tuple[str, Relationship]]:
        results = []
        for table in REL_TABLES.values():
            for r in self._rows(self._db.query(
                f"SELECT *, out.state AS neighbor_state FROM {table} WHERE in = type::thing('memory', $id)",
                {"id": memory_id},
            )):
                if active_only and r.get("neighbor_state") != "active":
                    continue
                neighbor_id = _extract_id(r["out"])
                results.append((neighbor_id, self._row_to_relationship(r, table)))

            for r in self._rows(self._db.query(
                f"SELECT *, in.state AS neighbor_state FROM {table} WHERE out = type::thing('memory', $id)",
                {"id": memory_id},
            )):
                if active_only and r.get("neighbor_state") != "active":
                    continue
                neighbor_id = _extract_id(r["in"])
                results.append((neighbor_id, self._row_to_relationship(r, table)))
        return results

    async def delete_auto_links(self, memory_id: str) -> None:
        self._db.query(
            "DELETE relates_to WHERE in = type::thing('memory', $id) AND strength < 1.0",
            {"id": memory_id},
        )

    async def bulk_update_stability(self, updates: list[tuple[float, str]]) -> None:
        if not updates:
            return
        self._db.query(
            """LET $updates = $updates;
               FOR $row IN $updates {
                   UPDATE type::thing('memory', $row.id) SET stability = $row.stability;
               };""",
            {
                "updates": [
                    {"id": mem_id, "stability": new_stability}
                    for new_stability, mem_id in updates
                ],
            },
        )

    async def has_incoming_supersedes(self, memory_id: str) -> bool:
        result = self._db.query(
            """SELECT count() AS cnt FROM supersedes
               WHERE out = type::thing('memory', $id) AND in.state = 'active'
               GROUP ALL""",
            {"id": memory_id},
        )
        rows = self._rows(result)
        if rows and isinstance(rows, list) and len(rows) > 0:
            return rows[0].get("cnt", 0) > 0
        return False

    async def get_contradictions_for(self, memory_id: str) -> list[tuple[str, str, float]]:
        result = self._db.query(
            """SELECT out AS other_id, out.content AS other_content, strength
               FROM contradicts WHERE in = type::thing('memory', $id) AND out.state = 'active'""",
            {"id": memory_id},
        )
        rows_out = self._rows(result)

        rows_in = self._rows(self._db.query(
            """SELECT in AS other_id, in.content AS other_content, strength
               FROM contradicts WHERE out = type::thing('memory', $id) AND in.state = 'active'""",
            {"id": memory_id},
        ))

        all_rows = rows_out + rows_in
        return [
            (_extract_id(r["other_id"]), str(r.get("other_content", ""))[:100], r["strength"])
            for r in all_rows
        ]

    # --- Consolidation Log ---

    async def insert_consolidation_log(self, entry: ConsolidationLogEntry) -> None:
        result = self._db.query(
            """CREATE type::thing('consolidation_log', $id) SET
                action = $action,
                source_ids = $source_ids,
                target_id = $target_id,
                reason = $reason,
                created_at = $created_at
            """,
            {
                "id": entry.id,
                "action": entry.action,
                "source_ids": entry.source_ids,
                "target_id": entry.target_id,
                "reason": entry.reason,
                "created_at": entry.created_at,
            },
        )
        self._check_result(result, "insert_consolidation_log")

    async def get_last_consolidation(self) -> dict | None:
        result = self._db.query(
            "SELECT * FROM consolidation_log ORDER BY created_at DESC LIMIT 1"
        )
        rows = self._rows(result)
        if not rows:
            return None
        r = rows[0]
        return {
            "id": _extract_id(r["id"]),
            "action": r["action"],
            "source_ids": r["source_ids"],
            "target_id": r.get("target_id"),
            "reason": r["reason"],
            "created_at": str(r["created_at"]),
        }

    async def get_consolidation_summary(self) -> dict:
        last = await self.get_last_consolidation()
        if last is None:
            return {"last_run": None, "promoted": 0, "merged": 0, "archived": 0}

        last_run = last["created_at"]
        result = self._db.query(
            "SELECT action, count() AS cnt FROM consolidation_log WHERE created_at >= $since GROUP BY action",
            {"since": last_run},
        )
        rows = self._rows(result)
        summary = {"last_run": last_run, "promoted": 0, "merged": 0, "archived": 0}
        for row in rows:
            action = row.get("action")
            cnt = row.get("cnt", 0)
            if action == "promote":
                summary["promoted"] = cnt
            elif action == "merge":
                summary["merged"] = cnt
            elif action == "archive":
                summary["archived"] = cnt
        return summary

    # --- Health report queries ---

    async def get_counts_by_type_and_state(self) -> dict:
        """Return counts keyed by (memory_type, state) tuples for engine reshaping."""
        result = self._db.query(
            "SELECT memory_type, state, count() AS cnt FROM memory GROUP BY memory_type, state"
        )
        rows = self._rows(result)
        return {(r["memory_type"], r["state"]): r["cnt"] for r in rows}

    async def get_active_memories_for_decay(self) -> list[dict]:
        """Fetch active memories with decay-relevant fields only (no embedding blob)."""
        result = self._db.query(
            """SELECT id, string::slice(content, 0, 120) AS content_preview,
                      memory_type, importance, stability, last_accessed, tags
               FROM memory
               WHERE state = 'active'"""
        )
        rows = self._rows(result)
        out = []
        for r in rows:
            out.append({
                "id": _extract_id(r["id"]),
                "content_preview": r.get("content_preview") or "",
                "memory_type": r["memory_type"],
                "importance": r["importance"],
                "stability": r["stability"],
                "last_accessed": self._parse_dt(r["last_accessed"]),
                "tags": r.get("tags") or [],
            })
        return out

    async def get_orphan_untagged(self) -> tuple[list[dict], int]:
        """Return active memories with no tags: (capped_list[:50], true_count)."""
        count_result = self._db.query(
            """SELECT count() AS cnt FROM memory
               WHERE state = 'active' AND (tags IS NONE OR array::len(tags) = 0)
               GROUP ALL"""
        )
        count_rows = self._rows(count_result)
        true_count = count_rows[0].get("cnt", 0) if count_rows else 0

        list_result = self._db.query(
            """SELECT id, string::slice(content, 0, 120) AS content_preview,
                      memory_type, created_at
               FROM memory
               WHERE state = 'active' AND (tags IS NONE OR array::len(tags) = 0)
               ORDER BY created_at ASC
               LIMIT 51"""
        )
        rows = self._rows(list_result)
        items = [
            {
                "id": _extract_id(r["id"]),
                "content_preview": r.get("content_preview") or "",
                "memory_type": r["memory_type"],
                "created_at": self._parse_dt(r["created_at"]).isoformat(),
            }
            for r in rows
        ]
        return (items[:50], true_count)

    async def get_orphan_unconnected(self) -> tuple[list[dict], int]:
        """Return active memories with no relationships in any edge table: (capped_list[:50], true_count).

        Collects the edge-endpoint set with one SELECT per edge table and side,
        then filters in Python.  Not a LET: the original bug WAS a planner
        assumption, and fixing it with another would trade one unverified
        belief for another.
        """
        # _EDGE_IDS in WHERE is a CORRELATED SUBQUERY — SurrealDB re-evaluates all
        # 16 SELECTs plus the flatten/distinct once PER CANDIDATE ROW.  On the
        # server backend that made memory_health hang past 300 s; the same shape
        # is here, so it is fixed here too rather than left as a latent trap.
        # Collect the endpoint set once, then filter.
        edge_ids: set[str] = set()
        for rel in REL_TABLES.values():
            for side in ("in", "out"):
                rows = self._db.query(f"SELECT VALUE {side} FROM {rel}")
                if isinstance(rows, list):
                    edge_ids.update(str(r) for r in rows if r is not None)

        list_result = self._db.query(
            "SELECT id, string::slice(content, 0, 120) AS content_preview, "
            "memory_type, tags, created_at "
            "FROM memory WHERE state = 'active' "
            "ORDER BY created_at ASC"
        )
        unconnected = [
            r
            for r in self._rows(list_result)
            if isinstance(r, dict) and "id" in r and str(r["id"]) not in edge_ids
        ]
        true_count = len(unconnected)
        if true_count == 0:
            return ([], 0)

        items = [
            {
                "id": _extract_id(r["id"]),
                "content_preview": r.get("content_preview") or "",
                "memory_type": r["memory_type"],
                "tags": r.get("tags") or [],
                "created_at": self._parse_dt(r["created_at"]).isoformat(),
            }
            for r in unconnected[:50]
        ]
        return (items, true_count)

    async def get_tag_frequencies(self) -> list[list[str]]:
        """Return all active memory tag arrays; engine flattens and counts (D7)."""
        result = self._db.query(
            "SELECT tags FROM memory WHERE state = 'active'"
        )
        rows = self._rows(result)
        return [r.get("tags") or [] for r in rows]

    async def get_health_consolidation_summary(self) -> dict | None:
        """Return last consolidation run summary grouped by UTC date.

        Named distinctly from get_last_consolidation() to avoid collision (D3).
        Returns None when consolidation_log is empty (never run).
        Engine is responsible for wrapping None into the never_run shape (D4).
        """
        result = self._db.query(
            "SELECT * FROM consolidation_log ORDER BY created_at DESC LIMIT 500"
        )
        rows = self._rows(result)
        if not rows:
            return None

        # Group rows by UTC date — same date = same run (no run_id column, D8)
        most_recent_date: str | None = None
        last_run_at: datetime | None = None
        action_counts: dict[str, int] = {}

        for r in rows:
            dt = self._parse_dt(r["created_at"])
            dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            date_str = dt_utc.date().isoformat()

            if most_recent_date is None:
                most_recent_date = date_str
                last_run_at = dt_utc

            if date_str == most_recent_date:
                action = r.get("action", "")
                action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "last_run_at": last_run_at.isoformat() if last_run_at else None,
            "last_run_summary": {
                "promote": action_counts.get("promote", 0),
                "archive": action_counts.get("archive", 0),
                "merge": action_counts.get("merge", 0),
                "flag_contradiction": action_counts.get("flag_contradiction", 0),
            },
        }

    # --- Config ---

    def get_config(self, key: str) -> Any | None:
        result = self._db.query(
            "SELECT val FROM preference WHERE id = type::thing('preference', $key)",
            {"key": key},
        )
        rows = self._rows(result)
        if not rows:
            return None
        return rows[0].get("val")

    def set_config(self, key: str, value: Any) -> None:
        now = datetime.now(timezone.utc)
        self._db.query(
            """UPSERT type::thing('preference', $key) SET
                val = $val,
                updated_at = $now
            """,
            {"key": key, "val": value, "now": now},
        )

    def get_all_config(self) -> dict[str, Any]:
        result = self._db.query("SELECT * FROM preference")
        rows = self._rows(result)
        return {_extract_id(r["id"]): r["val"] for r in rows}

    # --- Stats helpers ---

    async def get_counts_by_type(self) -> dict[str, int]:
        result = self._db.query(
            "SELECT memory_type, count() AS cnt FROM memory GROUP BY memory_type"
        )
        rows = self._rows(result)
        return {r["memory_type"]: r["cnt"] for r in rows}

    async def get_counts_by_state(self) -> dict[str, int]:
        result = self._db.query(
            "SELECT state, count() AS cnt FROM memory GROUP BY state"
        )
        rows = self._rows(result)
        return {r["state"]: r["cnt"] for r in rows}

    async def get_total_memory_count(self) -> int:
        result = self._db.query("SELECT count() AS cnt FROM memory GROUP ALL")
        rows = self._rows(result)
        if rows and isinstance(rows, list) and len(rows) > 0:
            return rows[0].get("cnt", 0)
        return 0

    async def get_db_size(self) -> int:
        if self._db_path.startswith("mem://"):
            return 0
        # For file-based, estimate from the data directory
        data_path = self._db_path.replace("surrealkv://", "").replace("file://", "")
        p = Path(data_path)
        if p.exists():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return 0

    # --- Batch / bulk methods (Phase 3-5 equivalents for embedded backend) ---
    # Implemented via Python-loop over per-id methods for simplicity; the embedded
    # backend is used in tests and development where absolute performance is not
    # critical.  The SurrealServerStorage equivalents use server-side bulk queries.

    async def get_neighbors_bulk(
        self, seed_ids: list[str]
    ) -> dict[str, list[tuple[str, Relationship]]]:
        """Return 1-hop neighbors for all seeds in one logical call.

        Uses per-seed get_neighbors() loops so the embedded SurrealDB sync
        client is not required to support multi-parameter IN clauses.
        """
        result: dict[str, list[tuple[str, Relationship]]] = {}
        for seed_id in seed_ids:
            result[seed_id] = await self.get_neighbors(seed_id, active_only=True)
        return result

    async def get_supersede_flags(self, ids: list[str]) -> set[str]:
        """Return subset of ids that have an incoming supersedes edge."""
        superseded: set[str] = set()
        for memory_id in ids:
            if await self.has_incoming_supersedes(memory_id):
                superseded.add(memory_id)
        return superseded

    async def get_contradictions_bulk(
        self, ids: list[str]
    ) -> dict[str, list[tuple[str, str, float]]]:
        """Return bidirectional contradictions for all ids, grouped by anchor id."""
        result: dict[str, list[tuple[str, str, float]]] = {}
        for memory_id in ids:
            result[memory_id] = await self.get_contradictions_for(memory_id)
        return result

    async def bulk_reinforce(self, updates: list[ReinforceUpdate]) -> None:
        """Apply reinforcement updates for multiple memories."""
        for u in updates:
            await self.update_memory_fields(
                u.memory_id,
                stability=u.stability,
                last_accessed=u.last_accessed,
                access_count=u.access_count,
            )

    async def spreading_activation_walk(
        self, seed_ids: list[str], max_depth: int = 3
    ) -> list[SpreadingActivationRow]:
        """BFS spreading-activation walk from seed_ids up to max_depth hops.

        Returns enriched rows (neighbor_id, depth, rel_strength, current_stability,
        state).  Uses per-node get_neighbors() internally — equivalent semantics to
        SurrealServerStorage but via Python BFS rather than a server-side walk query.
        Seed IDs are excluded from results.
        """
        from collections import deque

        seen: set[str] = set(seed_ids)
        queue: deque[tuple[str, int]] = deque((sid, 0) for sid in seed_ids)
        rows: list[SpreadingActivationRow] = []

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            neighbors = await self.get_neighbors(current_id, active_only=True)
            for neighbor_id, rel in neighbors:
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                # Fetch stability for this neighbor
                mem = await self.get_memory(neighbor_id)
                if mem is None or mem.state.value != "active":
                    continue
                rows.append(SpreadingActivationRow(
                    neighbor_id=neighbor_id,
                    depth=depth + 1,
                    rel_strength=rel.strength,
                    current_stability=mem.stability,
                    state=mem.state.value,
                ))
                queue.append((neighbor_id, depth + 1))

        return rows

    # --- Internal helpers ---

    def _parse_dt(self, val) -> datetime:
        """Parse datetime from SurrealDB — may be datetime object or ISO string."""
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val))

    def _row_to_memory(self, row: dict) -> Memory:
        return Memory(
            id=_extract_id(row["id"]),
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            state=MemoryState(row["state"]),
            importance=row["importance"],
            stability=row["stability"],
            retrievability=row["retrievability"],
            access_count=row["access_count"],
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
            last_accessed=self._parse_dt(row["last_accessed"]),
            source=row.get("source"),
            conversation_id=row.get("conversation_id"),
            tags=row.get("tags", []),
        )

    def _row_to_relationship(self, row: dict, table: str) -> Relationship:
        return Relationship(
            id=_extract_id(row["id"]),
            source_id=_extract_id(row["in"]),
            target_id=_extract_id(row["out"]),
            rel_type=RelType(table),
            strength=row.get("strength", 1.0),
            created_at=self._parse_dt(row["created_at"]),
        )
