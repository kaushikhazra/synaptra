"""SurrealServerStorage — server-backed storage adapter for CognitiveMemory.

Connects to a separately-running SurrealDB server process via async WebSocket
(AsyncWsSurrealConnection).  Implements the full StorageProtocol including all
six health methods required by MemoryEngine.get_health().

Connection model (design Component C):
- URL default: ws://127.0.0.1:8000/rpc
- Namespace/database: cognitive / memory
- Lazy connect on first use; coroutine-safe via asyncio.Lock
- Reconnect on transport drop with exponential backoff: 1s, 2s, 4s, 8s, 16s (5 max)
- Each retry is logged; after 5 failures a ConnectionError is raised to the MCP layer

get_db_size() semantics (W8):
- Reads directory size from SYNAPTRA_SURREAL_DATA_DIR env var
  (default: ~/.synaptra/surreal/)
- Returns -1 and logs warning when directory is inaccessible
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import shared helpers and constants from the embedded storage module to avoid duplication
from .surreal_storage import (
    SCHEMA_PATH,
    REL_TABLES,
    _extract_id,
    _rid,
    _to_iso,
    validate_edge_endpoints,
)

from surrealdb import AsyncSurreal, RecordID
from websockets.exceptions import ConnectionClosed, WebSocketException

from .models import (
    ConsolidationLogEntry,
    Memory,
    MemoryState,
    MemoryType,
    MemoryVersion,
    ReinforceUpdate,
    RelType,
    Relationship,
    SpreadingActivationRow,
)


# ──────────────────────────────────────────────────────────────────────────────
# Spreading-activation walk: SQL builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_layer_concat(frontier_var: str, excl_var: str) -> str:
    """Return a nested array::concat expression for one BFS depth layer.

    Produces 16 subqueries: 2 directions (outgoing + incoming) × 8 relation
    tables.  The result of each subquery is a dict with keys
    ``{neighbor, strength}``.  The 16 results are merged via 15 nested
    ``array::concat`` calls (SurrealDB 2.x has no variadic array-union).

    ``frontier_var``  — SurrealQL variable holding the current-hop record-IDs
                        (e.g. ``$seeds`` for depth 1, ``$d1_ids`` for depth 2)
    ``excl_var``      — SurrealQL variable holding all already-visited IDs
                        (seeds + all nodes seen at shallower depths)
    """
    tables = list(REL_TABLES.values())  # 8 tables in definition order
    subqueries: list[str] = []
    for table in tables:
        subqueries.append(
            f"(SELECT out AS neighbor, strength FROM {table}"
            f" WHERE in IN {frontier_var} AND out.state = 'active' AND out NOT IN {excl_var})"
        )
        subqueries.append(
            f"(SELECT in AS neighbor, strength FROM {table}"
            f" WHERE out IN {frontier_var} AND in.state = 'active' AND in NOT IN {excl_var})"
        )
    # Nest 16 items into 15 binary array::concat calls
    result = subqueries[0]
    for sq in subqueries[1:]:
        result = f"array::concat({result}, {sq})"
    return result


def _build_walk_sql(max_depth: int) -> str:
    """Build the full layered-LET spreading-activation walk query.

    Generates one LET block per depth level (1 … max_depth).  Each block:
    - computes the raw neighbor set for that depth via ``_build_layer_concat``
    - extracts the IDs for use as the next layer's frontier
    - computes the cumulative exclusion set so each node appears only at its
      shallowest depth

    After all depth layers a final GROUP-BY deduplicates (safety net) and
    a RETURN statement emits the result.

    The query expects a single parameter: ``$seeds`` — a list of
    ``memory:<id>`` record references.

    Returns rows with keys:
    ``{neighbor_id, depth, rel_strength, current_stability, state,
    last_accessed, memory_type}``
    """
    lines: list[str] = []

    for d in range(1, max_depth + 1):
        frontier = "$seeds" if d == 1 else f"$d{d - 1}_ids"
        excl = "$seeds" if d == 1 else f"$excl{d}"

        layer = _build_layer_concat(frontier, excl)
        lines.append(f"LET $d{d}_raw = {layer};")
        lines.append(f"LET $d{d}_ids = $d{d}_raw.neighbor;")

        # Build the exclusion set for the *next* layer (not needed after last layer)
        if d < max_depth:
            if d == 1:
                lines.append(f"LET $excl{d + 1} = array::concat($seeds, $d{d}_ids);")
            else:
                lines.append(f"LET $excl{d + 1} = array::concat($excl{d}, $d{d}_ids);")

    # Tag every raw layer with its depth and enrich with neighbor fields
    for d in range(1, max_depth + 1):
        lines.append(
            f"LET $d{d}_tagged = (SELECT neighbor.id AS neighbor_id, strength, {d} AS depth,"
            f" neighbor.stability AS current_stability,"
            f" neighbor.last_accessed AS last_accessed,"
            f" neighbor.memory_type AS memory_type,"
            f" 'active' AS state FROM $d{d}_raw);"
        )

    # Combine all tagged layers into $all
    combined = "$d1_tagged"
    for d in range(2, max_depth + 1):
        combined = f"array::concat({combined}, $d{d}_tagged)"
    lines.append(f"LET $all = {combined};")

    # GROUP BY dedup: keep shallowest depth and max strength/stability per neighbor
    lines.append(
        "LET $deduped = SELECT neighbor_id,"
        " math::min(depth) AS depth,"
        " math::max(strength) AS rel_strength,"
        " math::max(current_stability) AS current_stability,"
        " memory_type,"
        " last_accessed,"
        " 'active' AS state"
        # memory_type and last_accessed are properties of the neighbour record,
        # so they are constant within a neighbor_id group.  Grouping by them is
        # how they become selectable without an aggregate; it cannot split a
        # group, because the same node cannot carry two values.
        " FROM $all GROUP BY neighbor_id, memory_type, last_accessed;"
    )
    lines.append("RETURN $deduped;")

    return "\n".join(lines)


class SurrealServerStorage:
    """Server-backed SurrealDB storage adapter.

    Uses the async AsyncWsSurrealConnection (returned by AsyncSurreal("ws://...")).
    Compatible with StorageProtocol; implements all six health report methods.
    """

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8000/rpc",
    ) -> None:
        self._url = url
        self._db = None          # type: Any  # AsyncWsSurrealConnection | None
        self._lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Connection management
    # ──────────────────────────────────────────────────────────────────────────
    # DESIGN NOTE (cm-schema-init-fix, 2026-05-30):
    # Schema is provisioned once via scripts/provision_cm_schema.py.
    # Runtime connections NEVER run DDL. No _ensure_schema, no DEFINE
    # statements. Rationale: DEFINE INDEX OVERWRITE on every connection
    # was silently dropping and async-rebuilding HNSW + FTS indexes,
    # causing vector_search and fts_search to return 0 rows during
    # the rebuild window. Schema consistency checking without a recovery
    # plan is worse than not checking at all.

    async def _connect(self) -> Any:
        """Connect to SurrealDB server with exponential backoff.

        Attempts: delays 1s, 2s, 4s, 8s, 16s (5 max).
        Must be called while holding self._lock.
        Raises ConnectionError after all attempts are exhausted.
        """
        delays = [1, 2, 4, 8, 16]
        last_error: Exception | None = None

        for attempt, delay in enumerate(delays, 1):
            try:
                db = AsyncSurreal(self._url)
                await db.connect()
                await db.use("cognitive", "memory")
                logger.info("Connected to SurrealDB server at %s", self._url)
                return db
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "SurrealDB connect attempt %d/%d failed: %s",
                    attempt, len(delays), exc,
                )
                if attempt < len(delays):
                    await asyncio.sleep(delay)

        raise ConnectionError(
            f"Cannot connect to SurrealDB server at {self._url} after "
            f"{len(delays)} attempts. Last error: {last_error}"
        ) from last_error

    async def _connect_fast(self) -> Any:
        """Single connection attempt, no retry, for use after a transport drop.

        Unlike _connect(), this does not backoff-retry.  It either succeeds
        immediately or raises ConnectionError.  Used by _query() so that a
        reconnect after a transport drop does not consume the full 31-second
        backoff window before surfacing an error to the MCP caller.
        """
        try:
            db = AsyncSurreal(self._url)
            await db.connect()
            await db.use("cognitive", "memory")
            logger.info("Reconnected to SurrealDB server at %s", self._url)
            return db
        except Exception as exc:
            raise ConnectionError(
                f"SurrealDB at {self._url} unreachable after transport drop: {exc}"
            ) from exc

    async def _get_db(self) -> Any:
        """Return a live connection; lazy-init on first call (always-acquire pattern)."""
        async with self._lock:
            if self._db is None:
                self._db = await self._connect()
            return self._db

    def _is_transport_error(self, exc: Exception) -> bool:
        """Return True if exc indicates a dropped WebSocket transport.

        Async SDK behaviour (Spike 3): _recv_task swallows ConnectionClosed /
        WebSocketException and cancels pending query futures, so callers see
        asyncio.CancelledError.  isinstance guard covers this before the string
        heuristic fallback.
        """
        # isinstance guard — covers primary async SDK failure modes
        if isinstance(exc, (ConnectionClosed, WebSocketException,
                             asyncio.CancelledError, asyncio.TimeoutError,
                             OSError)):
            return True
        # String heuristic fallback — covers future SDK wrapping patterns
        type_name = type(exc).__name__.lower()
        msg = str(exc).lower()
        transport_keywords = (
            "connectionclosed", "connectionerror", "websocket",
            "connect", "transport", "eof", "disconnect", "broken",
            "socket", "closed", "reset", "refused",
            "cancelled",   # asyncio.CancelledError — future cancelled when WS drops
            "timeout",     # asyncio.TimeoutError   — query hung after WS gone
        )
        return any(k in type_name or k in msg for k in transport_keywords)

    async def _query(self, sql: str, params: dict | None = None) -> Any:
        """Execute a query; on transport error, fast-reconnect once and retry.

        Transport drops use _connect_fast() (single attempt, no backoff) so that
        an unreachable server surfaces a clear ConnectionError within < 1s rather
        than stalling the MCP client for the full 31-second backoff window.

        The 5-attempt exponential backoff in _connect() is reserved for the lazy
        first-init path in _get_db(), where we can afford to wait for the server
        to start up.
        """
        try:
            db = await self._get_db()
            return await db.query(sql, params) if params is not None else await db.query(sql)
        except Exception as exc:
            if self._is_transport_error(exc):
                logger.warning(
                    "Transport error on query, attempting fast reconnect: %s", exc
                )
                async with self._lock:
                    self._db = None
                # Single fast attempt — no backoff — then retry the query once
                db = await self._connect_fast()
                async with self._lock:
                    self._db = db
                return await db.query(sql, params) if params is not None else await db.query(sql)
            raise

    def _check_result(self, result: Any, operation: str) -> None:
        """Raise ValueError when SurrealDB signals a write failure in the result."""
        if isinstance(result, str):
            raise ValueError(f"SurrealDB {operation} failed: {result}")
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and first.get("status") == "ERR":
                raise ValueError(
                    f"SurrealDB {operation} failed: {first.get('result', 'unknown error')}"
                )

    async def close(self) -> None:
        async with self._lock:
            if self._db is not None:
                try:
                    await self._db.close()
                except Exception:
                    pass
                self._db = None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal result helpers (mirror SurrealStorage)
    # ──────────────────────────────────────────────────────────────────────────

    def _rows(self, result: Any) -> list[dict]:
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
                        flat.extend(inner if isinstance(inner, list) else [inner])
                    else:
                        flat.append(item)
            return flat
        if isinstance(result, dict):
            return [result]
        return []

    def _parse_dt(self, val: Any) -> datetime:
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
            rater=row.get("rater"),
            rated_at=(self._parse_dt(row["rated_at"]) if row.get("rated_at") else None),
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

    # ──────────────────────────────────────────────────────────────────────────
    # Memory CRUD
    # ──────────────────────────────────────────────────────────────────────────

    async def insert_memory(self, memory: Memory, embedding: list[float] | None = None) -> None:
        result = await self._query(
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
                rater = $rater,
                rated_at = $rated_at,
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
                "rater": memory.rater,
                "rated_at": memory.rated_at,
                "tags": memory.tags,
                "embedding": embedding,
            },
        )
        self._check_result(result, "insert_memory")

    async def get_memory(self, memory_id: str) -> Memory | None:
        result = await self._query(
            "SELECT * FROM type::thing('memory', $id)",
            {"id": memory_id},
        )
        rows = self._rows(result)
        if not rows:
            return None
        return self._row_to_memory(rows[0])

    async def update_memory_fields(self, memory_id: str, **fields: Any) -> None:
        if not fields:
            return
        set_parts = []
        params: dict[str, Any] = {"id": memory_id}
        for key, value in fields.items():
            param_name = f"f_{key}"
            if key == "memory_type" and isinstance(value, MemoryType):
                value = value.value
            elif key == "state" and isinstance(value, MemoryState):
                value = value.value
            set_parts.append(f"{key} = ${param_name}")
            params[param_name] = value
        set_clause = ", ".join(set_parts)
        await self._query(
            f"UPDATE type::thing('memory', $id) SET {set_clause}",
            params,
        )

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None:
        await self._query(
            "UPDATE type::thing('memory', $id) SET embedding = $embedding",
            {"id": memory_id, "embedding": embedding},
        )

    async def delete_memory(self, memory_id: str) -> None:
        for table in REL_TABLES.values():
            await self._query(
                f"DELETE {table} WHERE in = type::thing('memory', $id) OR out = type::thing('memory', $id)",
                {"id": memory_id},
            )
        await self._query(
            "DELETE memory_version WHERE memory_id = type::thing('memory', $id)",
            {"id": memory_id},
        )
        await self._query(
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
        rater_not: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        conditions: list[str] = []
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
        if rater_not is not None:
            # The NONE branch is not optional.  A pre-#8 row has no rater, so it
            # differs from every model and must be a candidate; relying on
            # `rater != $x` alone to cover unset fields is how this returns an
            # empty list against a store where every row qualifies.
            conditions.append("(rater IS NONE OR rater != $rater_not)")
            params["rater_not"] = rater_not

        where = " AND ".join(conditions) if conditions else "true"
        params["lim"] = limit
        params["off"] = offset

        result = await self._query(
            f"SELECT * FROM memory WHERE {where} ORDER BY created_at DESC LIMIT $lim START $off",
            params,
        )
        return [self._row_to_memory(r) for r in self._rows(result)]

    async def get_all_active_memories(self) -> list[Memory]:
        result = await self._query("SELECT * FROM memory WHERE state = 'active'")
        return [self._row_to_memory(r) for r in self._rows(result)]

    # ──────────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────────

    async def get_memories_by_ids(self, ids: list[str]) -> list[Memory]:
        if not ids:
            return []
        result = await self._query(
            "SELECT * FROM $records",
            {"records": [RecordID("memory", memory_id) for memory_id in ids]},
        )
        return [self._row_to_memory(r) for r in self._rows(result)]

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

        if tags:
            conditions.append("tags CONTAINSALL $tags")
            params["tags"] = tags

        if time_range:
            conditions.append("created_at >= $time_start")
            conditions.append("created_at <= $time_end")
            params["time_start"] = _to_iso(time_range[0])
            params["time_end"] = _to_iso(time_range[1])

        result = await self._query(
            f"""SELECT id, last_accessed FROM memory
                WHERE {' AND '.join(conditions)}
                ORDER BY last_accessed DESC
                LIMIT $limit""",
            params,
        )
        out: list[tuple[str, datetime]] = []
        for row in self._rows(result):
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
        params: dict[str, Any] = {"query": query, "state": state, "lim": limit}
        extra: list[str] = []

        if type_filter:
            extra.append("memory_type = $type_filter")
            params["type_filter"] = type_filter

        if tags:
            extra.append("tags CONTAINSALL $tags")
            params["tags"] = tags

        if time_range:
            extra.append("created_at >= $time_start")
            extra.append("created_at <= $time_end")
            params["time_start"] = _to_iso(time_range[0])
            params["time_end"] = _to_iso(time_range[1])

        extra_sql = "".join(f" AND {c}" for c in extra)
        result = await self._query(
            f"""SELECT id, search::score(1) AS score
               FROM memory
               WHERE content @1@ $query AND state = $state{extra_sql}
               ORDER BY score DESC
               LIMIT $lim""",
            params,
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
        top_k = max(1, int(top_k))
        fetch_k = top_k * 3
        # Step 1: Pure HNSW fetch — no extra AND predicates so the index is used.
        # SurrealDB silently returns 0 rows when HNSW is combined with any extra
        # AND predicate in the same WHERE clause (confirmed Spike-1 H2).
        result = await self._query(
            f"""SELECT id,
                       vector::similarity::cosine(embedding, $vec) AS score,
                       state,
                       memory_type,
                       tags,
                       created_at
                FROM memory
                WHERE embedding <|{fetch_k},40|> $vec
                ORDER BY score DESC
                LIMIT {fetch_k}""",
            {"vec": embedding},
        )
        rows = self._rows(result)

        # Step 2: Python post-filter — apply all predicates in-process.
        filtered: list[dict] = []
        for row in rows:
            if row.get("state") != state:
                continue
            if type_filter and row.get("memory_type") != type_filter:
                continue
            if tags:  # None or [] → no tag filter applied
                row_tags = row.get("tags") or []
                if not set(tags).issubset(set(row_tags)):
                    continue
            if time_range:
                created = self._parse_dt(row["created_at"])
                start, end = time_range
                # Coerce all to UTC-aware to prevent TypeError on mixed-tz compare.
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                if not (start <= created <= end):
                    continue
            filtered.append(row)

        # Step 3: Truncate to top_k.
        return [(_extract_id(r["id"]), r["score"]) for r in filtered[:top_k]]

    async def vector_search_for_memory(self, memory_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        top_k = max(1, int(top_k))
        fetch_k = top_k * 3
        # Step 1: Pure HNSW fetch — no extra AND predicates (same fix as vector_search).
        # Self-exclusion and state filter move to Python post-filter.
        result = await self._query(
            f"""LET $vec = (SELECT embedding FROM type::thing('memory', $id))[0].embedding;
                SELECT id,
                       vector::similarity::cosine(embedding, $vec) AS score,
                       state,
                       tags,
                       created_at
                FROM memory
                WHERE embedding <|{fetch_k},40|> $vec
                ORDER BY score DESC
                LIMIT {fetch_k}""",
            {"id": memory_id},
        )
        rows = result[-1] if result else []
        if isinstance(rows, dict):
            rows = [rows]

        # Step 2: Python post-filter — active only, exclude self.
        # _extract_id is required: row['id'] is a RecordID object; memory_id is a bare
        # UUID string. Direct != comparison would always be True (silent bug).
        filtered: list[dict] = []
        for row in rows:
            if not isinstance(row, dict) or "score" not in row:
                continue
            if row.get("state") != "active":
                continue
            if _extract_id(row["id"]) == memory_id:
                continue
            filtered.append(row)

        # Step 3: Truncate to top_k.
        return [(_extract_id(r["id"]), r["score"]) for r in filtered[:top_k]]

    # ──────────────────────────────────────────────────────────────────────────
    # Versions
    # ──────────────────────────────────────────────────────────────────────────

    async def insert_version(self, version: MemoryVersion) -> None:
        result = await self._query(
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
        result = await self._query(
            "SELECT * FROM memory_version WHERE memory_id = type::thing('memory', $id) ORDER BY created_at DESC",
            {"id": memory_id},
        )
        return [
            MemoryVersion(
                id=_extract_id(r["id"]),
                memory_id=memory_id,
                content=r["content"],
                metadata=r.get("metadata"),
                created_at=self._parse_dt(r["created_at"]),
            )
            for r in self._rows(result)
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Relationships
    # ──────────────────────────────────────────────────────────────────────────

    async def insert_relationship(self, rel: Relationship) -> None:
        # Single chokepoint — see validate_edge_endpoints.  Covers the MCP
        # path AND consolidation / _auto_link / _contradiction_check.
        await validate_edge_endpoints(self, rel)

        table = REL_TABLES[rel.rel_type.value]
        result = await self._query(
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
        await self._query(
            f"DELETE {table} WHERE in = type::thing('memory', $src) AND out = type::thing('memory', $tgt)",
            {"src": source_id, "tgt": target_id},
        )
        return True

    async def get_relationships_for(self, memory_id: str, rel_types: list[str] | None = None) -> list[Relationship]:
        tables = [REL_TABLES[rt] for rt in rel_types] if rel_types else list(REL_TABLES.values())
        all_rels: list[Relationship] = []
        for table in tables:
            result = await self._query(
                f"SELECT * FROM {table} WHERE in = type::thing('memory', $id) OR out = type::thing('memory', $id)",
                {"id": memory_id},
            )
            for r in self._rows(result):
                all_rels.append(self._row_to_relationship(r, table))
        return all_rels

    async def get_outgoing_relationships(self, memory_id: str) -> list[Relationship]:
        all_rels: list[Relationship] = []
        for table in REL_TABLES.values():
            result = await self._query(
                f"SELECT * FROM {table} WHERE in = type::thing('memory', $id)",
                {"id": memory_id},
            )
            for r in self._rows(result):
                all_rels.append(self._row_to_relationship(r, table))
        return all_rels

    async def get_incoming_relationships(self, memory_id: str, rel_type: str | None = None) -> list[Relationship]:
        tables = [REL_TABLES[rel_type]] if rel_type else list(REL_TABLES.values())
        all_rels: list[Relationship] = []
        for table in tables:
            result = await self._query(
                f"SELECT * FROM {table} WHERE out = type::thing('memory', $id)",
                {"id": memory_id},
            )
            for r in self._rows(result):
                all_rels.append(self._row_to_relationship(r, table))
        return all_rels

    async def get_neighbors(self, memory_id: str, active_only: bool = True) -> list[tuple[str, Relationship]]:
        results: list[tuple[str, Relationship]] = []
        for table in REL_TABLES.values():
            for r in self._rows(await self._query(
                f"SELECT *, out.state AS neighbor_state FROM {table} WHERE in = type::thing('memory', $id)",
                {"id": memory_id},
            )):
                if active_only and r.get("neighbor_state") != "active":
                    continue
                results.append((_extract_id(r["out"]), self._row_to_relationship(r, table)))

            for r in self._rows(await self._query(
                f"SELECT *, in.state AS neighbor_state FROM {table} WHERE out = type::thing('memory', $id)",
                {"id": memory_id},
            )):
                if active_only and r.get("neighbor_state") != "active":
                    continue
                results.append((_extract_id(r["in"]), self._row_to_relationship(r, table)))
        return results

    async def get_neighbors_bulk(
        self, seed_ids: list[str]
    ) -> dict[str, list[tuple[str, Relationship]]]:
        """
        Single round-trip per table: fetch neighbors for multiple seed memories.
        Queries all 8 relation tables in both directions (outgoing + incoming).
        Returns dict keyed by seed_id -> list of (neighbor_id, Relationship).
        """
        if not seed_ids:
            return {}
        result: dict[str, list[tuple[str, Relationship]]] = {sid: [] for sid in seed_ids}
        for table in REL_TABLES.values():
            # Outgoing: in IN seed_ids
            out_rows = self._rows(await self._query(
                f"""SELECT *, out.state AS neighbor_state
                    FROM {table}
                    WHERE in IN $seeds""",
                {"seeds": [RecordID("memory", sid) for sid in seed_ids]},
            ))
            for r in out_rows:
                if r.get("neighbor_state") != "active":
                    continue
                src_id = _extract_id(r["in"])
                if src_id in result:
                    result[src_id].append((_extract_id(r["out"]), self._row_to_relationship(r, table)))

            # Incoming: out IN seed_ids
            in_rows = self._rows(await self._query(
                f"""SELECT *, in.state AS neighbor_state
                    FROM {table}
                    WHERE out IN $seeds""",
                {"seeds": [RecordID("memory", sid) for sid in seed_ids]},
            ))
            for r in in_rows:
                if r.get("neighbor_state") != "active":
                    continue
                tgt_id = _extract_id(r["out"])
                if tgt_id in result:
                    result[tgt_id].append((_extract_id(r["in"]), self._row_to_relationship(r, table)))
        return result

    async def get_supersede_flags(self, ids: list[str]) -> set[str]:
        """
        Returns subset of input IDs that have been superseded
        (i.e., have an incoming supersedes edge: some other memory supersedes them).
        Single query: SELECT out FROM supersedes WHERE out IN $ids
        """
        if not ids:
            return set()
        rows = self._rows(await self._query(
            "SELECT out FROM supersedes WHERE out IN $ids",
            {"ids": [RecordID("memory", mid) for mid in ids]},
        ))
        return {_extract_id(r["out"]) for r in rows if r.get("out") is not None}

    async def get_contradictions_bulk(
        self, ids: list[str]
    ) -> dict[str, list[tuple[str, str, float]]]:
        """
        Batched bidirectional contradiction fetch.
        Mirrors get_contradictions_for semantics but for multiple IDs in one round-trip.
        Returns dict keyed by anchor_id -> list of (other_id, relationship_type, strength).
        Deduplicates pairs.
        """
        if not ids:
            return {}
        result: dict[str, list[tuple[str, str, float]]] = {mid: [] for mid in ids}
        seen: dict[str, set[str]] = {mid: set() for mid in ids}
        surreal_ids = [RecordID("memory", mid) for mid in ids]

        # Outgoing from anchor: in IN $ids
        rows_out = self._rows(await self._query(
            """SELECT in AS anchor_id, out AS other_id, out.content AS other_content,
                      out.state AS other_state, strength
               FROM contradicts WHERE in IN $ids""",
            {"ids": surreal_ids},
        ))
        for r in rows_out:
            if r.get("other_state") != "active":
                continue
            anchor = _extract_id(r["anchor_id"])
            other = _extract_id(r["other_id"])
            if anchor in result:
                pair_key = f"{anchor}:{other}"
                if pair_key not in seen[anchor]:
                    seen[anchor].add(pair_key)
                    result[anchor].append((other, str(r.get("other_content", ""))[:100], r["strength"]))

        # Incoming to anchor: out IN $ids
        rows_in = self._rows(await self._query(
            """SELECT out AS anchor_id, in AS other_id, in.content AS other_content,
                      in.state AS other_state, strength
               FROM contradicts WHERE out IN $ids""",
            {"ids": surreal_ids},
        ))
        for r in rows_in:
            if r.get("other_state") != "active":
                continue
            anchor = _extract_id(r["anchor_id"])
            other = _extract_id(r["other_id"])
            if anchor in result:
                pair_key = f"{anchor}:{other}"
                if pair_key not in seen[anchor]:
                    seen[anchor].add(pair_key)
                    result[anchor].append((other, str(r.get("other_content", ""))[:100], r["strength"]))

        return result

    async def bulk_reinforce(self, updates: list[ReinforceUpdate]) -> None:
        """
        Single SurrealQL FOR-loop to update stability, last_accessed, access_count
        for multiple memories. Returns immediately if updates is empty.
        """
        if not updates:
            return
        await self._query(
            """LET $updates = $updates;
               FOR $update IN $updates {
                   UPDATE type::thing('memory', $update.memory_id) SET
                       stability = $update.stability,
                       last_accessed = $update.last_accessed,
                       access_count = $update.access_count
               };""",
            {
                "updates": [
                    {
                        "memory_id": u.memory_id,
                        "stability": u.stability,
                        "last_accessed": u.last_accessed,
                        "access_count": u.access_count,
                    }
                    for u in updates
                ],
            },
        )

    async def spreading_activation_walk(
        self, seed_ids: list[str], max_depth: int = 3
    ) -> list[SpreadingActivationRow]:
        """Walk the relation graph up to *max_depth* hops from *seed_ids*.

        Single SurrealDB round-trip using the layered-LET query built by
        ``_build_walk_sql``.  All 8 relation tables are traversed in both
        directions per hop.

        Guarantees (enforced at SQL level; Python applies safety-net filters):
        - Seeds are absent from results.
        - Only ``state = 'active'`` neighbors are returned.
        - Each neighbor appears at most once, at its shallowest reachable depth.
        - ``rel_strength`` is the maximum-strength edge connecting the neighbor
          to its shallowest-depth discovery path.
        - ``current_stability`` is enriched from the neighbor record (no extra RT).

        Returns an empty list when *seed_ids* is empty or *max_depth* < 1.
        """
        if not seed_ids or max_depth < 1:
            return []

        seed_set = set(seed_ids)
        sql = _build_walk_sql(max_depth)

        result = await self._query(
            sql,
            {"seeds": [RecordID("memory", sid) for sid in seed_ids]},
        )

        # RETURN $deduped is the last statement — its value is result[-1]
        if not result:
            return []
        raw_rows = result[-1]
        if not raw_rows:
            return []
        if isinstance(raw_rows, dict):
            raw_rows = [raw_rows]

        # Parse rows; Python-side safety filters guard against SQL regressions
        best: dict[str, SpreadingActivationRow] = {}
        for r in raw_rows:
            if not isinstance(r, dict):
                continue
            raw_nid = r.get("neighbor_id")
            if raw_nid is None:
                continue
            neighbor_id = _extract_id(raw_nid)
            if not neighbor_id:
                continue
            # Safety: exclude seeds
            if neighbor_id in seed_set:
                continue
            # Safety: active-only
            state = str(r.get("state", "active"))
            if state != "active":
                continue
            depth = int(r.get("depth") or 0)
            # Enrichment for the boost damping.  A row that cannot be parsed
            # leaves last_accessed None, which makes the caller skip the boost
            # entirely -- the safe direction.
            raw_last_accessed = r.get("last_accessed")
            try:
                last_accessed = (
                    self._parse_dt(raw_last_accessed)
                    if raw_last_accessed is not None
                    else None
                )
            except (TypeError, ValueError):
                last_accessed = None
            row = SpreadingActivationRow(
                neighbor_id=neighbor_id,
                depth=depth,
                rel_strength=float(r.get("rel_strength") or 0.0),
                current_stability=float(r.get("current_stability") or 0.0),
                state=state,
                last_accessed=last_accessed,
                memory_type=str(r.get("memory_type") or ""),
            )
            # Safety: keep shallowest depth per neighbor
            existing = best.get(neighbor_id)
            if existing is None or depth < existing.depth:
                best[neighbor_id] = row

        return list(best.values())

    async def delete_auto_links(self, memory_id: str) -> None:
        await self._query(
            "DELETE relates_to WHERE in = type::thing('memory', $id) AND strength < 1.0",
            {"id": memory_id},
        )

    async def bulk_update_stability(self, updates: list[tuple[float, str]]) -> None:
        if not updates:
            return
        await self._query(
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
        result = await self._query(
            """SELECT count() AS cnt FROM supersedes
               WHERE out = type::thing('memory', $id) AND in.state = 'active'
               GROUP ALL""",
            {"id": memory_id},
        )
        rows = self._rows(result)
        if rows and isinstance(rows[0], dict):
            return rows[0].get("cnt", 0) > 0
        return False

    async def get_contradictions_for(self, memory_id: str) -> list[tuple[str, str, float]]:
        rows_out = self._rows(await self._query(
            """SELECT out AS other_id, out.content AS other_content, strength
               FROM contradicts WHERE in = type::thing('memory', $id) AND out.state = 'active'""",
            {"id": memory_id},
        ))
        rows_in = self._rows(await self._query(
            """SELECT in AS other_id, in.content AS other_content, strength
               FROM contradicts WHERE out = type::thing('memory', $id) AND in.state = 'active'""",
            {"id": memory_id},
        ))
        return [
            (_extract_id(r["other_id"]), str(r.get("other_content", ""))[:100], r["strength"])
            for r in rows_out + rows_in
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Consolidation Log
    # ──────────────────────────────────────────────────────────────────────────

    async def insert_consolidation_log(self, entry: ConsolidationLogEntry) -> None:
        result = await self._query(
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
        result = await self._query(
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
        result = await self._query(
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

    # ──────────────────────────────────────────────────────────────────────────
    # Health report queries (Component C requirement — 6 methods)
    # ──────────────────────────────────────────────────────────────────────────

    async def get_counts_by_type_and_state(self) -> dict:
        result = await self._query(
            "SELECT memory_type, state, count() AS cnt FROM memory GROUP BY memory_type, state"
        )
        rows = self._rows(result)
        return {(r["memory_type"], r["state"]): r["cnt"] for r in rows}

    async def get_active_memories_for_decay(self) -> list[dict]:
        result = await self._query(
            """SELECT id, string::slice(content, 0, 120) AS content_preview,
                      memory_type, importance, stability, last_accessed, tags
               FROM memory
               WHERE state = 'active'"""
        )
        out: list[dict] = []
        for r in self._rows(result):
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
        count_result = await self._query(
            """SELECT count() AS cnt FROM memory
               WHERE state = 'active' AND (tags IS NONE OR array::len(tags) = 0)
               GROUP ALL"""
        )
        count_rows = self._rows(count_result)
        true_count: int = count_rows[0].get("cnt", 0) if count_rows else 0

        list_result = await self._query(
            """SELECT id, string::slice(content, 0, 120) AS content_preview,
                      memory_type, created_at
               FROM memory
               WHERE state = 'active' AND (tags IS NONE OR array::len(tags) = 0)
               ORDER BY created_at ASC
               LIMIT 51"""
        )
        items = [
            {
                "id": _extract_id(r["id"]),
                "content_preview": r.get("content_preview") or "",
                "memory_type": r["memory_type"],
                "created_at": self._parse_dt(r["created_at"]).isoformat(),
            }
            for r in self._rows(list_result)
        ]
        return (items[:50], true_count)

    async def get_orphan_unconnected(self) -> tuple[list[dict], int]:
        # _EDGE_IDS is a CORRELATED SUBQUERY: placed in WHERE, SurrealDB does not
        # hoist it, so those 16 SELECTs plus the flatten/distinct were re-evaluated
        # once PER CANDIDATE ROW.  At ~2600 active memories against ~1800 endpoints
        # that never finished — memory_health hung past 300 s and was unusable.
        #
        # Collect the endpoint set ONCE, then filter.  Measured upstream on the
        # live store: 0.67 s total, against >120 s that never completed.
        edge_ids: set[str] = set()
        for rel in REL_TABLES.values():
            for side in ("in", "out"):
                rows = await self._query(f"SELECT VALUE {side} FROM {rel}")
                if isinstance(rows, list):
                    edge_ids.update(str(r) for r in rows if r is not None)

        list_result = await self._query(
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
        result = await self._query("SELECT tags FROM memory WHERE state = 'active'")
        return [r.get("tags") or [] for r in self._rows(result)]

    async def get_health_consolidation_summary(self) -> dict | None:
        result = await self._query(
            "SELECT * FROM consolidation_log ORDER BY created_at DESC LIMIT 500"
        )
        rows = self._rows(result)
        if not rows:
            return None

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

    # ──────────────────────────────────────────────────────────────────────────
    # Config
    # ──────────────────────────────────────────────────────────────────────────

    async def get_config(self, key: str) -> Any | None:
        result = await self._query(
            "SELECT val FROM preference WHERE id = type::thing('preference', $key)",
            {"key": key},
        )
        rows = self._rows(result)
        if not rows:
            return None
        return rows[0].get("val")

    async def set_config(self, key: str, value: Any) -> None:
        now = datetime.now(timezone.utc)
        await self._query(
            """UPSERT type::thing('preference', $key) SET
                val = $val,
                updated_at = $now
            """,
            {"key": key, "val": value, "now": now},
        )

    async def get_all_config(self) -> dict[str, Any]:
        result = await self._query("SELECT * FROM preference")
        return {_extract_id(r["id"]): r["val"] for r in self._rows(result)}

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    async def get_counts_by_type(self) -> dict[str, int]:
        result = await self._query(
            "SELECT memory_type, count() AS cnt FROM memory GROUP BY memory_type"
        )
        return {r["memory_type"]: r["cnt"] for r in self._rows(result)}

    async def get_counts_by_state(self) -> dict[str, int]:
        result = await self._query(
            "SELECT state, count() AS cnt FROM memory GROUP BY state"
        )
        return {r["state"]: r["cnt"] for r in self._rows(result)}

    async def get_total_memory_count(self) -> int:
        result = await self._query("SELECT count() AS cnt FROM memory GROUP ALL")
        rows = self._rows(result)
        if rows and isinstance(rows[0], dict):
            return rows[0].get("cnt", 0)
        return 0

    def get_db_size(self) -> int:
        """Return total bytes in the RocksDB data directory, or -1 if inaccessible.

        Reads the directory path from SYNAPTRA_SURREAL_DATA_DIR env var
        (default: ~/.synaptra/surreal/).  Returns -1 with a warning if the
        directory is absent or unreadable — this is expected before first run.
        """
        data_dir = os.environ.get(
            "SYNAPTRA_SURREAL_DATA_DIR",
            str(Path.home() / ".synaptra" / "surreal"),
        )
        p = Path(data_dir)
        try:
            if p.exists():
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            logger.warning(
                "SurrealDB data directory %s does not exist; get_db_size() returning -1",
                data_dir,
            )
            return -1
        except Exception as exc:
            logger.warning(
                "Cannot read SurrealDB data directory %s: %s; returning -1",
                data_dir, exc,
            )
            return -1
