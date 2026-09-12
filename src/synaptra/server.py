"""MCP server — Streamable HTTP via FastMCP + uvicorn."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .engine import MemoryEngine


class StrictArgumentFastMCP(FastMCP):
    """FastMCP that rejects unknown argument names instead of discarding them.

    The SDK builds each tool's argument model with Pydantic's default
    ``extra="ignore"`` and registers the dispatch handler with
    ``validate_input=False``. A misnamed *optional* argument is therefore
    dropped before the tool body runs and the call still reports success. A
    misnamed *required* argument errors only incidentally, because the required
    field then goes missing -- so the failure mode is invisible exactly where it
    is most likely.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tool_manager.get_tool(name)
        if tool is not None:
            declared = set(tool.parameters.get("properties", {}))
            # Underscore-prefixed keys are reserved by the MCP spec (``_meta``
            # and friends) and belong to the transport, not to the tool. They
            # are not the caller's typo, and rejecting them would fail every
            # call to every tool rather than catching a misnamed argument.
            supplied = {k for k in arguments if not k.startswith("_")}
            unknown = sorted(supplied - declared)
            if unknown:
                raise ValueError(
                    f"Unknown argument(s) for {name!r}: {', '.join(unknown)}. "
                    f"Accepted arguments: {', '.join(sorted(declared))}."
                )
        return await super().call_tool(name, arguments)


def _resolve_type_alias(type_: str | None, memory_type: str | None) -> str | None:
    """Resolve the ``type`` / ``memory_type`` pair to a single value.

    The stored field, the response field and the CLI flag are all
    ``memory_type``, while the tool argument is ``type``. A caller reading a
    response and passing the field name straight back is making the natural
    move, so ``memory_type`` is accepted here as an alias rather than treated
    as a mistake.
    """
    if type_ is not None and memory_type is not None and type_ != memory_type:
        raise ValueError(
            f"Conflicting type arguments: type={type_!r} and "
            f"memory_type={memory_type!r}. Pass one."
        )
    return type_ if type_ is not None else memory_type


# Create FastMCP server with Streamable HTTP
mcp = StrictArgumentFastMCP(
    "synaptra",
    streamable_http_path="/mcp",
    json_response=False,
    stateless_http=True,
)

# Engine instance — initialized on first use, guarded by lock for concurrent init
_engine: MemoryEngine | None = None
_engine_lock = threading.Lock()


def _get_engine() -> MemoryEngine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine

        backend = os.environ.get("SYNAPTRA_BACKEND", "surrealkv-file")
        config_path_str = os.environ.get("SYNAPTRA_CONFIG")
        config_path = Path(config_path_str) if config_path_str else None

        if backend == "surrealkv-file":
            # Legacy embedded SurrealKV path (default during rollback window)
            db_path = os.environ.get(
                "SYNAPTRA_DB",
                str(Path.home() / ".synaptra" / "data"),
            )
            if not db_path.startswith(("mem://", "surrealkv://", "file://")):
                Path(db_path).mkdir(parents=True, exist_ok=True)
                # SurrealDB URL parser requires forward slashes (Windows backslashes break it)
                db_path = f"surrealkv://{db_path.replace(os.sep, '/')}"
            _engine = MemoryEngine(
                db_path=db_path,
                config_path=config_path,
            )

        elif backend == "rocksdb-server":
            # New server-backed path: connect to a separately-running SurrealDB process
            from .surreal_server_storage import SurrealServerStorage

            surreal_url = os.environ.get(
                "SYNAPTRA_SURREAL_URL",
                "ws://127.0.0.1:8000/rpc",
            )
            storage = SurrealServerStorage(url=surreal_url)
            _engine = MemoryEngine(
                storage=storage,
                config_path=config_path,
            )

        else:
            raise ValueError(
                f"Unknown SYNAPTRA_BACKEND value: {backend!r}. "
                "Supported values: 'surrealkv-file' (default), 'rocksdb-server'."
            )

        return _engine


async def _get_engine_ready() -> MemoryEngine:
    """Return the engine with its config overrides loaded.

    `_get_engine` is synchronous, and on an async backend the overrides can only
    be read by awaiting storage. Without this, every config read silently serves
    the YAML default. The loader is idempotent, so this is a boolean check after
    the first call.
    """
    engine = _get_engine()
    await engine.load_config_overrides()
    return engine


def _response(data: Any = None, elapsed_ms: float = 0, **meta_extra) -> str:
    """Build standard tool response as JSON string."""
    result = {
        "success": True,
        "data": data,
        "meta": {"elapsed_ms": round(elapsed_ms, 2), **meta_extra},
    }
    return json.dumps(result, default=str)


def _error(message: str) -> str:
    result = {"success": False, "error": message, "data": None, "meta": {}}
    return json.dumps(result, default=str)


# === Tool Definitions ===


@mcp.tool()
async def memory_store(
    content: str,
    type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
    conversation_id: str | None = None,
    memory_type: str | None = None,
    rater: str | None = None,
) -> str:
    """Store a new memory with automatic classification and importance scoring. Agent can override type and importance. `memory_type` is accepted as an alias for `type`.

    `rater` is REQUIRED — the identifier of the model calling this tool, e.g.
    'claude-opus-5'. Importance is rater-relative: different models score the
    same content differently, so a score with no rater cannot be compared to
    any other score."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        mem = await engine.store_memory(
            content=content,
            memory_type=_resolve_type_alias(type, memory_type),
            importance=importance,
            tags=tags,
            source=source,
            conversation_id=conversation_id,
            rater=rater,
        )
        return _response(mem.model_dump(), (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_recall(
    query: str,
    type_filter: str | None = None,
    tags: list[str] | None = None,
    time_range: dict | None = None,
    limit: int | None = None,
) -> str:
    """Multi-strategy retrieval: semantic + keyword + graph + temporal, fused with RRF, decay-weighted. Returns ranked memories with provenance."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        tr = None
        if time_range:
            tr = (
                datetime.fromisoformat(time_range["start"]),
                datetime.fromisoformat(time_range["end"]),
            )
        results = await engine.recall(
            query=query,
            type_filter=type_filter,
            tags=tags,
            time_range=tr,
            limit=limit,
        )
        return _response(
            {"memories": [r.model_dump() for r in results]},
            (time.time() - start) * 1000,
        )
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_get(id: str) -> str:
    """Get a specific memory by ID with full metadata, relationships, version history, and on-the-fly retrievability. Read-only."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        result = await engine.get_memory(id)
        if result is None:
            return _error(f"Memory {id} not found")
        return _response(result.model_dump(), (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_update(
    id: str,
    content: str | None = None,
    type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    rater: str | None = None,
) -> str:
    """Update a memory's content or metadata. Creates a version snapshot, re-embeds if content changed, reinforces stability. `memory_type` is accepted as an alias for `type`.

    `rater` is required ONLY when changing `importance` — pass the identifier of
    the model producing the new score. An update to content, type or tags leaves
    the existing rater untouched, because the score did not change."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        mem = await engine.update_memory(
            memory_id=id,
            content=content,
            memory_type=_resolve_type_alias(type, memory_type),
            importance=importance,
            tags=tags,
            rater=rater,
        )
        if mem is None:
            return _error(f"Memory {id} not found")
        return _response(mem.model_dump(), (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_relate(
    source_id: str,
    target_id: str,
    rel_type: str,
    strength: float = 1.0,
) -> str:
    """Create a typed relationship between two memories. Default strength=1.0."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        rel = await engine.create_relationship(source_id, target_id, rel_type, strength)
        return _response(rel.model_dump(), (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_related(
    id: str,
    depth: int = 1,
    rel_types: list[str] | None = None,
) -> str:
    """Get related memories via graph traversal. Read-only, no side effects."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        results = await engine.get_related(id, depth=depth, rel_types=rel_types)
        return _response(results, (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_unrelate(source_id: str, target_id: str, rel_type: str) -> str:
    """Remove a relationship between two memories."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        success = await engine.delete_relationship(source_id, target_id, rel_type)
        return _response({"deleted": success}, (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_list(
    search: str | None = None,
    type: str | None = None,
    state: str | None = None,
    tags: list[str] | None = None,
    time_range: dict | None = None,
    importance_min: float | None = None,
    importance_max: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """Browse memories with filters and full-text search."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        tr = None
        if time_range:
            tr = (
                datetime.fromisoformat(time_range["start"]),
                datetime.fromisoformat(time_range["end"]),
            )
        memories = await engine.storage.list_memories(
            search=search,
            memory_type=type,
            state=state,
            tags=tags,
            time_range=tr,
            importance_min=importance_min,
            importance_max=importance_max,
            limit=limit,
            offset=offset,
        )
        return _response(
            {"memories": [m.model_dump() for m in memories]},
            (time.time() - start) * 1000,
        )
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_rerate_candidates(
    model: str,
    type: str | None = None,
    state: str | None = "active",
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List memories whose importance was set by a rater OTHER than `model` — the re-rating candidates.

    Importance is rater-relative: different models score the same content
    differently, so a score set by another rater is not comparable to one this
    model would give. Memories with NO rater are included — they predate rater
    tracking, so they differ from every model.

    Read-only. Nothing is re-scored, and access counts are not touched — looking
    for stale scores must not itself alter decay state.
    """
    start = time.time()
    engine = await _get_engine_ready()
    try:
        memories = await engine.storage.list_memories(
            memory_type=type,
            state=state,
            rater_not=model,
            limit=limit,
            offset=offset,
        )
        return _response(
            {
                "model": model,
                "candidates": [
                    {
                        "id": m.id,
                        "memory_type": m.memory_type.value,
                        "importance": m.importance,
                        "rater": m.rater,
                        "rated_at": m.rated_at.isoformat() if m.rated_at else None,
                        "first_line": m.content.splitlines()[0][:200]
                        if m.content
                        else "",
                    }
                    for m in memories
                ],
            },
            (time.time() - start) * 1000,
        )
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_archive(
    id: str | None = None,
    ids: list[str] | None = None,
    below_retrievability: float | None = None,
) -> str:
    """Archive memory/memories. Supports single ID, bulk IDs, or threshold-based."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        if id:
            success = await engine.archive_memory(id)
            return _response(
                {"archived": 1 if success else 0}, (time.time() - start) * 1000
            )
        elif ids:
            count = await engine.archive_bulk(ids)
            return _response({"archived": count}, (time.time() - start) * 1000)
        elif below_retrievability is not None:
            count = await engine.archive_below_retrievability(below_retrievability)
            return _response({"archived": count}, (time.time() - start) * 1000)
        return _error("Provide id, ids, or below_retrievability")
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_restore(
    id: str | None = None,
    ids: list[str] | None = None,
) -> str:
    """Restore archived memory/memories. Resets decay."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        if id:
            mem = await engine.restore_memory(id)
            return _response(
                mem.model_dump() if mem else None, (time.time() - start) * 1000
            )
        elif ids:
            results = []
            for mid in ids:
                m = await engine.restore_memory(mid)
                if m is not None:
                    results.append(mid)
            return _response({"restored": results}, (time.time() - start) * 1000)
        return _error("Provide id or ids")
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_delete(
    confirm: bool,
    id: str | None = None,
    ids: list[str] | None = None,
) -> str:
    """Permanently delete memory/memories. Cascades: relationships, versions, embeddings. Requires confirm=true."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        if not confirm:
            return _error("confirm must be true for permanent deletion")
        if id:
            success = await engine.delete_memory(id)
            return _response(
                {"deleted": 1 if success else 0}, (time.time() - start) * 1000
            )
        elif ids:
            count = 0
            for mid in ids:
                if await engine.delete_memory(mid):
                    count += 1
            return _response({"deleted": count}, (time.time() - start) * 1000)
        return _error("Provide id or ids")
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_stats() -> str:
    """System statistics: counts by type/state, average decay by type, consolidation history, storage usage."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        stats = await engine.get_stats()
        return _response(stats, (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_consolidate(dry_run: bool = False) -> str:
    """Trigger consolidation pipeline: decay update, promotion, archival, clustering, merging. Supports dry_run."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        actions = await engine.consolidate(dry_run=dry_run)
        return _response(
            {"actions": actions, "count": len(actions)}, (time.time() - start) * 1000
        )
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_self(
    query: str,
    tags: list[str] | None = None,
) -> str:
    """Recall identity memories — the agent's self-knowledge.

    Convenience wrapper over memory_recall with type_filter='identity'.
    Returns ranked identity memories matching the query.
    Supports optional tags filter for facet categories (e.g., 'origin', 'values', 'capability').
    """
    start = time.time()
    engine = await _get_engine_ready()
    try:
        results = await engine.recall(
            query=query,
            type_filter="identity",
            tags=tags,
        )
        return _response(
            {"memories": [r.model_dump() for r in results]},
            (time.time() - start) * 1000,
        )
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_who(
    person: str,
    query: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Recall what the agent knows about a specific person.

    Matches person:{name} tags (case-insensitive). If no query is provided,
    uses the person's name as the query to return everything about them ranked
    by relevance.

    Fallback behavior: if person-typed query returns zero results, re-queries
    without type_filter but keeps the person tag filter. If still empty, queries
    with just the person name (no type or tag filter) to handle cold-start
    scenarios where knowledge exists as episodic/semantic memories.
    """
    person_name = person.strip().lower()
    if not person_name:
        return _error("person name is required")
    start = time.time()
    engine = await _get_engine_ready()
    try:
        # Build tag filter: person:{name} (lowercase)
        person_tag = f"person:{person_name}"
        combined_tags = [person_tag] + (tags or [])

        # If no query provided, use the person's name as the query
        effective_query = query or person_name

        # Primary: person-typed memories with person tag
        results = await engine.recall(
            query=effective_query,
            type_filter="person",
            tags=combined_tags,
        )

        # W2 fallback: if no person-typed results, try without type filter
        if not results:
            results = await engine.recall(
                query=effective_query,
                tags=combined_tags,
            )

        # W2 fallback: if still empty, try with just the person name as query
        if not results:
            results = await engine.recall(
                query=effective_query,
            )

        return _response(
            {"memories": [r.model_dump() for r in results]},
            (time.time() - start) * 1000,
        )
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_health() -> str:
    """Return a diagnostic health report for the memory store.

    Includes: memory counts by type and state, at-risk decay list,
    orphan detection (untagged and unconnected), storage gaps,
    consolidation history, and backup staleness check.

    Backup fields:
    - most_recent_backup_age_days: int | None — days since the most recent backup,
      or None if no backups exist.
    - backup_is_stale: bool — True if age > 7 days or no backup exists.

    Note: The calling agent is responsible for surfacing
    backup_is_stale as a visible warning to the user. This tool only provides the data.

    Read-only — no side effects.
    """
    start = time.time()
    engine = await _get_engine_ready()
    try:
        report = await engine.get_health()

        # --- Stale backup check ---
        # O5 guard: if no backups exist, return None for age and True for is_stale.
        backup_age_days: int | None = None
        backup_is_stale: bool = True
        try:
            backups_root = Path.home() / ".synaptra" / "backups"
            most_recent_dt: datetime | None = None
            for manifest_path in backups_root.glob("*/manifest.json"):
                try:
                    manifest_data = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    ts_str = manifest_data.get("created_at", "")
                    dt = datetime.fromisoformat(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if most_recent_dt is None or dt > most_recent_dt:
                        most_recent_dt = dt
                except Exception:
                    pass
            if most_recent_dt is not None:
                delta = datetime.now(timezone.utc) - most_recent_dt
                backup_age_days = delta.days
                backup_is_stale = backup_age_days > 7
            # else: no backups found — age=None, stale=True (safe default)
        except Exception:
            pass  # filesystem error — leave stale=True as safe default

        report["most_recent_backup_age_days"] = backup_age_days
        report["backup_is_stale"] = backup_is_stale

        return _response(report, (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


@mcp.tool()
async def memory_config(
    key: str | None = None,
    value: Any = None,
) -> str:
    """View or update configuration. No params = return all. Key only = read. Key + value = write."""
    start = time.time()
    engine = await _get_engine_ready()
    try:
        if key and value is not None:
            await engine.set_config_async(key, value)
            # Report what is now stored, never the argument that came in. A
            # confirmation built from the request cannot detect a dropped write.
            stored = engine.get_config(key).get("value")
            return _response(
                {"key": key, "value": stored, "action": "set"},
                (time.time() - start) * 1000,
            )
        elif key:
            result = engine.get_config(key)
            return _response(result, (time.time() - start) * 1000)
        else:
            result = engine.get_config()
            return _response(result, (time.time() - start) * 1000)
    except Exception as e:
        return _error(str(e))


# === Server Entry Points ===


def get_app():
    """Get the Starlette ASGI app for uvicorn."""
    return mcp.streamable_http_app()


def main() -> None:
    """CLI entrypoint — run the MCP server (HTTP or stdio)."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Synaptra MCP server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="Transport to use: 'http' (default, uvicorn/streamable-HTTP) or 'stdio' (per-session MCP over stdin/stdout)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdio mode: FastMCP speaks MCP protocol over stdin/stdout.
        # Warm EVERYTHING at process start — identical to the HTTP branch below —
        # so the first recall does zero initialization on the request path.
        # Critical for stdio: the embedding model load is a synchronous, multi-
        # second CPU call; if it fired lazily during the first recall it would
        # run on anyio's event-loop thread and block the loop, hanging the call.
        # _get_engine() builds storage (eager surrealkv-file DB connect) + config;
        # embeddings.warmup() loads the model. After this, recall is query-only.
        _engine_instance = _get_engine()
        _engine_instance.embeddings.warmup()
        mcp.run(transport="stdio")
        return

    # http mode (default) — existing behaviour, unchanged.
    # When running headless (pythonw.exe / Task Scheduler), redirect logs to file
    log_dir = Path.home() / ".synaptra"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / "service.log")

    if not sys.stdout or not sys.stdout.writable():
        # Running under pythonw.exe — no console available
        sys.stdout = open(log_file, "a")
        sys.stderr = sys.stdout

    port = int(os.environ.get("SYNAPTRA_PORT", "8050"))
    host = os.environ.get("SYNAPTRA_HOST", "127.0.0.1")

    # Pre-initialize the engine so startup errors are caught early
    _engine_instance = _get_engine()
    # Eagerly load the embedding model — moves ~10 s cold-start out of first recall
    _engine_instance.embeddings.warmup()

    uvicorn.run(
        get_app(),
        host=host,
        port=port,
        log_level="info",
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": log_file,
                    "formatter": "default",
                },
            },
            "formatters": {
                "default": {
                    "fmt": "%(asctime)s [%(levelname)s] %(message)s",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["file"],
            },
        },
    )


if __name__ == "__main__":
    main()
