"""CLI for synaptra — connects to the MCP server via Streamable HTTP."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import click

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


# --- Constants ---

DEFAULT_URL = "http://127.0.0.1:8050/mcp"
CONNECT_TIMEOUT = 5  # seconds


# --- Connection helpers ---

def resolve_url(ctx: click.Context) -> str:
    """Get server URL from --url flag > env var > default."""
    return (
        ctx.obj.get("url")
        or os.environ.get("SYNAPTRA_URL")
        or DEFAULT_URL
    )


async def _call_tool_inner(url: str, tool_name: str, params: dict) -> dict:
    """Connect to MCP server, call one tool, return parsed response."""
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, params)
            if not result.content:
                raise ConnectionError("Server returned an empty response.")
            return json.loads(result.content[0].text)


async def call_tool(url: str, tool_name: str, params: dict) -> dict:
    """Timeout-guarded MCP tool call.

    Returns dict with keys: success, data, error (if failed), meta.
    Raises ConnectionError on network/timeout failures.
    """
    try:
        return await asyncio.wait_for(
            _call_tool_inner(url, tool_name, params),
            timeout=CONNECT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise ConnectionError(
            f"Server at {url} did not respond within {CONNECT_TIMEOUT} seconds."
        )
    except (ConnectionRefusedError, OSError) as e:
        raise ConnectionError(
            f"Cannot connect to synaptra server at {url}. "
            f"Is the server running?\nStart it with: synaptra"
        ) from e
    except json.JSONDecodeError:
        raise ConnectionError(
            f"Server at {url} returned an invalid response."
        )
    except Exception as e:
        # Catch httpx.ConnectError and other transport errors
        # Only treat as connection error if the exception type name suggests transport
        type_name = type(e).__name__.lower()
        if any(k in type_name for k in ("connect", "transport", "http", "network")):
            raise ConnectionError(
                f"Cannot connect to synaptra server at {url}. "
                f"Is the server running?\nStart it with: synaptra"
            ) from e
        raise


def run_tool(ctx: click.Context, tool_name: str, params: dict) -> dict:
    """Synchronous wrapper: resolve URL, call tool, handle errors."""
    url = resolve_url(ctx)
    try:
        response = asyncio.run(call_tool(url, tool_name, params))
    except ConnectionError as e:
        click.echo(str(e), err=True)
        sys.exit(2)
    except BaseException as e:
        # asyncio.run may wrap transport errors in ExceptionGroup
        # Check exception type names (not messages) to avoid masking programming errors
        type_names = [type(e).__name__.lower()]
        if hasattr(e, "exceptions"):
            type_names.extend(type(sub).__name__.lower() for sub in e.exceptions)
        is_transport = any(
            k in tn for tn in type_names for k in ("connect", "transport", "http", "network")
        )
        if is_transport:
            click.echo(
                f"Cannot connect to synaptra server at {url}. "
                f"Is the server running?\nStart it with: synaptra",
                err=True,
            )
            sys.exit(2)
        raise

    if not response.get("success"):
        click.echo(f"Error: {response.get('error', 'Unknown error')}", err=True)
        sys.exit(1)

    return response


# --- Parameter helpers ---

def split_tags(tags: str | None) -> list[str] | None:
    """'python,ai' -> ['python', 'ai']. None -> None."""
    if tags is None:
        return None
    return [t.strip() for t in tags.split(",") if t.strip()]


def build_time_range(start: str | None, end: str | None) -> dict | None:
    """Two ISO 8601 strings -> time_range dict. Requires both or neither."""
    if not start and not end:
        return None
    if bool(start) != bool(end):
        click.echo("Error: --time-start and --time-end must both be provided", err=True)
        sys.exit(2)
    return {"start": start, "end": end}


def route_ids(ids: tuple[str, ...]) -> dict:
    """Single ID -> {'id': x}. Multiple -> {'ids': list}."""
    if len(ids) == 1:
        return {"id": ids[0]}
    return {"ids": list(ids)}


def format_age(dt_str: str) -> str:
    """ISO datetime string -> relative age string."""
    try:
        dt = datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return "?"
    now = datetime.now(timezone.utc)
    delta = now - dt
    seconds = delta.total_seconds()
    if seconds < 0:
        return "future"
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 604800:
        return f"{int(seconds // 86400)}d ago"
    if seconds < 2592000:
        return f"{int(seconds // 604800)}w ago"
    return f"{int(seconds // 2592000)}mo ago"


# --- Output helpers ---

def output_json(response: dict) -> None:
    """Print raw MCP response as pretty JSON."""
    click.echo(json.dumps(response, indent=2, default=str))


def truncate(s: str, length: int) -> str:
    """Truncate string to length, adding ellipsis if needed."""
    if len(s) <= length:
        return s
    return s[: length - 1] + "\u2026"


def flatten_config(obj: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten nested config dict to sorted dot-notation key=value pairs."""
    items: list[tuple[str, str]] = []
    for k, v in obj.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.extend(flatten_config(v, key))
        else:
            items.append((key, str(v)))
    return sorted(items)


# --- Click group ---

@click.group()
@click.option("--url", default=None, help=f"Server URL (default: {DEFAULT_URL}, env: SYNAPTRA_URL)")
@click.option("--json", "json_mode", is_flag=True, help="Output raw JSON")
@click.version_option(package_name="synaptra")
@click.pass_context
def cli(ctx: click.Context, url: str | None, json_mode: bool) -> None:
    """Synaptra CLI -- browse, search, and manage memories."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["json"] = json_mode


# === Browsing commands ===


@cli.command("rerate-candidates")
@click.option(
    "--model",
    required=True,
    help="The model asking, e.g. 'claude-opus-5'. Memories rated by anyone else "
    "— including memories with no rater at all — are candidates.",
)
@click.option("--type", "mem_type", default=None, help="Filter by memory type")
@click.option("--state", default="active", help="Filter by state (default: active)")
@click.option("--limit", default=20, type=int, help="Max results (default: 20)")
@click.option("--offset", default=0, type=int, help="Skip first N results")
@click.pass_context
def rerate_candidates(ctx, model, mem_type, state, limit, offset):
    """List memories whose importance was set by a rater other than --model.

    Read-only: nothing is re-scored and no access counts are touched.
    """
    params = {"model": model, "limit": limit, "offset": offset, "state": state}
    if mem_type:
        params["type"] = mem_type

    response = run_tool(ctx, "memory_rerate_candidates", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    candidates = response.get("data", {}).get("candidates", [])
    if not candidates:
        click.echo(f"No re-rating candidates for {model}.")
        return

    for c in candidates:
        rater = c.get("rater") or "— (pre-rater-tracking)"
        click.echo(
            f"[{c.get('id', '')[:8]}] {c.get('memory_type', '?')} | "
            f"{c.get('importance', 0):.2f} | rated by {rater}"
        )
        click.echo(f"  {c.get('first_line', '')}")


@cli.command("list")
@click.option("--type", "mem_type", default=None, help="Filter by type (working, episodic, semantic, procedural)")
@click.option("--state", default=None, help="Filter by state (active, archived)")
@click.option("--tags", default=None, help="Filter by tags (comma-separated)")
@click.option("--search", default=None, help="Full-text search within list")
@click.option("--limit", default=20, type=int, help="Max results (default: 20)")
@click.option("--offset", default=0, type=int, help="Skip first N results")
@click.option("--importance-min", default=None, type=float, help="Min importance (0.0-1.0)")
@click.option("--importance-max", default=None, type=float, help="Max importance (0.0-1.0)")
@click.pass_context
def list_(ctx, mem_type, state, tags, search, limit, offset, importance_min, importance_max):
    """List memories with optional filters."""
    params = {"limit": limit, "offset": offset}
    if mem_type:
        params["type"] = mem_type
    if state:
        params["state"] = state
    if tags:
        params["tags"] = split_tags(tags)
    if search:
        params["search"] = search
    if importance_min is not None:
        params["importance_min"] = importance_min
    if importance_max is not None:
        params["importance_max"] = importance_max

    response = run_tool(ctx, "memory_list", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    memories = response.get("data", {}).get("memories", [])
    if not memories:
        click.echo("No memories found.")
        return

    for i, m in enumerate(memories):
        if i > 0:
            click.echo()
        mid = m.get("id", "")[:8]
        mtype = m.get("memory_type", "?")
        imp = f"{m.get('importance', 0):.2f}"
        age = format_age(m.get("created_at", ""))
        click.echo(f"[{mid}] {mtype} | {imp} | {age}")
        content = m.get("content", "")
        click.echo(f"  {content}")
        mtags = m.get("tags", [])
        if mtags:
            click.echo(f"  Tags: {', '.join(mtags)}")


@cli.command()
@click.argument("id")
@click.pass_context
def get(ctx, id):
    """Get a specific memory by ID."""
    response = run_tool(ctx, "memory_get", {"id": id})

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", {})
    mem = data.get("memory", data)

    click.echo(f"Memory {mem.get('id', id)}")
    click.echo()
    click.echo("Content:")
    click.echo(f"  {mem.get('content', '')}")
    click.echo()

    fields = [
        ("Type", mem.get("memory_type", "?")),
        ("State", mem.get("state", "?")),
        ("Importance", f"{mem.get('importance', 0):.2f}"),
        ("Stability", f"{mem.get('stability', 0):.2f}"),
        ("Retrievability", f"{mem.get('retrievability', 0):.2f}"),
        ("Access Count", str(mem.get("access_count", 0))),
        ("Tags", ", ".join(mem.get("tags", [])) or "\u2014"),
        ("Source", mem.get("source") or "\u2014"),
        ("Conversation", mem.get("conversation_id") or "\u2014"),
        ("Rater", mem.get("rater") or "\u2014"),
        ("Rated", mem.get("rated_at") or "\u2014"),
        ("Created", mem.get("created_at", "?")),
        ("Updated", mem.get("updated_at", "?")),
        ("Last Accessed", mem.get("last_accessed", "?")),
    ]
    max_label = max(len(f[0]) for f in fields)
    for label, value in fields:
        click.echo(f"{label + ':':<{max_label + 2}}{value}")

    # Relationships
    relationships = data.get("relationships", [])
    if relationships:
        click.echo()
        click.echo("Relationships:")
        for rel in relationships:
            direction = "->" if rel.get("direction") == "outgoing" else "<-"
            rel_type = rel.get("rel_type", "?")
            target = rel.get("target_id", "?")[:8]
            strength = rel.get("strength", 1.0)
            click.echo(f"  {direction} {rel_type}  {target} (strength: {strength:.2f})")

    # Versions
    versions = data.get("versions", [])
    if versions:
        click.echo()
        click.echo(f"Versions ({len(versions)}):")
        for i, v in enumerate(versions, 1):
            ts = v.get("created_at", "?")
            content = truncate(v.get("content", ""), 60)
            click.echo(f"  [{i}] {ts} \u2014 {content}")


@cli.command()
@click.pass_context
def stats(ctx):
    """View memory statistics."""
    response = run_tool(ctx, "memory_stats", {})

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", {})
    click.echo("Memory Statistics")
    click.echo()

    counts = data.get("counts", {})
    by_type = counts.get("by_type", {})
    if by_type:
        click.echo("By Type:")
        for t, c in by_type.items():
            click.echo(f"  {t + ':':<14}{c}")

    by_state = counts.get("by_state", {})
    if by_state:
        click.echo()
        click.echo("By State:")
        for s, c in by_state.items():
            click.echo(f"  {s + ':':<14}{c}")

    decay = data.get("decay", {})
    if decay:
        click.echo()
        click.echo("Decay (avg retrievability):")
        for t, d in decay.items():
            avg = d.get("avg_retrievability", 0) if isinstance(d, dict) else d
            click.echo(f"  {t + ':':<14}{avg:.2f}")

    total = counts.get("total", sum(by_type.values()) if by_type else 0)
    click.echo()
    click.echo(f"Total: {total} memories")


@cli.command()
@click.argument("id")
@click.option("--depth", default=1, type=int, help="Traversal depth (default: 1)")
@click.option("--rel-types", default=None, help="Filter by relationship types (comma-separated)")
@click.pass_context
def related(ctx, id, depth, rel_types):
    """View memories related to a given memory."""
    params = {"id": id, "depth": depth}
    if rel_types:
        params["rel_types"] = split_tags(rel_types)

    response = run_tool(ctx, "memory_related", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", [])
    if not data:
        click.echo("No related memories found.")
        return

    for i, item in enumerate(data):
        if i > 0:
            click.echo()
        direction = "->" if item.get("direction") == "outgoing" else "<-"
        rel_type = item.get("rel_type", "?")
        strength = item.get("strength", 1.0)
        target_id = item.get("target_id", item.get("id", "?"))[:8]
        click.echo(f"  {direction} {rel_type} [{target_id}] (strength: {strength:.2f})")
        content = item.get("content", "")
        if content:
            click.echo(f"     {content}")


# === Search commands ===


@cli.command()
@click.argument("query")
@click.option("--type", "mem_type", default=None, help="Filter by memory type")
@click.option("--tags", default=None, help="Filter by tags (comma-separated)")
@click.option("--limit", default=10, type=int, help="Max results (default: 10)")
@click.option("--time-start", default=None, help="Start of time range (ISO 8601)")
@click.option("--time-end", default=None, help="End of time range (ISO 8601)")
@click.pass_context
def recall(ctx, query, mem_type, tags, limit, time_start, time_end):
    """Search memories with multi-strategy retrieval."""
    params = {"query": query, "limit": limit}
    if mem_type:
        params["type_filter"] = mem_type
    if tags:
        params["tags"] = split_tags(tags)
    tr = build_time_range(time_start, time_end)
    if tr:
        params["time_range"] = tr

    response = run_tool(ctx, "memory_recall", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    memories = response.get("data", {}).get("memories", [])
    if not memories:
        click.echo("No memories match your query.")
        return

    for i, m in enumerate(memories, 1):
        if i > 1:
            click.echo()
        mid = m.get("id", "")[:8]
        score = m.get("score", 0)
        mtype = m.get("memory_type", "?")
        found_by = m.get("found_by", "?")
        if isinstance(found_by, list):
            found_by = ", ".join(found_by)
        click.echo(f"{i}. [{mid}] {mtype} | score: {score:.3f} | found by: {found_by}")
        content = m.get("content", "")
        click.echo(f"   {content}")
        mtags = m.get("tags", [])
        if mtags:
            click.echo(f"   Tags: {', '.join(mtags)}")


# === Management commands ===


@cli.command()
@click.argument("content", default="-")
@click.option("--type", "mem_type", default=None, help="Memory type (auto-classified if omitted)")
@click.option("--tags", default=None, help="Tags (comma-separated)")
@click.option("--importance", default=None, type=float, help="Importance (0.0-1.0)")
@click.option("--source", default=None, help="Source metadata")
@click.option(
    "--rater",
    default=None,
    help="REQUIRED. Who is setting importance, e.g. 'claude-opus-5'. "
    "Importance is rater-relative and a score with no rater is not comparable.",
)
@click.pass_context
def store(ctx, content, mem_type, tags, importance, source, rater):
    """Store a new memory. Use '-' to read from stdin."""
    if content == "-":
        content = click.get_text_stream("stdin").read().strip()
        if not content:
            click.echo("Error: No content provided via stdin", err=True)
            sys.exit(1)

    params = {"content": content}
    if mem_type:
        params["type"] = mem_type
    if tags:
        params["tags"] = split_tags(tags)
    if importance is not None:
        params["importance"] = importance
    if source:
        params["source"] = source
    if rater:
        params["rater"] = rater

    response = run_tool(ctx, "memory_store", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", {})
    mid = data.get("id", "?")[:8]
    mtype = data.get("memory_type", "?")
    imp = data.get("importance", 0)
    click.echo(f"Stored {mid} ({mtype}, importance: {imp:.2f})")


@cli.command()
@click.argument("id")
@click.option("--content", default=None, help="New content")
@click.option("--type", "mem_type", default=None, help="New memory type")
@click.option("--importance", default=None, type=float, help="New importance (0.0-1.0)")
@click.option("--tags", default=None, help="New tags (comma-separated, replaces existing)")
@click.option(
    "--rater",
    default=None,
    help="Required only with --importance. Who is producing the new score.",
)
@click.pass_context
def update(ctx, id, content, mem_type, importance, tags, rater):
    """Update a memory's content or metadata."""
    params = {"id": id}
    changed = []
    if content is not None:
        params["content"] = content
        changed.append("content")
    if mem_type is not None:
        params["type"] = mem_type
        changed.append("type")
    if importance is not None:
        params["importance"] = importance
        changed.append("importance")
    if tags is not None:
        params["tags"] = split_tags(tags)
        changed.append("tags")
    if rater:
        params["rater"] = rater

    if not changed:
        click.echo("Error: Provide at least one of --content, --type, --importance, --tags", err=True)
        ctx.exit(2)
        return

    response = run_tool(ctx, "memory_update", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    click.echo(f"Updated {id[:8]} ({', '.join(changed)})")


@cli.command()
@click.argument("ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete(ctx, ids, yes):
    """Permanently delete one or more memories."""
    count = len(ids)
    if not yes:
        if count == 1:
            prompt = f"Permanently delete memory {ids[0][:8]}?"
        else:
            prompt = f"Permanently delete {count} memories?"
        if not click.confirm(prompt, default=False):
            click.echo("Cancelled.")
            return

    params = {**route_ids(ids), "confirm": True}
    response = run_tool(ctx, "memory_delete", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    deleted = response.get("data", {}).get("deleted", 0)
    noun = "memory" if deleted == 1 else "memories"
    click.echo(f"Deleted {deleted} {noun}")


@cli.command()
@click.argument("ids", nargs=-1)
@click.option("--below-retrievability", default=None, type=float, help="Archive all below this retrievability threshold")
@click.pass_context
def archive(ctx, ids, below_retrievability):
    """Archive one or more memories, or all below a retrievability threshold."""
    if ids and below_retrievability is not None:
        click.echo("Error: Provide either IDs or --below-retrievability, not both", err=True)
        sys.exit(2)
    if not ids and below_retrievability is None:
        click.echo("Error: Provide IDs or --below-retrievability", err=True)
        sys.exit(2)

    if ids:
        params = route_ids(ids)
    else:
        params = {"below_retrievability": below_retrievability}

    response = run_tool(ctx, "memory_archive", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    archived = response.get("data", {}).get("archived", 0)
    noun = "memory" if archived == 1 else "memories"
    suffix = ""
    if below_retrievability is not None:
        suffix = f" (below retrievability {below_retrievability})"
    click.echo(f"Archived {archived} {noun}{suffix}")


@cli.command()
@click.argument("ids", nargs=-1, required=True)
@click.pass_context
def restore(ctx, ids):
    """Restore one or more archived memories."""
    params = route_ids(ids)
    response = run_tool(ctx, "memory_restore", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", {})
    if isinstance(data, dict) and "restored" in data:
        count = len(data["restored"])
    else:
        count = 1
    noun = "memory" if count == 1 else "memories"
    click.echo(f"Restored {count} {noun}")


@cli.command()
@click.argument("source_id")
@click.argument("target_id")
@click.option("--type", "rel_type", required=True, help="Relationship type (causes, follows, contradicts, supports, relates_to, supersedes, part_of)")
@click.option("--strength", default=1.0, type=float, help="Relationship strength (default: 1.0)")
@click.pass_context
def relate(ctx, source_id, target_id, rel_type, strength):
    """Create a relationship between two memories."""
    params = {
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rel_type,
        "strength": strength,
    }
    response = run_tool(ctx, "memory_relate", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    click.echo(f"Related {source_id[:8]} -> {target_id[:8]} ({rel_type}, strength: {strength:.2f})")


@cli.command()
@click.argument("source_id")
@click.argument("target_id")
@click.option("--type", "rel_type", required=True, help="Relationship type to remove")
@click.pass_context
def unrelate(ctx, source_id, target_id, rel_type):
    """Remove a relationship between two memories."""
    params = {
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rel_type,
    }
    response = run_tool(ctx, "memory_unrelate", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    click.echo(f"Unrelated {source_id[:8]} -> {target_id[:8]} ({rel_type})")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would happen without making changes")
@click.pass_context
def consolidate(ctx, dry_run):
    """Trigger the consolidation pipeline."""
    response = run_tool(ctx, "memory_consolidate", {"dry_run": dry_run})

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", {})
    actions = data.get("actions", [])
    count = data.get("count", len(actions))

    # Count by action type
    action_counts: dict[str, int] = {}
    for a in actions:
        action_type = a.get("action", "unknown") if isinstance(a, dict) else str(a)
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    if dry_run:
        parts = [f"{c} {t}" for t, c in action_counts.items()]
        click.echo(f"Would {', '.join(parts)}" if parts else "Nothing to do")
    else:
        parts = [f"{c} {t}" for t, c in action_counts.items()]
        click.echo(f"Consolidation complete: {', '.join(parts)}" if parts else "Consolidation complete: nothing to do")


# === Config command ===


@cli.command()
@click.argument("key", required=False, default=None)
@click.argument("value", required=False, default=None)
@click.pass_context
def config(ctx, key, value):
    """View or set configuration. No args=show all, key=read, key+value=set."""
    params = {}
    if key:
        params["key"] = key
    if value is not None:
        # Try to parse as number or bool
        try:
            params["value"] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            params["value"] = value

    response = run_tool(ctx, "memory_config", params)

    if ctx.obj["json"]:
        output_json(response)
        return

    data = response.get("data", {})

    # Set confirmation
    if key and value is not None:
        click.echo(f"Set {key} = {data.get('value', value)}")
        return

    # Single key read
    if key:
        v = data.get("value", data)
        if isinstance(v, dict):
            for k, val in flatten_config(v):
                click.echo(f"{k:<35}= {val}")
        else:
            click.echo(f"{key} = {v}")
        return

    # Show all config
    if isinstance(data, dict):
        for k, v in flatten_config(data):
            click.echo(f"{k:<35}= {v}")
    else:
        click.echo(str(data))


# --- Backup subgroup ---

from synaptra.backup.cli import backup_group  # noqa: E402
cli.add_command(backup_group)


# --- Entry point ---

def main() -> None:
    """CLI entry point for pyproject.toml."""
    cli()


if __name__ == "__main__":
    main()
