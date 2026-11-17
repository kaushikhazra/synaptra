"""Issue #13, stdio — a brain's connect does not wait for the embedding model.

Every second-brain runs Synaptra over stdio. The real ``server.main()`` runs here, with a
fake model that is slow to load (``tests/_slow_server.py --transport stdio``), and a real MCP
client talks to it over the process's stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    HAS_MCP = True
except ImportError:  # pragma: no cover
    HAS_MCP = False

needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="MCP client not installed")

SERVER = str(Path(__file__).parent / "_slow_server.py")


def _params(tmp_path, warm_s, wait_s):
    home = tmp_path / "home"
    home.mkdir()
    # Never inherit the owner's configuration: it points at the live store.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPTRA_")}
    env.update(
        USERPROFILE=str(home),
        HOME=str(home),
        SYNAPTRA_WARMUP_WAIT_S=str(wait_s),
        WARM_SECONDS=str(warm_s),
    )
    return StdioServerParameters(
        command=sys.executable, args=[SERVER, "--transport", "stdio"], env=env
    )


def _text(result) -> dict:
    return json.loads(result.content[0].text)


@needs_mcp
def test_stdio_connect_and_non_embedding_tools_do_not_wait_for_the_model(tmp_path):
    async def go():
        started = time.monotonic()
        async with stdio_client(_params(tmp_path, warm_s=25, wait_s=1)) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                stats = _text(await session.call_tool("memory_stats", {}))
                answered = time.monotonic() - started
                health = _text(await session.call_tool("memory_health", {}))
                store = _text(
                    await session.call_tool(
                        "memory_store", {"content": "x", "rater": "test"}
                    )
                )
                return answered, stats, health, store

    answered, stats, health, store = asyncio.run(go())
    assert stats["success"] is True, stats
    assert answered < 20, (
        f"connect + memory_stats took {answered:.1f} s (model load 25 s)"
    )
    embedding = health["data"]["embedding"]
    assert embedding["state"] == "warming", health  # the load was still under way
    # Past the bound the owner is told, plainly, and gets no degraded result.
    assert store["success"] is False and store["data"] is None, store
    assert "warming up" in store["error"] and "s so far" in store["error"], store


@needs_mcp
def test_stdio_embedding_tool_waits_then_succeeds_when_the_model_is_ready_in_time(
    tmp_path,
):
    async def go():
        async with stdio_client(_params(tmp_path, warm_s=4, wait_s=40)) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                t = time.monotonic()
                out = _text(
                    await session.call_tool(
                        "memory_store",
                        {"content": "stdio boot memory", "rater": "test"},
                    )
                )
                waited = time.monotonic() - t
                health = _text(await session.call_tool("memory_health", {}))
                return out, waited, health

    out, waited, health = asyncio.run(go())
    assert out["success"] is True, out
    assert waited >= 1.0, "it did not wait for the model"
    assert health["data"]["embedding"]["state"] == "ready"
