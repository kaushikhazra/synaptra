"""MCP HTTP test client — spawns the server and exercises all 14 tools via Streamable HTTP."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# We need to handle the case where mcp might not be installed
try:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("MCP SDK not installed or missing streamable_http client.")
    print("Install with: pip install 'mcp>=1.0.0'")

TEST_PORT = 52199
TEST_URL = f"http://127.0.0.1:{TEST_PORT}/mcp"


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll until the HTTP server is ready. Any HTTP response means the server is up."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            urllib.request.urlopen(req, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


async def run_mcp_tests():
    """Start the mock server, connect via Streamable HTTP, exercise all tools."""
    mock_server = str(Path(__file__).parent / "_mock_server.py")

    env = {
        **os.environ,
        "SYNAPTRA_DB": "mem://",
        "SYNAPTRA_PORT": str(TEST_PORT),
    }

    # Start the server subprocess
    proc = subprocess.Popen(
        [sys.executable, mock_server],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        print(f"Waiting for server on port {TEST_PORT}...")
        ready = wait_for_server(TEST_URL)
        if not ready:
            print("Server failed to start within timeout.")
            proc.kill()
            stdout, stderr = proc.communicate()
            print(f"STDOUT: {stdout.decode(errors='replace')}")
            print(f"STDERR: {stderr.decode(errors='replace')}")
            sys.exit(1)

        print("Server ready. Connecting MCP client...")

        async with streamablehttp_client(TEST_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                tools = await session.list_tools()
                print(f"Available tools: {len(tools.tools)}")
                for t in tools.tools:
                    print(f"  - {t.name}")

                assert len(tools.tools) == 16, f"Expected 16 tools, got {len(tools.tools)}"
                print(f"\nPASS: 16 tools registered")

                # memory_store
                result = await session.call_tool("memory_store", {
                    "content": "Python is a versatile programming language",
                    "type": "semantic",
                    "tags": ["programming", "python"],
                })
                store_data = json.loads(result.content[0].text)
                assert store_data["success"], f"Store failed: {store_data}"
                memory_id = store_data["data"]["id"]
                print(f"PASS: memory_store — id={memory_id[:8]}...")

                # memory_get
                result = await session.call_tool("memory_get", {"id": memory_id})
                get_data = json.loads(result.content[0].text)
                assert get_data["success"]
                assert get_data["data"]["memory"]["content"] == "Python is a versatile programming language"
                print(f"PASS: memory_get — content matches")

                # memory_update
                result = await session.call_tool("memory_update", {
                    "id": memory_id,
                    "content": "Python is a versatile and popular programming language",
                })
                update_data = json.loads(result.content[0].text)
                assert update_data["success"]
                print(f"PASS: memory_update — content updated")

                # Verify version was created
                result = await session.call_tool("memory_get", {"id": memory_id})
                get_data = json.loads(result.content[0].text)
                versions = get_data["data"].get("versions", [])
                assert len(versions) >= 1, f"Expected version, got {len(versions)}"
                print(f"PASS: memory_get — version history present ({len(versions)} versions)")

                # Store more for recall testing
                result = await session.call_tool("memory_store", {
                    "content": "JavaScript is the language of the web",
                    "type": "semantic",
                })
                js_data = json.loads(result.content[0].text)
                js_id = js_data["data"]["id"]

                # memory_recall
                result = await session.call_tool("memory_recall", {
                    "query": "programming language",
                    "limit": 5,
                })
                recall_data = json.loads(result.content[0].text)
                assert recall_data["success"]
                memories = recall_data["data"]["memories"]
                assert len(memories) > 0, "Recall returned no results"
                assert all("score" in m for m in memories)
                assert all("found_by" in m for m in memories)
                print(f"PASS: memory_recall — {len(memories)} results with scores and provenance")

                # memory_relate
                result = await session.call_tool("memory_relate", {
                    "source_id": memory_id,
                    "target_id": js_id,
                    "rel_type": "relates_to",
                    "strength": 1.0,
                })
                relate_data = json.loads(result.content[0].text)
                assert relate_data["success"]
                print(f"PASS: memory_relate — relationship created")

                # memory_related (read-only)
                result = await session.call_tool("memory_related", {
                    "id": memory_id,
                    "depth": 1,
                })
                related_data = json.loads(result.content[0].text)
                assert related_data["success"]
                print(f"PASS: memory_related — {len(related_data['data'])} neighbors found")

                # memory_unrelate
                result = await session.call_tool("memory_unrelate", {
                    "source_id": memory_id,
                    "target_id": js_id,
                    "rel_type": "relates_to",
                })
                unrelate_data = json.loads(result.content[0].text)
                assert unrelate_data["success"]
                print(f"PASS: memory_unrelate — relationship removed")

                # memory_list
                result = await session.call_tool("memory_list", {
                    "type": "semantic",
                    "limit": 10,
                })
                list_data = json.loads(result.content[0].text)
                assert list_data["success"]
                assert len(list_data["data"]["memories"]) >= 2
                print(f"PASS: memory_list — {len(list_data['data']['memories'])} memories listed")

                # memory_archive
                result = await session.call_tool("memory_archive", {"id": js_id})
                archive_data = json.loads(result.content[0].text)
                assert archive_data["success"]
                assert archive_data["data"]["archived"] == 1
                print(f"PASS: memory_archive — 1 memory archived")

                # memory_restore
                result = await session.call_tool("memory_restore", {"id": js_id})
                restore_data = json.loads(result.content[0].text)
                assert restore_data["success"]
                print(f"PASS: memory_restore — memory restored")

                # memory_stats
                result = await session.call_tool("memory_stats", {})
                stats_data = json.loads(result.content[0].text)
                assert stats_data["success"]
                assert "counts" in stats_data["data"]
                assert "decay" in stats_data["data"]
                print(f"PASS: memory_stats — complete stats returned")

                # memory_consolidate (dry run)
                result = await session.call_tool("memory_consolidate", {"dry_run": True})
                consol_data = json.loads(result.content[0].text)
                assert consol_data["success"]
                print(f"PASS: memory_consolidate — dry run returned {consol_data['data']['count']} actions")

                # memory_config
                result = await session.call_tool("memory_config", {
                    "key": "decay.growth_factor",
                    "value": 3.0,
                })
                config_data = json.loads(result.content[0].text)
                assert config_data["success"]
                print(f"PASS: memory_config — set growth_factor=3.0")

                # Read it back
                result = await session.call_tool("memory_config", {"key": "decay.growth_factor"})
                config_data = json.loads(result.content[0].text)
                assert config_data["data"]["value"] == 3.0
                print(f"PASS: memory_config — read back confirms 3.0")

                # memory_self — store identity memory, then recall via memory_self
                result = await session.call_tool("memory_store", {
                    "content": "I am Velasari, an AI agent built by Kaushik",
                    "type": "identity",
                    "tags": ["name", "origin"],
                })
                identity_data = json.loads(result.content[0].text)
                assert identity_data["success"]
                identity_id = identity_data["data"]["id"]
                print(f"PASS: memory_store — identity memory stored")

                result = await session.call_tool("memory_self", {
                    "query": "who am I",
                })
                self_data = json.loads(result.content[0].text)
                assert self_data["success"]
                assert len(self_data["data"]["memories"]) > 0
                print(f"PASS: memory_self — {len(self_data['data']['memories'])} identity memories found")

                # memory_who — store person memory, then recall via memory_who
                result = await session.call_tool("memory_store", {
                    "content": "Kaushik prefers explicit error handling over broad try/except",
                    "type": "person",
                    "tags": ["person:kaushik", "preferences", "coding-style"],
                })
                person_data = json.loads(result.content[0].text)
                assert person_data["success"]
                person_id = person_data["data"]["id"]
                print(f"PASS: memory_store — person memory stored")

                result = await session.call_tool("memory_who", {
                    "person": "kaushik",
                    "query": "coding preferences",
                })
                who_data = json.loads(result.content[0].text)
                assert who_data["success"]
                assert len(who_data["data"]["memories"]) > 0
                print(f"PASS: memory_who — {len(who_data['data']['memories'])} person memories found")

                # Clean up identity and person memories
                await session.call_tool("memory_delete", {"id": identity_id, "confirm": True})
                await session.call_tool("memory_delete", {"id": person_id, "confirm": True})

                # memory_delete
                result = await session.call_tool("memory_delete", {"id": js_id, "confirm": True})
                delete_data = json.loads(result.content[0].text)
                assert delete_data["success"]
                assert delete_data["data"]["deleted"] == 1
                print(f"PASS: memory_delete — memory permanently deleted")

                # Verify deleted
                result = await session.call_tool("memory_get", {"id": js_id})
                get_data = json.loads(result.content[0].text)
                assert not get_data["success"]
                print(f"PASS: memory_get after delete — correctly returns error")

                print(f"\n{'='*60}")
                print(f"ALL MCP TOOL TESTS PASSED")

    finally:
        # Clean up server process
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    if not HAS_MCP:
        sys.exit(1)
    asyncio.run(run_mcp_tests())
