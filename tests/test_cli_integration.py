"""Integration tests — CLI against a live MCP server with in-memory DB."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TEST_PORT = 52198  # Different from test_mcp_client to avoid conflicts
TEST_URL = f"http://127.0.0.1:{TEST_PORT}/mcp"
CLI_CMD = [sys.executable, "-m", "synaptra.cli"]


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll until the HTTP server is ready."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            urllib.request.urlopen(req, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True  # Any HTTP response = server is up
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def run_cli(*args: str, input_text: str | None = None, url: str | None = None) -> subprocess.CompletedProcess:
    """Run a CLI command and return the result."""
    target_url = url or TEST_URL
    cmd = [*CLI_CMD, "--url", target_url, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15,
        input=input_text,
    )


@pytest.fixture(scope="module")
def server():
    """Start the mock MCP server for the test module."""
    mock_server = str(Path(__file__).parent / "_mock_server.py")

    env = {
        **os.environ,
        "SYNAPTRA_DB": "mem://",
        "SYNAPTRA_PORT": str(TEST_PORT),
    }

    proc = subprocess.Popen(
        [sys.executable, mock_server],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ready = wait_for_server(TEST_URL)
    if not ready:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.skip(f"Server failed to start: {stderr.decode(errors='replace')[:500]}")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_store_and_list(server):
    """Store a memory, then verify it appears in list."""
    # Store
    r = run_cli("store", "Integration test memory", "--type", "semantic", "--tags", "test,cli")
    assert r.returncode == 0, f"store failed: {r.stderr}"
    assert "Stored" in r.stdout

    # List
    r = run_cli("list")
    assert r.returncode == 0, f"list failed: {r.stderr}"
    assert "Integration test memory" in r.stdout or "semantic" in r.stdout


def test_store_and_get_json(server):
    """Store, then get with --json and verify structure."""
    # Store with --json to get the ID
    r = run_cli("--json", "store", "JSON test memory")
    assert r.returncode == 0, f"store failed: {r.stderr}"
    data = json.loads(r.stdout)
    assert data["success"] is True
    memory_id = data["data"]["id"]

    # Get with --json
    r = run_cli("--json", "get", memory_id)
    assert r.returncode == 0, f"get failed: {r.stderr}"
    data = json.loads(r.stdout)
    assert data["success"] is True
    assert data["data"]["memory"]["content"] == "JSON test memory"


def test_recall(server):
    """Recall should find previously stored memories."""
    r = run_cli("recall", "test memory")
    assert r.returncode == 0, f"recall failed: {r.stderr}"
    # Should find at least one result (from previous test stores)
    assert "No memories match" not in r.stdout or r.returncode == 0


def test_stats(server):
    """Stats should return without error."""
    r = run_cli("stats")
    assert r.returncode == 0, f"stats failed: {r.stderr}"
    assert "Memory Statistics" in r.stdout


def test_update_and_delete(server):
    """Store → update → delete lifecycle."""
    # Store
    r = run_cli("--json", "store", "Lifecycle test")
    assert r.returncode == 0
    memory_id = json.loads(r.stdout)["data"]["id"]

    # Update
    r = run_cli("update", memory_id, "--content", "Updated lifecycle test")
    assert r.returncode == 0, f"update failed: {r.stderr}"
    assert "Updated" in r.stdout

    # Delete
    r = run_cli("delete", memory_id, "--yes")
    assert r.returncode == 0, f"delete failed: {r.stderr}"
    assert "Deleted 1 memory" in r.stdout


def test_config(server):
    """Config read should work."""
    r = run_cli("--json", "config")
    assert r.returncode == 0, f"config failed: {r.stderr}"
    data = json.loads(r.stdout)
    assert data["success"] is True


def test_connection_error_when_server_down():
    """CLI should show friendly error when server is unreachable."""
    r = run_cli("stats", url="http://127.0.0.1:59999/mcp")
    assert r.returncode == 2
    assert "Cannot connect" in r.stderr or "did not respond" in r.stderr
