"""Issue #13 — the server answers at once, and says plainly when the model is still loading.

Part 1 tests EmbeddingService on its own. Part 2 runs the real server (tests/_slow_server.py:
real ``main()``, fake model that is slow to load or fails) and talks to it over HTTP MCP, so the
order of events under test is the real one: bind the port, then warm the model in a thread.

Nothing here touches the running service or its store: the server is in-memory, on a free
port, with HOME pointed at a temp folder (``main()`` writes its log under the home directory).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from synaptra import embeddings as emb
from synaptra.embeddings import (
    EmbeddingService,
    EmbeddingsFailed,
    EmbeddingsWarming,
)

SERVER = str(Path(__file__).parent / "_slow_server.py")


# ------------------------------------------------------------------ part 1: the service itself
class _Slow(EmbeddingService):
    """A service whose model takes `delay` seconds to load, or fails with `fail`."""

    def __init__(self, delay=0.0, fail=None):
        super().__init__()
        self.delay, self.fail, self.loads = delay, fail, 0

    def _ensure_model(self):
        with self._load_lock:
            if self._model is None:
                self.loads += 1
                time.sleep(self.delay)
                if self.fail:
                    raise RuntimeError(self.fail)
                self._model = object()


def test_a_service_that_never_started_warming_does_not_make_callers_wait():
    svc = EmbeddingService()  # the lazy path of a CLI call or an older test
    t = time.monotonic()
    asyncio.run(svc.wait_ready(timeout=30))
    assert time.monotonic() - t < 0.5
    assert svc.status() == {"state": "cold"}


def test_start_warmup_returns_at_once_and_reports_warming_then_ready():
    svc = _Slow(delay=0.6)
    t = time.monotonic()
    svc.start_warmup()
    assert time.monotonic() - t < 0.3  # did not wait for the load
    assert svc.state == "warming" and svc.status()["state"] == "warming"
    asyncio.run(svc.wait_ready(timeout=10))
    assert svc.state == "ready"
    st = svc.status()
    assert st["state"] == "ready" and st["elapsed_s"] >= 0.5 and "cause" not in st


def test_wait_ready_gives_up_at_the_bound_with_a_warming_up_error():
    svc = _Slow(delay=3.0)
    svc.start_warmup()
    t = time.monotonic()
    with pytest.raises(EmbeddingsWarming) as e:
        asyncio.run(svc.wait_ready(timeout=0.3))
    assert 0.25 <= time.monotonic() - t < 1.5  # waited the bound, not the load
    assert "warming up" in str(e.value) and "s so far" in str(e.value)


def test_embed_while_warming_raises_instead_of_blocking():
    svc = _Slow(delay=3.0)
    svc.start_warmup()
    t = time.monotonic()
    with pytest.raises(EmbeddingsWarming):
        svc.embed("anything")
    with pytest.raises(EmbeddingsWarming):
        svc.embed_batch(["anything"])
    assert time.monotonic() - t < 0.5  # a blocking embed would have waited for the load


def test_a_failed_load_is_reported_with_its_cause_and_stays_failed():
    svc = _Slow(delay=0.05, fail="model files missing")
    svc.start_warmup()
    with pytest.raises(EmbeddingsFailed) as e:
        asyncio.run(svc.wait_ready(timeout=5))
    assert "model files missing" in str(e.value)
    st = svc.status()
    assert st["state"] == "failed" and "model files missing" in st["cause"]
    with pytest.raises(EmbeddingsFailed):
        svc.embed("anything")
    svc.start_warmup()  # does not silently retry
    assert svc.state == "failed"


def test_start_warmup_is_idempotent_and_loads_the_model_once():
    svc = _Slow(delay=0.3)
    for _ in range(5):
        svc.start_warmup()
    asyncio.run(svc.wait_ready(timeout=10))
    assert svc.loads == 1


def test_warmup_joins_a_load_already_under_way_instead_of_loading_twice():
    svc = _Slow(delay=0.5)
    svc.start_warmup()
    svc.warmup()  # synchronous caller (service.py) arriving mid-load
    assert svc.state == "ready" and svc.loads == 1


def test_the_real_loader_builds_the_model_once_when_two_callers_race(
    monkeypatch, tmp_path
):
    import types

    built = []

    def slow_constructor(name, **kwargs):
        built.append(name)
        time.sleep(0.4)
        return object()

    monkeypatch.setenv(emb.MODEL_DIR_ENV, str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        types.SimpleNamespace(TextEmbedding=slow_constructor),
    )
    # tests/test_client.py replaces EmbeddingService._ensure_model for the whole session when it
    # is imported, so load a private copy of the module to get the real loader under test.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cm_embeddings_pristine", emb.__file__
    )
    pristine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pristine)
    svc = pristine.EmbeddingService()  # the real _ensure_model
    svc.start_warmup()
    time.sleep(0.1)  # the background load is inside the constructor
    svc.warmup()  # a synchronous caller arrives mid-load
    asyncio.run(svc.wait_ready(timeout=5))
    assert built == [EmbeddingService.MODEL_NAME] and svc.state == "ready"


def test_the_wait_bound_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(emb.WAIT_ENV, raising=False)
    assert emb.wait_bound() == 60.0
    monkeypatch.setenv(emb.WAIT_ENV, "7.5")
    assert emb.wait_bound() == 7.5
    monkeypatch.setenv(emb.WAIT_ENV, "soon")
    assert emb.wait_bound() == 60.0  # a typo does not turn the wait off


# ------------------------------------------------------------------ part 2: the real server
try:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    HAS_MCP = True
except ImportError:  # pragma: no cover
    HAS_MCP = False

needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="MCP client not installed")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


class Server:
    def __init__(self, tmp_path, warm_s, wait_s, fail=None):
        self.port = _free_port()
        home = tmp_path / "home"
        home.mkdir()
        # Never inherit the owner's configuration: it points at the live store.
        env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPTRA_")}
        env.update(
            USERPROFILE=str(home),
            HOME=str(home),
            SYNAPTRA_PORT=str(self.port),
            SYNAPTRA_WARMUP_WAIT_S=str(wait_s),
            WARM_SECONDS=str(warm_s),
        )
        if fail:
            env["WARM_FAIL"] = fail
        self.spawned = time.monotonic()
        self.proc = subprocess.Popen(
            [sys.executable, SERVER],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.bound_after = None

    def wait_bound(self, timeout=45.0) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"the server exited early with {self.proc.returncode}"
                )
            if _listening(self.port):
                self.bound_after = time.monotonic() - self.spawned
                return self.bound_after
            time.sleep(0.05)
        raise AssertionError("the port never opened")

    def call(self, tool: str, **args) -> dict:
        async def go():
            async with streamable_http_client(f"http://127.0.0.1:{self.port}/mcp") as (
                r,
                w,
            ):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    res = await session.call_tool(tool, args)
                    return json.loads(res.content[0].text)

        return asyncio.run(go())

    def embedding(self) -> dict:
        return self.call("memory_health")["data"]["embedding"]

    def wait_state(self, state: str, timeout=30.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            e = self.embedding()
            if e["state"] == state:
                return e
            time.sleep(0.3)
        raise AssertionError(f"never reached {state!r}; last {self.embedding()}")

    def stop(self):
        self.proc.kill()
        self.proc.wait(timeout=10)


@pytest.fixture
def server(tmp_path):
    made = []

    def make(warm_s, wait_s=30, fail=None):
        s = Server(tmp_path, warm_s, wait_s, fail)
        made.append(s)
        return s

    yield make
    for s in made:
        s.stop()


@needs_mcp
def test_the_port_opens_while_the_model_is_still_loading_and_health_says_so(server):
    s = server(warm_s=14)
    bound = s.wait_bound()
    assert bound < 14, (
        f"the port opened {bound:.1f} s in, no earlier than the model load"
    )
    warming = s.embedding()
    assert warming["state"] == "warming" and isinstance(warming["elapsed_s"], float)
    ready = s.wait_state("ready", timeout=40)
    assert ready["elapsed_s"] >= 13  # the load really took the injected time
    print(f"\nport open {bound:.1f} s after the process started; model load 14 s")


@needs_mcp
def test_non_embedding_tools_answer_normally_while_warming(server):
    s = server(warm_s=20)
    s.wait_bound()
    for tool, args in (
        ("memory_list", {}),
        ("memory_stats", {}),
        ("memory_health", {}),
    ):
        t = time.monotonic()
        out = s.call(tool, **args)
        assert out["success"] is True, (tool, out)
        assert time.monotonic() - t < 5, f"{tool} was held up"
    t = time.monotonic()
    got = s.call(
        "memory_get", id="no-such-id"
    )  # answers (not found), does not wait for the model
    assert "warming" not in json.dumps(got) and time.monotonic() - t < 5
    assert s.embedding()["state"] == "warming"  # all of that happened mid-load


@needs_mcp
def test_an_embedding_tool_waits_for_the_model_and_then_succeeds(server):
    s = server(warm_s=5, wait_s=40)
    s.wait_bound()
    assert s.embedding()["state"] == "warming"
    t = time.monotonic()
    out = s.call("memory_store", content="the boot test memory", rater="test")
    waited = time.monotonic() - t
    assert out["success"] is True, out
    assert waited >= 1.0, "it did not wait for the model"
    assert s.embedding()["state"] == "ready"


@needs_mcp
def test_past_the_bound_an_embedding_tool_says_warming_up_never_an_empty_result(server):
    s = server(warm_s=25, wait_s=1)
    s.wait_bound()
    for tool, args in (
        ("memory_recall", {"query": "anything"}),
        ("memory_store", {"content": "x", "rater": "test"}),
        ("memory_self", {"query": "who"}),
    ):
        out = s.call(tool, **args)
        assert out["success"] is False and out["data"] is None, (tool, out)
        assert "warming up" in out["error"] and "s so far" in out["error"], (tool, out)
    assert (
        s.embedding()["state"] == "warming"
    )  # still loading: the error was about that


@needs_mcp
def test_a_failed_load_shows_failed_with_the_cause_and_the_port_stays_up(server):
    s = server(warm_s=1, wait_s=5, fail="model files missing")
    s.wait_bound()
    failed = s.wait_state("failed", timeout=30)
    assert "model files missing" in failed["cause"]
    for tool, args in (
        ("memory_recall", {"query": "anything"}),
        ("memory_store", {"content": "x", "rater": "test"}),
    ):
        out = s.call(tool, **args)
        assert out["success"] is False and "model files missing" in out["error"], (
            tool,
            out,
        )
    assert s.call("memory_list")["success"] is True  # everything else still answers
    assert _listening(s.port)
