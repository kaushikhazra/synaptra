"""Config override tests.

Covers the 2026-08-25 fix: on an async storage backend every override write was
discarded un-awaited and every read fell through to the YAML defaults, while the
CLI reported success. See .claude/specs/config-override-fix/fix.md.
"""

from __future__ import annotations

import pytest

from synaptra.config import Config


class SyncStorage:
    """Minimal synchronous config backend."""

    def __init__(self, initial=None):
        self.data = dict(initial or {})

    def get_config(self, key):
        return self.data.get(key)

    def set_config(self, key, value):
        self.data[key] = value

    def get_all_config(self):
        return dict(self.data)


class AsyncStorage:
    """Minimal asynchronous config backend, like SurrealServerStorage."""

    def __init__(self, initial=None):
        self.data = dict(initial or {})
        self.set_calls = 0

    async def get_config(self, key):
        return self.data.get(key)

    async def set_config(self, key, value):
        self.set_calls += 1
        self.data[key] = value

    async def get_all_config(self):
        return dict(self.data)


# === Async backend: the defect this fix exists for ===


@pytest.mark.asyncio
async def test_async_override_is_readable_after_load():
    storage = AsyncStorage({"decay.decay_influence": 0.0})
    cfg = Config(storage=storage)

    # Before loading, the YAML default wins and nothing blows up.
    assert cfg.get("decay.decay_influence", 0.5) == 0.5

    await cfg.load_overrides()
    assert cfg.get("decay.decay_influence") == 0.0


@pytest.mark.asyncio
async def test_async_write_actually_persists():
    storage = AsyncStorage()
    cfg = Config(storage=storage)
    await cfg.load_overrides()

    await cfg.set_async("decay.decay_influence", 0.0)

    # It reached storage, not just the cache.
    assert storage.set_calls == 1
    assert storage.data["decay.decay_influence"] == 0.0
    # And it is visible immediately, without a reload.
    assert cfg.get("decay.decay_influence") == 0.0


@pytest.mark.asyncio
async def test_async_write_survives_a_fresh_config_object():
    """The write must land in storage, not only in the in-memory cache."""
    storage = AsyncStorage()
    cfg = Config(storage=storage)
    await cfg.load_overrides()
    await cfg.set_async("retrieval.default_limit", 42)

    fresh = Config(storage=storage)
    await fresh.load_overrides()
    assert fresh.get("retrieval.default_limit") == 42


def test_sync_set_refuses_an_async_backend():
    """Silently dropping the coroutine here is the original defect."""
    cfg = Config(storage=AsyncStorage())
    with pytest.raises(RuntimeError, match="set_async"):
        cfg.set("decay.decay_influence", 0.0)


@pytest.mark.asyncio
async def test_load_overrides_is_idempotent():
    storage = AsyncStorage({"a.b": 1})
    cfg = Config(storage=storage)
    await cfg.load_overrides()

    storage.data["a.b"] = 2
    await cfg.load_overrides()  # no force — must not re-read
    assert cfg.get("a.b") == 1

    await cfg.load_overrides(force=True)
    assert cfg.get("a.b") == 2


@pytest.mark.asyncio
async def test_get_all_merges_overrides_over_defaults():
    storage = AsyncStorage({"decay.decay_influence": 0.0})
    cfg = Config(storage=storage)
    await cfg.load_overrides()

    merged = cfg.get_all()
    assert merged["decay"]["decay_influence"] == 0.0
    # A sibling default is preserved rather than clobbered by the merge.
    assert "growth_factor" in merged["decay"]


@pytest.mark.asyncio
async def test_unloaded_async_cache_warns_once(caplog):
    cfg = Config(storage=AsyncStorage({"a.b": 1}))
    with caplog.at_level("WARNING"):
        cfg.get("a.b")
        cfg.get("a.b")
    warnings = [r for r in caplog.records if "never loaded" in r.message]
    assert len(warnings) == 1


# === Sync backend: must be unaffected ===


def test_sync_backend_reads_without_loading():
    cfg = Config(storage=SyncStorage({"decay.decay_influence": 0.25}))
    assert cfg.get("decay.decay_influence") == 0.25


def test_sync_backend_write_round_trips():
    storage = SyncStorage()
    cfg = Config(storage=storage)
    cfg.set("decay.decay_influence", 0.25)
    assert storage.data["decay.decay_influence"] == 0.25
    assert cfg.get("decay.decay_influence") == 0.25


def test_no_storage_falls_back_to_yaml_defaults():
    cfg = Config(storage=None)
    assert cfg.get("decay.decay_influence") == 0.5
    with pytest.raises(RuntimeError):
        cfg.set("a.b", 1)


@pytest.mark.asyncio
async def test_no_storage_load_is_a_noop():
    cfg = Config(storage=None)
    await cfg.load_overrides()
    assert cfg.get("decay.decay_influence") == 0.5
