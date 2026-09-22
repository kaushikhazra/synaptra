"""Configuration management — loads defaults from YAML, overrides from the storage config table.

Reads must stay synchronous: `Config.get` is called from scoring loops in
`retrieval.py` and per-field in `engine.store_memory`. Awaiting storage per key
is not viable. So on async backends the overrides are loaded once into an
in-memory cache by `load_overrides()` and served from there.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import yaml


log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.default.yaml"


class Config:
    """Hierarchical config: YAML defaults, overridden by the storage config table."""

    def __init__(self, storage=None, config_path: Path | None = None):
        self._storage = storage
        self._overrides: dict[str, Any] = {}
        self._overrides_loaded = False
        self._warned_unloaded = False
        path = config_path or _DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                self._defaults = yaml.safe_load(f) or {}
        else:
            self._defaults = {}

    # === Async backend detection ===

    def _storage_is_async(self) -> bool:
        """True when the backend's config methods are coroutines.

        Checked on the function rather than by calling it, so no orphan
        coroutine is ever created.
        """
        if self._storage is None:
            return False
        return inspect.iscoroutinefunction(getattr(self._storage, "get_config", None))

    # === Override cache ===

    async def load_overrides(self, force: bool = False) -> None:
        """Load storage overrides into the in-memory cache. Idempotent.

        Must be awaited once before `get`/`get_all` can see overrides on an
        async backend. Cheap to call repeatedly — returns on a boolean check
        after the first load.
        """
        if self._overrides_loaded and not force:
            return
        if self._storage is None:
            self._overrides_loaded = True
            return

        overrides = self._storage.get_all_config()
        if inspect.isawaitable(overrides):
            overrides = await overrides
        self._overrides = dict(overrides or {})
        self._overrides_loaded = True

    def _cached_override(self, dotted_key: str) -> Any:
        """Return a cached override for the key, or None if absent.

        Warns once if the cache was never loaded on an async backend — that
        state silently serves YAML defaults and is the defect this module was
        rewritten to remove.
        """
        if self._overrides_loaded:
            return self._overrides.get(dotted_key)
        if self._storage_is_async() and not self._warned_unloaded:
            self._warned_unloaded = True
            log.warning(
                "Config overrides were never loaded on an async backend; serving YAML "
                "defaults. Await Config.load_overrides() during startup."
            )
        return None

    # === Reads ===

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value. Overrides win over YAML defaults.

        Keys use dot notation: 'decay.growth_factor', 'retrieval.weights.semantic'.
        """
        override = self._cached_override(dotted_key)
        if override is not None:
            return override

        # Sync backends can be read directly, no cache needed.
        if self._storage is not None and not self._storage_is_async():
            override = self._storage.get_config(dotted_key)
            if override is not None:
                return override

        # Walk the YAML defaults
        parts = dotted_key.split(".")
        node = self._defaults
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get_all(self) -> dict:
        """Return merged config: defaults with overrides applied."""
        result = dict(self._defaults)

        overrides: dict[str, Any] = dict(self._overrides)
        if self._storage is not None and not self._storage_is_async():
            direct = self._storage.get_all_config()
            if direct:
                overrides.update(direct)
        elif not self._overrides_loaded:
            self._cached_override("")  # emits the one-time warning

        for key, value in overrides.items():
            parts = key.split(".")
            node = result
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    # === Writes ===

    async def set_async(self, dotted_key: str, value: Any) -> None:
        """Persist a config override and update the cache. Use this on async backends."""
        if self._storage is None:
            raise RuntimeError("No storage backend for config writes")
        result = self._storage.set_config(dotted_key, value)
        if inspect.isawaitable(result):
            await result
        self._overrides[dotted_key] = value

    def set(self, dotted_key: str, value: Any) -> None:
        """Persist a config override on a synchronous backend.

        Raises on an async backend rather than dropping the write. Silently
        discarding an un-awaited coroutine here is what made every
        `synaptra-cli config <key> <value>` a no-op that reported success.
        """
        if self._storage is None:
            raise RuntimeError("No storage backend for config writes")
        if self._storage_is_async():
            raise RuntimeError(
                "Config.set cannot write to an async storage backend — the write "
                "would be discarded. Await Config.set_async() instead."
            )
        self._storage.set_config(dotted_key, value)
        self._overrides[dotted_key] = value
