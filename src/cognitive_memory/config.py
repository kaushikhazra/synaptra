"""Configuration management — loads defaults from YAML, overrides from storage config table."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.default.yaml"


class Config:
    """Hierarchical config: YAML defaults, overridden by SQLite config table at read time."""

    def __init__(self, storage=None, config_path: Path | None = None):
        self._storage = storage
        path = config_path or _DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                self._defaults = yaml.safe_load(f) or {}
        else:
            self._defaults = {}

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value. Checks storage overrides first, then YAML defaults.

        Keys use dot notation: 'decay.growth_factor', 'retrieval.weights.semantic'.
        If storage.get_config() is async (coroutine), it is discarded and YAML
        defaults are used — async config reads require the async get_config_async()
        path.
        """
        import inspect
        # Check storage override first
        if self._storage is not None:
            override = self._storage.get_config(dotted_key)
            # If get_config is async, the call returns a coroutine — close it and
            # fall through to YAML defaults (async backends must use the async path).
            if inspect.iscoroutine(override):
                override.close()
                override = None
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

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a config override in SQLite."""
        if self._storage is None:
            raise RuntimeError("No storage backend for config writes")
        self._storage.set_config(dotted_key, value)

    def get_all(self) -> dict:
        """Return merged config: defaults with storage overrides applied.

        If storage.get_all_config() is async (coroutine), it is discarded and
        only YAML defaults are returned.
        """
        import inspect
        result = dict(self._defaults)
        if self._storage is not None:
            overrides = self._storage.get_all_config()
            if inspect.iscoroutine(overrides):
                overrides.close()
                overrides = {}
            for key, value in overrides.items():
                parts = key.split(".")
                node = result
                for part in parts[:-1]:
                    if part not in node or not isinstance(node[part], dict):
                        node[part] = {}
                    node = node[part]
                node[parts[-1]] = value
        return result
