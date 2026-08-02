"""Tests for server.py _get_engine() backend branching (Phase 3 — Component D).

These tests verify:
- SYNAPTRA_BACKEND=surrealkv-file (default) → MemoryEngine with legacy db_path
- SYNAPTRA_BACKEND=rocksdb-server → MemoryEngine injected with SurrealServerStorage
- Unknown backend → ValueError with helpful message
- SYNAPTRA_SURREAL_URL env var is forwarded to SurrealServerStorage

All tests mock the actual storage/engine construction so no DB or network I/O occurs.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetEngineBackendBranching:
    """Tests for _get_engine() backend selection logic."""

    def setup_method(self):
        """Reset the global singleton before each test."""
        import synaptra.server as srv
        srv._engine = None

    def teardown_method(self):
        """Reset again after each test for isolation."""
        import synaptra.server as srv
        srv._engine = None

    def test_default_backend_is_surrealkv_file(self):
        """Default (no env var) → surrealkv-file path; MemoryEngine gets a db_path."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("SYNAPTRA_BACKEND", "SYNAPTRA_DB",
                            "SYNAPTRA_CONFIG")}
        env["SYNAPTRA_DB"] = "mem://"  # use in-memory DB so no filesystem I/O

        with patch.dict(os.environ, env, clear=True):
            from synaptra.server import _get_engine
            import synaptra.server as srv
            srv._engine = None  # reset singleton

            mock_engine = MagicMock()
            with patch("synaptra.server.MemoryEngine", return_value=mock_engine) as mock_me:
                engine = _get_engine()
                # db_path must have been passed (keyword or positional)
                call_kwargs = mock_me.call_args
                assert call_kwargs is not None
                # Verify it was called with db_path (mem://) not storage=
                assert "storage" not in (call_kwargs.kwargs or {}) or call_kwargs.kwargs.get("storage") is None

        assert engine is mock_engine

    def test_surrealkv_file_explicit(self):
        """Explicit SYNAPTRA_BACKEND=surrealkv-file → legacy path."""
        import synaptra.server as srv
        env = {"SYNAPTRA_BACKEND": "surrealkv-file", "SYNAPTRA_DB": "mem://"}
        with patch.dict(os.environ, env, clear=False):
            srv._engine = None
            mock_engine = MagicMock()
            with patch("synaptra.server.MemoryEngine", return_value=mock_engine):
                engine = srv._get_engine()
        assert engine is mock_engine

    def test_rocksdb_server_creates_surreal_server_storage(self):
        """SYNAPTRA_BACKEND=rocksdb-server → SurrealServerStorage injected."""
        import synaptra.server as srv
        env = {
            "SYNAPTRA_BACKEND": "rocksdb-server",
            "SYNAPTRA_SURREAL_URL": "ws://127.0.0.1:8000/rpc",
        }
        with patch.dict(os.environ, env, clear=False):
            srv._engine = None

            mock_storage = MagicMock()
            mock_engine = MagicMock()

            with patch("synaptra.surreal_server_storage.SurrealServerStorage",
                       return_value=mock_storage) as mock_sss:
                with patch("synaptra.server.MemoryEngine", return_value=mock_engine) as mock_me:
                    engine = srv._get_engine()

            # SurrealServerStorage must have been called with the URL
            mock_sss.assert_called_once_with(url="ws://127.0.0.1:8000/rpc")
            # MemoryEngine must have been called with storage=
            call_kwargs = mock_me.call_args.kwargs
            assert call_kwargs.get("storage") is mock_storage

        assert engine is mock_engine

    def test_rocksdb_server_default_url(self):
        """When SYNAPTRA_SURREAL_URL absent, defaults to ws://127.0.0.1:8000/rpc."""
        import synaptra.server as srv
        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("SYNAPTRA_BACKEND", "SYNAPTRA_SURREAL_URL",
                                  "SYNAPTRA_CONFIG")}
        env_clean["SYNAPTRA_BACKEND"] = "rocksdb-server"

        with patch.dict(os.environ, env_clean, clear=True):
            srv._engine = None

            with patch("synaptra.surreal_server_storage.SurrealServerStorage") as mock_sss:
                mock_sss.return_value = MagicMock()
                with patch("synaptra.server.MemoryEngine", return_value=MagicMock()):
                    srv._get_engine()

            mock_sss.assert_called_once_with(url="ws://127.0.0.1:8000/rpc")

    def test_unknown_backend_raises_value_error(self):
        """Unknown SYNAPTRA_BACKEND value → ValueError with supported values listed."""
        import synaptra.server as srv
        env = {"SYNAPTRA_BACKEND": "bogus-backend"}
        with patch.dict(os.environ, env, clear=False):
            srv._engine = None
            with pytest.raises(ValueError, match="bogus-backend"):
                srv._get_engine()

    def test_unknown_backend_message_lists_supported_values(self):
        import synaptra.server as srv
        env = {"SYNAPTRA_BACKEND": "xyzzy"}
        with patch.dict(os.environ, env, clear=False):
            srv._engine = None
            with pytest.raises(ValueError) as exc_info:
                srv._get_engine()
            msg = str(exc_info.value)
            assert "surrealkv-file" in msg
            assert "rocksdb-server" in msg
