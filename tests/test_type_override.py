"""Tests for the MCP type override (issue #2).

The override was reported as "accepted and then not applied". It was not: the
tool declares the argument as ``type`` while the engine parameter, the stored
field, the response field and the CLI flag are all ``memory_type``. The SDK's
FastMCP builds its argument model with Pydantic's default ``extra="ignore"``
and registers dispatch with ``validate_input=False``, so a misnamed *optional*
argument was discarded before the tool body ran and the call still reported
success.

These tests cover both halves of the fix:
- ``memory_type`` is accepted as an alias for ``type``
- genuinely unknown argument names are rejected loudly instead of dropped
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synaptra import decay
from synaptra.models import MemoryType
from synaptra.server import StrictArgumentFastMCP, _resolve_type_alias


class TestResolveTypeAlias:
    """`type` and `memory_type` collapse to one value, or fail loudly."""

    def test_type_only(self):
        assert _resolve_type_alias("procedural", None) == "procedural"

    def test_memory_type_alias_only(self):
        assert _resolve_type_alias(None, "procedural") == "procedural"

    def test_neither_returns_none_so_classification_runs(self):
        assert _resolve_type_alias(None, None) is None

    def test_both_agreeing_is_accepted(self):
        assert _resolve_type_alias("semantic", "semantic") == "semantic"

    def test_both_conflicting_raises(self):
        with pytest.raises(ValueError, match="Conflicting type arguments"):
            _resolve_type_alias("semantic", "procedural")

    def test_conflict_message_names_both_values(self):
        with pytest.raises(ValueError) as exc:
            _resolve_type_alias("semantic", "procedural")
        assert "semantic" in str(exc.value)
        assert "procedural" in str(exc.value)


class TestStrictArgumentRejection:
    """Unknown argument names must not be silently discarded."""

    def _server(self):
        mcp = StrictArgumentFastMCP("test-strict")

        @mcp.tool()
        async def sample(content: str, type: str | None = None) -> str:
            return f"{content}|{type}"

        return mcp

    @pytest.mark.asyncio
    async def test_declared_arguments_pass_through(self):
        mcp = self._server()
        result = await mcp.call_tool("sample", {"content": "x", "type": "semantic"})
        assert "semantic" in str(result)

    @pytest.mark.asyncio
    async def test_unknown_argument_raises(self):
        mcp = self._server()
        with pytest.raises(ValueError, match="Unknown argument"):
            await mcp.call_tool("sample", {"content": "x", "bogus": "y"})

    @pytest.mark.asyncio
    async def test_error_names_the_offending_argument(self):
        mcp = self._server()
        with pytest.raises(ValueError) as exc:
            await mcp.call_tool("sample", {"content": "x", "memory_type": "semantic"})
        assert "memory_type" in str(exc.value)

    @pytest.mark.asyncio
    async def test_error_lists_accepted_arguments(self):
        """The message has to be actionable, or it just moves the confusion."""
        mcp = self._server()
        with pytest.raises(ValueError) as exc:
            await mcp.call_tool("sample", {"content": "x", "bogus": "y"})
        assert "content" in str(exc.value)
        assert "type" in str(exc.value)

    @pytest.mark.asyncio
    async def test_reserved_underscore_keys_are_not_rejected(self):
        """``_meta`` belongs to the transport. Rejecting it would fail every call."""
        mcp = self._server()
        result = await mcp.call_tool(
            "sample", {"content": "x", "type": "semantic", "_meta": {"trace": 1}}
        )
        assert "semantic" in str(result)

    @pytest.mark.asyncio
    async def test_multiple_unknown_arguments_all_reported(self):
        mcp = self._server()
        with pytest.raises(ValueError) as exc:
            await mcp.call_tool("sample", {"content": "x", "aaa": 1, "zzz": 2})
        assert "aaa" in str(exc.value)
        assert "zzz" in str(exc.value)


class TestInitialStabilityByType:
    """Type is the decay class, so a wrong type is a durability bug.

    These are the values the override has to be able to reach; if the mapping
    moves, a memory stored as `procedural` no longer outlives an `episodic` one
    by the margin the type is chosen for.
    """

    EXPECTED = {
        "working": 0.04,
        "episodic": 2.0,
        "semantic": 14.0,
        "procedural": 60.0,
        "identity": 365.0,
        "person": 90.0,
    }

    @pytest.mark.parametrize("mem_type,expected", sorted(EXPECTED.items()))
    def test_initial_stability(self, mem_type, expected):
        assert decay.get_initial_stability(mem_type) == expected

    def test_every_enum_member_is_covered(self):
        """A new MemoryType must not silently inherit the 2.0 fallback."""
        assert {t.value for t in MemoryType} == set(self.EXPECTED)


class TestToolForwardsResolvedType:
    """The resolved type must reach the engine from either argument name."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs", [
        {"type": "procedural"},
        {"memory_type": "procedural"},
        {"type": "procedural", "memory_type": "procedural"},
    ])
    async def test_store_forwards_procedural(self, kwargs):
        from synaptra import server as srv

        engine = AsyncMock()
        engine.store_memory.return_value = MagicMock(
            **{"model_dump.return_value": {"memory_type": "procedural"}}
        )
        with patch.object(srv, "_get_engine", return_value=engine):
            await srv.memory_store(content="c", **kwargs)

        assert engine.store_memory.await_args.kwargs["memory_type"] == "procedural"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs", [
        {"type": "semantic"},
        {"memory_type": "semantic"},
    ])
    async def test_update_forwards_semantic(self, kwargs):
        from synaptra import server as srv

        engine = AsyncMock()
        engine.update_memory.return_value = MagicMock(
            **{"model_dump.return_value": {"memory_type": "semantic"}}
        )
        with patch.object(srv, "_get_engine", return_value=engine):
            await srv.memory_update(id="abc", **kwargs)

        assert engine.update_memory.await_args.kwargs["memory_type"] == "semantic"

    @pytest.mark.asyncio
    async def test_omitting_type_leaves_classification_to_the_engine(self):
        from synaptra import server as srv

        engine = AsyncMock()
        engine.store_memory.return_value = MagicMock(**{"model_dump.return_value": {}})
        with patch.object(srv, "_get_engine", return_value=engine):
            await srv.memory_store(content="c")

        assert engine.store_memory.await_args.kwargs["memory_type"] is None
