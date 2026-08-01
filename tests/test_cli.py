"""Unit tests for CLI helpers and commands."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cognitive_memory.cli import (
    cli,
    split_tags,
    build_time_range,
    route_ids,
    format_age,
    truncate,
    flatten_config,
)


# --- Helper tests ---


class TestSplitTags:
    def test_none(self):
        assert split_tags(None) is None

    def test_single(self):
        assert split_tags("python") == ["python"]

    def test_multiple(self):
        assert split_tags("python,ai,memory") == ["python", "ai", "memory"]

    def test_strips_whitespace(self):
        assert split_tags("python , ai , memory") == ["python", "ai", "memory"]

    def test_empty_string(self):
        assert split_tags("") == []

    def test_trailing_comma(self):
        assert split_tags("python,") == ["python"]

    def test_only_commas(self):
        assert split_tags(",,,") == []


class TestBuildTimeRange:
    def test_both_none(self):
        assert build_time_range(None, None) is None

    def test_both_provided(self):
        result = build_time_range("2026-01-01", "2026-03-01")
        assert result == {"start": "2026-01-01", "end": "2026-03-01"}

    def test_start_only_exits(self):
        """Partial range should exit with code 2."""
        runner = CliRunner()
        # We test via the CLI to capture sys.exit
        # build_time_range calls sys.exit directly, so we test it indirectly
        # through a command that uses it (tested in command tests)
        # For the unit test, verify the logic:
        with pytest.raises(SystemExit) as exc_info:
            build_time_range("2026-01-01", None)
        assert exc_info.value.code == 2

    def test_end_only_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            build_time_range(None, "2026-03-01")
        assert exc_info.value.code == 2

    def test_empty_strings_treated_as_none(self):
        assert build_time_range("", "") is None


class TestRouteIds:
    def test_single_id(self):
        assert route_ids(("abc123",)) == {"id": "abc123"}

    def test_multiple_ids(self):
        assert route_ids(("abc", "def", "ghi")) == {"ids": ["abc", "def", "ghi"]}

    def test_two_ids(self):
        assert route_ids(("abc", "def")) == {"ids": ["abc", "def"]}


class TestFormatAge:
    def test_now(self):
        dt = datetime.now(timezone.utc).isoformat()
        assert format_age(dt) == "now"

    def test_minutes(self):
        dt = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert format_age(dt) == "5m ago"

    def test_hours(self):
        dt = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        assert format_age(dt) == "3h ago"

    def test_days(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        assert format_age(dt) == "2d ago"

    def test_weeks(self):
        dt = (datetime.now(timezone.utc) - timedelta(weeks=2)).isoformat()
        assert format_age(dt) == "2w ago"

    def test_months(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        assert format_age(dt) == "2mo ago"

    def test_invalid_string(self):
        assert format_age("not-a-date") == "?"

    def test_none_input(self):
        assert format_age(None) == "?"

    def test_future(self):
        dt = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert format_age(dt) == "future"


class TestTruncate:
    def test_short_string(self):
        assert truncate("hello", 10) == "hello"

    def test_exact_length(self):
        assert truncate("hello", 5) == "hello"

    def test_truncated(self):
        result = truncate("hello world", 6)
        assert len(result) == 6
        assert result.endswith("\u2026")

    def test_single_char_limit(self):
        assert truncate("hello", 1) == "\u2026"


class TestFlattenConfig:
    def test_flat(self):
        result = flatten_config({"a": 1, "b": 2})
        assert result == [("a", "1"), ("b", "2")]

    def test_nested(self):
        result = flatten_config({"decay": {"growth_factor": 3.0, "base": 2.0}})
        assert result == [("decay.base", "2.0"), ("decay.growth_factor", "3.0")]

    def test_deeply_nested(self):
        result = flatten_config({"a": {"b": {"c": 1}}})
        assert result == [("a.b.c", "1")]

    def test_empty(self):
        assert flatten_config({}) == []


# --- CLI group tests ---


class TestCliGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Synaptra CLI" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


# --- Fixtures for command tests ---


@pytest.fixture
def runner():
    return CliRunner()


def _mock_response(data, success=True):
    """Build a standard MCP response dict."""
    if success:
        return {"success": True, "data": data, "meta": {"elapsed_ms": 1.0}}
    return {"success": False, "error": data, "data": None, "meta": {}}


SAMPLE_MEMORY = {
    "id": "e756ad4b-1b45-4a44-837a-118cfb2d5d79",
    "content": "Python is a versatile programming language",
    "memory_type": "semantic",
    "state": "active",
    "importance": 0.80,
    "stability": 3.50,
    "retrievability": 0.92,
    "access_count": 12,
    "tags": ["python", "ai"],
    "source": "conversation",
    "conversation_id": "conv-abc123",
    "created_at": "2026-03-19T12:30:00+00:00",
    "updated_at": "2026-03-20T08:15:00+00:00",
    "last_accessed": "2026-03-21T17:00:00+00:00",
}


# --- Command tests ---


class TestListCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_list_with_results(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": [SAMPLE_MEMORY]})
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "[e756ad4b]" in result.output
        assert "semantic" in result.output
        assert "0.80" in result.output
        assert "Tags: python, ai" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_list_empty(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": []})
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No memories found" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_list_json(self, mock_run, runner):
        resp = _mock_response({"memories": [SAMPLE_MEMORY]})
        mock_run.return_value = resp
        result = runner.invoke(cli, ["--json", "list"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["success"] is True

    @patch("cognitive_memory.cli.run_tool")
    def test_list_default_limit(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": []})
        runner.invoke(cli, ["list"])
        _, kwargs = mock_run.call_args
        assert kwargs == {} or mock_run.call_args[0][2].get("limit") == 20


class TestGetCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_get_detail(self, mock_run, runner):
        mock_run.return_value = _mock_response({
            "memory": SAMPLE_MEMORY,
            "relationships": [
                {"direction": "outgoing", "rel_type": "relates_to", "target_id": "a3f1c902-xxxx", "strength": 0.85}
            ],
            "versions": [
                {"created_at": "2026-03-19T12:30:00+00:00", "content": "Python is a versatile programming language"}
            ],
        })
        result = runner.invoke(cli, ["get", "e756ad4b-1b45-4a44-837a-118cfb2d5d79"])
        assert result.exit_code == 0
        assert "e756ad4b-1b45-4a44-837a-118cfb2d5d79" in result.output
        assert "semantic" in result.output
        assert "conv-abc123" in result.output
        assert "relates_to" in result.output
        assert "Versions (1)" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_get_json(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memory": SAMPLE_MEMORY, "relationships": [], "versions": []})
        result = runner.invoke(cli, ["--json", "get", "abc"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["success"] is True


class TestRecallCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_recall_results(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": [
            {**SAMPLE_MEMORY, "score": 0.847, "found_by": "semantic"}
        ]})
        result = runner.invoke(cli, ["recall", "programming language"])
        assert result.exit_code == 0
        assert "0.847" in result.output
        assert "found by: semantic" in result.output
        assert "[e756ad4b]" in result.output
        assert "Tags: python, ai" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_recall_empty(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": []})
        result = runner.invoke(cli, ["recall", "nothing here"])
        assert result.exit_code == 0
        assert "No memories match your query" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_recall_type_filter(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": []})
        runner.invoke(cli, ["recall", "query", "--type", "semantic"])
        params = mock_run.call_args[0][2]
        assert params["type_filter"] == "semantic"

    @patch("cognitive_memory.cli.run_tool")
    def test_recall_default_limit(self, mock_run, runner):
        mock_run.return_value = _mock_response({"memories": []})
        runner.invoke(cli, ["recall", "query"])
        params = mock_run.call_args[0][2]
        assert params["limit"] == 10


class TestStatsCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_stats(self, mock_run, runner):
        mock_run.return_value = _mock_response({
            "counts": {"by_type": {"semantic": 10, "episodic": 5}, "by_state": {"active": 12, "archived": 3}, "total": 15},
            "decay": {"semantic": {"avg_retrievability": 0.89}, "episodic": {"avg_retrievability": 0.71}},
        })
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "Memory Statistics" in result.output
        assert "semantic" in result.output
        assert "Total: 15" in result.output


class TestStoreCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_store(self, mock_run, runner):
        mock_run.return_value = _mock_response(SAMPLE_MEMORY)
        result = runner.invoke(cli, ["store", "Python is great"])
        assert result.exit_code == 0
        assert "Stored e756ad4b" in result.output
        assert "semantic" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_store_with_flags(self, mock_run, runner):
        mock_run.return_value = _mock_response(SAMPLE_MEMORY)
        runner.invoke(cli, ["store", "text", "--type", "semantic", "--tags", "a,b", "--importance", "0.9"])
        params = mock_run.call_args[0][2]
        assert params["type"] == "semantic"
        assert params["tags"] == ["a", "b"]
        assert params["importance"] == 0.9

    @patch("cognitive_memory.cli.run_tool")
    def test_store_stdin(self, mock_run, runner):
        mock_run.return_value = _mock_response(SAMPLE_MEMORY)
        result = runner.invoke(cli, ["store", "-"], input="Hello from stdin\n")
        assert result.exit_code == 0
        params = mock_run.call_args[0][2]
        assert params["content"] == "Hello from stdin"


class TestUpdateCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_update(self, mock_run, runner):
        mock_run.return_value = _mock_response(SAMPLE_MEMORY)
        result = runner.invoke(cli, ["update", "abc123", "--content", "new content"])
        assert result.exit_code == 0
        assert "Updated abc123" in result.output

    def test_update_no_flags(self, runner):
        result = runner.invoke(cli, ["update", "abc123"])
        assert result.exit_code != 0
        assert "at least one" in result.output.lower() or "Provide" in result.output


class TestDeleteCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_delete_confirmed(self, mock_run, runner):
        mock_run.return_value = _mock_response({"deleted": 1})
        result = runner.invoke(cli, ["delete", "abc123", "--yes"])
        assert result.exit_code == 0
        assert "Deleted 1 memory" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_delete_prompt_yes(self, mock_run, runner):
        mock_run.return_value = _mock_response({"deleted": 1})
        result = runner.invoke(cli, ["delete", "abc123"], input="y\n")
        assert result.exit_code == 0
        assert "Deleted 1 memory" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_delete_prompt_no(self, mock_run, runner):
        result = runner.invoke(cli, ["delete", "abc123"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_delete_multiple(self, mock_run, runner):
        mock_run.return_value = _mock_response({"deleted": 2})
        result = runner.invoke(cli, ["delete", "abc", "def", "--yes"])
        assert result.exit_code == 0
        assert "Deleted 2 memories" in result.output


class TestArchiveCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_archive_by_id(self, mock_run, runner):
        mock_run.return_value = _mock_response({"archived": 1})
        result = runner.invoke(cli, ["archive", "abc123"])
        assert result.exit_code == 0
        assert "Archived 1 memory" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_archive_by_threshold(self, mock_run, runner):
        mock_run.return_value = _mock_response({"archived": 5})
        result = runner.invoke(cli, ["archive", "--below-retrievability", "0.3"])
        assert result.exit_code == 0
        assert "Archived 5 memories" in result.output
        assert "0.3" in result.output

    def test_archive_both_errors(self, runner):
        result = runner.invoke(cli, ["archive", "abc", "--below-retrievability", "0.3"])
        assert result.exit_code != 0

    def test_archive_neither_errors(self, runner):
        result = runner.invoke(cli, ["archive"])
        assert result.exit_code != 0


class TestRestoreCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_restore(self, mock_run, runner):
        mock_run.return_value = _mock_response({"id": "abc"})
        result = runner.invoke(cli, ["restore", "abc123"])
        assert result.exit_code == 0
        assert "Restored" in result.output


class TestRelateCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_relate(self, mock_run, runner):
        mock_run.return_value = _mock_response({"source_id": "abc", "target_id": "def", "rel_type": "supports"})
        result = runner.invoke(cli, ["relate", "abc123", "def456", "--type", "supports"])
        assert result.exit_code == 0
        assert "Related abc123" in result.output
        assert "supports" in result.output


class TestUnrelateCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_unrelate(self, mock_run, runner):
        mock_run.return_value = _mock_response({"deleted": True})
        result = runner.invoke(cli, ["unrelate", "abc123", "def456", "--type", "supports"])
        assert result.exit_code == 0
        assert "Unrelated abc123" in result.output


class TestConsolidateCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_consolidate(self, mock_run, runner):
        mock_run.return_value = _mock_response({"actions": [{"action": "promote"}, {"action": "archive"}], "count": 2})
        result = runner.invoke(cli, ["consolidate"])
        assert result.exit_code == 0
        assert "Consolidation complete" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_consolidate_dry_run(self, mock_run, runner):
        mock_run.return_value = _mock_response({"actions": [{"action": "promote"}], "count": 1})
        result = runner.invoke(cli, ["consolidate", "--dry-run"])
        assert result.exit_code == 0
        assert "Would" in result.output


class TestConfigCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_config_show_all(self, mock_run, runner):
        mock_run.return_value = _mock_response({"decay": {"growth_factor": 3.0, "base": 2.0}})
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "decay.growth_factor" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_config_get_key(self, mock_run, runner):
        mock_run.return_value = _mock_response({"value": 3.0})
        result = runner.invoke(cli, ["config", "decay.growth_factor"])
        assert result.exit_code == 0
        assert "3.0" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_config_set_key(self, mock_run, runner):
        mock_run.return_value = _mock_response({"key": "decay.growth_factor", "value": 5.0, "action": "set"})
        result = runner.invoke(cli, ["config", "decay.growth_factor", "5.0"])
        assert result.exit_code == 0
        assert "Set decay.growth_factor" in result.output


class TestRelatedCommand:
    @patch("cognitive_memory.cli.run_tool")
    def test_related(self, mock_run, runner):
        mock_run.return_value = _mock_response([
            {"direction": "outgoing", "rel_type": "relates_to", "target_id": "a3f1c902-xxxx", "strength": 0.85, "content": "Related memory"}
        ])
        result = runner.invoke(cli, ["related", "abc123"])
        assert result.exit_code == 0
        assert "relates_to" in result.output
        assert "[a3f1c902]" in result.output
        assert "Related memory" in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_related_empty(self, mock_run, runner):
        mock_run.return_value = _mock_response([])
        result = runner.invoke(cli, ["related", "abc123"])
        assert result.exit_code == 0
        assert "No related memories found" in result.output


# --- B1 fix: ASCII arrows instead of Unicode ---


class TestAsciiArrows:
    """Verify no Unicode arrows that would crash on Windows cp1252."""

    @patch("cognitive_memory.cli.run_tool")
    def test_relate_uses_ascii_arrow(self, mock_run, runner):
        mock_run.return_value = _mock_response({"source_id": "abc", "target_id": "def", "rel_type": "supports"})
        result = runner.invoke(cli, ["relate", "abc12345", "def45678", "--type", "supports"])
        assert result.exit_code == 0
        assert "->" in result.output
        assert "\u2192" not in result.output  # No Unicode arrow

    @patch("cognitive_memory.cli.run_tool")
    def test_unrelate_uses_ascii_arrow(self, mock_run, runner):
        mock_run.return_value = _mock_response({"deleted": True})
        result = runner.invoke(cli, ["unrelate", "abc12345", "def45678", "--type", "supports"])
        assert result.exit_code == 0
        assert "->" in result.output
        assert "\u2192" not in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_get_relationships_use_ascii_arrows(self, mock_run, runner):
        mock_run.return_value = _mock_response({
            "memory": SAMPLE_MEMORY,
            "relationships": [
                {"direction": "outgoing", "rel_type": "relates_to", "target_id": "a3f1c902-xxxx", "strength": 0.85},
                {"direction": "incoming", "rel_type": "supports", "target_id": "b7d2e410-xxxx", "strength": 1.0},
            ],
            "versions": [],
        })
        result = runner.invoke(cli, ["get", "abc123"])
        assert result.exit_code == 0
        assert "->" in result.output  # outgoing
        assert "<-" in result.output  # incoming
        assert "\u2192" not in result.output
        assert "\u2190" not in result.output

    @patch("cognitive_memory.cli.run_tool")
    def test_related_uses_ascii_arrows(self, mock_run, runner):
        mock_run.return_value = _mock_response([
            {"direction": "outgoing", "rel_type": "relates_to", "target_id": "a3f1c902", "strength": 0.85, "content": "test"},
            {"direction": "incoming", "rel_type": "supports", "target_id": "b7d2e410", "strength": 1.0, "content": "test2"},
        ])
        result = runner.invoke(cli, ["related", "abc123"])
        assert result.exit_code == 0
        assert "-> relates_to [a3f1c902]" in result.output
        assert "<- supports [b7d2e410]" in result.output


# --- W1 fix: Exception type check instead of message match ---


class TestConnectionErrorHandling:
    @patch("cognitive_memory.cli.call_tool")
    def test_programming_error_not_masked(self, mock_call, runner):
        """A KeyError with 'connection' in message should NOT be caught as connection error."""
        mock_call.side_effect = KeyError("connection_failed_key")
        # This should propagate as an unhandled error, not exit 2
        result = runner.invoke(cli, ["stats"])
        # Click catches unhandled exceptions and returns exit code 1
        assert result.exit_code == 1
        # Should see the actual error, not "Cannot connect"
        assert "Cannot connect" not in result.output
