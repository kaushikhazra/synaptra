"""Issue #13: the time_range of memory_recall / memory_list.

A start-only range failed with the bare error 'end'; a date-only range reached the engine as naive
datetimes and then failed against the stored, timezone-aware created_at. One shared parser now reads
the range for both tools. Offline: a stub engine stands in for the store, the live service is never touched.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

UTC = timezone.utc


def parse(time_range):
    from synaptra.time_range import parse_time_range

    return parse_time_range(time_range)


# ---------------------------------------------------------------- forms of the range (criteria 1-4)
class TestForms:
    def test_start_only_runs_to_the_far_future(self):
        start, end = parse({"start": "2026-10-01T00:00:00Z"})
        assert start == datetime(2026, 10, 1, tzinfo=UTC)
        assert end > datetime(2100, 1, 1, tzinfo=UTC)

    def test_end_only_starts_before_any_memory(self):
        start, end = parse({"end": "2026-10-19T00:00:00Z"})
        assert end == datetime(2026, 10, 19, tzinfo=UTC)
        assert start < datetime(1971, 1, 1, tzinfo=UTC)

    def test_dates_cover_whole_days_the_end_date_included(self):
        start, end = parse({"start": "2026-10-01", "end": "2026-10-19"})
        assert start == datetime(2026, 10, 1, tzinfo=UTC)
        assert end == datetime(2026, 10, 19, 23, 59, 59, 999999, tzinfo=UTC)

    def test_offsets_select_the_same_instants_whatever_they_are_written_in(self):
        ist = parse(
            {"start": "2026-10-01T10:00:00+05:30", "end": "2026-10-02T10:00:00+05:30"}
        )
        zulu = parse({"start": "2026-10-01T04:30:00Z", "end": "2026-10-02T04:30:00Z"})
        assert ist == zulu

    def test_a_datetime_without_an_offset_is_read_as_utc(self):
        start, _ = parse({"start": "2026-10-01T12:00:00"})
        assert start == datetime(2026, 10, 1, 12, tzinfo=UTC)

    def test_the_tool_descriptions_say_how_the_range_is_read(self):
        from synaptra import server as srv

        for tool in (srv.memory_recall, srv.memory_list):
            assert "time_range" in tool.__doc__ and "UTC" in tool.__doc__

    @pytest.mark.parametrize("empty", [None, {}, {"start": None, "end": None}])
    def test_no_range_is_no_filter(self, empty):
        assert parse(empty) is None


# ---------------------------------------------------------------- errors (criterion 5)
class TestErrors:
    @pytest.mark.parametrize(
        "bad, field",
        [
            ({"start": "yesterday"}, "time_range.start"),
            ({"end": "2026-13-45"}, "time_range.end"),
            ({"start": 5}, "time_range.start"),
            ({"since": "2026-10-01"}, "since"),
            ("2026-10-01", "time_range"),
        ],
    )
    def test_the_message_names_the_field(self, bad, field):
        with pytest.raises(ValueError) as err:
            parse(bad)
        assert field in str(err.value)

    def test_the_message_quotes_the_offending_value(self):
        with pytest.raises(ValueError, match="yesterday"):
            parse({"start": "yesterday"})

    def test_start_after_end_is_refused(self):
        with pytest.raises(ValueError, match="after"):
            parse({"start": "2026-10-19", "end": "2026-10-01"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["memory_recall", "memory_list"])
    async def test_a_tool_returns_that_message_not_a_bare_key(self, tool):
        from synaptra import server as srv

        engine = AsyncMock()
        kwargs = {"query": "x"} if tool == "memory_recall" else {}
        with patch.object(srv, "_get_engine", return_value=engine):
            out = json.loads(
                await getattr(srv, tool)(time_range={"start": "yesterday"}, **kwargs)
            )
        assert out["success"] is False
        assert "time_range.start" in out["error"] and out["error"] != "'end'"


# ---------------------------------------------------------------- one parser for both tools (criterion 6)
class TestBothTools:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "time_range",
        [
            {"start": "2026-10-01"},
            {"end": "2026-10-19"},
            {"start": "2026-10-01", "end": "2026-10-19"},
            {"start": "2026-10-01T10:00:00+05:30", "end": "2026-10-19T00:00:00Z"},
        ],
    )
    async def test_recall_and_list_hand_the_engine_the_same_aware_range(
        self, time_range
    ):
        from synaptra import server as srv

        engine = AsyncMock()
        engine.recall.return_value = []
        engine.list_memories.return_value = []
        with patch.object(srv, "_get_engine", return_value=engine):
            recalled = json.loads(
                await srv.memory_recall(query="x", time_range=time_range)
            )
            listed = json.loads(await srv.memory_list(time_range=time_range))
        assert recalled["success"] and listed["success"]
        via_recall = engine.recall.await_args.kwargs["time_range"]
        via_list = engine.list_memories.await_args.kwargs["time_range"]
        assert via_recall == via_list
        assert all(t.tzinfo is not None for t in via_recall)


def memory(created):
    m = MagicMock()
    m.state.value = "active"
    m.memory_type.value = "episodic"
    m.tags = []
    m.created_at = created
    return m


class TestAgainstTheRealFilter:
    """The filter that raised the naive/aware TypeError, fed what the parser now produces."""

    def matches(self, time_range, created):
        from synaptra.retrieval import _memory_matches_filters

        return _memory_matches_filters(memory(created), None, None, parse(time_range))

    def test_a_memory_on_the_end_date_is_inside_a_date_range(self):
        window = {"start": "2026-10-19", "end": "2026-10-19"}
        assert self.matches(window, datetime(2026, 10, 19, 12, tzinfo=UTC))
        assert not self.matches(window, datetime(2026, 10, 20, 0, 0, 1, tzinfo=UTC))
        assert not self.matches(window, datetime(2026, 10, 18, 23, 59, tzinfo=UTC))

    def test_start_only_keeps_everything_after_it(self):
        assert self.matches({"start": "2026-10-01"}, datetime(2026, 10, 10, tzinfo=UTC))
        assert not self.matches(
            {"start": "2026-10-01"}, datetime(2026, 9, 30, tzinfo=UTC)
        )

    def test_end_only_keeps_everything_before_it(self):
        assert self.matches({"end": "2026-10-19"}, datetime(2026, 1, 1, tzinfo=UTC))
        assert not self.matches(
            {"end": "2026-10-19"}, datetime(2026, 10, 21, tzinfo=UTC)
        )


# ---------------------------------------------------------------- the CLI is not touched (criterion 7)
class TestCliUnchanged:
    def test_the_cli_still_wants_both_ends(self):
        from synaptra.cli import build_time_range

        assert build_time_range(None, None) is None
        with pytest.raises(SystemExit):
            build_time_range("2026-01-01", None)
