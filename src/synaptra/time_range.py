"""Reading the ``time_range`` argument of ``memory_recall`` and ``memory_list``.

One parser for both tools, so they cannot disagree about what a range means.

The range is an object with ``start`` and/or ``end``, each an ISO 8601 date or
datetime string:

* a missing end is open: no memory is older than 1970 or newer than year 9999;
* a date (``"2026-10-19"``) means the whole day, so as an ``end`` it includes
  everything created on that day;
* an offset (``+05:30``, ``Z``) is honoured, and the result is always UTC;
* a datetime with no offset is read as UTC.

The result is a ``(start, end)`` pair of timezone-aware datetimes, the shape the
engine and the storage backends take. Stored ``created_at`` values are timezone-aware, so
a naive datetime here would fail the comparison with ``can't compare
offset-naive and offset-aware datetimes``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc

# No memory is older than 1970 or newer than year 9999: these stand in for an open end.
OPEN_START = datetime(1970, 1, 1, tzinfo=UTC)
OPEN_END = datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)

_KEYS = ("start", "end")
_HAS_TIME = re.compile(r"[T ]", re.IGNORECASE)


def _bound(field: str, value: Any, end: bool) -> datetime:
    """One end of the range as an aware UTC datetime; ValueError names the field and the value."""
    if not isinstance(value, str):
        raise ValueError(
            f"time_range.{field}: {value!r} is not an ISO 8601 date or datetime string"
        )
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"time_range.{field}: {value!r} is not an ISO 8601 date or datetime"
        ) from None
    if not _HAS_TIME.search(text):  # a date: the whole day
        if end:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
        return parsed.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_time_range(time_range: Any) -> tuple[datetime, datetime] | None:
    """``{"start": ..., "end": ...}`` -> ``(start, end)`` aware UTC, or None for no filter."""
    if not time_range:
        return None
    if not isinstance(time_range, dict):
        raise ValueError(
            f"time_range must be an object with 'start' and/or 'end', got {time_range!r}"
        )
    unknown = sorted(str(k) for k in time_range if k not in _KEYS)
    if unknown:
        raise ValueError(
            f"time_range has unknown key(s) {unknown}; expected 'start' and/or 'end'"
        )
    raw_start, raw_end = time_range.get("start"), time_range.get("end")
    if raw_start is None and raw_end is None:
        return None
    start = OPEN_START if raw_start is None else _bound("start", raw_start, False)
    end = OPEN_END if raw_end is None else _bound("end", raw_end, True)
    if start > end:
        raise ValueError(
            f"time_range.start ({raw_start}) is after time_range.end ({raw_end})"
        )
    return start, end
