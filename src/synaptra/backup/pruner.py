"""CM Backup Retention Pruner.

Implements the 7-daily + 4-weekly + 6-monthly retention policy.
Also prunes data.pre-rollback-<ts> directories older than 7 days.

Callable standalone: prune_backups(backups_dir, policy) -> list[Path]
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_POLICY = {
    "daily": 7,
    "weekly": 4,
    "monthly": 6,
    "rollback_days": 7,
}

DEFAULT_BACKUPS_ROOT = Path.home() / ".synaptra" / "backups"


def _parse_manifest_date(backup_dir: Path) -> datetime | None:
    """Parse created_at from manifest.json. Returns None if not parseable."""
    manifest = backup_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        ts_str = data.get("created_at", "")
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def prune_backups(
    backups_dir: Path | None = None,
    policy: dict[str, int] | None = None,
) -> list[Path]:
    """Apply retention policy to the backups directory.

    Keeps:
    - Last `daily` backups by date.
    - Last `weekly` ISO-week backups beyond the daily window.
    - Last `monthly` calendar-month backups beyond the weekly window.

    Also prunes `data.pre-rollback-<ts>` directories older than `rollback_days`
    from ~/.synaptra/.

    Returns:
        List of deleted backup Paths.
    """
    if backups_dir is None:
        backups_dir = DEFAULT_BACKUPS_ROOT

    pol = {**DEFAULT_POLICY, **(policy or {})}

    if not backups_dir.exists():
        return []

    # Collect all complete backups (have manifest.json)
    backups: list[tuple[datetime, Path]] = []
    for d in backups_dir.iterdir():
        if not d.is_dir():
            continue
        dt = _parse_manifest_date(d)
        if dt is None:
            continue  # incomplete/orphaned — exporter already cleans up orphans
        backups.append((dt, d))

    if not backups:
        return []

    # Sort newest first
    backups.sort(key=lambda x: x[0], reverse=True)

    keep: set[Path] = set()

    # Daily tier: most recent `daily` backups
    for dt, path in backups[: pol["daily"]]:
        keep.add(path)

    daily_cutoff_dt = backups[pol["daily"] - 1][0] if len(backups) >= pol["daily"] else None

    # Weekly tier: one per ISO week, beyond the daily window
    seen_weeks: dict[tuple[int, int], Path] = {}
    for dt, path in backups:
        if daily_cutoff_dt and dt >= daily_cutoff_dt:
            continue  # still in daily window
        iso = dt.isocalendar()
        week_key = (iso.year, iso.week)
        if week_key not in seen_weeks:
            seen_weeks[week_key] = path

    weekly_candidates = sorted(seen_weeks.items(), key=lambda x: x[0], reverse=True)
    for _, path in weekly_candidates[: pol["weekly"]]:
        keep.add(path)

    # Monthly tier: one per calendar month, beyond the weekly window
    weekly_kept = {p for _, p in weekly_candidates[: pol["weekly"]]}
    seen_months: dict[tuple[int, int], Path] = {}
    for dt, path in backups:
        if path in keep:
            continue
        month_key = (dt.year, dt.month)
        if month_key not in seen_months:
            seen_months[month_key] = path

    monthly_candidates = sorted(seen_months.items(), key=lambda x: x[0], reverse=True)
    for _, path in monthly_candidates[: pol["monthly"]]:
        keep.add(path)

    # Delete everything not in keep
    deleted: list[Path] = []
    for _, path in backups:
        if path not in keep:
            try:
                shutil.rmtree(path)
                logger.info("Pruned backup: %s", path)
                deleted.append(path)
            except Exception as e:
                logger.warning("Could not prune %s: %s", path, e)

    # Prune data.pre-rollback-<ts> directories older than rollback_days
    cm_home = Path.home() / ".synaptra"
    if cm_home.exists():
        cutoff = datetime.now(timezone.utc) - timedelta(days=pol["rollback_days"])
        for d in cm_home.iterdir():
            if d.is_dir() and d.name.startswith("data.pre-rollback-"):
                try:
                    mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)
                    if mtime < cutoff:
                        shutil.rmtree(d)
                        logger.info("Pruned rollback snapshot: %s", d)
                        deleted.append(d)
                except Exception as e:
                    logger.warning("Could not prune rollback snapshot %s: %s", d, e)

    return deleted
