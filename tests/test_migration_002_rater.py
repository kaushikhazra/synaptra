"""Migration 002 must add rater / rated_at to a database that already ran 001.

The defect this guards against is a SILENT one, which is why every test here
first proves the columns are genuinely absent. A migration test that builds a
fresh schema and then checks the columns exist passes whether or not the
migration does anything at all.

Note the legacy database is built by running the REAL 001 unmodified. This repo
never edited 001, so 001 by itself is exactly the pre-rater schema — there is
nothing to strip. (Upstream has to ALTER TABLE ... DROP COLUMN to simulate it,
because upstream did edit 001; DROP COLUMN also needs SQLite 3.35+, which is
not guaranteed at this package's declared floor of Python 3.11.)
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "synaptra" / "migrations"


def _load(stem: str):
    path = MIGRATIONS_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(memory)")}


@pytest.fixture
def legacy_conn():
    """A database structurally identical to one that ran the original 001."""
    conn = sqlite3.connect(":memory:")
    _load("001_initial").migrate(conn)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    yield conn
    conn.close()


class TestLegacyDatabaseIsGenuinelyLegacy:
    """The guard that stops every test below from passing by testing nothing."""

    def test_rater_columns_absent_before_migrating(self, legacy_conn):
        cols = _columns(legacy_conn)
        assert "rater" not in cols
        assert "rated_at" not in cols

    def test_legacy_db_sits_at_user_version_1(self, legacy_conn):
        assert legacy_conn.execute("PRAGMA user_version").fetchone()[0] == 1


class TestMigrationAddsColumns:
    def test_both_columns_appear(self, legacy_conn):
        _load("002_rater").migrate(legacy_conn)
        cols = _columns(legacy_conn)
        assert "rater" in cols
        assert "rated_at" in cols

    def test_columns_are_text(self, legacy_conn):
        _load("002_rater").migrate(legacy_conn)
        types = {
            row[1]: row[2] for row in legacy_conn.execute("PRAGMA table_info(memory)")
        }
        assert types["rater"] == "TEXT"
        assert types["rated_at"] == "TEXT"

    def test_no_duplicate_columns_when_run_twice(self, legacy_conn):
        """ALTER TABLE ADD COLUMN has no IF NOT EXISTS; idempotence is explicit."""
        mod = _load("002_rater")
        mod.migrate(legacy_conn)
        mod.migrate(legacy_conn)
        cols = [row[1] for row in legacy_conn.execute("PRAGMA table_info(memory)")]
        assert cols.count("rater") == 1
        assert cols.count("rated_at") == 1

    def test_noop_on_missing_memory_table(self):
        """001 owns creating the table. 002 must not raise on a bare database."""
        conn = sqlite3.connect(":memory:")
        try:
            _load("002_rater").migrate(conn)  # must not raise
            assert _columns(conn) == set()
        finally:
            conn.close()

    def test_migration_never_calls_commit(self, legacy_conn):
        """The runner owns the transaction.

        If 002 committed, it would defeat the runner's rollback-on-error: a
        later failure could no longer undo this migration's DDL. Asserted by
        recording calls rather than by inspecting connection state, because
        sqlite3's legacy isolation does not open a transaction for DDL at all,
        so in_transaction would read False either way and prove nothing.
        """

        class CommitRecorder:
            def __init__(self, conn):
                self._conn = conn
                self.commits = 0

            def execute(self, *a, **kw):
                return self._conn.execute(*a, **kw)

            def commit(self):
                self.commits += 1
                return self._conn.commit()

        rec = CommitRecorder(legacy_conn)
        _load("002_rater").migrate(rec)
        assert rec.commits == 0
        assert "rater" in _columns(legacy_conn), "guard: the migration did run"


class TestExistingRowsReadBackNull:
    """No backfill: NULL means 'written before rater tracking'."""

    def _insert_legacy_row(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO memory (id, content, memory_type, state, importance, "
            "stability, retrievability, access_count, created_at, updated_at, "
            "last_accessed) VALUES "
            "('old-1', 'written before rater tracking', 'semantic', 'active', "
            "0.5, 14.0, 0.9, 0, '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()

    def test_preexisting_row_has_null_rater(self, legacy_conn):
        self._insert_legacy_row(legacy_conn)
        _load("002_rater").migrate(legacy_conn)
        row = legacy_conn.execute(
            "SELECT rater, rated_at FROM memory WHERE id = 'old-1'"
        ).fetchone()
        assert row == (None, None)

    def test_new_write_round_trips(self, legacy_conn):
        self._insert_legacy_row(legacy_conn)
        _load("002_rater").migrate(legacy_conn)
        legacy_conn.execute(
            "INSERT INTO memory (id, content, memory_type, state, importance, "
            "stability, retrievability, access_count, created_at, updated_at, "
            "last_accessed, rater, rated_at) VALUES "
            "('new-1', 'rated', 'semantic', 'active', 0.7, 14.0, 0.9, 0, "
            "'2026-09-12T00:00:00+00:00', '2026-09-12T00:00:00+00:00', "
            "'2026-09-12T00:00:00+00:00', 'claude-opus-5', "
            "'2026-09-12T00:00:00+00:00')"
        )
        row = legacy_conn.execute(
            "SELECT rater, rated_at FROM memory WHERE id = 'new-1'"
        ).fetchone()
        assert row == ("claude-opus-5", "2026-09-12T00:00:00+00:00")
        # and the old row is still NULL alongside it
        assert legacy_conn.execute(
            "SELECT rater FROM memory WHERE id = 'old-1'"
        ).fetchone() == (None,)


class TestRunnerActuallyFires002:
    """002 can be correct and still never run if the runner disagrees.

    The runner globs [0-9]*.py, takes int(stem.split('_')[0]) and gates on
    PRAGMA user_version — so a database at version 1 must reach version 2 and
    gain the columns without 001 being re-run.
    """

    def test_storage_upgrades_a_legacy_database_on_open(self, tmp_path):
        from synaptra.storage import Storage

        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        _load("001_initial").migrate(conn)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        assert "rater" not in _columns(conn)
        conn.close()

        Storage(str(db))  # opening runs the migration runner

        conn = sqlite3.connect(db)
        try:
            assert "rater" in _columns(conn)
            assert "rated_at" in _columns(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] >= 2
        finally:
            conn.close()

    def test_fresh_database_also_reaches_version_2(self, tmp_path):
        from synaptra.storage import Storage

        db = tmp_path / "fresh.db"
        Storage(str(db))

        conn = sqlite3.connect(db)
        try:
            assert "rater" in _columns(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] >= 2
        finally:
            conn.close()
