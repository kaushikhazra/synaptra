"""Add rater / rated_at to memory.

Importance is rater-relative: different models score the same content
differently, so a score is only comparable to another score from the same
rater.  `rater` records who set it and `rated_at` when.  See issue #8.

The private upstream originally added these two columns by EDITING
001_initial.py.  That is silent for any database that had already run the
original 001: the runner skips 001 because PRAGMA user_version is already 1,
so the columns never appear and nothing reports it.  Every test there built a
fresh schema, so nothing saw the gap either.  This repo does not edit 001 —
the columns arrive here, which is the only way an existing SQLite database
ever gets them.

NO BACKFILL.  An existing row reads back rater = NULL, which means exactly
"written before rater tracking" and is the correct value for it.

Idempotent by inspection of PRAGMA table_info rather than by assumption:
ALTER TABLE ... ADD COLUMN has no IF NOT EXISTS in SQLite, and this must be a
no-op on a table that already carries the columns.

The runner owns the transaction: it commits and bumps user_version, so there
is deliberately no conn.commit() here.
"""

import sqlite3

# column name -> SQLite declared type
_COLUMNS = {
    "rater": "TEXT",
    "rated_at": "TEXT",
}


def migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memory)")}

    # An empty table_info means there is no memory table at all.  001 owns
    # creating it; adding columns to a table that does not exist is not this
    # migration's job, and raising here would break a fresh database.
    if not existing:
        return

    for column, decl in _COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE memory ADD COLUMN {column} {decl}")
