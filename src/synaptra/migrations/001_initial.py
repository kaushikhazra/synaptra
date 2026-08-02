"""Initial schema: tables, FTS5, indexes."""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Create all tables for synaptra system."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory (
            id              TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            memory_type     TEXT NOT NULL CHECK(memory_type IN ('working', 'episodic', 'semantic', 'procedural')),
            state           TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('active', 'archived')),
            importance      REAL NOT NULL CHECK(importance >= 0.0 AND importance <= 1.0),
            stability       REAL NOT NULL,
            retrievability  REAL NOT NULL,
            access_count    INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            last_accessed   TEXT NOT NULL,
            source          TEXT,
            conversation_id TEXT,
            tags            TEXT DEFAULT '[]',
            embedding       BLOB
        );

        CREATE TABLE IF NOT EXISTS memory_version (
            id          TEXT PRIMARY KEY,
            memory_id   TEXT NOT NULL REFERENCES memory(id),
            content     TEXT NOT NULL,
            metadata    TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationship (
            id          TEXT PRIMARY KEY,
            source_id   TEXT NOT NULL REFERENCES memory(id),
            target_id   TEXT NOT NULL REFERENCES memory(id),
            rel_type    TEXT NOT NULL CHECK(rel_type IN ('causes', 'follows', 'contradicts', 'supports', 'relates_to', 'supersedes', 'part_of')),
            strength    REAL DEFAULT 1.0,
            created_at  TEXT NOT NULL,
            UNIQUE(source_id, target_id, rel_type)
        );

        CREATE TABLE IF NOT EXISTS consolidation_log (
            id          TEXT PRIMARY KEY,
            action      TEXT NOT NULL CHECK(action IN ('promote', 'merge', 'archive', 'flag_contradiction')),
            source_ids  TEXT NOT NULL,
            target_id   TEXT,
            reason      TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS config (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        -- FTS5 for full-text search
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            content,
            content_rowid='rowid',
            tokenize='unicode61'
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_memory_state ON memory(state);
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memory_state_type ON memory(state, memory_type);
        CREATE INDEX IF NOT EXISTS idx_memory_version_memory_id ON memory_version(memory_id);
        CREATE INDEX IF NOT EXISTS idx_relationship_source ON relationship(source_id);
        CREATE INDEX IF NOT EXISTS idx_relationship_target ON relationship(target_id);
        CREATE INDEX IF NOT EXISTS idx_consolidation_log_created ON consolidation_log(created_at);
    """)
