-- Cognitive Memory SurrealDB Schema
-- Schema definition for Cognitive Memory SurrealDB.
-- Applied once via scripts/provision_cm_schema.py (fresh instances).
-- Embedded SurrealStorage._ensure_schema() also uses this file for mem:// test DBs.
-- NEVER use OVERWRITE in this file. All statements use IF NOT EXISTS.

-- Memory records
DEFINE TABLE IF NOT EXISTS memory SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS content         ON memory TYPE string;
DEFINE FIELD IF NOT EXISTS memory_type     ON memory TYPE string ASSERT $value IN ['working', 'episodic', 'semantic', 'procedural', 'identity', 'person'];
DEFINE FIELD IF NOT EXISTS state           ON memory TYPE string DEFAULT 'active' ASSERT $value IN ['active', 'archived'];
DEFINE FIELD IF NOT EXISTS importance      ON memory TYPE float ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD IF NOT EXISTS stability       ON memory TYPE float;
DEFINE FIELD IF NOT EXISTS retrievability  ON memory TYPE float;
DEFINE FIELD IF NOT EXISTS access_count    ON memory TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS created_at      ON memory TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at      ON memory TYPE datetime;
DEFINE FIELD IF NOT EXISTS last_accessed   ON memory TYPE datetime;
DEFINE FIELD IF NOT EXISTS source          ON memory TYPE option<string>;
DEFINE FIELD IF NOT EXISTS conversation_id ON memory TYPE option<string>;
-- Who set `importance`, and when.  Importance is rater-relative: different
-- models score the same memory differently, so a score is only comparable to
-- another from the same rater.  NULL means the row predates rater tracking.
DEFINE FIELD IF NOT EXISTS rater           ON memory TYPE option<string>;
DEFINE FIELD IF NOT EXISTS rated_at        ON memory TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS tags            ON memory TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS embedding       ON memory TYPE option<array<float>>;

-- Vector index (HNSW, cosine distance, 384 dimensions for all-MiniLM-L6-v2)
DEFINE INDEX IF NOT EXISTS idx_memory_embedding ON memory FIELDS embedding HNSW DIMENSION 384 DIST COSINE;

-- Full-text search index
DEFINE ANALYZER IF NOT EXISTS memory_analyzer TOKENIZERS blank, class FILTERS lowercase, snowball(english);
DEFINE INDEX IF NOT EXISTS idx_memory_fts ON memory FIELDS content SEARCH ANALYZER memory_analyzer BM25;

-- Standard indexes
DEFINE INDEX IF NOT EXISTS idx_memory_state ON memory FIELDS state;
DEFINE INDEX IF NOT EXISTS idx_memory_type ON memory FIELDS memory_type;
DEFINE INDEX IF NOT EXISTS idx_memory_state_type ON memory FIELDS state, memory_type;

-- Graph edge tables (one per relationship type)
DEFINE TABLE IF NOT EXISTS causes SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON causes TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON causes TYPE datetime;

DEFINE TABLE IF NOT EXISTS follows SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON follows TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON follows TYPE datetime;

DEFINE TABLE IF NOT EXISTS contradicts SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON contradicts TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON contradicts TYPE datetime;

DEFINE TABLE IF NOT EXISTS supports SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON supports TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON supports TYPE datetime;

DEFINE TABLE IF NOT EXISTS relates_to SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON relates_to TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON relates_to TYPE datetime;

DEFINE TABLE IF NOT EXISTS supersedes SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON supersedes TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON supersedes TYPE datetime;

DEFINE TABLE IF NOT EXISTS part_of SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON part_of TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON part_of TYPE datetime;

DEFINE TABLE IF NOT EXISTS describes SCHEMAFULL TYPE RELATION FROM memory TO memory;
DEFINE FIELD IF NOT EXISTS strength   ON describes TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS created_at ON describes TYPE datetime;

-- Memory versions
DEFINE TABLE IF NOT EXISTS memory_version SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS memory_id  ON memory_version TYPE record<memory>;
DEFINE FIELD IF NOT EXISTS content     ON memory_version TYPE string;
DEFINE FIELD IF NOT EXISTS metadata    ON memory_version TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at  ON memory_version TYPE datetime;
DEFINE INDEX IF NOT EXISTS idx_version_memory ON memory_version FIELDS memory_id;

-- Consolidation log
DEFINE TABLE IF NOT EXISTS consolidation_log SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS action     ON consolidation_log TYPE string ASSERT $value IN ['promote', 'merge', 'archive', 'flag_contradiction'];
DEFINE FIELD IF NOT EXISTS source_ids ON consolidation_log TYPE array<string>;
DEFINE FIELD IF NOT EXISTS target_id  ON consolidation_log TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reason     ON consolidation_log TYPE string;
DEFINE FIELD IF NOT EXISTS created_at ON consolidation_log TYPE datetime;

-- Config (keyed by record ID, e.g., preference:⟨decay.growth_factor⟩)
DEFINE TABLE IF NOT EXISTS preference SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS val        ON preference TYPE any;
DEFINE FIELD IF NOT EXISTS updated_at ON preference TYPE datetime;

-- Migration tracking (operator-only, not application schema)
DEFINE TABLE IF NOT EXISTS schema_migration SCHEMALESS;
