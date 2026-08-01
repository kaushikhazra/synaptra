# Synaptra

A biologically-inspired synaptra memory system for AI agents, exposed as an [MCP](https://modelcontextprotocol.io/) server. Gives agents persistent memory with human-like properties: memories decay over time, strengthen with use, form relationships, and consolidate automatically.

## Features

- **Four memory types** with different decay rates — working (hours), episodic (days), semantic (weeks), procedural (months)
- **FSRS-inspired decay** — retrievability computed on-the-fly: `R(t) = e^(-t / 9S)`
- **Multi-strategy retrieval** — semantic search (HNSW cosine), BM25 keyword search, temporal recency, and graph traversal fused with Reciprocal Rank Fusion (RRF)
- **Spreading activation** — retrieving a memory strengthens its neighbors in the relationship graph
- **Automatic linking** — new memories are linked to similar existing ones via cosine similarity
- **Contradiction detection** — flags semantically similar memories with negation signals
- **Consolidation pipeline** — promotes working->episodic->semantic/procedural, archives forgotten memories, merges near-duplicates
- **Version history** — every update creates a snapshot for full audit trail
- **CLI tool** — browse, search, and manage memories from the terminal
- **Windows service** — runs as a background service via Task Scheduler (no admin required)

## Installation

Requires Python 3.11+.

```bash
pip install synaptra
```

This installs the MCP server, CLI tool, and all dependencies including `sentence-transformers` (all-MiniLM-L6-v2, 384d) and `SurrealDB` (embedded).

## Quick Start

### 1. Start the server

```bash
synaptra
```

This starts the Streamable HTTP MCP server on `http://127.0.0.1:8050/mcp`.

### 2. Connect from Claude Code

Add to your Claude Code MCP config (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "synaptra": {
      "command": "npx",
      "args": ["mcp-remote", "http://127.0.0.1:8050/mcp"]
    }
  }
}
```

### 3. Use the CLI

```bash
# Search memories
synaptra-cli recall "python programming"

# Browse
synaptra-cli list
synaptra-cli list --type semantic --tags "project,design"

# Get full details
synaptra-cli get <memory-id>

# Store a memory
synaptra-cli store "Python's GIL was removed in 3.13" --type semantic --tags "python,news"

# Pipe from stdin
echo "meeting notes here" | synaptra-cli store -

# System health
synaptra-cli stats
synaptra-cli consolidate --dry-run

# JSON output for scripting
synaptra-cli --json list | jq '.data.memories[].content'
```

Run `synaptra-cli --help` for all commands and flags.

## Windows Service

Run the server as a background service that auto-starts at logon:

```bash
synaptra-service install    # Register with Task Scheduler
synaptra-service start      # Start now
synaptra-service status     # Check health
synaptra-service stop       # Stop
synaptra-service remove     # Uninstall
synaptra-service debug      # Run in foreground (development)
```

No admin elevation or pywin32 required. Uses Task Scheduler with auto-restart on failure (3 attempts, 1 minute apart).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPTRA_DB` | `~/.synaptra/data` | SurrealDB data directory |
| `SYNAPTRA_PORT` | `8050` | HTTP server port |
| `SYNAPTRA_HOST` | `127.0.0.1` | HTTP server bind address |
| `SYNAPTRA_CONFIG` | bundled `config.default.yaml` | Config YAML override path |
| `SYNAPTRA_URL` | `http://127.0.0.1:8050/mcp` | CLI: server URL (overrides `--url`) |

## MCP Tools (14)

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory with auto-classification and importance scoring |
| `memory_recall` | Multi-strategy retrieval with RRF fusion and decay reranking |
| `memory_get` | Get a specific memory with relationships and version history |
| `memory_update` | Update content/metadata with versioning and re-embedding |
| `memory_relate` | Create typed relationships between memories |
| `memory_related` | Graph traversal to find connected memories |
| `memory_unrelate` | Remove a relationship |
| `memory_list` | Browse/filter memories with full-text search |
| `memory_archive` | Archive by ID, bulk IDs, or retrievability threshold |
| `memory_restore` | Restore archived memories with decay reset |
| `memory_delete` | Permanent deletion with cascade (requires `confirm: true`) |
| `memory_stats` | System statistics: counts, decay health, storage usage |
| `memory_consolidate` | Run consolidation pipeline (supports `dry_run`) |
| `memory_config` | View or update configuration |

## Architecture

```
cognitive_memory/
  server.py          Streamable HTTP MCP server (FastMCP + uvicorn)
  cli.py             CLI tool (click, connects via MCP client)
  service.py         Windows Task Scheduler service management
  engine.py          Central orchestrator
  surreal_storage.py SurrealDB embedded storage (HNSW vectors, BM25 FTS, graph edges)
  embeddings.py      Sentence-transformers embedding service
  retrieval.py       Two-phase RRF pipeline with spreading activation
  decay.py           FSRS-inspired decay engine (pure functions)
  consolidation.py   Promotion, archival, clustering, merging
  classification.py  Heuristic type classification + importance scoring
  config.py          YAML defaults + DB overrides
  models.py          Pydantic domain models
  protocols.py       Storage protocol (typing.Protocol)
  schema.surql       SurrealDB schema definition
```

## Configuration

All config uses dot-notation keys. View/set at runtime via `memory_config` tool or `synaptra-cli config`.

Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `decay.initial_stability.working` | 0.04 | Working memory S0 (~1 hour) |
| `decay.initial_stability.episodic` | 2.0 | Episodic memory S0 (~2 days) |
| `decay.initial_stability.semantic` | 14.0 | Semantic memory S0 (~2 weeks) |
| `decay.initial_stability.procedural` | 60.0 | Procedural memory S0 (~2 months) |
| `decay.growth_factor` | 2.0 | Reinforcement strength on access |
| `retrieval.weights.semantic` | 1.0 | Semantic search weight in RRF |
| `retrieval.weights.keyword` | 0.7 | BM25 keyword search weight |
| `retrieval.weights.graph` | 0.5 | Graph traversal weight |
| `auto_linking.similarity_threshold` | 0.75 | Min cosine similarity for auto-links |
| `consolidation.merge_threshold` | 0.90 | Min similarity to merge memories |

## Backup & Restore

CM provides a full backup/restore system via the `cm backup` subgroup. Backups are
logical NDJSON exports — backend-agnostic and inspectable without unpacking.

### Quick reference

```bash
# Create a backup (stops CM, exports, restarts CM)
cm backup create

# Verify a backup artifact (light check)
cm backup verify ~/.synaptra/backups/cm-20260515T040000Z

# Deep verify (loads into temp DB, runs HNSW query, ~30 s)
cm backup verify --deep ~/.synaptra/backups/cm-20260515T040000Z

# Restore into a fresh directory
cm backup restore ~/.synaptra/backups/cm-20260515T040000Z

# Restore into a specific target
cm backup restore ~/.synaptra/backups/cm-20260515T040000Z --target ~/myrestore
```

### Stop-CM ritual

Backups require exclusive access to the SurrealKV data directory. `cm backup create`
automatically:

1. Stops the `CognitiveMemory` Windows scheduled task.
2. Waits for the SurrealKV file lock to release (~5 s).
3. Opens SurrealKV directly and streams all data to NDJSON.
4. Restarts the CM service.
5. CM cold-start (SurrealKV clog replay) takes **~2 minutes** — expected behavior.

The ~2 min downtime is accepted. Backups run during `/dream` (a maintenance window)
or on explicit operator demand.

### Backup artifact layout

```
~/.synaptra/backups/cm-<timestamp>Z/
  manifest.json          # Metadata: counts, schema hash, version, timing
  schema.surql           # Snapshot of CM schema at backup time
  memory.ndjson          # All memory records (15 fields each, incl. embedding)
  memory_version.ndjson  # Edit history
  consolidation_log.ndjson
  preference.ndjson
  edges/
    causes.ndjson        # One file per relationship type
    follows.ndjson
    contradicts.ndjson
    supports.ndjson
    relates_to.ndjson
    supersedes.ndjson
    part_of.ndjson
    describes.ndjson
```

Each file is line-delimited JSON — `head memory.ndjson | python -m json.tool` works
without unpacking anything.

### Rollback procedure

Use `scripts/cm-rollback.ps1` for a full rollback to a previous backup:

```powershell
# Usage: cm-rollback.ps1 <backup_dir>
.\scripts\cm-rollback.ps1 "$env:USERPROFILE\.synaptra\backups\cm-20260515T040000Z"
```

The script uses atomic rename — live data is never directly overwritten. If restore
fails mid-way, CM restarts against the untouched live data. The old live data is
moved to `data.pre-rollback-<ts>` as a safety net (pruned after 7 days).

### Retention policy

The retention pruner runs automatically after `cm backup create`. Policy:

| Tier    | Keep | Selection                                          |
|---------|------|----------------------------------------------------|
| Daily   | 7    | Most recent 7 backups by timestamp                 |
| Weekly  | 4    | One per ISO week, most recent, beyond daily window |
| Monthly | 6    | One per calendar month, most recent, beyond weekly |

Pre-rollback safety copies (`data.pre-rollback-<ts>`) are pruned after 7 days.

Worst-case storage: 17 backups × ~10 MB ≈ 170 MB.

### Pre-dream integration

The `/dream` skill runs `cm backup create` + `cm backup verify --deep` as its
first step before any memory reshaping. If either fails, dream aborts. This
ensures every consolidation pass has a verified rollback point.

### Stale backup warning

The `memory_health` MCP tool exposes:
- `most_recent_backup_age_days`: days since the most recent backup (None if none exist).
- `backup_is_stale`: true if age > 7 days or no backups exist.

The session-start skill surfaces `backup_is_stale` as a visible warning.

## Development

```bash
pip install synaptra[dev]
pytest
```

## License

MIT
