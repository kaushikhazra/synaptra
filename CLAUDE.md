# Synaptra — Project Instructions

## What This Is

Biologically-inspired agent memory, exposed as an MCP server over Streamable HTTP. Decay-aware storage with multi-strategy retrieval, automatic linking, contradiction detection, consolidation and version history.

**This is a published PyPI package.** `pip install synaptra`, currently 2.0.0, MIT. People outside this machine have it installed. Every change here is a public change.

---

## ⚠ THE ONE THING THAT MAKES THIS REPO DIFFERENT

**Synaptra is the public mirror of a private upstream.**

```
   C:/Projects/cognitive-memory     upstream · internal · where changes are made first
            │
            │  same change, while it is fresh
            ▼
   C:/Projects/synaptra             this repo · public · PyPI
            │
            ▼
   C:/Projects/second-brain         workspace template built on the synaptra substrate
```

⇒ **Changes flow one way: cognitive-memory → synaptra.** Do not invent behaviour here that upstream does not have. If something is wrong here and also wrong upstream, it gets fixed in **both**, in the same pass.

⭐ **Why the coupling is strict:** a port that happens with the change is a port. A port that happens later is a migration, and migrations are where the two copies quietly diverge. They have diverged once already.

**Upstream is a plain git repo on this machine and you can read it directly.** `git -C C:/Projects/cognitive-memory log`, `show`, `diff`. Read the real commit rather than asking for a summary of it.

---

## ⛔ THE BRANDING DELTA — NEVER PORT OVER THESE

Upstream and this repo are **deliberately** different in the following places. A file-level copy from cognitive-memory will destroy every one of them. Port the *behaviour*, re-apply the names.

| | cognitive-memory | synaptra |
|---|---|---|
| package dir | `src/cognitive_memory/` | `src/synaptra/` |
| imports | `cognitive_memory.x` | `synaptra.x` |
| dist name | `cognitive-memory` | `synaptra` |
| console scripts | `cognitive-memory`, … | `synaptra`, `synaptra-service`, `synaptra-cli`, plus a `cm` alias |
| MCP self-name | — | `FastMCP("synaptra")` in `server.py` |
| env prefix | `COGNITIVE_MEMORY_*` | `SYNAPTRA_*` |
| data dir | `~/.cognitive-memory/` | `~/.synaptra/` |

**The full env surface here:** `SYNAPTRA_BACKEND` (default `surrealkv-file`) · `SYNAPTRA_DB` (default `~/.synaptra/data`) · `SYNAPTRA_CONFIG` · `SYNAPTRA_URL` · `SYNAPTRA_HOST` (`127.0.0.1`) · `SYNAPTRA_PORT` (`8050`) · `SYNAPTRA_SURREAL_URL` · `SYNAPTRA_SURREAL_DATA_DIR`.

⚠ After any port, grep for `cognitive_memory`, `cognitive-memory` and `COGNITIVE_MEMORY_` across `src/`, `pyproject.toml` and `README.md`. A clean-looking port that leaves one of these behind ships a broken package.

---

## Architecture

```
server.py         MCP entrypoint — FastMCP, Streamable HTTP at /mcp, stateless
  engine.py       Orchestrator — ingestion, update, delete, retrieval, consolidation
    storage.py                  storage protocol + SQLite backend
    surreal_storage.py          SurrealDB embedded
    surreal_server_storage.py   SurrealDB server backend
    embeddings.py   numpy matrix + lazy sentence-transformers (all-MiniLM-L6-v2, 384d)
    retrieval.py    semantic + keyword + temporal RRF fusion → graph traversal
                    → decay reranking → spreading activation
    decay.py        FSRS-inspired: R(t) = e^(-t / 9S). Pure functions.
    consolidation.py  promotion, archival, clustering, merging, contradiction detection
    classification.py heuristic type classification + importance scoring
    config.py       YAML defaults → stored overrides, dot-notation keys
    models.py       Pydantic domain models
    protocols.py    the storage interface both backends implement
    migrations/     numbered, run once via PRAGMA user_version
    schema.surql    SurrealDB schema, every field IF NOT EXISTS
  service.py        Windows background service via Task Scheduler, no admin
  cli.py            terminal browse / search / manage
  backup/           export, import, verify, prune
```

---

## Key Conventions

- **Memory types and decay:** `working` (hours) · `episodic` (days) · `semantic` (weeks) · `procedural` (months). Plus the longer classes upstream carries. Type is a **decay choice**, not a topic choice.
- **States:** `active`, `archived`.
- **Relationship types:** `causes`, `follows`, `contradicts`, `supports`, `relates_to`, `supersedes`, `part_of`, `describes`.
- **Config:** dot-notation (`decay.growth_factor`, `retrieval.weights.semantic`). Defaults in `config.default.yaml`.
- **Decay is computed on the fly**, not stored, except during consolidation sweeps.
- **Auto-created `relates_to` links carry `strength < 1.0`** (cosine similarity); manual links are `1.0`. `delete_auto_links` depends on that invariant.
- **The embedding model is lazy-loaded** on first `embed()`; cold start takes seconds. The HTTP server warms it at startup deliberately — do not remove that.

## ⛔ Rules that have already cost something

- **Migrations are numbered and run exactly once.** ⛔ **Never add a column by editing an existing migration.** A database that already ran `001` will never see the edit, and the failure is silent. Add `002_`, `003_`. The SurrealDB schema self-heals because every field is `IF NOT EXISTS`; **SQLite does not**, and this package has public SQLite installs.
- **Every SQLite write must maintain FTS5 sync** — insert into `memory_fts` on create, delete and re-insert on content update, delete on delete.
- **`mcp` is currently pinned `>=1.0.0,<2.0`.** MCP 2.0 renamed `streamablehttp_client` → `streamable_http_client`. Do not move the pin without asking; it breaks fresh installs either way and the direction is a decision, not a cleanup.
- **Version bumps are public.** A behaviour change is a minor bump, a schema or API change is a major one. Never publish from a dirty tree.

## Development

```bash
pip install -e ".[dev]"
pytest
synaptra                      # HTTP MCP server on http://127.0.0.1:8050/mcp
python -m synaptra.server
```

---

## Session start — do these two things first

**1. Join cross-chat.** This project talks to other Claude sessions on this machine over NATS.

```bash
crosschat register "C:/Projects/synaptra"
```

It prints `CROSSCHAT_REGISTERED synaptra`. Then start the listener with the **Monitor tool**, not a background shell, so each incoming line arrives as a live notification:

```
Monitor(command="crosschat monitor synaptra",
        description="synaptra crosschat listener",
        persistent=true, timeout_ms=3600000)
```

Start it **once** per session. It loops internally for the whole session. To send:

```bash
crosschat send synaptra <destination-id> "<message>"
```

Registered ids come from `crosschat list`. The two that matter: **`velasari`** and **`cognitive-memory`**.

**2. Report in.** Send `velasari` one line saying you are up and what state the working tree is in.

---

## Working with Velasari

**Velasari (`velasari`) holds the plan for this repo and you can trust it.** It carries the continuity that this session does not have: what upstream changed and why, which differences are deliberate, what Kaushik has already ruled on, and what has been tried before and failed.

```
   HOW    ──▶  yours. Implementation, structure, tests, the actual writing.
   WHAT   ──▶  Velasari's. Which changes cross, in what order.
   WHETHER ─▶  Kaushik's, and it reaches him through Velasari.
```

- **Take briefs from Velasari and act on them** without re-litigating the reasoning. If a brief looks wrong, say so once, plainly, and give your reason — then follow the answer.
- ⛔ **Do not expand scope on your own, and do not negotiate scope with the other build session.** Two build sessions left to agree scope will invent work for each other. Scope questions go up to Velasari.
- ⛔ **Never guess at a deliberate difference.** If you cannot tell whether something is a bug or a decision, ask. The branding delta above is the list of things that look like bugs and are not.
- **Trust does not transfer to your own output.** Velasari's judgement is trusted; your port is not, until the tests pass and you have re-read the diff from disk. ⚠ A check run from memory of having written the thing is not a check.
- **Report what actually happened.** If tests fail, say so with the output. If you skipped something, say that. A port reported clean that was not clean is the single most expensive thing that can happen in this repo, because the next step is publishing it.
