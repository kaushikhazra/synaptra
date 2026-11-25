# Changelog

Versions follow the repo rule: a behaviour change is a minor bump, a schema or API change is a major one.

## 3.1.0 - 2026-11-25

### Changed
- **The server answers within seconds and says plainly when it is still starting** (#13, PR #14).
  The HTTP port is bound first; the embedding model loads in a background thread afterwards
  (stdio: right before `mcp.run`). Tools that need embeddings (`memory_store`, `memory_update`
  with new content, `memory_recall`, `memory_consolidate`) wait for the model, bounded by
  `SYNAPTRA_WARMUP_WAIT_S` (default 60), then report "warming up". Every other tool answers at once.
- **`memory_health` reports the embedding state**: `embedding: {state, elapsed_s, cause?}`, where
  state is `cold`, `warming`, `ready` or `failed`.
- **fastembed (ONNX Runtime, CPU) replaces sentence-transformers.** Same model
  (`all-MiniLM-L6-v2`, 384 dimensions); torch and transformers are no longer installed or imported.
  Vectors match the previous stack (minimum cosine 0.99999988 on the golden fixture; recall top-1
  identical on a copy of a real store).
- **Model files live in `SYNAPTRA_MODEL_DIR`**, else `~/.synaptra/models`, instead of the system temp dir.

### Added
- `SYNAPTRA_WARMUP_WAIT_S` and `SYNAPTRA_MODEL_DIR` environment variables.

### Upgrade notes
- First start downloads the ~90 MB model once, in the background; the server is usable meanwhile.
- No schema change. Existing stores need no migration.

## 3.0.1 - 2026-11-04

### Fixed
- Fresh installs crashed: surrealdb 3.0.0 moved the embedded engine to an extra. Dependency now
  pinned `surrealdb>=1.0.8,<3` (#11, PR #12).

## 3.0.0 - 2026-11-02

### Changed
- **Breaking:** requires `mcp>=2.1.1,<3` and no longer runs on mcp 1.x (#8, PR #10). It conflicts
  with any software in the same Python environment that still needs mcp 1.

## Earlier

- `memory_recall` / `memory_list`: one parser for `time_range`, accepting open-ended and date-only
  ranges (PR #7).
- `memory_list` reports the same retrievability as `memory_get` (#4).
- 2.1.1, 2.1.0 and earlier: see the git history and tags.
