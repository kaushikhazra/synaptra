"""Multi-strategy retrieval pipeline — two-phase with RRF fusion and decay reranking."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

from . import decay as decay_mod
from .models import ContradictionInfo, RecallResult, ReinforceUpdate, SpreadingActivationRow

if TYPE_CHECKING:
    from .config import Config
    from .embeddings import EmbeddingService
    from .models import Memory
    from .surreal_server_storage import SurrealServerStorage as Storage


def _memory_matches_filters(
    mem: "Memory",
    type_filter: str | None,
    tags: list[str] | None,
    time_range: tuple[datetime, datetime] | None,
) -> bool:
    if mem.state.value != "active":
        return False
    if type_filter and mem.memory_type.value != type_filter:
        return False
    if tags and not all(tag in mem.tags for tag in tags):
        return False
    if time_range and not (time_range[0] <= mem.created_at <= time_range[1]):
        return False
    return True


def compute_spreading_boosts_from_walk(
    rows: list[SpreadingActivationRow],
    activation_strength: float,
    spread_factor: float,
    max_boost: float,
) -> dict[str, float]:
    """Compute max boost per neighbor from enriched spreading-activation walk rows.

    For each row, computes ``decay_mod.compute_spreading_boost`` using the row's
    ``rel_strength`` and ``depth``.  Deduplicates by ``neighbor_id``, keeping the
    maximum boost value across all paths that reach that neighbor.  Only rows
    where ``state == 'active'`` are included (Python safety filter — walk already
    enforces active-only server-side).

    Returns ``dict[neighbor_id, max_boost_value]``.
    """
    boosts: dict[str, float] = {}
    for row in rows:
        if row.state != "active":
            continue
        boost = decay_mod.compute_spreading_boost(
            row.rel_strength, activation_strength, row.depth, spread_factor, max_boost,
        )
        if boost <= 0:
            continue
        if row.neighbor_id not in boosts or boost > boosts[row.neighbor_id]:
            boosts[row.neighbor_id] = boost
    return boosts


async def recall(
    query: str,
    storage: "Storage",
    embeddings: "EmbeddingService",
    config: "Config",
    type_filter: str | None = None,
    tags: list[str] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    limit: int | None = None,
) -> list[RecallResult]:
    """Execute multi-strategy retrieval pipeline. Returns ranked results."""
    now = datetime.now(timezone.utc)
    limit = limit or config.get("retrieval.default_limit", 10)
    multiplier = config.get("retrieval.phase1_candidate_multiplier", 3)
    cap = config.get("retrieval.phase1_candidate_cap", 30)
    phase1_n = min(limit * multiplier, cap)
    timing_enabled = config.get("logging.recall_timing", False)

    # === Phase 1: concurrent strategies (semantic + keyword + temporal) ===
    # Filters pushed into storage queries (F1) — no Python-side hydration needed.

    query_vec = embeddings.embed(query)
    query_list = query_vec.astype(float).tolist()

    # Task 6.1: single asyncio.gather for all three Phase 1 strategies.
    # return_exceptions=True so that an FTS failure (malformed query) degrades
    # gracefully to an empty keyword list without cancelling the other two.
    if timing_enabled:
        _phase_start = time.perf_counter()
    phase1_raw = await asyncio.gather(
        storage.vector_search(
            query_list, top_k=phase1_n,
            type_filter=type_filter, tags=tags, time_range=time_range,
        ),
        storage.fts_search(
            query, state="active", limit=phase1_n,
            type_filter=type_filter, tags=tags, time_range=time_range,
        ),
        storage.get_recent_active_ids(
            limit=phase1_n,
            type_filter=type_filter, tags=tags, time_range=time_range,
        ),
        return_exceptions=True,
    )

    if timing_enabled:
        log.debug(f"recall.phase1_gather: {(time.perf_counter() - _phase_start) * 1000:.1f}ms")

    semantic_raw: list[tuple[str, float]] = (
        phase1_raw[0] if not isinstance(phase1_raw[0], BaseException) else []
    )
    fts_raw: list[tuple[str, float]] = (
        phase1_raw[1] if not isinstance(phase1_raw[1], BaseException) else []
    )
    temporal_raw: list[tuple[str, datetime]] = (
        phase1_raw[2] if not isinstance(phase1_raw[2], BaseException) else []
    )

    # Normalise FTS BM25 scores (can be negative) to positive values.
    keyword_results: list[tuple[str, float]] = [
        (mid, -score if score < 0 else score) for mid, score in fts_raw
    ]

    # Compute recency scores for temporal results.
    temporal_results: list[tuple[str, float]] = []
    for mid, last_accessed in temporal_raw:
        elapsed_days = max(0, (now - last_accessed).total_seconds()) / 86400.0
        recency_score = math.exp(-elapsed_days / 30.0)  # 30-day half-life
        temporal_results.append((mid, recency_score))

    # === Phase 1 RRF Fusion ===
    k = config.get("retrieval.rrf_k", 60)
    w_semantic = config.get("retrieval.weights.semantic", 1.0)
    w_keyword = config.get("retrieval.weights.keyword", 0.7)
    w_temporal = config.get("retrieval.weights.temporal", 0.3)

    phase1_scores: dict[str, float] = defaultdict(float)
    phase1_found_by: dict[str, set] = defaultdict(set)

    for rank, (mid, _) in enumerate(semantic_raw):
        phase1_scores[mid] += w_semantic / (k + rank + 1)
        phase1_found_by[mid].add("semantic")

    for rank, (mid, _) in enumerate(keyword_results):
        phase1_scores[mid] += w_keyword / (k + rank + 1)
        phase1_found_by[mid].add("keyword")

    for rank, (mid, _) in enumerate(temporal_results):
        phase1_scores[mid] += w_temporal / (k + rank + 1)
        phase1_found_by[mid].add("temporal")

    # Top-N from Phase 1
    phase1_ranked = sorted(phase1_scores.items(), key=lambda x: x[1], reverse=True)[:phase1_n]

    # === Phase 2: Graph traversal from top-5 seeds (Task 6.2) ===
    # Single batched call replaces the per-seed get_neighbors loop.
    seed_count = config.get("retrieval.phase2_seed_count", 5)
    seeds = [mid for mid, _ in phase1_ranked[:seed_count]]

    if timing_enabled:
        _phase_start = time.perf_counter()
    seed_neighbors = await storage.get_neighbors_bulk(seeds)
    if timing_enabled:
        log.debug(f"recall.phase2_neighbors_bulk: {(time.perf_counter() - _phase_start) * 1000:.1f}ms")

    # Flatten neighbors; track visited to skip seeds and duplicates.
    visited: set[str] = set(seeds)
    phase2_neighbor_scores: dict[str, float] = {}
    for _seed_id, neighbors_list in seed_neighbors.items():
        for neighbor_id, rel in neighbors_list:
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            phase2_neighbor_scores[neighbor_id] = max(
                rel.strength,
                phase2_neighbor_scores.get(neighbor_id, 0.0),
            )

    # === Final RRF Fusion (Phase 1 + Graph) ===
    w_graph = config.get("retrieval.weights.graph", 0.5)
    final_scores: dict[str, float] = defaultdict(float)
    final_found_by: dict[str, set] = defaultdict(set)

    # Phase 1 keeps its combined score (weight 1.0 preserves Phase 1 ranking)
    for mid, score in phase1_ranked:
        final_scores[mid] += score
        final_found_by[mid] = phase1_found_by.get(mid, set()).copy()

    # Graph discoveries (Phase 2 neighbours not already in Phase 1)
    graph_ranked = sorted(phase2_neighbor_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (mid, _) in enumerate(graph_ranked):
        final_scores[mid] += w_graph / (k + rank + 1)
        final_found_by[mid].add("graph")

    # === Post-RRF: single hydration pass + supersede flags concurrent (Task 6.3) ===
    # D2: collapse three former hydration sites to one.
    # O3 micro-win: hydration and supersede lookup are disjoint reads — run concurrently.
    final_id_list = list(final_scores.keys())
    if timing_enabled:
        _phase_start = time.perf_counter()
    memories_raw, supersede_set = await asyncio.gather(
        storage.get_memories_by_ids(final_id_list),
        storage.get_supersede_flags(final_id_list),
    )
    if timing_enabled:
        log.debug(f"recall.hydration_supersede_gather: {(time.perf_counter() - _phase_start) * 1000:.1f}ms")
    memories_by_id: dict[str, Memory] = {mem.id: mem for mem in memories_raw}

    # === Decay-Weighted Reranking + Supersede Penalty ===
    # Phase 2 IDs came from graph edges (not filter-pushed queries) — apply
    # Python-side filter for them.  Phase 1 IDs were already filtered server-side.
    phase2_ids = set(phase2_neighbor_scores.keys())
    decay_influence = config.get("decay.decay_influence", 0.5)
    supersede_penalty = config.get("retrieval.supersede_penalty", 0.3)

    if timing_enabled:
        _phase_start = time.perf_counter()
    decay_scored: list[tuple[str, float]] = []
    for mid, rrf_score in final_scores.items():
        mem = memories_by_id.get(mid)
        if mem is None:
            continue
        # Python-side filter for Phase 2 neighbours only.
        if mid in phase2_ids and not _memory_matches_filters(mem, type_filter, tags, time_range):
            continue
        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
        score = rrf_score * (r ** decay_influence)
        if mid in supersede_set:
            score *= supersede_penalty
        decay_scored.append((mid, score))

    # === Top-K Selection ===
    decay_scored.sort(key=lambda x: x[1], reverse=True)
    top_k = decay_scored[:limit]
    top_k_ids = [mid for mid, _ in top_k]
    if timing_enabled:
        log.debug(f"recall.scoring: {(time.perf_counter() - _phase_start) * 1000:.1f}ms")

    # === Config for spreading activation ===
    growth_factor = config.get("decay.growth_factor", 2.0)
    activation_strength = config.get("spreading_activation.activation_strength", 0.3)
    spread_factor = config.get("spreading_activation.spread_factor", 0.5)
    max_depth = config.get("spreading_activation.max_depth", 3)
    max_boost = config.get("spreading_activation.max_boost", 0.5)

    # === Contradictions + Spreading Walk concurrent (Task 6.4) ===
    # D4: server-side bulk walk replaces recursive _spread_from.
    # D6: get_contradictions_bulk replaces per-result get_contradictions_for.
    # Both are independent reads — run concurrently.
    if timing_enabled:
        _phase_start = time.perf_counter()
    contradictions_map, spread_rows = await asyncio.gather(
        storage.get_contradictions_bulk(top_k_ids),
        storage.spreading_activation_walk(top_k_ids, max_depth=max_depth),
    )
    if timing_enabled:
        log.debug(f"recall.contradictions_spreading_gather: {(time.perf_counter() - _phase_start) * 1000:.1f}ms")

    spread_boosts = compute_spreading_boosts_from_walk(
        spread_rows, activation_strength, spread_factor, max_boost,
    )

    # === Reinforce Write-Back + Spreading Stability Update concurrent (Task 6.5) ===
    # D7: bulk_reinforce replaces per-result update_memory_fields.
    # top_k IDs and spread_boosts keys are disjoint (walk excludes seeds).
    reinforce_updates: list[ReinforceUpdate] = []
    for mid, _ in top_k:
        mem = memories_by_id.get(mid)
        if mem is None:
            continue
        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
        new_stability = decay_mod.reinforce(mem.stability, r, growth_factor)
        reinforce_updates.append(ReinforceUpdate(
            memory_id=mid,
            stability=new_stability,
            last_accessed=now,
            access_count=mem.access_count + 1,
        ))

    # Build (new_stability, neighbor_id) pairs for bulk_update_stability.
    # SpreadingActivationRow.current_stability gives pre-boost stability directly —
    # no separate hydration needed (D2a enriched walk return).
    neighbor_stability: dict[str, float] = {}
    for row in spread_rows:
        # Keep first occurrence per neighbor_id (walk deduplicates to shallowest depth).
        if row.neighbor_id not in neighbor_stability:
            neighbor_stability[row.neighbor_id] = row.current_stability

    spread_boost_pairs: list[tuple[float, str]] = []
    for neighbor_id, boost in spread_boosts.items():
        cur_s = neighbor_stability.get(neighbor_id)
        if cur_s is None:
            continue
        new_s = decay_mod.apply_spreading_boost(cur_s, boost)
        spread_boost_pairs.append((new_s, neighbor_id))

    if timing_enabled:
        _phase_start = time.perf_counter()
    await asyncio.gather(
        storage.bulk_reinforce(reinforce_updates),
        storage.bulk_update_stability(spread_boost_pairs),
    )
    if timing_enabled:
        log.debug(f"recall.reinforce_stability_gather: {(time.perf_counter() - _phase_start) * 1000:.1f}ms")

    # === Build Results ===
    results: list[RecallResult] = []
    for mid, score in top_k:
        mem = memories_by_id.get(mid)
        if mem is None:
            continue
        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)

        raw_contradictions = contradictions_map.get(mid, [])
        contradictions = [
            ContradictionInfo(memory_id=cid, content_preview=preview, strength=strength)
            for cid, preview, strength in raw_contradictions
        ]

        results.append(RecallResult(
            id=mem.id,
            content=mem.content,
            memory_type=mem.memory_type,
            importance=mem.importance,
            retrievability=r,
            score=score,
            found_by=sorted(final_found_by.get(mid, set())),
            tags=mem.tags,
            created_at=mem.created_at,
            last_accessed=mem.last_accessed,
            contradictions=contradictions,
        ))

    return results
