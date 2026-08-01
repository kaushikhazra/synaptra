"""Consolidation pipeline — decay update, promotion, archival, clustering, merging."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import decay as decay_mod
from .models import (
    ConsolidationLogEntry,
    MemoryState,
    MemoryType,
    RelType,
    Relationship,
)

if TYPE_CHECKING:
    from .config import Config
    from .embeddings import EmbeddingService
    from .surreal_storage import SurrealStorage as Storage


async def consolidate(
    storage: "Storage",
    embeddings: "EmbeddingService",
    config: "Config",
    dry_run: bool = False,
) -> list[dict]:
    """Run the full consolidation pipeline. Returns list of actions taken."""
    now = datetime.now(timezone.utc)
    actions: list[dict] = []

    # Stage 1: Decay Update
    active_memories = await storage.get_all_active_memories()
    if not dry_run:
        for mem in active_memories:
            r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
            await storage.update_memory_fields(mem.id, retrievability=r)

    # Stage 2: Promotion Pass
    active_memories = await storage.get_all_active_memories()
    promotion_actions = await _promotion_pass(active_memories, storage, config, now, dry_run)
    actions.extend(promotion_actions)

    # Stage 3: Archive Pass
    forget_threshold = config.get("decay.thresholds.forgotten", 0.2)
    active_memories = await storage.get_all_active_memories()
    archive_actions = await _archive_pass(active_memories, storage, embeddings, config, now, forget_threshold, dry_run)
    actions.extend(archive_actions)

    # Stage 4+5: Cluster Scan + Merge Pass
    active_memories = await storage.get_all_active_memories()
    merge_actions = await _cluster_and_merge(active_memories, storage, embeddings, config, now, dry_run)
    actions.extend(merge_actions)

    # Stage 6: Log
    if not dry_run:
        for action in actions:
            entry = ConsolidationLogEntry(
                id=str(uuid.uuid4()),
                action=action["action"],
                source_ids=action.get("source_ids", []),
                target_id=action.get("target_id"),
                reason=action.get("reason", ""),
                created_at=now,
            )
            await storage.insert_consolidation_log(entry)

    return actions


async def _promotion_pass(
    memories: list,
    storage: "Storage",
    config: "Config",
    now: datetime,
    dry_run: bool,
) -> list[dict]:
    actions = []

    # Config
    w2e_access = config.get("consolidation.promotion.working_to_episodic.min_access_count", 3)
    w2e_importance = config.get("consolidation.promotion.working_to_episodic.min_importance", 0.4)
    w2e_rels = config.get("consolidation.promotion.working_to_episodic.min_relationships", 2)
    e2s_access = config.get("consolidation.promotion.episodic_to_semantic.min_access_count", 5)
    e2s_r = config.get("consolidation.promotion.episodic_to_semantic.min_retrievability", 0.6)
    e2p_access = config.get("consolidation.promotion.episodic_to_procedural.min_access_count", 3)
    e2p_patterns = config.get("consolidation.promotion.episodic_to_procedural.patterns", [
        "how to", "steps", "workflow", "procedure", "when .+ do", "first .+ then",
    ])
    to_person_access = config.get("consolidation.promotion.to_person.min_access_count", 3)

    for mem in memories:
        new_type = None
        reason = ""

        if mem.memory_type == MemoryType.WORKING:
            # Working → Episodic
            rels = await storage.get_relationships_for(mem.id)
            access_ok = mem.access_count >= w2e_access and mem.importance >= w2e_importance
            rels_ok = len(rels) >= w2e_rels
            if access_ok or rels_ok:
                new_type = MemoryType.EPISODIC
                reason = f"Working→Episodic: access={mem.access_count}, importance={mem.importance:.2f}, rels={len(rels)}"

        elif mem.memory_type == MemoryType.EPISODIC:
            # Episodic → Person (check first — most specific, requires person: tag)
            person_tags = [t for t in mem.tags if t.startswith("person:")]
            if person_tags and mem.access_count >= to_person_access:
                new_type = MemoryType.PERSON
                reason = f"Episodic→Person: access={mem.access_count}, has person tags {person_tags}"
            else:
                # Episodic → Procedural (check next, more specific than semantic)
                content_lower = mem.content.lower()
                pattern_match = any(re.search(p, content_lower) for p in e2p_patterns)
                if pattern_match and mem.access_count >= e2p_access:
                    new_type = MemoryType.PROCEDURAL
                    reason = f"Episodic→Procedural: access={mem.access_count}, procedural content pattern"
                else:
                    # Episodic → Semantic
                    r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
                    if mem.access_count >= e2s_access and r > e2s_r:
                        new_type = MemoryType.SEMANTIC
                        reason = f"Episodic→Semantic: access={mem.access_count}, R={r:.2f}"

        elif mem.memory_type == MemoryType.SEMANTIC:
            # Semantic → Person (requires person: tag)
            person_tags = [t for t in mem.tags if t.startswith("person:")]
            if person_tags and mem.access_count >= to_person_access:
                new_type = MemoryType.PERSON
                reason = f"Semantic→Person: access={mem.access_count}, has person tags {person_tags}"

        if new_type is not None:
            action = {
                "action": "promote",
                "source_ids": [mem.id],
                "target_id": mem.id,
                "reason": reason,
                "from_type": mem.memory_type.value,
                "to_type": new_type.value,
            }
            actions.append(action)

            if not dry_run:
                new_s0 = decay_mod.get_initial_stability(new_type.value)
                await storage.update_memory_fields(
                    mem.id,
                    memory_type=new_type,
                    stability=new_s0,
                    updated_at=now,
                )

    return actions


async def _archive_pass(
    memories: list,
    storage: "Storage",
    embeddings: "EmbeddingService",
    config: "Config",
    now: datetime,
    forget_threshold: float,
    dry_run: bool,
) -> list[dict]:
    actions = []
    person_threshold = config.get("consolidation.person_archive_threshold", 0.05)

    for mem in memories:
        # Identity memories are never auto-archived
        if mem.memory_type == MemoryType.IDENTITY:
            continue

        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)

        # Person memories use a stricter (lower) threshold — harder to archive
        if mem.memory_type == MemoryType.PERSON:
            if r < person_threshold:
                action = {
                    "action": "archive",
                    "source_ids": [mem.id],
                    "reason": f"R={r:.4f} < person_threshold={person_threshold}",
                }
                actions.append(action)
                if not dry_run:
                    await storage.update_memory_fields(mem.id, state=MemoryState.ARCHIVED, updated_at=now)
        else:
            if r < forget_threshold:
                action = {
                    "action": "archive",
                    "source_ids": [mem.id],
                    "reason": f"R={r:.4f} < threshold={forget_threshold}",
                }
                actions.append(action)
                if not dry_run:
                    await storage.update_memory_fields(mem.id, state=MemoryState.ARCHIVED, updated_at=now)

    return actions


async def _cluster_and_merge(
    memories: list,
    storage: "Storage",
    embeddings: "EmbeddingService",
    config: "Config",
    now: datetime,
    dry_run: bool,
) -> list[dict]:
    actions = []
    merge_threshold = config.get("consolidation.merge_threshold", 0.90)
    contradiction_threshold = config.get("contradiction.similarity_threshold", 0.80)
    negation_signals = config.get("contradiction.negation_signals", [
        "not", "never", "no longer", "changed", "wrong", "actually", "incorrect", "false",
    ])

    # Use vector search per memory to find similar pairs
    if len(memories) < 2:
        return actions

    merged_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for mem in memories:
        if mem.id in merged_ids:
            continue
        # Find similar memories via vector search
        similar = await storage.vector_search_for_memory(mem.id, top_k=10)
        for other_id, sim in similar:
            if other_id == mem.id or other_id in merged_ids:
                continue
            pair = tuple(sorted([mem.id, other_id]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if sim < contradiction_threshold:
                continue

            mem_b = await storage.get_memory(other_id)
            if mem_b is None:
                continue

            has_negation = _has_negation_signals(mem.content, mem_b.content, negation_signals)

            if sim >= merge_threshold and not has_negation:
                # Identity memories are foundational — never merge or auto-archive them
                if mem.memory_type == MemoryType.IDENTITY or mem_b.memory_type == MemoryType.IDENTITY:
                    continue
                # Merge
                score_a = mem.importance * mem.access_count
                score_b = mem_b.importance * mem_b.access_count
                if score_a >= score_b:
                    primary, secondary = mem, mem_b
                else:
                    primary, secondary = mem_b, mem

                actions.append({
                    "action": "merge",
                    "source_ids": [primary.id, secondary.id],
                    "target_id": primary.id,
                    "reason": f"Similarity={sim:.3f}, merged into {primary.id}",
                })

                if not dry_run:
                    await _execute_merge(storage, embeddings, primary, secondary, now)
                merged_ids.add(secondary.id)

            elif has_negation:
                # Flag contradiction
                actions.append({
                    "action": "flag_contradiction",
                    "source_ids": [mem.id, mem_b.id],
                    "reason": f"Similarity={sim:.3f}, negation signals detected",
                })
                if not dry_run:
                    await _create_contradicts(storage, mem.id, mem_b.id, sim, now)

    return actions


def _has_negation_signals(content_a: str, content_b: str, signals: list[str]) -> bool:
    """Check if two similar texts contain negation signals suggesting contradiction."""
    combined = (content_a + " " + content_b).lower()
    return any(signal in combined for signal in signals)


async def _create_contradicts(storage: "Storage", id_a: str, id_b: str, strength: float, now: datetime) -> None:
    rel = Relationship(
        id=str(uuid.uuid4()),
        source_id=id_a,
        target_id=id_b,
        rel_type=RelType.CONTRADICTS,
        strength=strength,
        created_at=now,
    )
    await storage.insert_relationship(rel)


async def _execute_merge(
    storage: "Storage",
    embeddings: "EmbeddingService",
    primary,
    secondary,
    now: datetime,
) -> None:
    """Execute a merge: append content, re-embed, transfer relationships, archive secondary."""
    # Step 2: Append unique content
    merged_content = f"{primary.content}\n\n[Merged from {secondary.id}]: {secondary.content}"

    # Step 3: Re-embed
    new_embedding = embeddings.embed(merged_content)
    embedding_list = new_embedding.astype(float).tolist()

    await storage.update_memory_fields(primary.id, content=merged_content, updated_at=now)
    await storage.update_embedding(primary.id, embedding_list)

    # Step 4: Transfer relationships with dedup
    secondary_rels = await storage.get_relationships_for(secondary.id)
    for rel in secondary_rels:
        if rel.source_id == secondary.id:
            # Re-point source from secondary to primary
            new_source = primary.id
            new_target = rel.target_id
        else:
            # Re-point target from secondary to primary
            new_source = rel.source_id
            new_target = primary.id

        # Skip self-referential
        if new_source == new_target:
            continue

        # Check for existing relationship with same (source, target, type)
        existing = await storage.get_relationships_for(primary.id, [rel.rel_type.value])
        duplicate = None
        for ex in existing:
            if ex.source_id == new_source and ex.target_id == new_target:
                duplicate = ex
                break

        if duplicate:
            # Keep higher strength
            if rel.strength > duplicate.strength:
                await storage.delete_relationship(duplicate.source_id, duplicate.target_id, duplicate.rel_type.value)
                new_rel = Relationship(
                    id=str(uuid.uuid4()),
                    source_id=new_source,
                    target_id=new_target,
                    rel_type=rel.rel_type,
                    strength=rel.strength,
                    created_at=now,
                )
                await storage.insert_relationship(new_rel)
        else:
            new_rel = Relationship(
                id=str(uuid.uuid4()),
                source_id=new_source,
                target_id=new_target,
                rel_type=rel.rel_type,
                strength=rel.strength,
                created_at=now,
            )
            await storage.insert_relationship(new_rel)

    # Step 5: Archive secondary with supersedes link
    await storage.update_memory_fields(secondary.id, state=MemoryState.ARCHIVED, updated_at=now)

    supersedes_rel = Relationship(
        id=str(uuid.uuid4()),
        source_id=primary.id,
        target_id=secondary.id,
        rel_type=RelType.SUPERSEDES,
        strength=1.0,
        created_at=now,
    )
    await storage.insert_relationship(supersedes_rel)
