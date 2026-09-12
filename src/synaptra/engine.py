"""Memory engine — orchestrates ingestion, update, restore, delete, and retrieval."""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Optional

from . import decay as decay_mod
from . import retrieval as retrieval_mod
from . import consolidation as consolidation_mod
from .classification import classify, score_importance
from .config import Config
from .embeddings import EmbeddingService
from .models import (
    AUTO_RATER,
    ContradictionInfo,
    Memory,
    MemoryGetResponse,
    MemoryState,
    MemoryType,
    MemoryVersion,
    RelType,
    Relationship,
    RelationshipInfo,
    RecallResult,
    StatsResponse,
    ToolResponse,
)
from .protocols import StorageProtocol
from .surreal_storage import SurrealStorage


def _require_rater(rater: str | None) -> None:
    """Reject a write that sets `importance` without naming who rated it.

    Importance drives retrieval ranking, and different models score the same
    content differently — so a score whose rater is unknown cannot be compared
    to any other score. Silently defaulting would re-open the NULL set, which
    is meant to be closed and to mean exactly "written before rater tracking".
    """
    if rater is None or not rater.strip():
        raise ValueError(
            "rater is required when setting importance. Pass the identifier of "
            "the model producing the score, e.g. rater='claude-opus-5'. "
            "Importance is rater-relative: a score with no rater cannot be "
            "compared to any other score."
        )


# --- Health report private helpers (module-level, not class methods) ---

_ALL_MEMORY_TYPES = ["working", "episodic", "semantic", "procedural", "identity", "person"]


def _build_totals(counts_raw: dict) -> dict:
    """Reshape (type, state) count dict into the totals section of the health report.

    Ensures all 6 memory types are always present, defaulting to 0 (US-1.1).
    counts_raw keys are (memory_type_str, state_str) tuples.
    """
    by_type: dict[str, dict[str, int]] = {t: {"active": 0, "archived": 0} for t in _ALL_MEMORY_TYPES}

    for (mem_type, state), cnt in counts_raw.items():
        if mem_type in by_type and state in ("active", "archived"):
            by_type[mem_type][state] = cnt

    total_active = sum(v["active"] for v in by_type.values())
    total_archived = sum(v["archived"] for v in by_type.values())

    return {
        "by_type": by_type,
        "by_state": {"active": total_active, "archived": total_archived},
        "total": total_active + total_archived,
    }


def _build_decay_report(active_memories: list[dict], now: datetime) -> dict:
    """Compute fresh retrievability for every active memory and build the decay section.

    - at_risk: memories with R < 0.3, identity excluded (D2, D4), sorted ascending by R, capped at 50 (D6)
    - by_type: per-type averages; types with zero active members get null averages (US-2.2)
    """
    AT_RISK_THRESHOLD = 0.3
    IDENTITY_TYPE = "identity"

    at_risk_all: list[dict] = []
    by_type_buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"imp": [], "stab": [], "ret": []})

    for mem in active_memories:
        r = decay_mod.compute_retrievability(mem["last_accessed"], mem["stability"], now)
        t = mem["memory_type"]
        by_type_buckets[t]["imp"].append(mem["importance"])
        by_type_buckets[t]["stab"].append(mem["stability"])
        by_type_buckets[t]["ret"].append(r)

        if r < AT_RISK_THRESHOLD and t != IDENTITY_TYPE:
            at_risk_all.append({
                "id": mem["id"],
                "content_preview": mem["content_preview"],
                "memory_type": t,
                "retrievability": round(r, 4),
                "last_accessed": mem["last_accessed"].isoformat(),
                "stability": mem["stability"],
                "tags": mem["tags"],
            })

    at_risk_all.sort(key=lambda x: x["retrievability"])

    by_type_summary: dict[str, dict] = {}
    for mem_type, buckets in by_type_buckets.items():
        if buckets["ret"]:
            by_type_summary[mem_type] = {
                "avg_importance": round(mean(buckets["imp"]), 4),
                "avg_stability": round(mean(buckets["stab"]), 4),
                "avg_retrievability": round(mean(buckets["ret"]), 4),
                "count": len(buckets["ret"]),
            }

    # Backfill types absent from active memories with null averages (US-2.2)
    for mem_type in _ALL_MEMORY_TYPES:
        if mem_type not in by_type_summary:
            by_type_summary[mem_type] = {
                "avg_importance": None,
                "avg_stability": None,
                "avg_retrievability": None,
                "count": 0,
            }

    return {
        "at_risk": at_risk_all[:50],
        "at_risk_count": len(at_risk_all),
        "by_type": by_type_summary,
    }


def _build_gaps(totals: dict, tag_rows: list[list[str]], untagged_count: int) -> dict:
    """Identify storage gaps: empty types, sparse types, and tag coverage.

    - empty_types: types with zero active memories (all excluded types included)
    - sparse_types: types with < 5% of total active, identity excluded (D3)
    - tag_coverage: top-20 tags + unique count, using Counter flatten (D7)
    """
    SPARSE_THRESHOLD_PCT = 5.0
    EXCLUDED_FROM_SPARSE = {"identity"}

    total_active = totals["by_state"]["active"]
    empty_types = [t for t, v in totals["by_type"].items() if v["active"] == 0]

    sparse_types = []
    for mem_type, counts in totals["by_type"].items():
        if mem_type in EXCLUDED_FROM_SPARSE:
            continue
        active = counts["active"]
        if active == 0:
            continue  # already in empty_types
        pct = (active / total_active * 100) if total_active > 0 else 0.0
        if pct < SPARSE_THRESHOLD_PCT:
            sparse_types.append({
                "type": mem_type,
                "active_count": active,
                "pct_of_active": round(pct, 1),
            })

    # Flatten tag arrays and count (D7) — no SurrealDB aggregation complexity
    counter = Counter(tag for tags in tag_rows for tag in tags if tag)
    total_unique = len(counter)
    top_tags = sorted(
        [{"tag": t, "count": c} for t, c in counter.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:20]

    return {
        "empty_types": empty_types,
        "sparse_types": sparse_types,
        "tag_coverage": {
            "untagged_count": untagged_count,
            "top_tags": top_tags,
            "total_unique_tags": total_unique,
        },
    }


class MemoryEngine:
    """Central orchestrator for the synaptra system."""

    def __init__(
        self,
        db_path: str = "mem://",
        config_path: Path | None = None,
        storage: StorageProtocol | None = None,
    ):
        self.storage = storage if storage is not None else SurrealStorage(db_path)
        self.config = Config(storage=self.storage, config_path=config_path)
        self.embeddings = EmbeddingService()

    def close(self) -> None:
        self.storage.close()

    # === Ingestion ===

    async def store_memory(
        self,
        content: str,
        memory_type: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        conversation_id: str | None = None,
        *,
        rater: str,
    ) -> Memory:
        """Store a new memory with classification, embedding, auto-linking, and contradiction check.

        `rater` is mandatory. Importance is not an absolute quantity — different
        models score the same content differently — so a score is only comparable
        to another score from the same rater. MCP carries no model identity, so
        the caller is the only thing that knows who is rating. See issue #8.
        """
        _require_rater(rater)
        now = datetime.now(timezone.utc)
        memory_id = str(uuid.uuid4())

        # Step 1: Classification
        if memory_type:
            mem_type = MemoryType(memory_type)
        else:
            mem_type, _ = classify(content, source)

        # Step 2: Importance scoring
        importance_cfg = {
            "base_score": self.config.get("importance.base_score", 0.5),
            "named_entity_bonus": self.config.get("importance.named_entity_bonus", 0.1),
            "relational_bonus": self.config.get("importance.relational_bonus", 0.1),
            "length_bonus": self.config.get("importance.length_bonus", 0.1),
            "length_threshold": self.config.get("importance.length_threshold", 200),
            "working_penalty": self.config.get("importance.working_penalty", 0.1),
            "identity_bonus": self.config.get("importance.identity_bonus", 0.2),
            "person_bonus": self.config.get("importance.person_bonus", 0.15),
            "min": self.config.get("importance.min", 0.1),
            "max": self.config.get("importance.max", 1.0),
        }
        imp = score_importance(content, mem_type, importance, importance_cfg)

        # Step 3: Embedding
        embedding = self.embeddings.embed(content)
        embedding_list = embedding.astype(float).tolist()

        # Initial decay values
        s0 = decay_mod.get_initial_stability(mem_type.value)

        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=mem_type,
            state=MemoryState.ACTIVE,
            importance=imp,
            stability=s0,
            retrievability=1.0,
            access_count=0,
            created_at=now,
            updated_at=now,
            last_accessed=now,
            source=source,
            conversation_id=conversation_id,
            # If the agent supplied importance, the agent rated it. If it did
            # not, `score_importance` fell back to its heuristic — so the
            # heuristic is the rater, not the caller.
            rater=rater if importance is not None else AUTO_RATER,
            rated_at=now,
            tags=tags or [],
        )

        await self.storage.insert_memory(memory, embedding_list)

        # Step 4: Auto-linking
        await self._auto_link(memory_id, embedding_list, now)

        # Step 5: Contradiction check
        await self._contradiction_check(memory_id, embedding_list, content, now)

        return memory

    async def _auto_link(self, memory_id: str, embedding: list[float], now: datetime) -> None:
        """Find similar active memories and create relates_to links."""
        threshold = self.config.get("auto_linking.similarity_threshold", 0.75)
        max_links = self.config.get("auto_linking.max_links", 5)

        results = await self.storage.vector_search(embedding, top_k=max_links + 1)
        linked = 0
        for mid, score in results:
            if mid == memory_id:
                continue
            if score < threshold:
                continue
            if linked >= max_links:
                break

            rel = Relationship(
                id=str(uuid.uuid4()),
                source_id=memory_id,
                target_id=mid,
                rel_type=RelType.RELATES_TO,
                strength=score,
                created_at=now,
            )
            await self.storage.insert_relationship(rel)
            linked += 1

    async def _contradiction_check(self, memory_id: str, embedding: list[float], content: str, now: datetime) -> None:
        """Check for contradictions with existing memories."""
        threshold = self.config.get("contradiction.similarity_threshold", 0.80)
        negation_signals = self.config.get("contradiction.negation_signals", [
            "not", "never", "no longer", "changed", "wrong", "actually",
        ])

        results = await self.storage.vector_search(embedding, top_k=10)
        for mid, score in results:
            if mid == memory_id:
                continue
            if score < threshold:
                continue

            other = await self.storage.get_memory(mid)
            if other is None:
                continue

            combined = (content + " " + other.content).lower()
            if any(signal in combined for signal in negation_signals):
                rel = Relationship(
                    id=str(uuid.uuid4()),
                    source_id=memory_id,
                    target_id=mid,
                    rel_type=RelType.CONTRADICTS,
                    strength=score,
                    created_at=now,
                )
                await self.storage.insert_relationship(rel)

    # === Update ===

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        rater: str | None = None,
    ) -> Memory | None:
        """Update a memory with versioning, re-embedding, reinforcement.

        `rater` is required ONLY when `importance` changes. An update that
        touches content, type or tags leaves `rater` and `rated_at` untouched —
        refreshing them on every update would relabel an old score as freshly
        rated, destroying the signal issue #8 exists to create.
        """
        if importance is not None:
            _require_rater(rater)
        mem = await self.storage.get_memory(memory_id)
        if mem is None:
            return None

        now = datetime.now(timezone.utc)
        content_changed = content is not None and content != mem.content
        type_changed = memory_type is not None and memory_type != mem.memory_type.value

        # Version snapshot
        version = MemoryVersion(
            id=str(uuid.uuid4()),
            memory_id=memory_id,
            content=mem.content,
            metadata={
                "changed": [],
                **({"old_content": mem.content[:200]} if content_changed else {}),
                **({"old_type": mem.memory_type.value} if type_changed else {}),
            },
            created_at=now,
        )
        if content_changed:
            version.metadata["changed"].append("content")
        if type_changed:
            version.metadata["changed"].append("type")
        if importance is not None:
            version.metadata["changed"].append("importance")
        if tags is not None:
            version.metadata["changed"].append("tags")
        await self.storage.insert_version(version)

        # Apply changes
        fields = {}
        if content is not None:
            fields["content"] = content
        if memory_type is not None:
            fields["memory_type"] = MemoryType(memory_type)
        if importance is not None:
            fields["importance"] = max(0.1, min(1.0, importance))
            # Only re-stamp the rater when the score itself changed.
            fields["rater"] = rater
            fields["rated_at"] = now
        if tags is not None:
            fields["tags"] = tags

        # Re-embed if content changed
        if content_changed:
            new_embedding = self.embeddings.embed(content)
            await self.storage.update_embedding(memory_id, new_embedding.astype(float).tolist())

        # Reinforce (or reset stability on type change)
        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
        if type_changed:
            fields["stability"] = decay_mod.get_initial_stability(memory_type)
        else:
            growth_factor = self.config.get("decay.growth_factor", 2.0)
            fields["stability"] = decay_mod.reinforce(mem.stability, r, growth_factor)

        fields["updated_at"] = now
        fields["last_accessed"] = now

        await self.storage.update_memory_fields(memory_id, **fields)

        # Re-scan auto-links if content changed
        if content_changed:
            await self.storage.delete_auto_links(memory_id)
            new_embedding = self.embeddings.embed(content)
            await self._auto_link(memory_id, new_embedding.astype(float).tolist(), now)

        return await self.storage.get_memory(memory_id)

    # === Restore ===

    async def restore_memory(self, memory_id: str) -> Memory | None:
        """Restore an archived memory with decay reset."""
        mem = await self.storage.get_memory(memory_id)
        if mem is None or mem.state != MemoryState.ARCHIVED:
            return None

        now = datetime.now(timezone.utc)
        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
        growth_factor = self.config.get("decay.growth_factor", 2.0)
        new_stability = decay_mod.reinforce(mem.stability, r, growth_factor)

        await self.storage.update_memory_fields(
            memory_id,
            state=MemoryState.ACTIVE,
            last_accessed=now,
            stability=new_stability,
            updated_at=now,
        )

        return await self.storage.get_memory(memory_id)

    # === Delete ===

    async def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory with full cascade."""
        mem = await self.storage.get_memory(memory_id)
        if mem is None:
            return False
        await self.storage.delete_memory(memory_id)
        return True

    # === Retrieval ===

    async def recall(
        self,
        query: str,
        type_filter: str | None = None,
        tags: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int | None = None,
    ) -> list[RecallResult]:
        """Multi-strategy retrieval."""
        return await retrieval_mod.recall(
            query, self.storage, self.embeddings, self.config,
            type_filter=type_filter, tags=tags, time_range=time_range, limit=limit,
        )

    # === Get (read-only) ===

    async def get_memory(self, memory_id: str) -> MemoryGetResponse | None:
        """Get full inspection view — memory + relationships + versions. Read-only."""
        mem = await self.storage.get_memory(memory_id)
        if mem is None:
            return None

        now = datetime.now(timezone.utc)
        r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
        mem.retrievability = r

        rels = await self.storage.get_relationships_for(mem.id)
        rel_infos = []
        for rel in rels:
            if rel.source_id == mem.id:
                rel_infos.append(RelationshipInfo(
                    memory_id=rel.target_id, rel_type=rel.rel_type,
                    strength=rel.strength, direction="outgoing",
                ))
            else:
                rel_infos.append(RelationshipInfo(
                    memory_id=rel.source_id, rel_type=rel.rel_type,
                    strength=rel.strength, direction="incoming",
                ))

        versions = await self.storage.get_versions(mem.id)

        return MemoryGetResponse(memory=mem, relationships=rel_infos, versions=versions)

    # === Relationships ===

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        strength: float = 1.0,
    ) -> Relationship:
        now = datetime.now(timezone.utc)
        rel = Relationship(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            rel_type=RelType(rel_type),
            strength=strength,
            created_at=now,
        )
        await self.storage.insert_relationship(rel)
        return rel

    async def delete_relationship(self, source_id: str, target_id: str, rel_type: str) -> bool:
        return await self.storage.delete_relationship(source_id, target_id, rel_type)

    async def get_related(
        self,
        memory_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> list[dict]:
        """Read-only graph traversal. No side effects."""
        visited: set[str] = {memory_id}
        results: list[dict] = []
        frontier = [memory_id]

        for d in range(depth):
            next_frontier = []
            for mid in frontier:
                neighbors = await self.storage.get_neighbors(mid, active_only=True)
                for neighbor_id, rel in neighbors:
                    if neighbor_id in visited:
                        continue
                    if rel_types and rel.rel_type.value not in rel_types:
                        continue
                    visited.add(neighbor_id)
                    mem = await self.storage.get_memory(neighbor_id)
                    if mem is None:
                        continue
                    now = datetime.now(timezone.utc)
                    r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
                    results.append({
                        "memory": mem.model_dump(),
                        "relationship": rel.model_dump(),
                        "depth": d + 1,
                        "retrievability": r,
                    })
                    next_frontier.append(neighbor_id)
            frontier = next_frontier

        return results

    # === Archive ===

    async def archive_memory(self, memory_id: str) -> bool:
        mem = await self.storage.get_memory(memory_id)
        if mem is None or mem.state == MemoryState.ARCHIVED:
            return False
        now = datetime.now(timezone.utc)
        await self.storage.update_memory_fields(memory_id, state=MemoryState.ARCHIVED, updated_at=now)
        return True

    async def archive_bulk(self, memory_ids: list[str]) -> int:
        count = 0
        for mid in memory_ids:
            if await self.archive_memory(mid):
                count += 1
        return count

    async def archive_below_retrievability(self, threshold: float) -> int:
        now = datetime.now(timezone.utc)
        active = await self.storage.get_all_active_memories()
        count = 0
        for mem in active:
            r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
            if r < threshold:
                await self.archive_memory(mem.id)
                count += 1
        return count

    # === Consolidation ===

    async def consolidate(self, dry_run: bool = False) -> list[dict]:
        return await consolidation_mod.consolidate(
            self.storage, self.embeddings, self.config, dry_run=dry_run,
        )

    # === Stats ===

    async def get_stats(self) -> dict:
        now = datetime.now(timezone.utc)
        counts_by_type = await self.storage.get_counts_by_type()
        counts_by_state = await self.storage.get_counts_by_state()

        active = await self.storage.get_all_active_memories()
        r_by_type: dict[str, list[float]] = {}
        fading_count = 0
        forgotten_count = 0
        healthy_threshold = self.config.get("decay.thresholds.healthy", 0.5)
        fading_threshold = self.config.get("decay.thresholds.fading", 0.2)

        for mem in active:
            r = decay_mod.compute_retrievability(mem.last_accessed, mem.stability, now)
            t = mem.memory_type.value
            r_by_type.setdefault(t, []).append(r)
            if r < fading_threshold:
                forgotten_count += 1
            elif r < healthy_threshold:
                fading_count += 1

        avg_r = {t: sum(rs) / len(rs) if rs else 0.0 for t, rs in r_by_type.items()}
        consolidation_summary = await self.storage.get_consolidation_summary()

        return {
            "counts": {
                "by_type": counts_by_type,
                "by_state": counts_by_state,
            },
            "decay": {
                "avg_retrievability_by_type": avg_r,
                "fading_count": fading_count,
                "forgotten_count": forgotten_count,
            },
            "consolidation": consolidation_summary,
            "storage": {
                "db_size_bytes": await self.storage.get_db_size(),
                "memory_count": await self.storage.get_total_memory_count(),
            },
        }

    # === Health ===

    async def get_health(self) -> dict:
        """Return a full diagnostic health report for the memory store.

        Pure read-only — no side effects. Assembles six storage queries,
        computes fresh retrievability in Python (D1), and returns a structured
        report dict ready for the MCP tool response (D9).
        """
        now = datetime.now(timezone.utc)

        # 1. Counts by type and state
        counts_raw = await self.storage.get_counts_by_type_and_state()
        totals = _build_totals(counts_raw)

        # 2. Decay — fetch active memories, compute fresh retrievability
        active_memories = await self.storage.get_active_memories_for_decay()
        decay_report = _build_decay_report(active_memories, now)

        # 3. Orphans
        untagged, untagged_count = await self.storage.get_orphan_untagged()
        unconnected, unconnected_count = await self.storage.get_orphan_unconnected()

        # 4. Storage gaps
        tag_rows = await self.storage.get_tag_frequencies()
        gaps = _build_gaps(totals, tag_rows, untagged_count)

        # 5. Consolidation — engine owns the None→never_run shape (D4)
        consolidation = await self.storage.get_health_consolidation_summary()
        if consolidation is None:
            consolidation = {"never_run": True, "last_run_at": None, "last_run_summary": None}
        else:
            consolidation["never_run"] = False

        return {
            "generated_at": now.isoformat(),
            "totals": totals,
            "decay": decay_report,
            "orphans": {
                "no_tags": untagged,
                "no_tags_count": untagged_count,
                "no_relations": unconnected,
                "no_relations_count": unconnected_count,
            },
            "gaps": gaps,
            "consolidation": consolidation,
        }

    # === Config ===

    def get_config(self, key: str | None = None) -> dict:
        if key:
            return {"key": key, "value": self.config.get(key)}
        return self.config.get_all()

    def set_config(self, key: str, value) -> None:
        self.config.set(key, value)
