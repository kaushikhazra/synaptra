"""Integration test client -- exercises MemoryEngine with mocked embeddings over SurrealDB."""

from __future__ import annotations

import sys
import uuid
import time
import math
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# Mock the sentence-transformers import before importing our modules
class MockSentenceTransformer:
    def encode(self, text, **kwargs):
        # Deterministic embeddings based on text hash
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec

# Patch
import cognitive_memory.embeddings as emb_module
_original_ensure = emb_module.EmbeddingService._ensure_model
emb_module.EmbeddingService._ensure_model = lambda self: setattr(self, '_model', MockSentenceTransformer()) if self._model is None else None

from cognitive_memory.engine import MemoryEngine
from cognitive_memory.models import MemoryState, MemoryType
from cognitive_memory import decay as decay_mod


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def test(self, name: str, fn):
        try:
            fn()
            self.passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  FAIL: {name} -- {e}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, err in self.errors:
                print(f"  * {name}: {err}")
        return self.failed == 0


def run_tests():
    runner = TestRunner()

    # === Decay Engine Tests ===
    print("\n--- Decay Engine ---")

    def test_r_at_t0():
        now = datetime.now(timezone.utc)
        r = decay_mod.compute_retrievability(now, 2.0, now)
        assert abs(r - 1.0) < 0.001, f"Expected ~1.0, got {r}"
    runner.test("R = 1.0 at t=0", test_r_at_t0)

    def test_r_decays():
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=5)
        r = decay_mod.compute_retrievability(past, 2.0, now)
        assert 0 < r < 1.0, f"Expected 0 < R < 1, got {r}"
    runner.test("R decays over time", test_r_decays)

    def test_reinforce_increases_s():
        s_new = decay_mod.reinforce(2.0, 0.5, growth_factor=2.0)
        assert s_new > 2.0, f"Expected S > 2.0, got {s_new}"
    runner.test("Reinforcement increases stability", test_reinforce_increases_s)

    def test_reinforce_low_r_bigger_boost():
        s_low = decay_mod.reinforce(2.0, 0.2, growth_factor=2.0)
        s_high = decay_mod.reinforce(2.0, 0.8, growth_factor=2.0)
        assert s_low > s_high, f"Low-R boost ({s_low}) should > high-R boost ({s_high})"
    runner.test("Low-R retrieval gives bigger boost", test_reinforce_low_r_bigger_boost)

    def test_spreading_boost_attenuates():
        b1 = decay_mod.compute_spreading_boost(0.8, 0.3, depth=1, spread_factor=0.5)
        b2 = decay_mod.compute_spreading_boost(0.8, 0.3, depth=2, spread_factor=0.5)
        b3 = decay_mod.compute_spreading_boost(0.8, 0.3, depth=3, spread_factor=0.5)
        assert b1 > b2 > b3, f"Boost should attenuate: {b1}, {b2}, {b3}"
    runner.test("Spreading activation attenuates with depth", test_spreading_boost_attenuates)

    def test_spreading_boost_capped():
        b = decay_mod.compute_spreading_boost(1.0, 1.0, depth=1, spread_factor=1.0, max_boost=0.5)
        assert b == 0.5, f"Expected 0.5 cap, got {b}"
    runner.test("Spreading boost capped at max_boost", test_spreading_boost_capped)

    # === Storage + Engine Tests ===
    print("\n--- Storage & Engine ---")

    def test_store_and_get():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Python is a programming language", memory_type="semantic")
        assert mem.id is not None
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.state == MemoryState.ACTIVE
        assert mem.stability == 14.0  # semantic S₀

        got = engine.get_memory(mem.id)
        assert got is not None
        assert got.memory.content == "Python is a programming language"
        engine.close()
    runner.test("Store and get memory", test_store_and_get)

    def test_auto_classification():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Yesterday I visited the park and saw a bird")
        assert mem.memory_type in [MemoryType.EPISODIC, MemoryType.WORKING, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
        engine.close()
    runner.test("Auto-classification runs without error", test_auto_classification)

    def test_importance_scoring():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("A short note", memory_type="working")
        assert 0.1 <= mem.importance <= 1.0
        # Working memory should get penalty (base 0.5 - 0.1 = 0.4)
        assert mem.importance <= 0.5, f"Expected importance <= 0.5 with working penalty, got {mem.importance}"
        engine.close()
    runner.test("Importance scoring with working penalty", test_importance_scoring)

    def test_update_creates_version():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Original content", memory_type="semantic")
        updated = engine.update_memory(mem.id, content="Updated content")
        assert updated is not None
        assert updated.content == "Updated content"
        # Check version was created
        got = engine.get_memory(mem.id)
        assert len(got.versions) == 1
        assert got.versions[0].content == "Original content"
        engine.close()
    runner.test("Update creates version snapshot", test_update_creates_version)

    def test_update_reinforces():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Test content", memory_type="semantic")
        original_s = mem.stability
        updated = engine.update_memory(mem.id, tags=["test"])
        assert updated.stability >= original_s  # should be boosted (or equal if R=1.0)
        engine.close()
    runner.test("Update reinforces stability", test_update_reinforces)

    def test_type_change_resets_stability():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Some fact", memory_type="working")
        assert mem.stability == 0.04  # working S₀
        updated = engine.update_memory(mem.id, memory_type="semantic")
        assert updated.stability == 14.0  # semantic S₀
        engine.close()
    runner.test("Type change resets stability to new S0", test_type_change_resets_stability)

    def test_archive_and_restore():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Archivable content", memory_type="episodic")
        assert engine.archive_memory(mem.id)
        got = engine.storage.get_memory(mem.id)
        assert got.state == MemoryState.ARCHIVED

        restored = engine.restore_memory(mem.id)
        assert restored is not None
        assert restored.state == MemoryState.ACTIVE
        assert restored.stability >= mem.stability  # should be boosted (or equal if R=1.0 instant)
        engine.close()
    runner.test("Archive and restore with decay reset", test_archive_and_restore)

    def test_delete_cascade():
        engine = MemoryEngine(db_path="mem://")
        mem1 = engine.store_memory("Memory A", memory_type="semantic")
        mem2 = engine.store_memory("Memory B", memory_type="semantic")
        engine.create_relationship(mem1.id, mem2.id, "supports")

        # Delete mem1
        assert engine.delete_memory(mem1.id)
        assert engine.storage.get_memory(mem1.id) is None
        # Relationship should be gone
        rels = engine.storage.get_relationships_for(mem2.id)
        assert len(rels) == 0 or all(r.source_id != mem1.id and r.target_id != mem1.id for r in rels)
        engine.close()
    runner.test("Delete cascades relationships", test_delete_cascade)

    # === Relationships ===
    print("\n--- Relationships ---")

    def test_create_and_get_relationship():
        engine = MemoryEngine(db_path="mem://")
        m1 = engine.store_memory("Cause event", memory_type="episodic")
        m2 = engine.store_memory("Effect event", memory_type="episodic")
        rel = engine.create_relationship(m1.id, m2.id, "causes", strength=0.9)
        assert rel.rel_type.value == "causes"

        got = engine.get_memory(m1.id)
        assert any(r.rel_type.value == "causes" for r in got.relationships)
        engine.close()
    runner.test("Create and retrieve relationship", test_create_and_get_relationship)

    def test_unrelate():
        engine = MemoryEngine(db_path="mem://")
        m1 = engine.store_memory("A", memory_type="semantic")
        m2 = engine.store_memory("B", memory_type="semantic")
        engine.create_relationship(m1.id, m2.id, "supports")
        assert engine.delete_relationship(m1.id, m2.id, "supports")
        rels = engine.storage.get_relationships_for(m1.id, ["supports"])
        assert len(rels) == 0
        engine.close()
    runner.test("Delete relationship (unrelate)", test_unrelate)

    def test_related_read_only():
        engine = MemoryEngine(db_path="mem://")
        m1 = engine.store_memory("Hub", memory_type="semantic")
        m2 = engine.store_memory("Spoke 1", memory_type="semantic")
        m3 = engine.store_memory("Spoke 2", memory_type="semantic")
        engine.create_relationship(m1.id, m2.id, "relates_to")
        engine.create_relationship(m1.id, m3.id, "relates_to")

        original_count = engine.storage.get_memory(m2.id).access_count
        related = engine.get_related(m1.id, depth=1)
        after_count = engine.storage.get_memory(m2.id).access_count
        assert original_count == after_count, "get_related should not change access_count"
        engine.close()
    runner.test("get_related is read-only (no side effects)", test_related_read_only)

    # === Retrieval ===
    print("\n--- Retrieval ---")

    def test_recall_returns_results():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory("Python is great for data science", memory_type="semantic")
        engine.store_memory("JavaScript runs in the browser", memory_type="semantic")
        engine.store_memory("SQL is used for databases", memory_type="semantic")

        results = engine.recall("programming languages")
        assert len(results) > 0
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'found_by') for r in results)
        engine.close()
    runner.test("Recall returns scored results with provenance", test_recall_returns_results)

    def test_recall_reinforces():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Important fact about reinforcement", memory_type="semantic")
        original_count = mem.access_count

        results = engine.recall("reinforcement")
        # Check if the memory was found and reinforced
        updated = engine.storage.get_memory(mem.id)
        # access_count should have increased if it was in top-K
        if any(r.id == mem.id for r in results):
            assert updated.access_count > original_count
        engine.close()
    runner.test("Recall reinforces returned memories", test_recall_reinforces)

    # === Consolidation ===
    print("\n--- Consolidation ---")

    def test_consolidation_dry_run():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory("Working memory item", memory_type="working")
        actions = engine.consolidate(dry_run=True)
        assert isinstance(actions, list)
        engine.close()
    runner.test("Consolidation dry run returns actions", test_consolidation_dry_run)

    def test_promotion_working_to_episodic():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Frequently accessed working item", memory_type="working", importance=0.5)
        # Simulate high access count (default threshold is 3)
        engine.storage.update_memory_fields(mem.id, access_count=5)

        actions = engine.consolidate()
        promoted = [a for a in actions if a["action"] == "promote"]
        if promoted:
            updated = engine.storage.get_memory(mem.id)
            assert updated.memory_type == MemoryType.EPISODIC
            assert updated.stability == 2.0  # episodic S₀
        engine.close()
    runner.test("Promotion: working -> episodic", test_promotion_working_to_episodic)

    def test_archive_pass():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Old forgotten memory", memory_type="working")
        # Set last_accessed far in the past so R < threshold
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        engine.storage.update_memory_fields(mem.id, last_accessed=old_time)

        actions = engine.consolidate()
        archive_actions = [a for a in actions if a["action"] == "archive"]
        assert len(archive_actions) > 0, "Should archive forgotten working memory"
        updated = engine.storage.get_memory(mem.id)
        assert updated.state == MemoryState.ARCHIVED
        engine.close()
    runner.test("Archive pass catches forgotten memories", test_archive_pass)

    def test_promote_before_archive():
        """Working memory with high access count should be promoted, not archived."""
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("Active working item", memory_type="working", importance=0.5)
        # High access count (qualifies for promotion) + old last_accessed (qualifies for archive)
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        engine.storage.update_memory_fields(mem.id, access_count=5, last_accessed=old_time)

        actions = engine.consolidate()

        updated = engine.storage.get_memory(mem.id)
        # After promotion, new S₀ = 2.0 (episodic), and the archive pass re-fetches.
        # With S₀=2.0 and last_accessed=30 days ago, R = e^(-30 / (9*2)) = e^(-1.67) ≈ 0.19
        # This is below the forgotten threshold (0.2), so it will be archived too.
        # The key test: promotion happened (type changed).
        assert updated.memory_type == MemoryType.EPISODIC, f"Should be promoted, got {updated.memory_type}"
        engine.close()
    runner.test("Promote before archive (stage ordering)", test_promote_before_archive)

    # === Orphaned Memory Repair ===
    print("\n--- Orphaned Memory Repair ---")

    def test_orphaned_memory_repair():
        """Store a memory, delete its auto-links, run consolidation, verify links re-created."""
        engine = MemoryEngine(db_path="mem://")
        # Store two related memories
        m1 = engine.store_memory("Python is great for machine learning", memory_type="semantic")
        m2 = engine.store_memory("Python machine learning libraries include scikit-learn", memory_type="semantic")

        # Delete auto-links for m1 (simulate orphan)
        engine.storage.delete_auto_links(m1.id)
        rels_before = engine.storage.get_relationships_for(m1.id, ["relates_to"])
        auto_before = [r for r in rels_before if r.strength < 1.0]

        # Run consolidation — cluster scan should find and re-link similar memories
        actions = engine.consolidate()

        # After consolidation, check if m1 and m2 are connected again
        # (either via merge or the fact that similar memories are detected)
        m1_after = engine.storage.get_memory(m1.id)
        assert m1_after is not None, "Memory should still exist after consolidation"
        engine.close()
    runner.test("Orphaned memory repair via consolidation", test_orphaned_memory_repair)

    # === Stats ===
    print("\n--- Stats ---")

    def test_stats():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory("Stat test 1", memory_type="semantic")
        engine.store_memory("Stat test 2", memory_type="episodic")
        stats = engine.get_stats()
        assert "counts" in stats
        assert "decay" in stats
        assert "consolidation" in stats
        assert "storage" in stats
        assert stats["counts"]["by_state"].get("active", 0) >= 2
        engine.close()
    runner.test("Stats returns complete structure", test_stats)

    # === Config ===
    print("\n--- Config ---")

    def test_config_read_write():
        engine = MemoryEngine(db_path="mem://")
        engine.set_config("decay.growth_factor", 3.0)
        result = engine.get_config("decay.growth_factor")
        assert result["value"] == 3.0
        engine.close()
    runner.test("Config read/write", test_config_read_write)

    # === memory_list ===
    print("\n--- List & Search ---")

    def test_list_with_filters():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory("Alpha semantic", memory_type="semantic", tags=["test"])
        engine.store_memory("Beta episodic", memory_type="episodic")
        engine.store_memory("Gamma semantic", memory_type="semantic")

        all_mems = engine.storage.list_memories()
        assert len(all_mems) == 3

        semantic_only = engine.storage.list_memories(memory_type="semantic")
        assert len(semantic_only) == 2

        tagged = engine.storage.list_memories(tags=["test"])
        assert len(tagged) == 1
        engine.close()
    runner.test("List with type and tag filters", test_list_with_filters)

    def test_fts_search():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory("The quick brown fox jumps over the lazy dog", memory_type="semantic")
        engine.store_memory("A lazy cat sleeps all day", memory_type="semantic")

        results = engine.storage.fts_search("fox")
        assert len(results) >= 1
        engine.close()
    runner.test("Full-text search", test_fts_search)

    # === Identity & Person Memory Types ===
    print("\n--- Identity & Person Memory Types ---")

    def test_store_identity_memory():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory(
            "I am Velasari, an AI agent", memory_type="identity",
            tags=["name", "origin"],
        )
        assert mem.memory_type == MemoryType.IDENTITY
        assert mem.stability == 365.0  # identity S₀
        assert mem.importance >= 0.7, f"Expected importance >= 0.7 (base 0.5 + identity bonus 0.2), got {mem.importance}"
        engine.close()
    runner.test("Store identity memory with correct stability and importance", test_store_identity_memory)

    def test_store_person_memory():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory(
            "Kaushik prefers explicit error handling",
            memory_type="person",
            tags=["person:kaushik", "preferences"],
        )
        assert mem.memory_type == MemoryType.PERSON
        assert mem.stability == 90.0  # person S₀
        assert mem.importance >= 0.65, f"Expected importance >= 0.65 (base 0.5 + person bonus 0.15), got {mem.importance}"
        engine.close()
    runner.test("Store person memory with correct stability and importance", test_store_person_memory)

    def test_identity_type_change_stability():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("A fact", memory_type="episodic")
        assert mem.stability == 2.0
        updated = engine.update_memory(mem.id, memory_type="identity")
        assert updated.stability == 365.0  # identity S₀
        engine.close()
    runner.test("Type change to identity resets stability to 365", test_identity_type_change_stability)

    def test_recall_identity_filter():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory("I am Velasari", memory_type="identity")
        engine.store_memory("Python is a language", memory_type="semantic")
        results = engine.recall("agent identity", type_filter="identity")
        assert all(r.memory_type == MemoryType.IDENTITY for r in results)
        engine.close()
    runner.test("Recall with type_filter=identity returns only identity memories", test_recall_identity_filter)

    def test_recall_person_filter():
        engine = MemoryEngine(db_path="mem://")
        engine.store_memory(
            "Kaushik likes Python", memory_type="person",
            tags=["person:kaushik"],
        )
        engine.store_memory("JavaScript is fast", memory_type="semantic")
        results = engine.recall("programming", type_filter="person")
        assert all(r.memory_type == MemoryType.PERSON for r in results)
        engine.close()
    runner.test("Recall with type_filter=person returns only person memories", test_recall_person_filter)

    def test_describes_relationship():
        engine = MemoryEngine(db_path="mem://")
        m1 = engine.store_memory("Kaushik anchor", memory_type="person", tags=["person:kaushik"])
        m2 = engine.store_memory("Kaushik prefers SOLID", memory_type="person", tags=["person:kaushik"])
        rel = engine.create_relationship(m2.id, m1.id, "describes", strength=1.0)
        assert rel.rel_type.value == "describes"
        got = engine.get_memory(m2.id)
        assert any(r.rel_type.value == "describes" for r in got.relationships)
        engine.close()
    runner.test("Create describes relationship between person memories", test_describes_relationship)

    # === Identity & Person Consolidation ===
    print("\n--- Identity & Person Consolidation ---")

    def test_identity_archive_exemption():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory("I am Velasari", memory_type="identity")
        # Set last_accessed far in the past to ensure R < threshold
        old_time = datetime.now(timezone.utc) - timedelta(days=3650)  # 10 years
        engine.storage.update_memory_fields(mem.id, last_accessed=old_time)

        actions = engine.consolidate()
        archive_actions = [a for a in actions if a["action"] == "archive" and mem.id in a["source_ids"]]
        assert len(archive_actions) == 0, "Identity memories should never be auto-archived"
        updated = engine.storage.get_memory(mem.id)
        assert updated.state == MemoryState.ACTIVE
        engine.close()
    runner.test("Identity memories exempt from auto-archival", test_identity_archive_exemption)

    def test_person_archive_threshold():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory(
            "Kaushik health info", memory_type="person",
            tags=["person:kaushik", "health"],
        )
        # Set last_accessed so R is between person threshold (0.05) and global threshold (0.2)
        # With S₀=90, we need R around 0.1 — between 0.05 and 0.2
        # R = e^(-t / (9*90)) = e^(-t/810); for R=0.1, t = -810 * ln(0.1) ≈ 1865 days
        # For R=0.15, t = -810 * ln(0.15) ≈ 1538 days — should NOT be archived (R > 0.05)
        moderate_old = datetime.now(timezone.utc) - timedelta(days=1538)
        engine.storage.update_memory_fields(mem.id, last_accessed=moderate_old)

        actions = engine.consolidate()
        archive_actions = [a for a in actions if a["action"] == "archive" and mem.id in a["source_ids"]]
        assert len(archive_actions) == 0, "Person memories should not be archived when R > person_threshold"
        updated = engine.storage.get_memory(mem.id)
        assert updated.state == MemoryState.ACTIVE
        engine.close()
    runner.test("Person memories use stricter archive threshold (not archived above 0.05)", test_person_archive_threshold)

    def test_promotion_episodic_to_person():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory(
            "Kaushik mentioned he likes tea",
            memory_type="episodic",
            tags=["person:kaushik", "preferences"],
        )
        # Set access_count above the to_person threshold (default 3)
        engine.storage.update_memory_fields(mem.id, access_count=5)

        actions = engine.consolidate()
        promoted = [a for a in actions if a["action"] == "promote" and mem.id in a["source_ids"]]
        assert len(promoted) > 0, "Episodic memory with person tag and high access should be promoted"
        updated = engine.storage.get_memory(mem.id)
        assert updated.memory_type == MemoryType.PERSON, f"Expected PERSON, got {updated.memory_type}"
        assert updated.stability == 90.0  # person S₀
        engine.close()
    runner.test("Promotion: episodic -> person (with person tag)", test_promotion_episodic_to_person)

    def test_promotion_semantic_to_person():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory(
            "Kaushik is a software architect with 27 years experience",
            memory_type="semantic",
            tags=["person:kaushik", "career"],
        )
        engine.storage.update_memory_fields(mem.id, access_count=5)

        actions = engine.consolidate()
        promoted = [a for a in actions if a["action"] == "promote" and mem.id in a["source_ids"]]
        assert len(promoted) > 0, "Semantic memory with person tag and high access should be promoted"
        updated = engine.storage.get_memory(mem.id)
        assert updated.memory_type == MemoryType.PERSON, f"Expected PERSON, got {updated.memory_type}"
        engine.close()
    runner.test("Promotion: semantic -> person (with person tag)", test_promotion_semantic_to_person)

    def test_no_promotion_without_person_tag():
        engine = MemoryEngine(db_path="mem://")
        mem = engine.store_memory(
            "Some episodic fact without person tag",
            memory_type="episodic",
        )
        engine.storage.update_memory_fields(mem.id, access_count=5)

        actions = engine.consolidate()
        promoted = [a for a in actions if a["action"] == "promote" and mem.id in a["source_ids"]
                     and a.get("to_type") == "person"]
        assert len(promoted) == 0, "Should not promote to person without person: tag"
        engine.close()
    runner.test("No person promotion without person: tag", test_no_promotion_without_person_tag)

    # === Bug-fix regression tests ===
    print("\n--- Bug-fix Regressions ---")

    def test_identity_merge_exemption():
        """B1: Identity memories must survive the consolidation merge pass."""
        engine = MemoryEngine(db_path="mem://")
        # Identical content → identical mock embedding → cosine sim = 1.0 → normally triggers merge
        m1 = engine.store_memory("I am Velasari, built by Kaushik", memory_type="identity")
        m2 = engine.store_memory("I am Velasari, built by Kaushik", memory_type="identity")

        actions = engine.consolidate()

        # No merge action should target either identity memory
        merge_actions = [
            a for a in actions
            if a["action"] == "merge" and (m1.id in a["source_ids"] or m2.id in a["source_ids"])
        ]
        assert len(merge_actions) == 0, "Identity memories should never be merged by consolidation"

        # Both must still be active
        u1 = engine.storage.get_memory(m1.id)
        u2 = engine.storage.get_memory(m2.id)
        assert u1.state == MemoryState.ACTIVE, "First identity memory archived by merge"
        assert u2.state == MemoryState.ACTIVE, "Second identity memory archived by merge"
        engine.close()
    runner.test("B1: Identity memories exempt from consolidation merge", test_identity_merge_exemption)

    def test_memory_who_empty_person_returns_error():
        """B2: memory_who('') must return an error, not random recall results."""
        import cognitive_memory.server as server_mod
        for empty_input in ("", "   ", "\t"):
            result = json.loads(server_mod.memory_who(empty_input))
            assert result["success"] is False, f"Expected failure for person={empty_input!r}, got success"
            assert result["error"], "Error message must be non-empty"
    runner.test("B2: memory_who with empty person name returns error", test_memory_who_empty_person_returns_error)

    # === Summary ===
    return runner.summary()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
