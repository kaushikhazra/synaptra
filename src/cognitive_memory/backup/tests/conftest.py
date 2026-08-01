"""Shared fixtures for backup tests.

Provides a synthetic 50-memory SurrealKV database for round-trip testing.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


def make_embedding(dim: int = 384) -> list[float]:
    """Generate a random unit-normalized embedding vector."""
    rng = [random.gauss(0, 1) for _ in range(dim)]
    magnitude = sum(x ** 2 for x in rng) ** 0.5
    return [x / magnitude for x in rng]


def make_memory(i: int) -> dict:
    """Create a synthetic memory record dict."""
    mem_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    types = ["working", "episodic", "semantic", "procedural", "identity"]
    return {
        "id": mem_id,
        "content": f"Test memory {i}: This is synthetic content for testing purposes. Index={i}",
        "memory_type": types[i % len(types)],
        "state": "active",
        "importance": round(random.uniform(0.1, 1.0), 4),
        "stability": round(random.uniform(1.0, 20.0), 4),
        "retrievability": round(random.uniform(0.5, 1.0), 4),
        "access_count": random.randint(0, 50),
        "source": "test",
        "conversation_id": f"conv-{i}",
        "tags": [f"tag-{i}", "test"],
        "embedding": make_embedding(),
        "created_at": now,
        "updated_at": now,
        "last_accessed": now,
    }


@pytest.fixture(scope="session")
def synthetic_memories() -> list[dict]:
    """50 synthetic memory records."""
    random.seed(42)
    return [make_memory(i) for i in range(50)]


@pytest.fixture
def surreal_db_with_memories(synthetic_memories, tmp_path):
    """Populate a SurrealKV instance with 50 synthetic memories + a few edges.

    Yields (db_path, memories) — the connection is CLOSED before yielding so
    tests that need to open the same path (e.g., via the exporter) can do so
    without file-lock conflicts. The connection is garbage-collected before yield.
    """
    import gc
    from surrealdb import Surreal
    from pathlib import Path

    schema_path = Path(__file__).parent.parent.parent / "schema.surql"

    db_path = tmp_path / "test-db"
    db_url = f"surrealkv://{str(db_path).replace(os.sep, '/')}"

    db = Surreal(db_url)
    db.connect()
    db.use("cognitive", "memory")

    # Apply schema
    schema_text = schema_path.read_text(encoding="utf-8")
    for stmt in schema_text.split(";"):
        lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
        clean = "\n".join(lines).strip()
        if clean:
            try:
                db.query(clean)
            except Exception:
                pass

    # Insert memories with datetime objects for datetime fields
    for mem in synthetic_memories:
        coerced = dict(mem)
        for field in ("created_at", "updated_at", "last_accessed"):
            if isinstance(coerced.get(field), str):
                coerced[field] = datetime.fromisoformat(coerced[field])
        db.query(
            "CREATE type::thing('memory', $id) CONTENT $content",
            {"id": coerced["id"], "content": coerced},
        )

    # Insert a few relates_to edges between first 5 memories
    mems = synthetic_memories
    edges = [
        (mems[0]["id"], mems[1]["id"]),
        (mems[1]["id"], mems[2]["id"]),
        (mems[2]["id"], mems[3]["id"]),
    ]
    for src, tgt in edges:
        db.query(
            """LET $from = type::thing('memory', $src);
               LET $to = type::thing('memory', $tgt);
               RELATE $from->relates_to->$to
               SET strength = $strength, created_at = $created_at""",
            {
                "src": src,
                "tgt": tgt,
                "strength": 0.8,
                "created_at": datetime.now(timezone.utc),
            },
        )

    # Close and garbage-collect the connection before yielding so callers can
    # open the same path without file-lock conflicts.
    try:
        db.close()
    except Exception:
        pass
    del db
    gc.collect()

    yield db_path, synthetic_memories
