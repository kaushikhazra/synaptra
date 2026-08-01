"""Tests for provisioning script idempotency.

Verifies that schema.surql applied via the provisioning logic is safe to run
twice on the same database instance — the second run must be a no-op with
no errors (guaranteed by IF NOT EXISTS throughout schema.surql).
"""

from pathlib import Path

import pytest
from surrealdb import Surreal


SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "cognitive_memory"
    / "schema.surql"
)


def _run_schema(db: Surreal, schema_sql: str) -> list[tuple[str, str]]:
    """Execute each semicolon-delimited statement from schema_sql against db.

    Returns a list of (statement_preview, error_message) for any failures.
    An empty list means all statements succeeded.
    """
    errors: list[tuple[str, str]] = []
    for stmt in schema_sql.split(";"):
        lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
        clean = "\n".join(lines).strip()
        if not clean:
            continue
        try:
            db.query(clean)
        except Exception as exc:
            errors.append((clean[:80], str(exc)))
    return errors


def test_provision_idempotent():
    """Provisioning logic is safe to run twice on the same mem:// DB instance.

    Both runs must complete with zero errors. The second run exercises
    IF NOT EXISTS idempotency: every DEFINE statement must be a no-op when
    the schema already exists.
    """
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    db = Surreal("mem://")
    db.connect()
    db.use("cognitive", "memory")

    try:
        # First run — creates all tables, fields, indexes, analyzers
        errors1 = _run_schema(db, schema_sql)
        assert errors1 == [], f"First provision run had errors: {errors1}"

        # Verify schema exists after first run: memory table must be present
        info_result = db.query("INFO FOR DB")
        info = info_result[0] if isinstance(info_result, list) else info_result
        if isinstance(info, dict):
            tables = info.get("tables", {})
            assert "memory" in str(tables), (
                f"'memory' table not found in INFO FOR DB after first provision.\n"
                f"Tables: {tables}"
            )

        # Second run — all IF NOT EXISTS statements must skip silently (no-op)
        errors2 = _run_schema(db, schema_sql)
        assert errors2 == [], f"Second provision run had errors (idempotency broken): {errors2}"

    finally:
        db.close()
