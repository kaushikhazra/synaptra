"""Repository-wide pytest fixtures."""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest


_TMP_ROOT = Path(__file__).parent / "temp_pytest_runtime"
_TMP_ROOT.mkdir(exist_ok=True)


@pytest.fixture
def tmp_path() -> Path:
    """Provide a repo-local temp directory to avoid Windows sandbox temp issues."""
    path = _TMP_ROOT / f"pytest-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
