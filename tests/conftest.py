"""Shared test fixtures."""

from __future__ import annotations

import _pytest.pathlib
import _pytest.tmpdir


def _ignore_dead_symlink_cleanup(root) -> None:
    """Avoid Windows sandbox permission failures in pytest temp cleanup."""
    return None


_pytest.pathlib.cleanup_dead_symlinks = _ignore_dead_symlink_cleanup
_pytest.tmpdir.cleanup_dead_symlinks = _ignore_dead_symlink_cleanup
