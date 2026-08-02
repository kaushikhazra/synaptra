"""FSRS-inspired dual-strength decay engine. Pure functions, no side effects."""

from __future__ import annotations

import math
from datetime import datetime, timezone


def compute_retrievability(last_accessed: datetime, stability: float, now: datetime | None = None) -> float:
    """Compute R on-the-fly. R(t) = e^(-t / (9 * S)). Returns 0.0-1.0."""
    if now is None:
        now = datetime.now(timezone.utc)
    elapsed_seconds = max(0, (now - last_accessed).total_seconds())
    elapsed_days = elapsed_seconds / 86400.0
    if stability <= 0:
        return 0.0
    return math.exp(-elapsed_days / (9.0 * stability))


def reinforce(stability: float, retrievability: float, growth_factor: float = 2.0) -> float:
    """Apply reinforcement on access. Memories retrieved at low R get bigger boost."""
    r_clamped = max(0.0, min(1.0, retrievability))
    return stability * (1.0 + growth_factor * (1.0 - r_clamped))


def get_initial_stability(memory_type: str, config_map: dict[str, float] | None = None) -> float:
    """Get S₀ for a memory type."""
    defaults = {
        "working": 0.04,
        "episodic": 2.0,
        "semantic": 14.0,
        "procedural": 60.0,
        "identity": 365.0,
        "person": 90.0,
    }
    if config_map:
        return config_map.get(memory_type, defaults.get(memory_type, 2.0))
    return defaults.get(memory_type, 2.0)


def compute_spreading_boost(
    relationship_strength: float,
    activation_strength: float = 0.3,
    depth: int = 1,
    spread_factor: float = 0.5,
    max_boost: float = 0.5,
) -> float:
    """Compute stability boost for a neighbor at given hop depth.

    1-hop: boost = activation_strength * relationship_strength
    2-hop: boost = (1-hop) * spread_factor
    N-hop: boost = (N-1 hop) * spread_factor

    Capped at max_boost.
    """
    if depth < 1:
        return 0.0
    base_boost = activation_strength * relationship_strength
    boost = base_boost * (spread_factor ** (depth - 1))
    return min(boost, max_boost)


def apply_spreading_boost(stability: float, boost: float) -> float:
    """Apply spreading activation boost to a neighbor's stability.
    S_new = S_old * (1 + boost)
    """
    return stability * (1.0 + boost)


def classify_decay_state(retrievability: float, healthy_threshold: float = 0.5, fading_threshold: float = 0.2) -> str:
    """Classify a memory's decay health: healthy, fading, or forgotten."""
    if retrievability > healthy_threshold:
        return "healthy"
    elif retrievability > fading_threshold:
        return "fading"
    else:
        return "forgotten"
