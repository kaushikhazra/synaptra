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


def apply_spreading_boost(
    stability: float,
    boost: float,
    retrievability: float,
    ceiling: float | None = None,
) -> float:
    """Apply spreading activation boost to a neighbor's stability.

    S_new = min(S_old * (1 + boost * (1 - R)), max(ceiling, S_old))

    Two guards.  BOTH are required — that is the non-obvious part.

    ``retrievability`` mirrors :func:`reinforce`.  A neighbour that is already
    fully retrievable has nothing to consolidate, so it gains nothing — exactly
    as a *retrieved* memory at R = 1.0 gains nothing.  Without this term the
    boost is unconditional and compounds on every recall forever: the defect
    that took live stability values to 5.4e25 while the memories actually being
    retrieved gained zero.

    ``ceiling`` is NOT merely defensive, and it is tempting to assume it is.
    The R term looks self-limiting — growth in S drives R toward 1.0, which
    drives the boost toward 0 — but a spreading boost does not update the
    neighbour's ``last_accessed``, so elapsed time keeps growing too and pushes
    R back down.  The two race, and elapsed wins slowly.  Measured over n
    recalls at boost 0.3, S converges on ``n**2 / 60``:

        n=100 -> 175      n=500 -> 4.1e3      n=2000 -> 6.7e4

    So the R term reduces the growth order from exponential to quadratic — at
    n=2000 that is 1.1e229 down to 6.7e4 — and quadratic is still unbounded.
    The ceiling is what actually bounds it, and the multiplier is chosen to
    preserve the decay class rather than to be generous: at 10x a type's
    initial stability a semantic memory halves at ~2.4 years, where at 100x it
    would take ~24 and stop being mortal at all.

    Applied as ``max(ceiling, stability)``, so it caps *growth* and can never
    shrink a memory that legitimately reinforced its way above it — a boost
    must never be negative.
    """
    r_clamped = max(0.0, min(1.0, retrievability))
    new_stability = stability * (1.0 + boost * (1.0 - r_clamped))
    if ceiling is not None:
        new_stability = min(new_stability, max(ceiling, stability))
    return new_stability


def classify_decay_state(retrievability: float, healthy_threshold: float = 0.5, fading_threshold: float = 0.2) -> str:
    """Classify a memory's decay health: healthy, fading, or forgotten."""
    if retrievability > healthy_threshold:
        return "healthy"
    elif retrievability > fading_threshold:
        return "fading"
    else:
        return "forgotten"
