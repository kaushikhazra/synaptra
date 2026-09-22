"""Regression tests for the spreading-activation stability boost.

Written 2026-08-29 after `apply_spreading_boost` was found compounding a
neighbour's stability by 1.3x on every recall, unconditionally and without a
ceiling, while the memory actually retrieved gained exactly zero at R = 1.0.
Live values had reached 5.4e25.

The defect survived five months of daily use with a fully green suite, because
the suite tested `compute_spreading_boost` (how large the boost is) and never
`apply_spreading_boost` (what happens when you apply it).  These tests cover the
second question.
"""

from __future__ import annotations

import math

import pytest

from synaptra import decay as decay_mod


# ── The R term ────────────────────────────────────────────────────────────────


def test_fully_retrievable_neighbour_gains_nothing():
    """R = 1.0 means nothing to consolidate — the same rule `reinforce` follows.

    This is the assertion whose absence let the runaway happen.
    """
    assert decay_mod.apply_spreading_boost(100.0, 0.3, retrievability=1.0) == 100.0


def test_faded_neighbour_gains_the_full_boost():
    assert decay_mod.apply_spreading_boost(
        100.0, 0.3, retrievability=0.0
    ) == pytest.approx(130.0)


def test_boost_scales_with_how_faded_the_neighbour_is():
    high = decay_mod.apply_spreading_boost(100.0, 0.3, retrievability=0.9)
    low = decay_mod.apply_spreading_boost(100.0, 0.3, retrievability=0.1)
    assert low > high > 100.0


def test_retrievability_is_clamped():
    assert decay_mod.apply_spreading_boost(100.0, 0.3, retrievability=5.0) == 100.0
    assert decay_mod.apply_spreading_boost(
        100.0, 0.3, retrievability=-5.0
    ) == pytest.approx(130.0)


def test_boost_is_never_negative():
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert decay_mod.apply_spreading_boost(50.0, 0.3, retrievability=r) >= 50.0


# ── The ceiling ───────────────────────────────────────────────────────────────


def test_ceiling_caps_growth():
    out = decay_mod.apply_spreading_boost(100.0, 0.5, retrievability=0.0, ceiling=110.0)
    assert out == 110.0


def test_ceiling_never_shrinks_a_memory_already_above_it():
    """A memory legitimately reinforced past the ceiling must not be pulled down.

    `apply_spreading_boost` applies a *boost*; a boost that reduces stability is
    a different operation wearing this one's name.
    """
    out = decay_mod.apply_spreading_boost(500.0, 0.3, retrievability=0.0, ceiling=110.0)
    assert out == 500.0


def test_no_ceiling_means_no_cap():
    out = decay_mod.apply_spreading_boost(100.0, 0.5, retrievability=0.0, ceiling=None)
    assert out == pytest.approx(150.0)


# ── The runaway itself ────────────────────────────────────────────────────────


def _simulate_recalls(
    *,
    initial_stability: float,
    n_recalls: int,
    boost: float = 0.3,
    days_between_recalls: float = 1.0,
    ceiling: float | None = None,
) -> float:
    """Repeatedly boost one neighbour, the way a daily session actually does.

    Models the real loop faithfully in the one respect that matters: a spreading
    boost writes stability and does NOT touch ``last_accessed``.  So the
    neighbour's elapsed time keeps growing while its stability is lifted, and
    R has to be recomputed from both on every pass.
    """
    stability = initial_stability
    for i in range(1, n_recalls + 1):
        elapsed_days = i * days_between_recalls
        r = math.exp(-elapsed_days / (9.0 * stability))
        stability = decay_mod.apply_spreading_boost(stability, boost, r, ceiling)
    return stability


def test_repeated_recalls_do_not_run_away():
    """139 daily recalls is what took a `person` memory from 90 to 6.08e17.

    Under the old rule this returned 90 * 1.3**139 ~ 6e17.  It must now stay
    within touching distance of where it started.
    """
    final = _simulate_recalls(initial_stability=90.0, n_recalls=139)
    assert final < 90.0 * 10, f"neighbour stability ran away to {final:.4g}"


def test_five_months_of_daily_recalls_stays_bounded_for_every_type():
    for mem_type in (
        "working",
        "episodic",
        "semantic",
        "procedural",
        "identity",
        "person",
    ):
        s0 = decay_mod.get_initial_stability(mem_type)
        ceiling = (
            10.0 * s0
        )  # production default, spreading_activation.stability_ceiling_multiple
        final = _simulate_recalls(initial_stability=s0, n_recalls=150, ceiling=ceiling)
        assert final <= ceiling, (
            f"{mem_type}: {final:.4g} exceeded ceiling {ceiling:.4g}"
        )


def test_r_term_reduces_growth_from_exponential_to_quadratic():
    """The R term changes the growth ORDER. It does not stop growth.

    A spreading boost does not update the neighbour's ``last_accessed``, so
    elapsed time keeps rising and pushes R back down even as growing S pushes it
    up.  Elapsed wins slowly, and S settles on ``n**2 / 60`` at boost 0.3.
    """
    for n in (200, 500, 1000, 2000):
        s = _simulate_recalls(initial_stability=14.0, n_recalls=n, ceiling=None)
        assert s / (n**2) == pytest.approx(1.0 / 60.0, rel=0.05), (
            f"n={n}: expected quadratic growth ~n^2/60, got {s:.4g}"
        )
        # ...and astronomically below what the unguarded rule produced.
        assert s < 14.0 * (1.3**n)


def test_r_term_alone_does_not_bound_growth_so_the_ceiling_is_required():
    """Guards against someone later concluding the ceiling is redundant.

    It is not.  Quadratic is smaller than exponential and still unbounded — at
    2000 recalls an unceilinged neighbour reaches ~6.7e4 from a start of 14.
    """
    unbounded = _simulate_recalls(initial_stability=14.0, n_recalls=2000, ceiling=None)
    assert unbounded > 14.0 * 100, (
        "if this now passes under a ceiling-free run, the R term changed and the "
        "ceiling rationale needs revisiting"
    )

    ceilinged = _simulate_recalls(
        initial_stability=14.0, n_recalls=2000, ceiling=10.0 * 14.0
    )
    assert ceilinged <= 10.0 * 14.0


def test_a_neighbour_never_overtakes_a_directly_retrieved_memory():
    """The defect inverted the intended relationship — neighbours outgrew seeds.

    Under identical conditions, being retrieved must be worth at least as much
    as being adjacent to something retrieved.
    """
    stability, r = 14.0, 0.4
    seed = decay_mod.reinforce(stability, r, growth_factor=2.0)
    neighbour = decay_mod.apply_spreading_boost(stability, 0.3, r)
    assert neighbour <= seed
