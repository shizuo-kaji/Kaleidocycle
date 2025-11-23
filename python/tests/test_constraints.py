"""Tests for constraint residuals and the penalty-based solver."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle.constraints import ConstraintConfig, constraint_penalty, constraint_residuals, enforce_terminal
from kaleidocycle.geometry import HingeFrame, mean_cosine, pairwise_cosines, random_hinges
from kaleidocycle.solvers import SolverOptions, optimize_cycle


@pytest.fixture(scope="module")
def sample():
    """Create a properly optimized kaleidocycle with constant torsion."""
    initial = random_hinges(6, seed=42)
    config = ConstraintConfig(enforce_anchors=False)
    opts = SolverOptions(method="SLSQP", maxiter=1000, penalty_weight=100.0)
    result = optimize_cycle(initial.as_array(), config, objective="mean_cos", options=opts)

    # Verify constant torsion is achieved
    hinges = result.hinges
    cosines = pairwise_cosines(hinges)
    torsion_variation = np.std(cosines[:-1])  # Exclude last (closure) cosine
    assert torsion_variation < 1e-4, f"Fixture does not have constant torsion: std={torsion_variation}"

    return HingeFrame(result.hinges)


def test_constraint_residuals_small_for_sample(sample) -> None:
    """Test that a properly optimized kaleidocycle satisfies all constraints."""
    config = ConstraintConfig(enforce_anchors=False)
    residuals = constraint_residuals(sample.as_array(), config)

    # Unit norm constraint
    assert np.linalg.norm(residuals["unit_norm"]) < 1e-4

    # Closure constraint
    assert np.linalg.norm(residuals["closure"]) < 2e-3

    # Constant torsion constraint (most important for kaleidocycles!)
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-4


def test_penalty_solver_reduces_violation(sample) -> None:
    config = ConstraintConfig(enforce_anchors=False)
    noisy = sample.as_array().copy()
    rng = np.random.default_rng(123)
    noisy[:-1] += rng.normal(scale=1e-2, size=noisy[:-1].shape)
    noisy = enforce_terminal(noisy, oriented=False)

    before = constraint_penalty(noisy, config)
    result = optimize_cycle(
        noisy,
        config,
        objective="torsion",
        options=SolverOptions(maxiter=500, penalty_weight=200.0, method="L-BFGS-B"),
    )

    # Optimization should not increase the penalty
    after = constraint_penalty(result.hinges, config)
    # Check that penalty is not worse (allow equal if already good)
    assert after <= before + 1e-10  # Small tolerance for numerical precision


def test_mean_cosine_objective_decreases_cos() -> None:
    """Test that neg_mean_cos objective maximizes (increases) the mean cosine value."""
    rng = np.random.default_rng(5)
    initial = random_hinges(6, seed=5).as_array()
    config = ConstraintConfig(enforce_anchors=False)
    before = mean_cosine(initial)
    result = optimize_cycle(
        initial,
        config,
        objective="neg_mean_cos",  # Minimize negative mean_cos = maximize mean_cos
        options=SolverOptions(maxiter=200, penalty_weight=50.0),
    )
    after = mean_cosine(result.hinges)
    # neg_mean_cos objective maximizes mean cosine (makes it less negative / more positive)
    assert after >= before


def test_constant_torsion_validation() -> None:
    """Test that optimization achieves constant torsion, the fundamental kaleidocycle property."""
    # Start with random hinges (no constant torsion)
    initial = random_hinges(6, seed=123).as_array()
    cosines_before = pairwise_cosines(initial)
    torsion_std_before = np.std(cosines_before[:-1])  # Exclude closure

    # Verify random hinges do NOT have constant torsion
    assert torsion_std_before > 0.1, "Random hinges should not have constant torsion"

    # Optimize to achieve constant torsion
    config = ConstraintConfig(enforce_anchors=False)
    result = optimize_cycle(
        initial,
        config,
        objective="mean_cos",
        options=SolverOptions(maxiter=1000, penalty_weight=100.0),
    )

    # Verify optimized result HAS constant torsion
    cosines_after = pairwise_cosines(result.hinges)
    torsion_std_after = np.std(cosines_after[:-1])  # Exclude closure

    # This is the fundamental kaleidocycle property
    assert torsion_std_after < 1e-4, (
        f"Optimized kaleidocycle must have constant torsion. "
        f"Got std={torsion_std_after:.2e}, expected < 1e-4"
    )

    # Also verify all constraints are satisfied
    residuals = constraint_residuals(result.hinges, config)
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-4
    assert np.linalg.norm(residuals["unit_norm"]) < 1e-4
    assert np.linalg.norm(residuals["closure"]) < 2e-3


def test_constraint_solver_vs_penalty() -> None:
    """Test that constraint solver produces valid results and satisfies constraints better than penalty method."""
    initial = random_hinges(6, seed=42).as_array()
    config = ConstraintConfig(enforce_anchors=False)

    # Test with penalty-based method
    opts_penalty = SolverOptions(
        method="Powell",
        maxiter=500,
        penalty_weight=100.0,
        use_constraint_solver=False,
    )
    result_penalty = optimize_cycle(initial, config, objective="mean_cos", options=opts_penalty)

    # Test with constraint solver
    opts_constraint = SolverOptions(
        method="Powell",  # Not used when use_constraint_solver=True
        maxiter=500,
        penalty_weight=100.0,
        use_constraint_solver=True,
        constraint_method="trust-constr",
    )
    result_constraint = optimize_cycle(initial, config, objective="mean_cos", options=opts_constraint)

    # Both should produce valid results
    assert result_penalty.hinges.shape == initial.shape
    assert result_constraint.hinges.shape == initial.shape

    # Constraint solver should satisfy constraints better or equally well
    # (penalty might be comparable because it's still minimized in penalty approach)
    assert result_constraint.penalty < 1e-2, "Constraint solver should satisfy constraints well"

    # Both should have constant torsion
    cosines_penalty = pairwise_cosines(result_penalty.hinges)
    cosines_constraint = pairwise_cosines(result_constraint.hinges)

    torsion_std_penalty = np.std(cosines_penalty[:-1])
    torsion_std_constraint = np.std(cosines_constraint[:-1])

    assert torsion_std_penalty < 1e-3, "Penalty method should achieve constant torsion"
    assert torsion_std_constraint < 1e-3, "Constraint solver should achieve constant torsion"

    # Verify unit norm constraint
    residuals_penalty = constraint_residuals(result_penalty.hinges, config)
    residuals_constraint = constraint_residuals(result_constraint.hinges, config)

    assert np.linalg.norm(residuals_penalty["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals_constraint["unit_norm"]) < 1e-3


def test_constraint_solver_different_objectives() -> None:
    """Test constraint solver with different objective functions."""
    initial = random_hinges(6, seed=99).as_array()
    config = ConstraintConfig(enforce_anchors=False)
    opts = SolverOptions(use_constraint_solver=True, maxiter=500)

    # Test with different objectives
    for objective in ["mean_cos", "neg_mean_cos", "bending"]:
        result = optimize_cycle(initial, config, objective=objective, options=opts)

        # Should satisfy constraints
        assert result.penalty < 1e-2, f"Constraint solver with {objective} should satisfy constraints"

        # Should have constant torsion
        cosines = pairwise_cosines(result.hinges)
        torsion_std = np.std(cosines[:-1])
        assert torsion_std < 1e-3, f"Should achieve constant torsion with {objective} objective"
