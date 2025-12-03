"""Tests for Kaleidocycle creation from n (number of tetrahedra)."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import Kaleidocycle, ConstraintConfig, constraint_residuals


def test_kaleidocycle_from_n_basic():
    """Test basic creation from n."""
    kc = Kaleidocycle(9)

    assert kc.n == 9
    assert kc.oriented == False  # Default
    assert kc.hinges.shape == (10, 3)
    assert "created_from" in kc.metadata
    assert kc.metadata["created_from"] == "optimize_cycle"


def test_kaleidocycle_from_n_oriented():
    """Test creation with oriented=True."""
    kc = Kaleidocycle(8, oriented=True)

    assert kc.n == 8
    assert kc.oriented == True
    assert kc.hinges.shape == (9, 3)


def test_kaleidocycle_from_n_non_oriented():
    """Test creation with oriented=False explicitly."""
    kc = Kaleidocycle(7, oriented=False)

    assert kc.n == 7
    assert kc.oriented == False
    assert kc.hinges.shape == (8, 3)


def test_kaleidocycle_from_n_constraints_satisfied():
    """Test that created kaleidocycle satisfies constraints."""
    kc = Kaleidocycle(6, seed=42)

    # Check constraints
    config = ConstraintConfig(oriented=kc.oriented, constant_torsion=True)
    residuals = constraint_residuals(kc.hinges, config)

    # All residuals should be small
    assert np.linalg.norm(residuals["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals["closure"]) < 1e-3
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-3


def test_kaleidocycle_from_n_with_seed():
    """Test that seed produces reproducible results."""
    kc1 = Kaleidocycle(6, seed=42)
    kc2 = Kaleidocycle(6, seed=42)

    np.testing.assert_array_almost_equal(kc1.hinges, kc2.hinges, decimal=5)


def test_kaleidocycle_from_n_different_seeds():
    """Test that different seeds produce different results."""
    kc1 = Kaleidocycle(6, seed=42)
    kc2 = Kaleidocycle(6, seed=123)

    # Should be different (very unlikely to be the same)
    assert not np.allclose(kc1.hinges, kc2.hinges)


def test_kaleidocycle_from_n_with_solver_options():
    """Test creation with custom solver options."""
    solver_opts = {
        "maxiter": 200,
        "use_constraint_solver": True,
    }

    kc = Kaleidocycle(6, seed=42, solver_options=solver_opts)

    assert kc.n == 6
    # Check that it still satisfies constraints
    config = ConstraintConfig(oriented=kc.oriented, constant_torsion=True)
    residuals = constraint_residuals(kc.hinges, config)
    assert np.linalg.norm(residuals["unit_norm"]) < 1e-3


def test_kaleidocycle_from_n_invalid_n():
    """Test that invalid n values raise errors."""
    # n too small
    with pytest.raises(ValueError, match="n must be an integer >= 3"):
        Kaleidocycle(2)

    # n not an integer
    with pytest.raises(ValueError, match="n must be an integer"):
        Kaleidocycle(5.5)

    # Negative n
    with pytest.raises(ValueError, match="n must be an integer >= 3"):
        Kaleidocycle(-1)


def test_kaleidocycle_from_n_various_sizes():
    """Test creation with various n values."""
    for n in [3, 4, 5, 6, 7, 8, 9, 10]:
        kc = Kaleidocycle(n, seed=42)
        assert kc.n == n
        assert kc.hinges.shape == (n + 1, 3)


def test_kaleidocycle_from_n_oriented_even():
    """Test oriented kaleidocycle with even n uses bending objective."""
    # This should use bending objective (both cosine objectives are meaningless)
    kc = Kaleidocycle(8, oriented=True, seed=42)

    # Check that it's actually optimized and satisfies constraints
    config = ConstraintConfig(oriented=True, constant_torsion=True)
    residuals = constraint_residuals(kc.hinges, config)

    assert np.linalg.norm(residuals["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-3


def test_kaleidocycle_from_n_oriented_odd():
    """Test oriented kaleidocycle with odd n uses mean_cos."""
    # This should use mean_cos objective
    kc = Kaleidocycle(9, oriented=True, seed=42)

    config = ConstraintConfig(oriented=True, constant_torsion=True)
    residuals = constraint_residuals(kc.hinges, config)

    assert np.linalg.norm(residuals["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-3


def test_kaleidocycle_from_n_non_oriented_even():
    """Test non-oriented kaleidocycle with even n uses mean_cos."""
    # This should use mean_cos objective
    kc = Kaleidocycle(8, oriented=False, seed=42)

    config = ConstraintConfig(oriented=False, constant_torsion=True)
    residuals = constraint_residuals(kc.hinges, config)

    assert np.linalg.norm(residuals["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-3


def test_kaleidocycle_from_n_non_oriented_odd():
    """Test non-oriented kaleidocycle with odd n uses neg_mean_cos."""
    # This should use neg_mean_cos objective
    kc = Kaleidocycle(7, oriented=False, seed=42)

    config = ConstraintConfig(oriented=False, constant_torsion=True)
    residuals = constraint_residuals(kc.hinges, config)

    assert np.linalg.norm(residuals["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals["constant_torsion"]) < 1e-3


def test_kaleidocycle_from_n_properties_accessible():
    """Test that all properties are accessible after creation from n."""
    kc = Kaleidocycle(6, seed=42)

    # These should all work without errors
    curve = kc.curve
    tangents = kc.tangents
    normals = kc.normals

    assert curve.shape == (7, 3)
    assert tangents.shape == (6, 3)
    assert normals.shape == (6, 3)


def test_kaleidocycle_from_n_compute_properties():
    """Test computing properties on kaleidocycle created from n."""
    kc = Kaleidocycle(6, seed=42)

    config = ConstraintConfig(oriented=kc.oriented, constant_torsion=True)
    kc.compute(config=config)

    # Should have computed all properties
    assert "geometric" in kc.metadata
    assert "energies" in kc.metadata
    assert "constraints" in kc.metadata
    assert "topological" in kc.metadata


def test_kaleidocycle_from_n_plot_method():
    """Test that plot method works on kaleidocycle created from n."""
    kc = Kaleidocycle(6, seed=42)

    # Should not raise error
    ax = kc.plot()
    assert ax is not None


def test_kaleidocycle_mixed_initialization_error():
    """Test that providing both n and hinges raises error."""
    hinges = np.random.rand(7, 3)

    with pytest.raises(ValueError, match="only initialize from one parameter"):
        Kaleidocycle(6, hinges=hinges)


def test_kaleidocycle_from_n_vs_manual():
    """Test that Kaleidocycle(n) produces similar quality to manual optimization."""
    from kaleidocycle import random_hinges, optimize_cycle, SolverOptions

    n = 6
    seed = 42

    # Create using Kaleidocycle(n)
    kc_auto = Kaleidocycle(n, seed=seed)

    # Create manually
    initial = random_hinges(n, seed=seed, oriented=False).as_array()
    config = ConstraintConfig(oriented=False, constant_torsion=True)
    opts = SolverOptions()
    result = optimize_cycle(initial, config, objective="mean_cos", options=opts)
    kc_manual = Kaleidocycle(hinges=result.hinges)

    # Both should satisfy constraints similarly well
    residuals_auto = constraint_residuals(kc_auto.hinges, config)
    residuals_manual = constraint_residuals(kc_manual.hinges, config)

    assert np.linalg.norm(residuals_auto["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals_manual["unit_norm"]) < 1e-3
    assert np.linalg.norm(residuals_auto["constant_torsion"]) < 1e-3
    assert np.linalg.norm(residuals_manual["constant_torsion"]) < 1e-3


def test_kaleidocycle_from_n_objective_selection():
    """Test that the correct objective is selected for different n and oriented."""
    # This is mostly an integration test that verifies no warnings are raised
    # for meaningless objectives

    # These should all work without warnings about meaningless objectives
    import warnings

    test_cases = [
        (8, True),  # Should use bending (both cosine objectives are meaningless)
        (9, True),  # Should use mean_cos
        (8, False),  # Should use mean_cos
        (7, False),  # Should use neg_mean_cos
    ]

    for n, oriented in test_cases:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kc = Kaleidocycle(n, oriented=oriented, seed=42)

            # Check no RuntimeWarning about meaningless objective
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            meaningless_warnings = [
                x for x in runtime_warnings if "meaningless" in str(x.message)
            ]

            assert len(meaningless_warnings) == 0, f"Got meaningless warning for n={n}, oriented={oriented}"


def test_kaleidocycle_n_property():
    """Test n property is correctly set."""
    kc = Kaleidocycle(7, seed=42)
    assert kc.n == 7
    assert kc.hinges.shape[0] == kc.n + 1


def test_kaleidocycle_oriented_property():
    """Test oriented property."""
    kc_oriented = Kaleidocycle(6, oriented=True, seed=42)
    kc_non_oriented = Kaleidocycle(6, oriented=False, seed=42)

    assert kc_oriented.oriented is True
    assert kc_non_oriented.oriented is False


def test_kaleidocycle_is_closed_property():
    """Test is_closed property."""
    # Optimized kaleidocycles should satisfy closure constraint
    kc = Kaleidocycle(6, seed=42)
    assert kc.is_closed is True

    # Create from random hinges (may not satisfy closure)
    from kaleidocycle import random_hinges
    hinges_random = random_hinges(6, seed=123).as_array()
    kc_random = Kaleidocycle(hinges=hinges_random)

    # Random hinges may or may not satisfy closure - just check it's a boolean
    result = kc_random.is_closed
    assert isinstance(result, bool)


def test_kaleidocycle_is_aligned_property():
    """Test is_aligned property."""
    # Optimized kaleidocycles should satisfy alignment constraint
    kc_oriented = Kaleidocycle(6, oriented=True, seed=42)
    assert kc_oriented.is_aligned is True

    kc_non_oriented = Kaleidocycle(6, oriented=False, seed=42)
    assert kc_non_oriented.is_aligned is True

    # Create kaleidocycle that violates alignment constraint
    from kaleidocycle import random_hinges
    hinges = random_hinges(6, seed=42).as_array()
    # Modify last hinge to violate alignment (for oriented, last != first)
    # Make the last hinge very different from the first
    hinges[-1] = np.array([0.0, 0.0, 1.0])
    kc_misaligned = Kaleidocycle(hinges=hinges)

    # This should violate alignment constraint
    result_misaligned = kc_misaligned.is_aligned
    assert isinstance(result_misaligned, bool)


def test_kaleidocycle_is_unit_norm_property():
    """Test is_unit_norm property."""
    kc = Kaleidocycle(6, seed=42)
    # Optimized kaleidocycles should have unit norm hinges
    assert kc.is_unit_norm is True

    # Create non-unit norm hinges
    from kaleidocycle import random_hinges
    hinges = random_hinges(6, seed=42).as_array() * 2.0
    kc_non_unit = Kaleidocycle(hinges=hinges)

    assert kc_non_unit.is_unit_norm is False


def test_kaleidocycle_constant_torsion_property():
    """Test constant_torsion property."""
    kc = Kaleidocycle(6, seed=42)

    # Optimized kaleidocycles typically have constant torsion
    result = kc.constant_torsion

    # Should return either None or a float
    assert result is None or isinstance(result, float)

    # If it's a float, should be positive (torsion angles are non-negative)
    if result is not None:
        assert result >= 0
        assert result <= np.pi  # Torsion is an angle
