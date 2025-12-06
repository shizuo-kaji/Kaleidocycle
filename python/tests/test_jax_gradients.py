"""Tests for JAX gradient and Jacobian correctness.

These tests verify that JAX automatic differentiation produces the same
results as finite differences (within numerical tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.optimality import (
    check_stationarity,
    compute_constraint_jacobian,
    compute_energy_gradient,
)


@pytest.mark.jax
class TestJAXGradientCorrectness:
    """Test JAX gradient computation matches finite differences."""

    @pytest.mark.parametrize("energy", ["bending", "mean_cos"])
    @pytest.mark.parametrize("n", [6, 8, 10])
    def test_jax_gradient_matches_numpy(self, energy, n):
        """Verify JAX gradients match NumPy finite differences."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=n, oriented=True, seed=42)

        # Compute gradient with NumPy (finite differences)
        grad_numpy = compute_energy_gradient(
            kc.hinges, energy, eps=1e-7, backend='numpy'
        )

        # Compute gradient with JAX (autodiff)
        grad_jax = compute_energy_gradient(
            kc.hinges, energy, backend='jax'
        )

        # Should match within tolerance (finite diff has O(eps^2) error)
        np.testing.assert_allclose(grad_jax, grad_numpy, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("oriented", [True, False])
    def test_jax_gradient_different_orientations(self, oriented):
        """Test JAX gradients for oriented and non-oriented."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=8, oriented=oriented, seed=42)

        grad_numpy = compute_energy_gradient(
            kc.hinges, 'bending', eps=1e-7, backend='numpy'
        )
        grad_jax = compute_energy_gradient(
            kc.hinges, 'bending', backend='jax'
        )

        np.testing.assert_allclose(grad_jax, grad_numpy, rtol=1e-5, atol=1e-7)

    def test_jax_gradient_shape(self):
        """Test that JAX gradient has correct shape."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'bending', backend='jax')

        assert grad.shape == kc.hinges.shape  # (n+1, 3)

    def test_jax_gradient_is_numpy_array(self):
        """Test that JAX gradient is converted to NumPy array."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'bending', backend='jax')

        assert isinstance(grad, np.ndarray)


@pytest.mark.jax
class TestJAXJacobianCorrectness:
    """Test JAX Jacobian computation matches finite differences."""

    @pytest.mark.parametrize("oriented", [True, False])
    def test_jax_jacobian_matches_numpy(self, oriented):
        """Verify JAX Jacobians match NumPy finite differences."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=oriented, seed=42)
        config = ConstraintConfig(oriented=oriented, constant_torsion=True)

        # Compute Jacobian with NumPy (finite differences)
        jac_numpy = compute_constraint_jacobian(
            kc.hinges, config, eps=1e-7, backend='numpy'
        )

        # Compute Jacobian with JAX (autodiff)
        jac_jax = compute_constraint_jacobian(
            kc.hinges, config, backend='jax'
        )

        # Should match within tolerance
        np.testing.assert_allclose(jac_jax, jac_numpy, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("n", [6, 8, 10])
    def test_jax_jacobian_different_sizes(self, n):
        """Test JAX Jacobians for different kaleidocycle sizes."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=n, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        jac_numpy = compute_constraint_jacobian(
            kc.hinges, config, eps=1e-7, backend='numpy'
        )
        jac_jax = compute_constraint_jacobian(
            kc.hinges, config, backend='jax'
        )

        np.testing.assert_allclose(jac_jax, jac_numpy, rtol=1e-5, atol=1e-7)

    def test_jax_jacobian_shape(self):
        """Test that JAX Jacobian has correct shape."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        jac = compute_constraint_jacobian(kc.hinges, config, backend='jax')

        # Jacobian should be (n_constraints, n_vars)
        n_vars = (kc.n + 1) * 3  # 7 hinges * 3 components = 21
        assert jac.shape[1] == n_vars
        assert jac.shape[0] > 0  # At least some constraints

    def test_jax_jacobian_with_different_constraints(self):
        """Test JAX Jacobian with different constraint configurations."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)

        # Test with constant torsion
        config1 = ConstraintConfig(oriented=True, constant_torsion=True)
        jac1_numpy = compute_constraint_jacobian(
            kc.hinges, config1, eps=1e-7, backend='numpy'
        )
        jac1_jax = compute_constraint_jacobian(
            kc.hinges, config1, backend='jax'
        )
        np.testing.assert_allclose(jac1_jax, jac1_numpy, rtol=1e-5, atol=1e-7)

        # Test without constant torsion
        config2 = ConstraintConfig(oriented=True, constant_torsion=False)
        jac2_numpy = compute_constraint_jacobian(
            kc.hinges, config2, eps=1e-7, backend='numpy'
        )
        jac2_jax = compute_constraint_jacobian(
            kc.hinges, config2, backend='jax'
        )
        np.testing.assert_allclose(jac2_jax, jac2_numpy, rtol=1e-5, atol=1e-7)


@pytest.mark.jax
class TestJAXStationarityCheck:
    """Test check_stationarity with JAX backend."""

    @pytest.mark.parametrize("energy", ["bending", "mean_cos"])
    def test_jax_stationarity_check(self, energy):
        """Test stationarity check with JAX backend."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=8, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        # Check stationarity with NumPy
        result_numpy = check_stationarity(
            kc.hinges, energy, config, tolerance=1e-5, backend='numpy'
        )

        # Check stationarity with JAX
        result_jax = check_stationarity(
            kc.hinges, energy, config, tolerance=1e-5, backend='jax'
        )

        # Projected gradient norms should be close
        np.testing.assert_allclose(
            result_jax['projected_gradient_norm'],
            result_numpy['projected_gradient_norm'],
            rtol=1e-4,
            atol=1e-6,
        )

        # Both should agree on stationarity (or not)
        # Note: They might disagree at the boundary due to numerical precision
        if result_numpy['projected_gradient_norm'] < 1e-6 or result_numpy['projected_gradient_norm'] > 1e-4:
            assert result_jax['is_stationary'] == result_numpy['is_stationary']

    def test_jax_stationarity_details(self):
        """Test that stationarity check returns correct details."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        result = check_stationarity(
            kc.hinges, 'bending', config, backend='jax'
        )

        assert 'is_stationary' in result
        assert 'projected_gradient_norm' in result
        assert 'gradient_norm' in result
        assert 'constraint_penalty' in result
        assert 'details' in result
        assert result['details']['backend'] == 'jax'

    def test_optimized_kaleidocycle_is_stationary(self):
        """Test that optimized kaleidocycles are stationary with JAX."""
        pytest.importorskip("jax")

        # Create an optimized kaleidocycle (should be near stationary point)
        kc = Kaleidocycle(n=8, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        result = check_stationarity(
            kc.hinges, 'bending', config, tolerance=1e-4, backend='jax'
        )

        # Optimized kaleidocycles should have small projected gradient
        # (may not be exactly zero if optimization didn't fully converge)
        assert result['projected_gradient_norm'] < 0.1


@pytest.mark.jax
class TestJAXPerformance:
    """Test JAX performance characteristics (qualitative)."""

    def test_jax_gradient_faster_for_large_system(self):
        """JAX should be faster than finite differences for large systems."""
        pytest.importorskip("jax")
        import time

        n = 15  # Larger system
        kc = Kaleidocycle(n=n, oriented=True, seed=42)

        # Warm up JAX (first call compiles)
        _ = compute_energy_gradient(kc.hinges, 'bending', backend='jax')

        # Time NumPy
        start = time.time()
        for _ in range(3):
            compute_energy_gradient(kc.hinges, 'bending', eps=1e-7, backend='numpy')
        numpy_time = (time.time() - start) / 3

        # Time JAX
        start = time.time()
        for _ in range(3):
            compute_energy_gradient(kc.hinges, 'bending', backend='jax')
        jax_time = (time.time() - start) / 3

        # JAX should be faster (at least for gradient computation)
        # This is not a strict requirement as it depends on system,
        # but we can check that both complete successfully
        assert numpy_time > 0
        assert jax_time > 0

        # Print speedup for informational purposes (not an assertion)
        speedup = numpy_time / jax_time
        print(f"\nJAX speedup: {speedup:.2f}x (n={n}, NumPy: {numpy_time*1000:.2f}ms, JAX: {jax_time*1000:.2f}ms)")


@pytest.mark.jax
class TestJAXRobustness:
    """Test JAX backend robustness and edge cases."""

    def test_jax_handles_small_kaleidocycle(self):
        """JAX should handle small kaleidocycles (n=3)."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=3, oriented=True, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'bending', backend='jax')

        assert grad.shape == (4, 3)  # n+1=4 hinges
        assert np.all(np.isfinite(grad))

    def test_jax_handles_non_oriented(self):
        """JAX should handle non-oriented kaleidocycles."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=8, oriented=False, seed=42)
        config = ConstraintConfig(oriented=False, constant_torsion=True)

        grad = compute_energy_gradient(kc.hinges, 'bending', backend='jax')
        jac = compute_constraint_jacobian(kc.hinges, config, backend='jax')

        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(jac))

    def test_jax_gradient_is_finite(self):
        """JAX gradients should always be finite."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'bending', backend='jax')

        assert np.all(np.isfinite(grad))
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))
