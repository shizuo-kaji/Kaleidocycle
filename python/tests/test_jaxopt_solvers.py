"""Tests for JAXopt-based optimization."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.solvers import SolverOptions, optimize_cycle


@pytest.mark.jax
class TestJAXoptOptimization:
    """Test JAXopt-based optimization."""

    def test_jaxopt_optimizer_runs(self):
        """Test that JAX+scipy optimizer completes without error."""
        pytest.importorskip("jax")

        # Create initial configuration
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=100)

        # Run optimization with JAX backend (uses scipy with JAX autodiff)
        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        assert result.hinges.shape == kc.hinges.shape
        assert result.result is not None
        assert result.backend_name == 'scipy'  # JAX backend uses scipy with autodiff

    @pytest.mark.parametrize("objective", ["bending", "mean_cos", "neg_mean_cos"])
    def test_jaxopt_different_objectives(self, objective):
        """Test JAXopt with different objective functions."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=50)

        result = optimize_cycle(
            kc.hinges,
            config,
            objective=objective,
            options=options,
            backend='jax'
        )

        assert result.hinges.shape == kc.hinges.shape
        assert np.all(np.isfinite(result.hinges))
        assert np.isfinite(result.energy)

    def test_jaxopt_respects_constraints(self):
        """Test that JAXopt optimization respects constraints."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=200, penalty_weight=100.0)

        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        # Check unit norm constraint (approximately)
        norms = np.linalg.norm(result.hinges[:-1], axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-3, atol=1e-3)

        # Check alignment constraint
        if config.oriented:
            np.testing.assert_allclose(result.hinges[0], result.hinges[-1], rtol=1e-3, atol=1e-3)
        else:
            np.testing.assert_allclose(result.hinges[0], -result.hinges[-1], rtol=1e-3, atol=1e-3)

        # Check penalty is small
        assert result.penalty < 1e-2

    @pytest.mark.parametrize("oriented", [True, False])
    def test_jaxopt_oriented_vs_nonoriented(self, oriented):
        """Test JAXopt with oriented and non-oriented configurations."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=6, oriented=oriented, seed=42)
        config = ConstraintConfig(oriented=oriented, constant_torsion=True)
        options = SolverOptions(maxiter=100)

        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        assert result.hinges.shape == kc.hinges.shape
        assert result.success

    def test_jaxopt_reduces_objective(self):
        """Test that JAXopt reduces the objective function."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=200)

        # Compute initial energy
        from kaleidocycle.geometry import mean_cosine
        initial_energy = mean_cosine(kc.hinges, wrap=False)

        # Optimize
        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        final_energy = mean_cosine(result.hinges, wrap=False)

        # Energy should be reduced (or similar if already optimal)
        assert final_energy <= initial_energy + 1e-6


@pytest.mark.jax
class TestJAXoptVsScipy:
    """Compare JAXopt and SciPy optimization results."""

    def test_jaxopt_vs_scipy_convergence(self):
        """Test that JAX and NumPy backends converge to similar solutions."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        # Use constrained optimization for better convergence
        options = SolverOptions(maxiter=200, use_constraint_solver=True)

        # Optimize with NumPy/SciPy
        result_numpy = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='numpy'
        )

        # Optimize with JAX/scipy (using JAX autodiff)
        result_jax = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        # Energies should be similar (both should find local minimum)
        np.testing.assert_allclose(
            result_jax.energy,
            result_numpy.energy,
            rtol=5e-2,
            atol=5e-2
        )

        # Both should satisfy constraints
        assert result_jax.penalty < 1e-2
        assert result_numpy.penalty < 1e-2

    def test_jaxopt_scipy_both_complete(self):
        """Test that both backends complete optimization."""
        pytest.importorskip("jax")

        kc = Kaleidocycle(n=8, oriented=True, seed=123)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        # Use constrained optimization for better convergence
        options = SolverOptions(maxiter=200, use_constraint_solver=True)

        result_numpy = optimize_cycle(
            kc.hinges, config, objective='mean_cos', options=options, backend='numpy'
        )
        result_jax = optimize_cycle(
            kc.hinges, config, objective='mean_cos', options=options, backend='jax'
        )

        # Both should complete and produce valid results
        assert np.all(np.isfinite(result_numpy.hinges))
        assert np.all(np.isfinite(result_jax.hinges))

        # Energies should be similar (both use scipy with same method)
        np.testing.assert_allclose(
            result_jax.energy,
            result_numpy.energy,
            rtol=5e-2,
            atol=1e-2
        )


@pytest.mark.jax
class TestJAXoptEdgeCases:
    """Test JAXopt edge cases and robustness."""

    def test_jaxopt_small_kaleidocycle(self):
        """Test JAXopt with small kaleidocycle (n=3)."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=3, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=100)

        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        assert result.hinges.shape == (4, 3)  # n+1 hinges
        assert np.all(np.isfinite(result.hinges))

    def test_jaxopt_different_penalty_weights(self):
        """Test JAXopt with different penalty weights."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        for penalty_weight in [10.0, 100.0, 1000.0]:
            options = SolverOptions(maxiter=100, penalty_weight=penalty_weight)

            result = optimize_cycle(
                kc.hinges,
                config,
                objective='mean_cos',
                options=options,
                backend='jax'
            )

            # Higher penalty weight should give lower constraint penalty
            # (but this is not strictly enforced in test)
            assert result.penalty < 0.1  # Should satisfy constraints

    def test_jaxopt_handles_nan_gracefully(self):
        """Test that JAXopt handles potential numerical issues."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        # Create a potentially problematic initial configuration
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=50)

        # Add some noise to make it less optimal
        noisy_hinges = kc.hinges + 0.1 * np.random.RandomState(42).randn(*kc.hinges.shape)

        result = optimize_cycle(
            noisy_hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        # Should complete without NaN
        assert np.all(np.isfinite(result.hinges))
        assert np.isfinite(result.energy)


@pytest.mark.jax
class TestJAXoptPerformance:
    """Test JAXopt performance characteristics."""

    def test_jaxopt_converges_in_reasonable_iterations(self):
        """Test that JAXopt converges within reasonable iterations."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")

        kc = Kaleidocycle(n=8, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=500)

        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )

        # Should converge before max iterations for this problem
        assert result.success
        assert result.penalty < 1e-2

    def test_jaxopt_runtime_reasonable(self):
        """Test that JAXopt runs in reasonable time."""
        pytest.importorskip("jax")
        pytest.importorskip("jaxopt")
        import time

        kc = Kaleidocycle(n=10, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        options = SolverOptions(maxiter=100)

        start = time.time()
        result = optimize_cycle(
            kc.hinges,
            config,
            objective='mean_cos',
            options=options,
            backend='jax'
        )
        elapsed = time.time() - start

        # Should complete in reasonable time (< 10 seconds for n=10)
        assert elapsed < 10.0
        assert result.success
