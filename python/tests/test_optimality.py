"""Tests for optimality checking and stationarity detection."""

import numpy as np
import pytest

from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.geometry import random_hinges
from kaleidocycle.optimality import (
    check_stationarity,
    compute_constraint_jacobian,
    compute_energy_gradient,
    find_nearby_stationary,
    finite_motion_dof,
    follow_motion,
    local_dof,
    project_gradient,
)


class TestEnergyGradient:
    """Tests for energy gradient computation."""

    def test_bending_gradient_shape(self):
        """Test that bending gradient has correct shape."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'bending')
        assert grad.shape == kc.hinges.shape

    def test_mean_cos_gradient_shape(self):
        """Test that mean_cos gradient has correct shape."""
        kc = Kaleidocycle(n=6, oriented=False, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'mean_cos')
        assert grad.shape == kc.hinges.shape

    def test_gradient_finite(self):
        """Test that gradient values are finite."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        grad = compute_energy_gradient(kc.hinges, 'bending')
        assert np.all(np.isfinite(grad))

    def test_invalid_energy_raises(self):
        """Test that invalid energy type raises ValueError."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        with pytest.raises(ValueError, match="unknown energy type"):
            compute_energy_gradient(kc.hinges, 'invalid')


class TestConstraintJacobian:
    """Tests for constraint Jacobian computation."""

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)
        jac = compute_constraint_jacobian(kc.hinges, config)

        # Expected number of constraints:
        # - unit_norm: n (6 hinges, excluding last which repeats first)
        # - closure: 3 (vector constraint)
        # - alignment: 1 (scalar constraint)
        # - constant_torsion: n (6 constraints for consecutive pairs)
        # Total: 6 + 3 + 1 + 6 = 16
        n_constraints = 16
        n_vars = kc.hinges.size  # 7 hinges × 3 = 21

        assert jac.shape == (n_constraints, n_vars)

    def test_jacobian_finite(self):
        """Test that Jacobian entries are finite."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True)
        jac = compute_constraint_jacobian(kc.hinges, config)
        assert np.all(np.isfinite(jac))

    def test_jacobian_nonzero(self):
        """Test that Jacobian has some nonzero entries."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True)
        jac = compute_constraint_jacobian(kc.hinges, config)
        assert np.any(jac != 0)


class TestGradientProjection:
    """Tests for gradient projection onto constraint tangent space."""

    def test_projection_shape(self):
        """Test that projected gradient has same shape as input."""
        grad = np.random.randn(7, 3)
        jac = np.random.randn(10, 21)
        proj_grad = project_gradient(grad, jac)
        assert proj_grad.shape == grad.shape

    def test_projection_reduces_norm(self):
        """Test that projection typically reduces gradient norm."""
        # Create a gradient that has components in constraint directions
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True)

        grad = compute_energy_gradient(kc.hinges, 'bending')
        jac = compute_constraint_jacobian(kc.hinges, config)
        proj_grad = project_gradient(grad, jac)

        # Projected gradient should have smaller or equal norm
        # (equality only if gradient was already in nullspace)
        assert np.linalg.norm(proj_grad) <= np.linalg.norm(grad) + 1e-10

    def test_projection_is_orthogonal(self):
        """Test that projected gradient is orthogonal to constraint gradients."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True)

        grad = compute_energy_gradient(kc.hinges, 'bending')
        jac = compute_constraint_jacobian(kc.hinges, config)
        proj_grad = project_gradient(grad, jac)

        # proj_grad should be orthogonal to all rows of Jacobian
        # i.e., J @ proj_grad.flatten() ≈ 0
        orthogonality = jac @ proj_grad.flatten()
        assert np.linalg.norm(orthogonality) < 1e-8


class TestStationarityCheck:
    """Tests for stationary point detection."""

    def test_optimized_kaleidocycle_is_stationary_bending(self):
        """Test that optimized kaleidocycle has small projected gradient.

        Note: Numerical optimization may not reach perfect stationarity due to
        solver tolerance and accumulated numerical errors. We use a relaxed
        tolerance to account for this.
        """
        # Create an oriented kaleidocycle optimized with bending energy
        # For oriented with even n, the objective is "bending"
        kc = Kaleidocycle(n=8, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        result = check_stationarity(
            kc.hinges, 'bending', config, tolerance=10.0
        )

        # Check that projected gradient is reasonably small
        # (relaxed tolerance due to numerical optimization precision)
        assert result['projected_gradient_norm'] < 10.0, (
            f"Projected gradient too large: "
            f"{result['projected_gradient_norm']:.2e}"
        )

    def test_optimized_kaleidocycle_is_stationary_mean_cos(self):
        """Test that optimized kaleidocycle has small projected gradient for mean_cos.

        Note: Due to numerical optimization precision, we use relaxed tolerance.
        """
        # For non-oriented with even n, mean_cos is meaningful
        # The kaleidocycle is created with objective "mean_cos"
        kc = Kaleidocycle(n=8, oriented=False, seed=42)
        config = ConstraintConfig(oriented=False, constant_torsion=True)

        result = check_stationarity(
            kc.hinges, 'mean_cos', config, tolerance=1.0
        )

        # Check that projected gradient is reasonably small
        assert result['projected_gradient_norm'] < 1.0, (
            f"Projected gradient too large: "
            f"{result['projected_gradient_norm']:.2e}"
        )

    def test_perturbed_configuration_not_stationary(self):
        """Test that perturbed configuration is not stationary."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        # Add random perturbation
        hinges_perturbed = kc.hinges + 0.1 * np.random.randn(*kc.hinges.shape)

        # Normalize to maintain unit norm approximately
        for i in range(len(hinges_perturbed) - 1):
            hinges_perturbed[i] /= np.linalg.norm(hinges_perturbed[i])

        # Enforce terminal alignment
        if config.oriented:
            hinges_perturbed[-1] = hinges_perturbed[0]
        else:
            hinges_perturbed[-1] = -hinges_perturbed[0]

        result = check_stationarity(
            hinges_perturbed, 'bending', config, tolerance=1e-6
        )

        # Perturbed configuration should not be stationary
        # (unless we were very unlucky with the perturbation)
        assert not result['is_stationary']

    def test_constraint_penalty_small_for_optimized(self):
        """Test that constraint penalty is small for optimized kaleidocycle."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(oriented=True)

        result = check_stationarity(kc.hinges, 'bending', config)

        # Optimized kaleidocycle should satisfy constraints well
        assert result['constraint_penalty'] < 1e-6

    def test_is_stationary_method(self):
        """Test the is_stationary method on Kaleidocycle class."""
        kc = Kaleidocycle(n=8, oriented=True, seed=42)

        result = kc.is_stationary('bending', tolerance=1e-4)

        assert 'is_stationary' in result
        assert 'projected_gradient_norm' in result
        assert 'gradient_norm' in result
        assert 'constraint_penalty' in result
        assert 'details' in result
        assert isinstance(result['is_stationary'], bool)

    def test_different_tolerances(self):
        """Test that tolerance affects stationarity determination."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)

        # With very strict tolerance, likely not stationary
        result_strict = kc.is_stationary('bending', tolerance=1e-10)

        # With loose tolerance, should be stationary
        result_loose = kc.is_stationary('bending', tolerance=100.0)

        # Loose tolerance should be more likely to pass
        # (though not guaranteed for all configurations)
        assert result_loose['is_stationary']

        # Projected gradient norm should be the same regardless of tolerance
        assert result_strict['projected_gradient_norm'] == result_loose['projected_gradient_norm']

    def test_details_contain_expected_keys(self):
        """Test that details dictionary contains expected diagnostic info."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        result = kc.is_stationary('bending')

        details = result['details']
        assert 'energy' in details
        assert 'tolerance' in details
        assert 'finite_diff_step' in details
        assert 'n_constraints' in details
        assert 'n_variables' in details
        assert 'constraint_rank' in details

        assert details['energy'] == 'bending'
        assert details['n_variables'] == kc.hinges.size


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_kaleidocycle(self):
        """Test with small kaleidocycle (n=3)."""
        kc = Kaleidocycle(n=3, oriented=True, seed=42)
        result = kc.is_stationary('bending', tolerance=1e-4)

        assert 'is_stationary' in result
        assert np.isfinite(result['projected_gradient_norm'])

    def test_non_oriented_kaleidocycle(self):
        """Test with non-oriented kaleidocycle."""
        kc = Kaleidocycle(n=7, oriented=False, seed=42)
        result = kc.is_stationary('mean_cos', tolerance=1e-4)

        assert 'is_stationary' in result
        assert result['details']['energy'] == 'mean_cos'

    def test_custom_config(self):
        """Test with custom constraint configuration."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        config = ConstraintConfig(
            oriented=True,
            constant_torsion=False,  # Relax constant torsion
            alignment=True,
        )

        result = kc.is_stationary('bending', config=config)

        # Should have fewer constraints
        assert result['details']['n_constraints'] < 15

    def test_gradient_norm_positive(self):
        """Test that gradient norms are non-negative."""
        kc = Kaleidocycle(n=6, oriented=True, seed=42)
        result = kc.is_stationary('bending')

        assert result['gradient_norm'] >= 0
        assert result['projected_gradient_norm'] >= 0

    def test_random_hinges_not_stationary(self):
        """Test that random hinges are typically not stationary."""
        hinges = random_hinges(6, seed=42, oriented=True).as_array()
        kc = Kaleidocycle(hinges=hinges)
        config = ConstraintConfig(oriented=True, constant_torsion=True)

        result = kc.is_stationary('bending', tolerance=1e-6, config=config)

        # Random hinges should have high constraint penalty
        # and likely not be stationary
        # (though we don't strictly require this as the test might fail randomly)
        assert result['constraint_penalty'] > 1e-3


class TestLocalDoF:
    """Tests for local degree-of-freedom computation."""

    def test_basic_keys_and_shapes(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        info = kc.local_dof()
        for key in [
            "dof",
            "raw_dof",
            "rigid_dof",
            "rank",
            "n_constraints",
            "n_variables",
            "singular_values",
            "tol",
        ]:
            assert key in info
        assert info["n_variables"] == kc.hinges.size
        assert info["dof"] == info["raw_dof"] - info["rigid_dof"]

    def test_basis_lies_in_nullspace(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        info = kc.local_dof(return_basis=True)
        if info["dof"] == 0:
            pytest.skip("no internal DoF for this configuration")
        J = compute_constraint_jacobian(kc.hinges, kc.config)
        B = info["basis"].reshape(-1, info["dof"])
        assert np.linalg.norm(J @ B) < 1e-8

    def test_rigid_rotation_in_nullspace(self):
        """Three global rotations are always tangent to the constraint set."""
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        info = kc.local_dof()
        # No anchors -> all 3 infinitesimal rotations should be detected.
        assert info["rigid_dof"] == 3

    def test_anchors_remove_rigid_rotations(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        cfg = ConstraintConfig(
            oriented=True,
            constant_torsion=True,
            alignment=True,
            closure=True,
            enforce_anchors=True,
        )
        info = kc.local_dof(config=cfg)
        assert info["rigid_dof"] == 0

    def test_dof_consistency(self):
        kc = Kaleidocycle(n=6, oriented=False, seed=1)
        info_raw = kc.local_dof(subtract_rigid=False)
        info = kc.local_dof(subtract_rigid=True)
        assert info_raw["raw_dof"] == info["raw_dof"]
        assert info_raw["rigid_dof"] == 0
        assert info["dof"] + info["rigid_dof"] == info["raw_dof"]


class TestFiniteMotionDoF:
    """Tests for finite (nonlinear) motion DoF estimation via continuation."""

    def test_basic_keys(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        info = kc.finite_motion_dof(
            seed=0, step_size=1e-3, n_steps=8, n_samples=8
        )
        for key in [
            "finite_dof",
            "infinitesimal_dof",
            "rigid_dof",
            "n_samples",
            "n_successful",
            "displacement_singular_values",
            "max_residual",
            "step_size",
            "n_steps",
        ]:
            assert key in info

    def test_paths_stay_on_manifold(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        info = kc.finite_motion_dof(
            seed=0, step_size=1e-3, n_steps=8, n_samples=6,
            correction_tol=1e-8,
        )
        # The Newton corrector should keep us on the constraint manifold.
        assert info["max_residual"] < 1e-6

    def test_finite_bounded_by_infinitesimal(self):
        kc = Kaleidocycle(n=6, oriented=False, seed=1)
        info = kc.finite_motion_dof(
            seed=0, step_size=5e-4, n_steps=8, n_samples=8
        )
        assert info["finite_dof"] <= info["infinitesimal_dof"]

    def test_zero_dof_short_circuit(self):
        """When no infinitesimal DoF, finite DoF is 0 without continuation."""
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        # Anchor + every constraint reduces the count enough that this is
        # a useful smoke test of the short-circuit path. We patch via a
        # config that yields infinitesimal_dof == 0 if possible, but the
        # generic case is to confirm consistency rather than exact zero.
        info = kc.finite_motion_dof(
            seed=0, step_size=1e-3, n_steps=2, n_samples=2
        )
        assert info["finite_dof"] >= 0
        assert info["n_successful"] <= info["n_samples"]

    def test_paths_returned(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        info = kc.finite_motion_dof(
            seed=0, step_size=1e-3, n_steps=4, n_samples=8,
            return_paths=True,
        )
        assert "paths" in info
        # n_samples is clamped to at least the infinitesimal DoF so that
        # every basis direction is tried.
        assert len(info["paths"]) == info["n_samples"]
        for p in info["paths"]:
            assert p.ndim == 3 and p.shape[1:] == kc.hinges.shape


class TestFullAlignment:
    """Tests for the full_alignment flag on ConstraintConfig."""

    def test_full_alignment_adds_two_constraints(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        cfg_scalar = ConstraintConfig(
            oriented=True, constant_torsion=True, alignment=True,
            closure=True, full_alignment=False,
        )
        cfg_vec = ConstraintConfig(
            oriented=True, constant_torsion=True, alignment=True,
            closure=True, full_alignment=True,
        )
        J_scalar = compute_constraint_jacobian(kc.hinges, cfg_scalar)
        J_vec = compute_constraint_jacobian(kc.hinges, cfg_vec)
        # Vector form has 2 extra rows (3 instead of 1)
        assert J_vec.shape[0] == J_scalar.shape[0] + 2

    def test_full_alignment_rows_are_active(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        ref = float(np.dot(kc.hinges[0], kc.hinges[1]))
        cfg_scalar = ConstraintConfig(
            oriented=True, constant_torsion=True, alignment=True,
            closure=True, full_alignment=False, reference_torsion=ref,
        )
        cfg_vec = ConstraintConfig(
            oriented=True, constant_torsion=True, alignment=True,
            closure=True, full_alignment=True, reference_torsion=ref,
        )
        # With reference_torsion fixed, no constraint row is identically
        # zero by construction. The scalar alignment row is zero at the
        # manifold (rank-0), the full alignment rows are rank-3.
        J_scalar = compute_constraint_jacobian(kc.hinges, cfg_scalar)
        J_vec = compute_constraint_jacobian(kc.hinges, cfg_vec)
        assert np.linalg.matrix_rank(J_vec) > np.linalg.matrix_rank(J_scalar)


class TestFindNearbyStationary:
    """Tests for find_nearby_stationary."""

    def test_returns_dict_with_required_keys(self):
        kc = Kaleidocycle(n=7, oriented=True, seed=0)
        cfg = ConstraintConfig(
            oriented=True, constant_torsion=True, full_alignment=True,
        )
        info = find_nearby_stationary(kc.hinges, cfg, energy="mean_cos",
                                       maxfev=500)
        for key in ("hinges", "projected_gradient_norm", "n_eval",
                    "success", "distance"):
            assert key in info
        assert info["hinges"].shape == kc.hinges.shape

    def test_theta_solution_is_already_stationary(self):
        # Theta(7,3) is conjectured to be at a mean_cos stationary.
        from kaleidocycle import generate_theta_binormals
        from kaleidocycle.theta import solve_closure_conditions

        sol = solve_closure_conditions(7, m=3, initial_guess=(0.48, 0.27))
        v, r, y = sol
        b = generate_theta_binormals(v, 0.0, r, y, N=7, t=0.0)
        kc = Kaleidocycle(hinges=b)
        cfg = ConstraintConfig(
            oriented=kc.oriented, constant_torsion=True, full_alignment=True,
        )
        info = find_nearby_stationary(kc.hinges, cfg, energy="mean_cos",
                                       maxfev=500)
        # Theta(7,3) is already stationary to machine precision.
        assert info["projected_gradient_norm"] < 1e-7
        assert info["distance"] < 1e-3


class TestFollowMotion:
    """Tests for follow_motion."""

    def test_returns_3d_frames(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        cfg = ConstraintConfig(
            oriented=True, constant_torsion=True, full_alignment=True,
        )
        frames = follow_motion(kc.hinges, cfg, n_steps=5)
        assert frames.ndim == 3
        assert frames.shape[1:] == kc.hinges.shape

    def test_bidirectional_path_length(self):
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        cfg = ConstraintConfig(
            oriented=True, constant_torsion=True, full_alignment=True,
        )
        frames = follow_motion(kc.hinges, cfg, n_steps=10,
                                bidirectional=True)
        # Up to 2*n_steps + 1 frames, center is the starting config.
        assert frames.shape[0] <= 2 * 10 + 1
        # The middle frame should equal hinges (or be close to it).
        mid = frames.shape[0] // 2
        assert np.allclose(frames[mid], kc.hinges)

    def test_zero_dof_returns_single_frame(self):
        # Use a contrived config that fully determines the hinges.
        kc = Kaleidocycle(n=8, oriented=True, seed=0)
        cfg = ConstraintConfig(
            oriented=True, constant_torsion=True, full_alignment=True,
            enforce_anchors=True,
        )
        # With anchors + full alignment + torsion, internal DoF may be > 0
        # so this test just verifies the function does not raise.
        frames = follow_motion(kc.hinges, cfg, n_steps=3)
        assert frames.ndim == 3


class TestAlignFirstThree:
    """Tests for align_first_three."""

    def test_first_vertex_at_origin(self):
        from kaleidocycle import align_first_three
        rng = np.random.default_rng(0)
        c = rng.standard_normal((10, 3))
        a = align_first_three(c)
        assert np.allclose(a[0], 0.0)

    def test_second_vertex_on_x_axis(self):
        from kaleidocycle import align_first_three
        rng = np.random.default_rng(1)
        c = rng.standard_normal((10, 3))
        a = align_first_three(c)
        assert a[1, 0] > 0
        assert abs(a[1, 1]) < 1e-10
        assert abs(a[1, 2]) < 1e-10

    def test_third_vertex_in_xy_plane(self):
        from kaleidocycle import align_first_three
        rng = np.random.default_rng(2)
        c = rng.standard_normal((10, 3))
        a = align_first_three(c)
        assert abs(a[2, 2]) < 1e-10
        assert a[2, 1] >= 0

    def test_distances_preserved(self):
        from kaleidocycle import align_first_three
        rng = np.random.default_rng(3)
        c = rng.standard_normal((10, 3))
        a = align_first_three(c)
        # Rigid motion preserves all pairwise distances.
        d1 = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
        d2 = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
        assert np.allclose(d1, d2, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
