"""Tests for the Jacobi elliptic critical families (Theorems B-E of the paper)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipe, ellipj, ellipk

from kaleidocycle import Kaleidocycle, compute_linking_number
from kaleidocycle.elliptic import (
    closing_angle_curve,
    closing_moduli,
    continuum_constants,
    elliptic_family,
    fixed_torsion_jacobian,
    jacobi_operator,
    jacobi_zeta,
    minimum_torsion_angle,
    principal_family,
    quadratic_inertia,
    torsion_hessian,
)
from kaleidocycle.integrable import (
    critical_multiplier,
    critical_torsion_cosine,
    first_hamiltonian,
    mkdv_hierarchy_field,
    qrt_invariant,
    variational_recurrence_residual,
)

# mu_N / pi of the principal family, from the table in the paper's remark
# comparing with Schoenke-Fried.
PAPER_MU_OVER_PI = {
    7: 0.40455,
    8: 0.34427,
    9: 0.30101,
    10: 0.26800,
    11: 0.24180,
    12: 0.22043,
    15: 0.17459,
    21: 0.12369,
}


def test_jacobi_zeta_matches_quadrature() -> None:
    # Includes the point where scipy.special.ellipeinc is wrong
    # (scipy 1.17-1.18), which the Carlson evaluation must avoid.
    for k in (0.3, 0.8465722693560513, 0.95):
        parameter = k * k
        quarter = ellipk(parameter)
        for u in (0.2, 0.75 * quarter, 1.3 * quarter, -2.5 * quarter):
            integral = quad(lambda t, p=parameter: ellipj(t, p)[2] ** 2, 0.0, u)[0]
            expected = integral - ellipe(parameter) / quarter * u
            assert jacobi_zeta(u, k) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(("n", "m", "k"), [(7, 3, 0.4), (10, 4, 0.7), (13, 5, 0.9)])
def test_family_solves_stationary_recurrence(n: int, m: int, k: float) -> None:
    family = elliptic_family(n, m, k)
    kappa = family.curvatures(0.37)
    sign = family.sign
    assert sign == (-1) ** m
    residual = variational_recurrence_residual(kappa, family.multiplier, sign=sign)
    assert np.max(np.abs(residual)) < 1e-12
    assert np.ptp(qrt_invariant(kappa, family.multiplier, sign=sign)) < 1e-12
    # The zeta formula for c_m equals (lambda - E2/N) / 2.
    lam = critical_multiplier(kappa, family.torsion_cosine, sign=sign)
    assert lam == pytest.approx(family.multiplier, abs=1e-12)


def test_closing_angle_is_monotone_with_paper_endpoints() -> None:
    for n, m in [(9, 3), (13, 6)]:
        _, theta = closing_angle_curve(n, m)
        assert np.all(np.diff(theta) < 0.0)
        assert abs(theta[0]) < 1e-2
        assert theta[-1] == pytest.approx(-m * np.pi, abs=1e-3)
        assert len(closing_moduli(n, m)) == (m - 1) // 2


@pytest.mark.parametrize("n", sorted(PAPER_MU_OVER_PI))
def test_principal_family_reproduces_paper_table(n: int) -> None:
    family = principal_family(n)
    configuration = family.configuration(0.3)
    assert np.linalg.norm(configuration.closure_residual) < 1e-12
    assert np.linalg.norm(configuration.monodromy_residual) < 1e-12
    assert family.torsion_angle / np.pi == pytest.approx(PAPER_MU_OVER_PI[n], abs=6e-6)
    # The c formula of the variational-recurrence theorem.
    kappa = family.curvatures(0.3)
    assert critical_torsion_cosine(kappa, sign=-1) == pytest.approx(
        family.torsion_cosine, abs=1e-12
    )


def test_minimum_torsion_decreases_and_self_linking_is_three_halves() -> None:
    angles = [minimum_torsion_angle(n) for n in range(7, 14)]
    assert np.all(np.diff(angles) < 0.0)
    family = principal_family(9)
    for phase in np.linspace(0.0, family.phase_period, 4, endpoint=False):
        # compute_linking_number is 2 SL in the paper's normalisation.
        lk = compute_linking_number(family.binormals(phase))
        assert lk == pytest.approx(3.0, abs=1e-6)


def test_seven_hinge_energy_is_function_of_torsion() -> None:
    family = principal_family(7)
    mu = family.torsion_angle
    expected = 2.0 * np.log(1024.0 / 5.0) + 14.0 * np.log(np.sin(mu / 2.0))
    for phase in (0.0, 0.9, 2.1):
        assert first_hamiltonian(family.curvatures(phase)) == pytest.approx(
            expected, abs=1e-12
        )


def test_fixed_torsion_jacobian_loses_one_rank_at_critical_points() -> None:
    family = principal_family(9)
    singular = np.linalg.svd(
        fixed_torsion_jacobian(family.binormals(0.4), sign=-1), compute_uv=False
    )
    assert np.sum(singular > 1e-9 * singular[0]) == 2 * 9 + 2
    generic = Kaleidocycle(
        9, oriented=False, solver_options={"mode": "random_feasible"}, seed=3
    )
    singular = np.linalg.svd(
        fixed_torsion_jacobian(generic.hinges, sign=-1), compute_uv=False
    )
    assert np.sum(singular > 1e-9 * singular[0]) == 2 * 9 + 3


@pytest.mark.parametrize(
    ("n", "m"), [(7, 3), (9, 3), (10, 4), (11, 5), (13, 6), (15, 7)]
)
def test_torsion_index_theorem(n: int, m: int) -> None:
    for k in closing_moduli(n, m):
        family = elliptic_family(n, m, k)
        hessian = torsion_hessian(family.binormals(0.37), sign=family.sign)
        assert hessian.stationarity_residual < 1e-10
        assert hessian.inertia == (n - m - 3, m - 3, 1)
        assert np.allclose(hessian.torsion_multipliers, -1.0 / n, atol=1e-12)
        jacobi = jacobi_operator(
            family.curvatures(0.37), family.multiplier, sign=family.sign
        )
        assert quadratic_inertia(jacobi) == (m, n - m - 1, 1)


def test_multiplier_vector_and_jacobi_kernel() -> None:
    family = principal_family(10)
    hessian = torsion_hessian(family.binormals(0.2), sign=-1)
    np.testing.assert_allclose(
        hessian.closure_multiplier, family.multiplier_vector(0.2), atol=1e-12
    )
    kappa = family.curvatures(0.2)
    first_field = mkdv_hierarchy_field(kappa, 1, sign=-1)
    jacobi = jacobi_operator(kappa, family.multiplier, sign=-1)
    assert np.linalg.norm(jacobi @ first_field) < 1e-12 * np.linalg.norm(first_field)


def test_continuum_constants_and_rates() -> None:
    k_inf, c_inf = continuum_constants()
    assert k_inf == pytest.approx(0.6309247559, abs=1e-9)
    assert c_inf == pytest.approx(8.0934956856, abs=1e-9)
    family = principal_family(60)
    assert abs(family.modulus - k_inf) * 60**2 < 10.0
    assert abs(60 * family.torsion_angle - c_inf) * 60**2 < 40.0


def test_local_dof_default_counts_design_space_dimension() -> None:
    for n in (9, 10):
        kc = Kaleidocycle(
            n, oriented=False, solver_options={"mode": "random_feasible"}, seed=1
        )
        assert kc.local_dof()["dof"] == n - 5


def test_finite_motion_dof_matches_index_theorem() -> None:
    from kaleidocycle import ConstraintConfig
    from kaleidocycle.optimality import finite_motion_dof

    family = principal_family(9)
    config = ConstraintConfig(
        oriented=False, full_alignment=True, reference_torsion=family.torsion_cosine
    )
    # The exact (JAX) Jacobian matters here: near a critical circle the
    # fixed-torsion Jacobian is nearly singular and amplifies the noise of a
    # finite-difference Jacobian.
    info = finite_motion_dof(family.binormals(0.3), config, seed=0, backend="jax")
    # Linearised count N - 5 (rank loss), but the fibre is one circle.
    assert info["infinitesimal_dof"] == 4
    assert info["finite_dof"] == 1

    generic = Kaleidocycle(
        10, oriented=False, solver_options={"mode": "random_feasible"}, seed=0
    )
    torsion = float(generic.hinges[0] @ generic.hinges[1])
    config = ConstraintConfig(
        oriented=False, full_alignment=True, reference_torsion=torsion
    )
    info = generic.finite_motion_dof(config=config, seed=0, backend="jax")
    assert info["infinitesimal_dof"] == info["finite_dof"] == 10 - 6


@pytest.mark.parametrize(
    ("n", "m", "guess"),
    [(7, 3, (0.48, 0.27)), (9, 3, (0.34, 0.29)), (15, 5, (0.40, 0.69))],
)
def test_theta_solutions_are_closed_elliptic_families(n, m, guess) -> None:
    from kaleidocycle import generate_theta_binormals
    from kaleidocycle.theta import solve_closure_conditions

    v, r, y = solve_closure_conditions(n, m=m, initial_guess=guess)
    # t = 0 places zero-curvature vertices exactly on samples for (9, 3) and
    # (15, 5), where the theta formula has a removable singularity.
    binormals = generate_theta_binormals(v, 0.0, r, y, n, t=0.0)
    dots = np.einsum("ij,ij->i", binormals[:-1], binormals[1:])
    assert np.ptp(dots) < 1e-8
    torsion = float(np.arccos(np.mean(dots)))
    closed = [elliptic_family(n, m, k).torsion_angle for k in closing_moduli(n, m)]
    assert min(abs(torsion - angle) for angle in closed) < 1e-7
