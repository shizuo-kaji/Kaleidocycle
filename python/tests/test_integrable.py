"""Tests for the integrable curvature-coordinate implementation."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle.energies import bending_energy
from kaleidocycle.integrable import (
    cayley_curvatures,
    critical_multiplier,
    critical_torsion_cosine,
    curvature_angles,
    curvature_weights,
    first_hamiltonian,
    framed_polygon_from_binormals,
    hierarchy_hamiltonian_gradients,
    hierarchy_hamiltonians,
    hierarchy_orbit_jacobian,
    integrate_curvature_flow,
    lift_coefficients,
    lifted_velocities,
    mkdv1_field,
    mkdv2_field,
    mkdv_hierarchy_field,
    poisson_operator,
    qrt_invariant,
    reconstruct_framed_polygon,
    second_hamiltonian,
    sine_gordon_field,
    sine_gordon_potential,
    spectral_curve_diagnostics,
    spectral_integral_gradients,
    spectral_integrals,
    twisted_shift,
    twisted_trace_polynomial,
)


@pytest.fixture
def anti_oriented_binormals() -> np.ndarray:
    """A closed, constant-torsion, anti-oriented 10-kaleidocycle."""

    return np.array(
        [
            [-0.197210754958889, -0.736268758601003, 0.647314632336303],
            [-0.090868831545127, -0.643711301536718, -0.759854338493588],
            [-0.846918417946784, 0.451354241300241, -0.281084581937182],
            [0.259694269524323, 0.812408626358509, 0.522064277837997],
            [0.729208721526919, 0.189447700220193, -0.657544074059189],
            [-0.034710105016990, -0.949435119356560, -0.312038719940498],
            [0.387669042828578, -0.300570630020176, 0.871418389524601],
            [-0.273163189350037, 0.865424864800357, 0.420025803218565],
            [-0.960227768263853, -0.271553598112736, -0.064971350663561],
            [0.238455153539984, -0.676477142203325, -0.696791084778390],
            [0.197210754958889, 0.736268758601003, -0.647314632336303],
        ]
    )


@pytest.fixture
def generic_anti_oriented_binormals() -> np.ndarray:
    """A non-critical closed anti-oriented 10-kaleidocycle."""

    return np.array(
        [
            [-0.795909536484102, 0.209891863157631, 0.567867427762748],
            [0.762955642631478, 0.646223372332325, 0.017147607069958],
            [0.120557631349841, -0.843151320264701, -0.523986363046986],
            [-0.209079870325557, 0.834606416986250, -0.509625094113331],
            [-0.033931436443466, 0.047673237992738, 0.998286491945159],
            [0.895599587297800, 0.086576727442052, -0.436355187315832],
            [-0.776685476425861, 0.443047293797991, -0.447737385268876],
            [0.715282974790076, -0.380756055013228, -0.585999225721522],
            [0.288008403235251, 0.571634867896427, 0.768299900735979],
            [-0.848503026326882, 0.247905210461928, -0.467531411714710],
            [0.795909536484102, -0.209891863157631, -0.567867427762748],
        ]
    )


@pytest.fixture
def generic_k15_anti_oriented_binormals() -> np.ndarray:
    """A non-critical closed anti-oriented 15-kaleidocycle."""

    return np.array(
        [
            [0.096909308257180, 0.399943796085817, -0.911401967271073],
            [-0.101429107168040, -0.889802748881254, -0.444930561214285],
            [0.422531335769680, -0.477967346268420, 0.770074338095884],
            [-0.886753094132892, -0.183999864160197, 0.424043629871704],
            [0.358729410857937, 0.170022085064301, 0.917826617818368],
            [-0.143302148845240, -0.950568436905805, 0.275470762322958],
            [-0.510303096827173, -0.206855953854991, -0.834746287037767],
            [0.752659433592397, -0.542263317768722, -0.373435765864534],
            [-0.163908343071642, 0.337446132533754, -0.926965027770477],
            [-0.989714933244920, 0.024433015565180, 0.140951689107989],
            [0.078625902565157, -0.414933603329930, 0.906448052716456],
            [-0.795735960758465, -0.585458527198868, -0.155056743439085],
            [0.488227399114590, -0.552706302695304, -0.675388591637948],
            [0.116231136708072, -0.756209277090423, 0.643923793706934],
            [-0.900848493298941, -0.375959323153869, -0.217086571336673],
            [-0.096909308257180, -0.399943796085817, 0.911401967271073],
        ]
    )


def test_twisted_shift_handles_both_boundaries() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        twisted_shift(values, 1, sign=-1), [2.0, 3.0, 4.0, -1.0]
    )
    np.testing.assert_array_equal(
        twisted_shift(values, -1, sign=-1), [-4.0, 1.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(
        twisted_shift(values, 5, sign=-1), [-2.0, -3.0, -4.0, 1.0]
    )


def test_curvature_coordinate_round_trip() -> None:
    curvatures = np.array([-3.0, -0.4, 0.0, 0.8, 4.0])
    np.testing.assert_allclose(
        cayley_curvatures(curvature_angles(curvatures)), curvatures
    )


def test_one_dimensional_bending_energy_is_first_hamiltonian() -> None:
    curvatures = np.array([0.2, -0.7, 1.1, 0.4, -0.3])
    assert bending_energy(curvatures) == pytest.approx(first_hamiltonian(curvatures))


def test_reconstruction_satisfies_local_frenet_relations() -> None:
    curvatures = np.array([0.2, -0.7, 1.1, 0.4, -0.3])
    mu = 0.83
    polygon = reconstruct_framed_polygon(curvatures, mu, sign=-1)

    np.testing.assert_allclose(np.linalg.det(polygon.frames), 1.0, atol=1e-13)
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", polygon.binormals[:-1], polygon.binormals[1:]),
        np.cos(mu),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.diff(polygon.vertices, axis=0), polygon.tangents, atol=1e-13
    )


def test_binormal_curvature_reconstruction_round_trip(
    anti_oriented_binormals: np.ndarray,
) -> None:
    polygon = framed_polygon_from_binormals(anti_oriented_binormals)
    reconstructed = reconstruct_framed_polygon(
        polygon.curvatures,
        polygon.torsion_angle,
        sign=polygon.sign,
        initial_frame=polygon.frames[0],
    )

    assert polygon.sign == -1
    np.testing.assert_allclose(reconstructed.binormals, anti_oriented_binormals)
    assert np.linalg.norm(reconstructed.closure_residual) < 1e-12
    assert np.linalg.norm(reconstructed.monodromy_residual) < 1e-12


@pytest.mark.parametrize("sign", [1, -1])
def test_mkdv_fields_are_hamiltonian(sign: int) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    d = curvature_weights(kappa)
    grad_e1 = kappa / (2.0 * d)
    grad_e2 = 0.5 * (
        twisted_shift(kappa, -1, sign=sign) + twisted_shift(kappa, 1, sign=sign)
    )

    np.testing.assert_allclose(
        poisson_operator(kappa, grad_e1, sign=sign),
        mkdv1_field(kappa, sign=sign),
    )
    np.testing.assert_allclose(
        poisson_operator(kappa, grad_e2, sign=sign),
        mkdv2_field(kappa, sign=sign),
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_spectral_series_recovers_low_order_hamiltonians(sign: int) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    integrals = spectral_integrals(kappa, 3, sign=sign)
    gradients = spectral_integral_gradients(kappa, 3, sign=sign)
    z = kappa / 2.0
    expected_i3 = np.sum(
        z**2 * twisted_shift(z, 1, sign=sign) ** 2
        + 2.0 * z * twisted_shift(z, 1, sign=sign) ** 2 * twisted_shift(z, 2, sign=sign)
        + 2.0 * z * twisted_shift(z, 2, sign=sign)
    )

    assert integrals[0] == pytest.approx(first_hamiltonian(kappa))
    assert integrals[1] == pytest.approx(second_hamiltonian(kappa, sign=sign))
    assert integrals[2] == pytest.approx(expected_i3)

    epsilon = 1e-6
    finite_difference = np.empty_like(gradients)
    for index in range(kappa.size):
        perturbation = np.zeros_like(kappa)
        perturbation[index] = epsilon
        finite_difference[:, index] = (
            spectral_integrals(kappa + perturbation, 3, sign=sign)
            - spectral_integrals(kappa - perturbation, 3, sign=sign)
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(gradients, finite_difference, atol=2e-9)


@pytest.mark.parametrize("sign", [1, -1])
def test_hierarchy_extends_existing_flows_and_hamiltonians(sign: int) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    integrals = spectral_integrals(kappa, 5, sign=sign)
    hamiltonians = hierarchy_hamiltonians(kappa, 5, sign=sign)

    assert hamiltonians[0] == pytest.approx(first_hamiltonian(kappa))
    assert hamiltonians[1] == pytest.approx(second_hamiltonian(kappa, sign=sign))
    assert hamiltonians[2] == pytest.approx(integrals[2] + 2.0 * integrals[0])
    assert hamiltonians[3] == pytest.approx(integrals[3] + 3.0 * integrals[1])
    assert hamiltonians[4] == pytest.approx(
        integrals[4] + 4.0 * integrals[2] + 6.0 * integrals[0]
    )
    np.testing.assert_allclose(
        mkdv_hierarchy_field(kappa, 1, sign=sign),
        mkdv1_field(kappa, sign=sign),
    )
    np.testing.assert_allclose(
        mkdv_hierarchy_field(kappa, 2, sign=sign),
        mkdv2_field(kappa, sign=sign),
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_hierarchy_hamiltonians_are_in_involution(sign: int) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    gradients = hierarchy_hamiltonian_gradients(kappa, 5, sign=sign)
    fields = np.array(
        [mkdv_hierarchy_field(kappa, order, sign=sign) for order in range(1, 6)]
    )

    np.testing.assert_allclose(gradients @ fields.T, 0.0, atol=6e-14)


def test_hierarchy_orbit_jacobian_reaches_cayley_hamilton_bound() -> None:
    kappa = np.array([-1.2, 0.4, 1.1, -0.7, 0.2, 1.5, -0.9, 0.8, -0.3, 0.6])
    jacobian = hierarchy_orbit_jacobian(kappa, sign=-1)

    assert jacobian.shape == (10, 10)
    for order in range(1, 11):
        np.testing.assert_allclose(
            jacobian[:, order - 1],
            mkdv_hierarchy_field(kappa, order, sign=-1),
        )
    normalised = jacobian / np.linalg.norm(jacobian, axis=0)
    assert np.linalg.matrix_rank(normalised, tol=1e-10) == 5


def test_closed_k10_flow_rank_matches_spectral_prym_dimension(
    generic_anti_oriented_binormals: np.ndarray,
) -> None:
    polygon = framed_polygon_from_binormals(generic_anti_oriented_binormals)
    jacobian = hierarchy_orbit_jacobian(polygon.curvatures, sign=-1)
    normalised = jacobian / np.linalg.norm(jacobian, axis=0)
    singular_values = np.linalg.svd(normalised, compute_uv=False)
    spectral = spectral_curve_diagnostics(
        polygon.curvatures,
        sign=-1,
        torsion_angle=polygon.torsion_angle,
    )

    assert np.linalg.matrix_rank(normalised, tol=1e-10) == 3
    assert singular_values[2] > 1e-1
    assert singular_values[3] < 1e-12
    assert spectral.arithmetic_genus == 9
    assert spectral.geometric_genus == 5
    assert spectral.quotient_genus == 2
    assert spectral.prym_dimension == 3
    assert spectral.singular_factor_residual is not None
    assert spectral.singular_factor_residual < 1e-12
    assert spectral.reciprocal_residual < 1e-12
    assert spectral.minimum_root_separation > 1e-2


def test_closed_k15_flow_rank_matches_spectral_prym_dimension(
    generic_k15_anti_oriented_binormals: np.ndarray,
) -> None:
    polygon = framed_polygon_from_binormals(generic_k15_anti_oriented_binormals)
    jacobian = hierarchy_orbit_jacobian(polygon.curvatures, sign=-1)
    normalised = jacobian / np.linalg.norm(jacobian, axis=0)
    singular_values = np.linalg.svd(normalised, compute_uv=False)
    spectral = spectral_curve_diagnostics(
        polygon.curvatures,
        sign=-1,
        torsion_angle=polygon.torsion_angle,
    )

    assert np.linalg.matrix_rank(normalised, tol=1e-10) == 5
    assert singular_values[4] > 1e-1
    assert singular_values[5] < 1e-12
    assert spectral.arithmetic_genus == 14
    assert spectral.geometric_genus == 10
    assert spectral.quotient_genus == 5
    assert spectral.prym_dimension == 5
    assert spectral.singular_factor_residual is not None
    assert spectral.singular_factor_residual < 1e-12
    assert spectral.reciprocal_residual < 1e-12
    assert spectral.minimum_root_separation > 1e-2


def test_twisted_trace_polynomial_has_expected_reciprocity() -> None:
    kappa = np.array([-1.2, 0.4, 1.1, -0.7, 0.2, 1.5, -0.9, 0.8, -0.3, 0.6])

    periodic = twisted_trace_polynomial(kappa, sign=1)
    anti_periodic = twisted_trace_polynomial(kappa, sign=-1)

    np.testing.assert_allclose(periodic[::-1], periodic, atol=1e-14)
    np.testing.assert_allclose(anti_periodic[::-1], -anti_periodic, atol=1e-14)


@pytest.mark.parametrize("sign", [1, -1])
def test_third_hierarchy_field_matches_local_recursion_formula(sign: int) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    z = kappa / 2.0
    weight = 1.0 + z**2
    k1 = weight * (twisted_shift(z, 1, sign=sign) - twisted_shift(z, -1, sign=sign))
    k2 = weight * (
        twisted_shift(weight, 1) * (z + twisted_shift(z, 2, sign=sign))
        - twisted_shift(weight, -1) * (twisted_shift(z, -2, sign=sign) + z)
    )
    z_m1 = twisted_shift(z, -1, sign=sign)
    z_m2 = twisted_shift(z, -2, sign=sign)
    z_p1 = twisted_shift(z, 1, sign=sign)
    primitive = (
        2.0 * (z_m1 * z_p1 + z_m2 * z)
        + 2.0 * z_m1**2 * z**2
        + 2.0 * z_m1 * z**2 * z_p1
        + 2.0 * z_m2 * z_m1**2 * z
    )
    expected = (
        weight * twisted_shift(k2, 1, sign=sign)
        + 2.0 * z * z_p1 * k2
        + weight * twisted_shift(k2, -1, sign=sign)
        + k1 * primitive
    )

    np.testing.assert_allclose(
        mkdv_hierarchy_field(kappa, 3, sign=sign), expected, atol=2e-14
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("field", [mkdv1_field, mkdv2_field])
def test_mkdv_fields_preserve_both_hamiltonians(sign: int, field) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    d = curvature_weights(kappa)
    velocity = field(kappa, sign=sign)
    grad_e1 = kappa / (2.0 * d)
    grad_e2 = 0.5 * (
        twisted_shift(kappa, -1, sign=sign) + twisted_shift(kappa, 1, sign=sign)
    )

    assert abs(np.dot(grad_e1, velocity)) < 1e-13
    assert abs(np.dot(grad_e2, velocity)) < 1e-13


def test_sine_gordon_potential_and_conservation() -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    u = sine_gordon_potential(kappa)
    phi = curvature_angles(kappa)
    np.testing.assert_allclose(twisted_shift(u, -1, sign=-1) - u, phi, atol=1e-14)

    velocity = sine_gordon_field(kappa, 0.9)
    d = curvature_weights(kappa)
    grad_e1 = kappa / (2.0 * d)
    grad_e2 = 0.5 * (
        twisted_shift(kappa, -1, sign=-1) + twisted_shift(kappa, 1, sign=-1)
    )
    assert abs(np.dot(grad_e1, velocity)) < 1e-13
    assert abs(np.dot(grad_e2, velocity)) < 1e-13


@pytest.mark.parametrize("flow", ["mkdv1", "mkdv2", "mkdv3"])
def test_lift_coefficients_satisfy_compatibility(flow: str) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    mu = 0.9
    coefficients = lift_coefficients(kappa, mu, flow=flow, sign=-1)
    field = mkdv_hierarchy_field(kappa, int(flow[4:]), sign=-1)
    angles = curvature_angles(kappa)
    d = curvature_weights(kappa)

    for n in range(kappa.size - 1):
        cosine = np.cos(angles[n + 1])
        sine = np.sin(angles[n + 1])
        r1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(mu), -np.sin(mu)],
                [0.0, np.sin(mu), np.cos(mu)],
            ]
        )
        r3 = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        transition = r1 @ r3
        expected_angular = transition.T @ coefficients.angular[n] + np.array(
            [0.0, 0.0, field[n + 1] / d[n + 1]]
        )
        np.testing.assert_allclose(
            coefficients.angular[n + 1], expected_angular, atol=2e-14
        )

        omega_e1 = np.array(
            [0.0, coefficients.angular[n, 2], -coefficients.angular[n, 1]]
        )
        np.testing.assert_allclose(
            transition @ coefficients.vertex[n + 1] - coefficients.vertex[n],
            omega_e1,
            atol=2e-14,
        )


def test_lifted_velocity_has_twisted_terminal_value(
    anti_oriented_binormals: np.ndarray,
) -> None:
    polygon = framed_polygon_from_binormals(anti_oriented_binormals)
    vertex_velocity, binormal_velocity = lifted_velocities(polygon, flow="sine-gordon")
    np.testing.assert_allclose(vertex_velocity[-1], vertex_velocity[0])
    np.testing.assert_allclose(binormal_velocity[-1], -binormal_velocity[0])


def test_qrt_invariant_for_generated_recurrence() -> None:
    multiplier = 1.3
    sequence = [0.2, 0.7]
    for _ in range(7):
        previous, current = sequence[-2:]
        sequence.append(multiplier * current / (1.0 + current**2 / 4.0) - previous)
    invariant = qrt_invariant(np.array(sequence), multiplier)
    np.testing.assert_allclose(invariant[1:], invariant[1], atol=2e-14)


def test_critical_torsion_and_multiplier_formulas(
    anti_oriented_binormals: np.ndarray,
) -> None:
    polygon = framed_polygon_from_binormals(anti_oriented_binormals)
    cosine = critical_torsion_cosine(polygon.curvatures, sign=-1)
    multiplier = critical_multiplier(polygon.curvatures, cosine, sign=-1)

    assert cosine == pytest.approx(np.cos(polygon.torsion_angle), abs=1e-12)
    assert multiplier == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        mkdv2_field(polygon.curvatures, sign=-1),
        multiplier * mkdv1_field(polygon.curvatures, sign=-1),
        atol=2e-7,
    )


@pytest.mark.parametrize("flow", ["mkdv1", "mkdv2", "mkdv3", "sine-gordon"])
def test_numerical_flow_preserves_closed_configuration(
    anti_oriented_binormals: np.ndarray, flow: str
) -> None:
    initial = framed_polygon_from_binormals(anti_oriented_binormals)
    evolution = integrate_curvature_flow(
        initial.curvatures,
        initial.torsion_angle,
        np.linspace(0.0, 0.1, 5),
        flow=flow,
        sign=-1,
        initial_frame=initial.frames[0],
    )
    assert np.ptp(evolution.first_hamiltonian) < 2e-10
    assert np.ptp(evolution.second_hamiltonian) < 2e-10
    assert first_hamiltonian(evolution.curvatures[0]) == pytest.approx(
        evolution.first_hamiltonian[0]
    )
    assert second_hamiltonian(evolution.curvatures[0], sign=-1) == pytest.approx(
        evolution.second_hamiltonian[0]
    )
    if flow == "mkdv3":
        assert np.max(np.ptp(evolution.hierarchy_hamiltonians(3), axis=0)) < 3e-10
    for configuration in evolution.configurations():
        assert np.linalg.norm(configuration.closure_residual) < 2e-9
        assert np.linalg.norm(configuration.monodromy_residual) < 2e-9
