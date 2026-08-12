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
    integrate_curvature_flow,
    lift_coefficients,
    lifted_velocities,
    mkdv1_field,
    mkdv2_field,
    poisson_operator,
    qrt_invariant,
    reconstruct_framed_polygon,
    second_hamiltonian,
    sine_gordon_field,
    sine_gordon_potential,
    twisted_shift,
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
    assert bending_energy(curvatures) == pytest.approx(
        first_hamiltonian(curvatures)
    )


def test_reconstruction_satisfies_local_frenet_relations() -> None:
    curvatures = np.array([0.2, -0.7, 1.1, 0.4, -0.3])
    mu = 0.83
    polygon = reconstruct_framed_polygon(curvatures, mu, sign=-1)

    np.testing.assert_allclose(
        np.linalg.det(polygon.frames), 1.0, atol=1e-13
    )
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
        twisted_shift(kappa, -1, sign=sign)
        + twisted_shift(kappa, 1, sign=sign)
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
@pytest.mark.parametrize("field", [mkdv1_field, mkdv2_field])
def test_mkdv_fields_preserve_both_hamiltonians(sign: int, field) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    d = curvature_weights(kappa)
    velocity = field(kappa, sign=sign)
    grad_e1 = kappa / (2.0 * d)
    grad_e2 = 0.5 * (
        twisted_shift(kappa, -1, sign=sign)
        + twisted_shift(kappa, 1, sign=sign)
    )

    assert abs(np.dot(grad_e1, velocity)) < 1e-13
    assert abs(np.dot(grad_e2, velocity)) < 1e-13


def test_sine_gordon_potential_and_conservation() -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    u = sine_gordon_potential(kappa)
    phi = curvature_angles(kappa)
    np.testing.assert_allclose(
        twisted_shift(u, -1, sign=-1) - u, phi, atol=1e-14
    )

    velocity = sine_gordon_field(kappa, 0.9)
    d = curvature_weights(kappa)
    grad_e1 = kappa / (2.0 * d)
    grad_e2 = 0.5 * (
        twisted_shift(kappa, -1, sign=-1)
        + twisted_shift(kappa, 1, sign=-1)
    )
    assert abs(np.dot(grad_e1, velocity)) < 1e-13
    assert abs(np.dot(grad_e2, velocity)) < 1e-13


@pytest.mark.parametrize("flow", ["mkdv1", "mkdv2"])
def test_lift_coefficients_satisfy_compatibility(flow: str) -> None:
    kappa = np.array([0.3, -0.8, 1.2, 0.1, -0.4, 0.7])
    mu = 0.9
    coefficients = lift_coefficients(kappa, mu, flow=flow, sign=-1)
    field = mkdv1_field(kappa, sign=-1) if flow == "mkdv1" else mkdv2_field(
        kappa, sign=-1
    )
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
        r3 = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        transition = r1 @ r3
        expected_angular = (
            transition.T @ coefficients.angular[n]
            + np.array([0.0, 0.0, field[n + 1] / d[n + 1]])
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
    vertex_velocity, binormal_velocity = lifted_velocities(
        polygon, flow="sine-gordon"
    )
    np.testing.assert_allclose(vertex_velocity[-1], vertex_velocity[0])
    np.testing.assert_allclose(binormal_velocity[-1], -binormal_velocity[0])


def test_qrt_invariant_for_generated_recurrence() -> None:
    multiplier = 1.3
    sequence = [0.2, 0.7]
    for _ in range(7):
        previous, current = sequence[-2:]
        sequence.append(
            multiplier * current / (1.0 + current**2 / 4.0) - previous
        )
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


@pytest.mark.parametrize("flow", ["mkdv1", "mkdv2", "sine-gordon"])
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
    for configuration in evolution.configurations():
        assert np.linalg.norm(configuration.closure_residual) < 2e-9
        assert np.linalg.norm(configuration.monodromy_residual) < 2e-9
