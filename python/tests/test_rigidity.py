"""Tests for rigidity and finite-motion diagnostics."""

from __future__ import annotations

import numpy as np

from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.rigidity import (
    calladine_summary,
    connelly_second_order_test,
    constraint_residual_vector,
    pellegrino_calladine_analysis,
    rigidity_svd,
    stress_hessian,
    trace_finite_motion,
)


def _unit_only_config() -> ConstraintConfig:
    return ConstraintConfig(
        closure=False,
        alignment=False,
        constant_torsion=False,
    )


def test_rigidity_svd_finds_mechanisms_for_unit_constraints() -> None:
    hinges = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    analysis = rigidity_svd(hinges, _unit_only_config())

    assert analysis.n_constraints == 2
    assert analysis.n_variables == 9
    assert analysis.rank == 2
    assert analysis.nullity == 7
    assert analysis.self_stress_count == 0
    assert np.linalg.norm(analysis.jacobian @ analysis.mechanism_basis) < 1e-10


def test_scalar_alignment_rank_zero_row_creates_self_stress() -> None:
    hinges = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    config = ConstraintConfig(
        closure=False,
        alignment=True,
        full_alignment=False,
        oriented=True,
        constant_torsion=False,
    )

    analysis = rigidity_svd(hinges, config)

    assert analysis.n_constraints == 3
    assert analysis.rank == 2
    assert analysis.self_stress_count == 1
    assert (
        np.linalg.norm(analysis.jacobian.T @ analysis.self_stress_basis)
        < analysis.tolerance
    )


def test_calladine_summary_matches_raw_count() -> None:
    hinges = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    summary = calladine_summary(
        hinges,
        _unit_only_config(),
        subtract_rigid=False,
    )
    alias = pellegrino_calladine_analysis(
        hinges,
        _unit_only_config(),
        subtract_rigid=False,
    )

    assert summary.calladine_index == summary.expected_index
    assert summary.mechanisms - summary.self_stresses == (
        summary.n_variables - summary.n_constraints
    )
    assert alias.calladine_index == summary.calladine_index


def test_stress_hessian_for_unit_norm_constraint() -> None:
    hinges = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    hessian = stress_hessian(
        hinges,
        _unit_only_config(),
        stress=np.array([1.0, 0.0]),
        eps=1e-5,
    )

    expected_diag = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.diag(hessian), expected_diag, atol=1e-4)
    off_diag = hessian - np.diag(np.diag(hessian))
    assert np.linalg.norm(off_diag) < 1e-6


def test_connelly_second_order_test_accepts_explicit_stress_and_direction() -> None:
    hinges = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tangent = np.zeros(hinges.size)
    tangent[1] = 1.0

    result = connelly_second_order_test(
        hinges,
        _unit_only_config(),
        stress=np.array([1.0, 0.0]),
        tangent_basis=tangent,
        eps=1e-5,
    )

    assert result.quadratic_form.shape == (1, 1)
    assert result.eigenvalues[0] > 1.0
    assert result.positive_definite


def test_trace_finite_motion_stays_on_constraint_manifold() -> None:
    hinges = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    config = _unit_only_config()

    frames = trace_finite_motion(
        hinges,
        config,
        direction_index=0,
        step_size=1e-3,
        n_steps=4,
        bidirectional=False,
        subtract_rigid=False,
        correction_tol=1e-10,
    )

    assert frames.ndim == 3
    assert frames.shape[1:] == hinges.shape
    assert frames.shape[0] >= 2
    for frame in frames:
        assert np.linalg.norm(constraint_residual_vector(frame, config)) < 1e-8
