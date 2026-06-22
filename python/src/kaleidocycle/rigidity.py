"""Rigidity and finite-motion diagnostics for kaleidocycle constraints.

This module exposes the linear and second-order layers commonly used in
finite-motion analysis:

- Pellegrino/Calladine: rigidity matrix rank, mechanisms, and self-stress.
- Connelly: stress-weighted second variation on infinitesimal mechanisms.
- Whiteley-style bookkeeping: explicit constraint/variable count diagnostics.

The actual constraints are still defined by :mod:`kaleidocycle.constraints`;
this module only differentiates and interprets them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .constraints import ConstraintConfig, constraint_residuals
from .optimality import (
    compute_constraint_jacobian,
    finite_motion_dof,
    follow_motion,
)


@dataclass(slots=True)
class RigiditySVD:
    """SVD decomposition of the constraint rigidity matrix.

    The matrix is ``J = dF/dq`` where ``F`` is the flattened constraint
    residual vector and ``q`` is the flattened hinge array. Columns of
    ``mechanism_basis`` span ``ker J``. Columns of ``self_stress_basis``
    span ``ker J.T``.
    """

    jacobian: NDArray[np.float64]
    residual: NDArray[np.float64]
    residual_labels: tuple[str, ...]
    singular_values: NDArray[np.float64]
    rank: int
    tolerance: float
    mechanism_basis: NDArray[np.float64]
    self_stress_basis: NDArray[np.float64]

    @property
    def n_constraints(self) -> int:
        """Number of scalar constraints."""

        return int(self.jacobian.shape[0])

    @property
    def n_variables(self) -> int:
        """Number of scalar variables."""

        return int(self.jacobian.shape[1])

    @property
    def nullity(self) -> int:
        """Dimension of the infinitesimal mechanism space."""

        return int(self.mechanism_basis.shape[1])

    @property
    def self_stress_count(self) -> int:
        """Dimension of the self-stress space."""

        return int(self.self_stress_basis.shape[1])


@dataclass(slots=True)
class CalladineSummary:
    """Pellegrino/Calladine rank and mobility count."""

    n_variables: int
    n_constraints: int
    rank: int
    mechanisms: int
    self_stresses: int
    rigid_mechanisms: int
    internal_mechanisms: int
    calladine_index: int
    expected_index: int
    singular_values: NDArray[np.float64]
    tolerance: float


@dataclass(slots=True)
class SecondOrderStressTest:
    """Connelly-style stress-weighted second-order test result."""

    stress: NDArray[np.float64]
    hessian: NDArray[np.float64]
    tangent_basis: NDArray[np.float64]
    quadratic_form: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    positive_definite: bool
    negative_definite: bool
    semidefinite: bool
    tolerance: float


def flatten_residuals(
    residuals: dict[str, NDArray[np.float64] | float],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Flatten grouped residuals in deterministic key order.

    Returns both the flat numeric vector and a parallel tuple of group labels.
    The ordering intentionally matches ``optimality._flatten_residuals``.
    """

    values: list[NDArray[np.float64]] = []
    labels: list[str] = []
    for key in sorted(residuals):
        arr = np.asarray(residuals[key], dtype=float).reshape(-1)
        values.append(arr)
        labels.extend([key] * arr.size)
    if not values:
        return np.array([], dtype=float), tuple()
    return np.concatenate(values), tuple(labels)


def constraint_residual_vector(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
) -> NDArray[np.float64]:
    """Return the flattened constraint residual vector ``F(q)``."""

    residual, _labels = flatten_residuals(constraint_residuals(hinges, config))
    return residual


def rigidity_matrix(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> NDArray[np.float64]:
    """Return the rigidity matrix ``dF/dq`` for the selected constraints."""

    return compute_constraint_jacobian(
        hinges,
        config,
        eps=finite_diff_step,
        backend=backend,
    )


def _svd_tolerance(
    singular_values: NDArray[np.float64],
    shape: tuple[int, int],
    tol: float | None,
) -> float:
    if tol is not None:
        return float(tol)
    max_sv = float(singular_values[0]) if singular_values.size else 0.0
    return float(max(shape) * np.finfo(float).eps * max(max_sv, 1.0))


def rigidity_svd(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    tol: float | None = None,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> RigiditySVD:
    """Compute mechanisms and self-stresses from the constraint Jacobian.

    This is the numerical core of the Pellegrino/Calladine analysis.
    """

    hinges_arr = np.asarray(hinges, dtype=float)
    if hinges_arr.ndim != 2 or hinges_arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {hinges_arr.shape}"
        raise ValueError(msg)

    residuals = constraint_residuals(hinges_arr, config)
    residual, labels = flatten_residuals(residuals)
    jacobian = rigidity_matrix(
        hinges_arr,
        config,
        finite_diff_step=finite_diff_step,
        backend=backend,
    )

    u, singular_values, vt = np.linalg.svd(jacobian, full_matrices=True)
    tol_eff = _svd_tolerance(singular_values, jacobian.shape, tol)
    if tol is None and backend != "jax":
        tol_eff = max(tol_eff, 10.0 * finite_diff_step)
    rank = int(np.sum(singular_values > tol_eff))

    mechanism_basis = vt[rank:].T.copy()
    self_stress_basis = u[:, rank:].copy()

    return RigiditySVD(
        jacobian=jacobian,
        residual=residual,
        residual_labels=labels,
        singular_values=singular_values,
        rank=rank,
        tolerance=tol_eff,
        mechanism_basis=mechanism_basis,
        self_stress_basis=self_stress_basis,
    )


def _rigid_rotation_basis(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return infinitesimal global rotations ``delta h_i = omega x h_i``."""

    h = np.asarray(hinges, dtype=float)
    basis = np.zeros((h.size, 3), dtype=float)
    for i, axis in enumerate(np.eye(3)):
        basis[:, i] = np.cross(axis, h).reshape(-1)

    u, s, _vt = np.linalg.svd(basis, full_matrices=False)
    if not s.size:
        return np.zeros((h.size, 0), dtype=float)
    rank = int(np.sum(s > 1e-12 * max(float(s[0]), 1.0)))
    return u[:, :rank]


def _subspace_overlap_dimension(
    basis: NDArray[np.float64],
    candidates: NDArray[np.float64],
    *,
    tol: float = 1e-6,
) -> int:
    """Count independent candidate columns contained in a subspace basis."""

    if basis.size == 0 or candidates.size == 0:
        return 0
    projected = basis @ (basis.T @ candidates)
    residual = candidates - projected
    mask = np.linalg.norm(residual, axis=0) < tol
    accepted = candidates[:, mask]
    if accepted.size == 0:
        return 0
    _u, s, _vt = np.linalg.svd(accepted, full_matrices=False)
    if not s.size:
        return 0
    return int(np.sum(s > 1e-10 * max(float(s[0]), 1.0)))


def calladine_summary(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    subtract_rigid: bool = True,
    tol: float | None = None,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> CalladineSummary:
    """Return the Pellegrino/Calladine mobility and self-stress count.

    The raw identity is
    ``mechanisms - self_stresses = n_variables - n_constraints``.
    With ``subtract_rigid=True``, global rotations contained in the
    mechanism space are quotiented out of the reported internal mechanism
    count.
    """

    analysis = rigidity_svd(
        hinges,
        config,
        tol=tol,
        finite_diff_step=finite_diff_step,
        backend=backend,
    )
    mechanisms = analysis.nullity
    self_stresses = analysis.self_stress_count

    rigid_mechanisms = 0
    if subtract_rigid and not config.enforce_anchors and mechanisms > 0:
        rigid = _rigid_rotation_basis(np.asarray(hinges, dtype=float))
        rigid_mechanisms = _subspace_overlap_dimension(
            analysis.mechanism_basis,
            rigid,
        )

    internal = max(mechanisms - rigid_mechanisms, 0)
    calladine_index = internal - self_stresses
    expected_index = (
        analysis.n_variables - analysis.n_constraints - rigid_mechanisms
    )

    return CalladineSummary(
        n_variables=analysis.n_variables,
        n_constraints=analysis.n_constraints,
        rank=analysis.rank,
        mechanisms=mechanisms,
        self_stresses=self_stresses,
        rigid_mechanisms=rigid_mechanisms,
        internal_mechanisms=internal,
        calladine_index=int(calladine_index),
        expected_index=int(expected_index),
        singular_values=analysis.singular_values,
        tolerance=analysis.tolerance,
    )


def pellegrino_calladine_analysis(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    **kwargs,
) -> CalladineSummary:
    """Alias for :func:`calladine_summary` using the literature names."""

    return calladine_summary(hinges, config, **kwargs)


def stress_hessian(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    stress: NDArray[np.float64],
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Finite-difference Hessian of the stress-weighted constraint residual.

    For stress vector ``w`` this returns the Hessian of
    ``phi(q) = w · F(q)``. In Connelly's second-order test, ``w`` is
    normally a self-stress, i.e. a vector in ``ker J.T``.
    """

    h0 = np.asarray(hinges, dtype=float)
    if h0.ndim != 2 or h0.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {h0.shape}"
        raise ValueError(msg)

    stress_arr = np.asarray(stress, dtype=float).reshape(-1)
    residual0 = constraint_residual_vector(h0, config)
    if stress_arr.shape != residual0.shape:
        msg = (
            f"stress has length {stress_arr.size}, but residual has "
            f"length {residual0.size}"
        )
        raise ValueError(msg)

    x0 = h0.reshape(-1)
    n_vars = x0.size
    hessian = np.zeros((n_vars, n_vars), dtype=float)

    def phi(x: NDArray[np.float64]) -> float:
        h = x.reshape(h0.shape)
        return float(stress_arr @ constraint_residual_vector(h, config))

    for i in range(n_vars):
        ei = np.zeros(n_vars, dtype=float)
        ei[i] = eps
        for j in range(i, n_vars):
            ej = np.zeros(n_vars, dtype=float)
            ej[j] = eps
            value = (
                phi(x0 + ei + ej)
                - phi(x0 + ei - ej)
                - phi(x0 - ei + ej)
                + phi(x0 - ei - ej)
            ) / (4.0 * eps * eps)
            hessian[i, j] = value
            hessian[j, i] = value

    return 0.5 * (hessian + hessian.T)


def _as_direction_matrix(
    directions: NDArray[np.float64],
    n_variables: int,
) -> NDArray[np.float64]:
    arr = np.asarray(directions, dtype=float)
    if arr.ndim == 1:
        if arr.size != n_variables:
            msg = f"direction has length {arr.size}, expected {n_variables}"
            raise ValueError(msg)
        return arr.reshape(n_variables, 1)
    if arr.ndim == 2:
        if arr.shape[0] == n_variables:
            return arr
        if arr.size == n_variables:
            return arr.reshape(n_variables, 1)
    if arr.ndim == 3 and arr.shape[0] * arr.shape[1] == n_variables:
        return arr.reshape(n_variables, arr.shape[2])
    msg = f"cannot interpret direction array with shape {arr.shape}"
    raise ValueError(msg)


def second_order_form(
    hessian: NDArray[np.float64],
    directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Restrict a Hessian to the span of the given direction columns."""

    hess = np.asarray(hessian, dtype=float)
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
        msg = f"hessian must be square, got shape {hess.shape}"
        raise ValueError(msg)
    basis = _as_direction_matrix(directions, hess.shape[0])
    return basis.T @ hess @ basis


def connelly_second_order_test(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    stress: NDArray[np.float64] | None = None,
    stress_index: int = 0,
    tangent_basis: NDArray[np.float64] | None = None,
    eps: float = 1e-5,
    tol: float | None = None,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> SecondOrderStressTest:
    """Evaluate a Connelly-style stress quadratic form on mechanisms.

    If ``stress`` is omitted, a self-stress from ``ker J.T`` is selected
    by ``stress_index``. If ``tangent_basis`` is omitted, the full
    infinitesimal mechanism basis ``ker J`` is used.
    """

    analysis = rigidity_svd(
        hinges,
        config,
        tol=tol,
        finite_diff_step=finite_diff_step,
        backend=backend,
    )

    if stress is None:
        if analysis.self_stress_count == 0:
            raise ValueError("no self-stress is available for this constraint set")
        stress_arr = analysis.self_stress_basis[
            :, stress_index % analysis.self_stress_count
        ]
    else:
        stress_arr = np.asarray(stress, dtype=float).reshape(-1)

    if tangent_basis is None:
        basis = analysis.mechanism_basis
    else:
        basis = _as_direction_matrix(tangent_basis, analysis.n_variables)

    hessian = stress_hessian(hinges, config, stress_arr, eps=eps)
    quadratic = second_order_form(hessian, basis)
    quadratic = 0.5 * (quadratic + quadratic.T)
    eigenvalues = np.linalg.eigvalsh(quadratic) if quadratic.size else np.array([])

    if tol is None:
        eig_tol = 1e-8 * max(
            float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0,
            1.0,
        )
    else:
        eig_tol = float(tol)

    positive = bool(eigenvalues.size and np.all(eigenvalues > eig_tol))
    negative = bool(eigenvalues.size and np.all(eigenvalues < -eig_tol))
    semidefinite = bool(
        eigenvalues.size
        and (
            np.all(eigenvalues >= -eig_tol)
            or np.all(eigenvalues <= eig_tol)
        )
    )

    return SecondOrderStressTest(
        stress=stress_arr,
        hessian=hessian,
        tangent_basis=basis,
        quadratic_form=quadratic,
        eigenvalues=eigenvalues,
        positive_definite=positive,
        negative_definite=negative,
        semidefinite=semidefinite,
        tolerance=eig_tol,
    )


def trace_finite_motion(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    direction_index: int = 0,
    step_size: float = 5e-4,
    n_steps: int = 80,
    bidirectional: bool = True,
    correction_tol: float = 1e-8,
    max_newton_iter: int = 50,
    subtract_rigid: bool = True,
    nullspace_tol: float | None = None,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> NDArray[np.float64]:
    """Trace a finite motion branch with predictor-corrector continuation."""

    return follow_motion(
        hinges,
        config,
        direction_index=direction_index,
        step_size=step_size,
        n_steps=n_steps,
        bidirectional=bidirectional,
        correction_tol=correction_tol,
        max_newton_iter=max_newton_iter,
        subtract_rigid=subtract_rigid,
        nullspace_tol=nullspace_tol,
        finite_diff_step=finite_diff_step,
        backend=backend,
    )


def finite_motion_analysis(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    subtract_rigid: bool = True,
    seed: int | None = None,
    **kwargs,
) -> dict:
    """Combine Calladine counting with sampled nonlinear continuation."""

    return {
        "calladine": calladine_summary(
            hinges,
            config,
            subtract_rigid=subtract_rigid,
        ),
        "finite_motion": finite_motion_dof(
            hinges,
            config,
            subtract_rigid=subtract_rigid,
            seed=seed,
            **kwargs,
        ),
    }


__all__ = [
    "CalladineSummary",
    "RigiditySVD",
    "SecondOrderStressTest",
    "calladine_summary",
    "connelly_second_order_test",
    "constraint_residual_vector",
    "finite_motion_analysis",
    "flatten_residuals",
    "pellegrino_calladine_analysis",
    "rigidity_matrix",
    "rigidity_svd",
    "second_order_form",
    "stress_hessian",
    "trace_finite_motion",
]
