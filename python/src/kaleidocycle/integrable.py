"""Integrable deformations of regular kaleidocycles.

This module implements the curvature-coordinate model and the compatible
semi-discrete flows described in ``Integrable deformations of kaleidocycles``.
The fundamental coordinate is the Cayley curvature

``kappa[n] = 2 * tan(phi[n] / 2)``,

not the curvature angle ``phi`` itself.  Boundary values are evaluated with
the twisted convention ``x[n + N] = sign * x[n]``, where ``sign=1`` denotes
an oriented kaleidocycle and ``sign=-1`` an anti-oriented one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]
FlowName = Literal["mkdv1", "mkdv2", "sine-gordon"]
FlowSpec = FlowName | str | int


def _vector(values: ArrayLike, *, name: str = "values") -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if array.size < 3:
        raise ValueError(f"{name} must contain at least three entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _sign(sign: int) -> int:
    if sign not in (-1, 1):
        raise ValueError("sign must be 1 (oriented) or -1 (anti-oriented)")
    return sign


def _frame(value: ArrayLike | None) -> FloatArray:
    if value is None:
        return np.eye(3)
    frame = np.asarray(value, dtype=float)
    if frame.shape != (3, 3):
        raise ValueError("initial_frame must have shape (3, 3)")
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-8):
        raise ValueError("initial_frame must be orthogonal")
    if not np.isclose(np.linalg.det(frame), 1.0, atol=1e-8):
        raise ValueError("initial_frame must have determinant 1")
    return frame


def _torsion_angle(value: float) -> float:
    angle = float(value)
    if not np.isfinite(angle) or not 0.0 < angle < np.pi:
        raise ValueError("torsion_angle must lie strictly between 0 and pi")
    return angle


def twisted_shift(values: ArrayLike, offset: int, *, sign: int = 1) -> FloatArray:
    """Shift a finite sequence using ``x[n + N] = sign * x[n]``.

    The returned entry at index ``n`` is the extended value ``x[n+offset]``.
    Unlike ``numpy.roll``, this handles anti-periodic wraparound correctly for
    arbitrary positive and negative offsets.
    """

    array = _vector(values)
    boundary_sign = _sign(sign)
    indices = np.arange(array.size, dtype=int) + int(offset)
    quotients, remainders = np.divmod(indices, array.size)
    factors = np.where(
        boundary_sign == 1,
        1.0,
        np.where(quotients % 2 == 0, 1.0, -1.0),
    )
    return factors * array[remainders]


def curvature_angles(curvatures: ArrayLike) -> FloatArray:
    """Convert Cayley curvatures to signed angles in ``(-pi, pi)``."""

    return 2.0 * np.arctan(_vector(curvatures, name="curvatures") / 2.0)


def cayley_curvatures(angles: ArrayLike) -> FloatArray:
    """Convert signed curvature angles to Cayley curvatures."""

    phi = _vector(angles, name="angles")
    if np.any(np.abs(phi) >= np.pi):
        raise ValueError("curvature angles must lie strictly between -pi and pi")
    return 2.0 * np.tan(phi / 2.0)


def curvature_weights(curvatures: ArrayLike) -> FloatArray:
    """Return ``D[n] = 1 + kappa[n]**2 / 4``."""

    kappa = _vector(curvatures, name="curvatures")
    return 1.0 + kappa**2 / 4.0


def rotation_1(angle: float) -> FloatArray:
    """Rotation about the first coordinate axis."""

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def rotation_3(angle: float) -> FloatArray:
    """Rotation about the third coordinate axis."""

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True, slots=True)
class FramedPolygon:
    """Based reconstruction of a regular kaleidocycle from curvature data."""

    curvatures: FloatArray
    torsion_angle: float
    sign: int
    frames: FloatArray
    vertices: FloatArray

    @property
    def angles(self) -> FloatArray:
        """Signed curvature angles."""

        return curvature_angles(self.curvatures)

    @property
    def tangents(self) -> FloatArray:
        """Unit edge tangents, one per curvature coordinate."""

        return self.frames[:-1, :, 0]

    @property
    def normals(self) -> FloatArray:
        """Frame normals, including the terminal frame."""

        return self.frames[:, :, 1]

    @property
    def binormals(self) -> FloatArray:
        """Unit binormals, including the terminal frame."""

        return self.frames[:, :, 2]

    @property
    def closure_residual(self) -> FloatArray:
        """Displacement after one period."""

        return self.vertices[-1] - self.vertices[0]

    @property
    def monodromy_residual(self) -> FloatArray:
        """Residual of ``F[N] = F[0] diag(1, sign, sign)``."""

        target = self.frames[0] @ np.diag([1.0, self.sign, self.sign])
        return self.frames[-1] - target


@dataclass(frozen=True, slots=True)
class LiftCoefficients:
    """Moving-frame coefficients for a compatible lifted flow."""

    vertex: FloatArray
    angular: FloatArray


@dataclass(frozen=True, slots=True)
class SpectralCurveDiagnostics:
    """Genus data for the twisted-monodromy spectral curve.

    Polynomial coefficients use descending powers of
    ``lambda = zeta**2``.  ``geometric_genus`` is the genus after removing
    the closed-kaleidocycle square factor when a torsion angle is supplied.
    The hierarchy linearises on the Prym part of the reciprocal involution,
    whose dimension is ``prym_dimension``.
    """

    trace_coefficients: FloatArray
    branch_coefficients: FloatArray
    normalised_branch_coefficients: FloatArray
    singular_factor_residual: float | None
    reciprocal_residual: float
    minimum_root_separation: float
    arithmetic_genus: int
    geometric_genus: int
    quotient_genus: int
    prym_dimension: int


def reconstruct_framed_polygon(
    curvatures: ArrayLike,
    torsion_angle: float,
    *,
    sign: int = 1,
    initial_frame: ArrayLike | None = None,
    initial_vertex: ArrayLike | None = None,
) -> FramedPolygon:
    """Reconstruct frames and vertices using the discrete Frenet equation.

    The indexing follows the paper exactly:
    ``F[n+1] = F[n] R1(mu) R3(phi[n+1])``.  Thus the first recursion step
    uses the twisted value at curvature index 1.
    """

    kappa = _vector(curvatures, name="curvatures")
    boundary_sign = _sign(sign)
    mu = _torsion_angle(torsion_angle)

    frame_0 = _frame(initial_frame)

    if initial_vertex is None:
        vertex_0 = np.zeros(3)
    else:
        vertex_0 = np.asarray(initial_vertex, dtype=float)
        if vertex_0.shape != (3,):
            raise ValueError("initial_vertex must have shape (3,)")

    n_vertices = kappa.size
    frames = np.empty((n_vertices + 1, 3, 3), dtype=float)
    vertices = np.empty((n_vertices + 1, 3), dtype=float)
    frames[0] = frame_0
    vertices[0] = vertex_0
    shifted_angles = curvature_angles(twisted_shift(kappa, 1, sign=boundary_sign))
    torsion_rotation = rotation_1(mu)

    for n in range(n_vertices):
        vertices[n + 1] = vertices[n] + frames[n, :, 0]
        frames[n + 1] = frames[n] @ torsion_rotation @ rotation_3(shifted_angles[n])

    return FramedPolygon(
        curvatures=kappa.copy(),
        torsion_angle=mu,
        sign=boundary_sign,
        frames=frames,
        vertices=vertices,
    )


def framed_polygon_from_binormals(
    binormals: ArrayLike,
    *,
    sign: int | None = None,
    torsion_tolerance: float = 1e-6,
) -> FramedPolygon:
    """Extract curvature coordinates and a moving frame from binormals.

    ``binormals`` must contain the terminal value, so its shape is
    ``(N + 1, 3)``.  Constant torsion is validated before reconstruction.
    """

    vectors = np.asarray(binormals, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] < 4:
        raise ValueError("binormals must have shape (N + 1, 3), N >= 3")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("binormals must be nonzero")
    vectors = vectors / norms[:, None]

    if sign is None:
        same = np.linalg.norm(vectors[-1] - vectors[0])
        opposite = np.linalg.norm(vectors[-1] + vectors[0])
        boundary_sign = 1 if same <= opposite else -1
    else:
        boundary_sign = _sign(sign)
    if not np.allclose(vectors[-1], boundary_sign * vectors[0], atol=1e-5):
        raise ValueError("terminal binormal does not match the requested monodromy")

    dots = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
    if np.ptp(dots) > torsion_tolerance:
        raise ValueError("adjacent binormals do not have constant torsion")
    torsion_cosine = float(np.clip(np.mean(dots), -1.0, 1.0))
    if abs(torsion_cosine) >= 1.0 - 1e-12:
        raise ValueError("torsion angle must lie strictly between 0 and pi")
    mu = float(np.arccos(torsion_cosine))

    cross_products = np.cross(vectors[:-1], vectors[1:])
    cross_norms = np.linalg.norm(cross_products, axis=1)
    if np.any(cross_norms < 1e-12):
        raise ValueError("adjacent binormals must not be parallel")
    tangents = cross_products / cross_norms[:, None]
    normals = np.cross(vectors[:-1], tangents)
    previous_tangents = _twisted_shift_rows(tangents, -1, sign=1)
    sine = np.einsum("ij,ij->i", tangents, np.cross(vectors[:-1], previous_tangents))
    cosine = np.einsum("ij,ij->i", previous_tangents, tangents)
    angles = np.arctan2(sine, cosine)
    kappa = cayley_curvatures(angles)
    initial_frame = np.column_stack((tangents[0], normals[0], vectors[0]))
    vertices = np.empty((kappa.size + 1, 3), dtype=float)
    vertices[0] = 0.0
    vertices[1:] = np.cumsum(tangents, axis=0)

    frames = np.empty((kappa.size + 1, 3, 3), dtype=float)
    frames[:-1, :, 0] = tangents
    frames[:-1, :, 1] = normals
    frames[:-1, :, 2] = vectors[:-1]
    frames[-1] = initial_frame @ np.diag([1.0, boundary_sign, boundary_sign])
    return FramedPolygon(kappa, mu, boundary_sign, frames, vertices)


def _twisted_shift_rows(values: ArrayLike, offset: int, *, sign: int = 1) -> FloatArray:
    """Twisted shift along the first axis of an array."""

    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("values must have a nonempty first axis")
    boundary_sign = _sign(sign)
    indices = np.arange(array.shape[0], dtype=int) + int(offset)
    quotients, remainders = np.divmod(indices, array.shape[0])
    factors = np.where(
        boundary_sign == 1,
        1.0,
        np.where(quotients % 2 == 0, 1.0, -1.0),
    )
    return factors.reshape((-1,) + (1,) * (array.ndim - 1)) * array[remainders]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _normalised_twisted_monodromy_series(
    curvatures: FloatArray,
    max_degree: int,
    *,
    sign: int,
    base_index: int = 0,
) -> FloatArray:
    """Return coefficients of the normalised twisted monodromy.

    Factoring ``zeta`` from every Ablowitz--Ladik matrix and putting
    ``q = zeta**-2`` turns the monodromy into a matrix polynomial in
    ``q``.  This helper returns the coefficients of ``J_sigma T / zeta**N``.
    """

    kappa = _vector(curvatures, name="curvatures")
    boundary_sign = _sign(sign)
    coefficients = np.zeros((max_degree + 1, 2, 2), dtype=float)
    coefficients[0] = np.eye(2)
    n_sites = kappa.size

    for index in range(base_index, base_index + n_sites):
        quotient, remainder = divmod(index, n_sites)
        factor_sign = -1.0 if boundary_sign == -1 and quotient % 2 else 1.0
        z_value = factor_sign * kappa[remainder] / 2.0
        scale = 1.0 / np.sqrt(1.0 + z_value**2)
        factor_0 = scale * np.array([[1.0, 0.0], [-z_value, 0.0]], dtype=float)
        factor_1 = scale * np.array([[0.0, z_value], [0.0, 1.0]], dtype=float)

        updated = np.zeros_like(coefficients)
        for degree in range(max_degree + 1):
            updated[degree] = factor_0 @ coefficients[degree]
            if degree:
                updated[degree] += factor_1 @ coefficients[degree - 1]
        coefficients = updated

    twist = np.diag([1.0, float(boundary_sign)])
    return np.einsum("ab,mbc->mac", twist, coefficients)


def twisted_trace_polynomial(curvatures: ArrayLike, *, sign: int = 1) -> FloatArray:
    """Return the polynomial ``P(lambda)`` in the spectral curve.

    With ``lambda = zeta**2``, the twisted trace is

    ``Delta_sigma(zeta) = lambda**(-N/2) P(lambda)``.

    The returned array contains the coefficients of ``P`` in descending
    powers.  It has length ``N + 1`` and satisfies the reciprocal relation
    ``P(lambda) = sign * lambda**N * P(1/lambda)``.
    """

    kappa = _vector(curvatures, name="curvatures")
    boundary_sign = _sign(sign)
    monodromy = _normalised_twisted_monodromy_series(
        kappa, kappa.size, sign=boundary_sign
    )
    return np.trace(monodromy, axis1=1, axis2=2)


def spectral_curve_diagnostics(
    curvatures: ArrayLike,
    *,
    sign: int = 1,
    torsion_angle: float | None = None,
    factor_tolerance: float = 1e-8,
) -> SpectralCurveDiagnostics:
    """Construct the spectral curve and compute its genus diagnostics.

    The hyperelliptic model is

    ``y**2 = F(lambda) = P(lambda)**2 - 4*sign*lambda**N``.

    Its generic arithmetic genus is ``N - 1``.  For a closed
    constant-torsion kaleidocycle, supplying ``torsion_angle`` removes the
    square factor

    ``(lambda**2 - 2*cos(mu)*lambda + 1)**4``.

    ``minimum_root_separation`` reports the numerical square-free check for
    the remaining polynomial.  The reciprocal involution on its normalisation
    has quotient genus ``quotient_genus``; the positive mKdV hierarchy moves
    on the associated Prym variety of dimension ``prym_dimension``.
    """

    kappa = _vector(curvatures, name="curvatures")
    boundary_sign = _sign(sign)
    tolerance = float(factor_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("factor_tolerance must be a positive finite number")

    n_sites = kappa.size
    trace = twisted_trace_polynomial(kappa, sign=boundary_sign)
    branch = np.polymul(trace, trace)
    branch[n_sites] -= 4.0 * boundary_sign
    normalised = branch.copy()
    factor_residual: float | None = None

    if torsion_angle is not None:
        mu = _torsion_angle(torsion_angle)
        quadratic = np.array([1.0, -2.0 * np.cos(mu), 1.0])
        singular_factor = np.array([1.0])
        for _ in range(4):
            singular_factor = np.polymul(singular_factor, quadratic)
        quotient, remainder = np.polydiv(branch, singular_factor)
        factor_residual = float(
            np.linalg.norm(remainder)
            / max(np.linalg.norm(branch), np.finfo(float).tiny)
        )
        if factor_residual > tolerance:
            raise ValueError(
                "the closed-kaleidocycle spectral factor is not present: "
                f"relative remainder {factor_residual:.3e}"
            )
        normalised = np.asarray(np.real_if_close(quotient), dtype=float)

    degree = normalised.size - 1
    if degree < 4 or degree % 2:
        raise ValueError("the normalised branch polynomial must have even degree >= 4")
    half_degree = degree // 2
    arithmetic_genus = n_sites - 1
    geometric_genus = half_degree - 1
    quotient_genus = (half_degree - 1) // 2
    prym_dimension = geometric_genus - quotient_genus

    reciprocal_residual = float(
        np.linalg.norm(normalised - normalised[::-1])
        / max(np.linalg.norm(normalised), np.finfo(float).tiny)
    )
    roots = np.roots(normalised)
    pairwise_distances = np.abs(roots[:, None] - roots[None, :])
    np.fill_diagonal(pairwise_distances, np.inf)
    minimum_root_separation = float(np.min(pairwise_distances))

    return SpectralCurveDiagnostics(
        trace_coefficients=trace,
        branch_coefficients=branch,
        normalised_branch_coefficients=normalised,
        singular_factor_residual=factor_residual,
        reciprocal_residual=reciprocal_residual,
        minimum_root_separation=minimum_root_separation,
        arithmetic_genus=arithmetic_genus,
        geometric_genus=geometric_genus,
        quotient_genus=quotient_genus,
        prym_dimension=prym_dimension,
    )


def _floquet_multiplier_series(
    trace: FloatArray, n_sites: int, *, sign: int
) -> FloatArray:
    """Solve ``rho**2 - trace*rho + sign*q**N = 0`` as a series."""

    multiplier = np.zeros_like(trace)
    multiplier[0] = trace[0]
    for degree in range(1, trace.size):
        remainder = sum(
            (multiplier[index] - trace[index]) * multiplier[degree - index]
            for index in range(1, degree)
        )
        if degree == n_sites:
            remainder += sign
        multiplier[degree] = trace[degree] - remainder / multiplier[0]
    return multiplier


def _scalar_series_inverse(series: FloatArray) -> FloatArray:
    inverse = np.zeros_like(series)
    inverse[0] = 1.0 / series[0]
    for degree in range(1, series.size):
        inverse[degree] = (
            -sum(
                series[index] * inverse[degree - index]
                for index in range(1, degree + 1)
            )
            / series[0]
        )
    return inverse


def _spectral_integrals_and_gradients(
    curvatures: ArrayLike, count: int, *, sign: int
) -> tuple[FloatArray, FloatArray]:
    """Expand the logarithmic Floquet multiplier and its exact gradient."""

    kappa = _vector(curvatures, name="curvatures")
    n_integrals = _positive_integer(count, name="count")
    boundary_sign = _sign(sign)
    max_degree = n_integrals - 1
    n_sites = kappa.size
    z_values = kappa / 2.0

    coefficients = np.zeros((n_integrals, 2, 2), dtype=float)
    coefficient_gradients = np.zeros((n_integrals, 2, 2, n_sites), dtype=float)
    coefficients[0] = np.eye(2)

    for site, z_value in enumerate(z_values):
        weight = 1.0 + z_value**2
        scale = weight**-0.5
        scale_gradient = -kappa[site] / (4.0 * weight**1.5)
        factor_0_base = np.array([[1.0, 0.0], [-z_value, 0.0]], dtype=float)
        factor_1_base = np.array([[0.0, z_value], [0.0, 1.0]], dtype=float)
        factor_0 = scale * factor_0_base
        factor_1 = scale * factor_1_base
        factor_0_gradient = scale_gradient * factor_0_base + scale * np.array(
            [[0.0, 0.0], [-0.5, 0.0]], dtype=float
        )
        factor_1_gradient = scale_gradient * factor_1_base + scale * np.array(
            [[0.0, 0.5], [0.0, 0.0]], dtype=float
        )

        updated = np.zeros_like(coefficients)
        updated_gradients = np.zeros_like(coefficient_gradients)
        for degree in range(max_degree + 1):
            updated[degree] = factor_0 @ coefficients[degree]
            updated_gradients[degree] = np.einsum(
                "ab,bcn->acn", factor_0, coefficient_gradients[degree]
            )
            updated_gradients[degree, :, :, site] += (
                factor_0_gradient @ coefficients[degree]
            )
            if degree:
                updated[degree] += factor_1 @ coefficients[degree - 1]
                updated_gradients[degree] += np.einsum(
                    "ab,bcn->acn",
                    factor_1,
                    coefficient_gradients[degree - 1],
                )
                updated_gradients[degree, :, :, site] += (
                    factor_1_gradient @ coefficients[degree - 1]
                )
        coefficients = updated
        coefficient_gradients = updated_gradients

    trace = coefficients[:, 0, 0] + boundary_sign * coefficients[:, 1, 1]
    trace_gradient = (
        coefficient_gradients[:, 0, 0, :]
        + boundary_sign * coefficient_gradients[:, 1, 1, :]
    )

    multiplier = np.zeros(n_integrals, dtype=float)
    multiplier_gradient = np.zeros((n_integrals, n_sites), dtype=float)
    multiplier[0] = trace[0]
    multiplier_gradient[0] = trace_gradient[0]
    for degree in range(1, n_integrals):
        remainder = 0.0
        remainder_gradient = np.zeros(n_sites, dtype=float)
        for index in range(1, degree):
            left = multiplier[index] - trace[index]
            left_gradient = multiplier_gradient[index] - trace_gradient[index]
            right = multiplier[degree - index]
            right_gradient = multiplier_gradient[degree - index]
            remainder += left * right
            remainder_gradient += left_gradient * right + left * right_gradient
        if degree == n_sites:
            remainder += boundary_sign
        multiplier[degree] = trace[degree] - remainder / multiplier[0]
        multiplier_gradient[degree] = (
            trace_gradient[degree]
            - remainder_gradient / multiplier[0]
            + remainder * multiplier_gradient[0] / multiplier[0] ** 2
        )

    logarithm = np.zeros(n_integrals, dtype=float)
    logarithm_gradient = np.zeros((n_integrals, n_sites), dtype=float)
    logarithm[0] = np.log(multiplier[0])
    logarithm_gradient[0] = multiplier_gradient[0] / multiplier[0]
    for degree in range(1, n_integrals):
        numerator = degree * multiplier[degree]
        numerator_gradient = degree * multiplier_gradient[degree]
        for index in range(1, degree):
            numerator -= index * logarithm[index] * multiplier[degree - index]
            numerator_gradient -= index * (
                logarithm_gradient[index] * multiplier[degree - index]
                + logarithm[index] * multiplier_gradient[degree - index]
            )
        denominator = degree * multiplier[0]
        denominator_gradient = degree * multiplier_gradient[0]
        logarithm[degree] = numerator / denominator
        logarithm_gradient[degree] = (
            numerator_gradient * denominator - numerator * denominator_gradient
        ) / denominator**2

    return -2.0 * logarithm, -2.0 * logarithm_gradient


def spectral_integrals(
    curvatures: ArrayLike, count: int, *, sign: int = 1
) -> FloatArray:
    """Return ``I_1, ..., I_count`` from the twisted Floquet expansion.

    The expansion is the one in equation (5.12) of the manuscript,
    ``-2 log(rho_sigma(zeta) / zeta**N)`` at ``zeta = infinity``.
    """

    values, _ = _spectral_integrals_and_gradients(curvatures, count, sign=sign)
    return values


def spectral_integral_gradients(
    curvatures: ArrayLike, count: int, *, sign: int = 1
) -> FloatArray:
    """Return coordinate gradients of the first ``count`` spectral integrals."""

    _, gradients = _spectral_integrals_and_gradients(curvatures, count, sign=sign)
    return gradients


def _hierarchy_hamiltonian_weights(order: int) -> FloatArray:
    """Weights expressing ``E_order`` through ``I_1, ..., I_order``."""

    hierarchy_order = _positive_integer(order, name="order")
    power = hierarchy_order - 1
    weights = np.zeros(hierarchy_order, dtype=float)
    for index in range((power - 1) // 2 + 1 if power else 0):
        weights[power - 2 * index] += comb(power, index)
    if power % 2 == 0:
        weights[0] += comb(power, power // 2)
    return weights


def hierarchy_hamiltonians(
    curvatures: ArrayLike, count: int, *, sign: int = 1
) -> FloatArray:
    """Return the Hamiltonians ``E_1, ..., E_count`` of the hierarchy."""

    n_hamiltonians = _positive_integer(count, name="count")
    integrals = spectral_integrals(curvatures, n_hamiltonians, sign=sign)
    return np.array(
        [
            np.dot(_hierarchy_hamiltonian_weights(order), integrals[:order])
            for order in range(1, n_hamiltonians + 1)
        ],
        dtype=float,
    )


def hierarchy_hamiltonian_gradients(
    curvatures: ArrayLike, count: int, *, sign: int = 1
) -> FloatArray:
    """Return gradients of ``E_1, ..., E_count`` in curvature coordinates."""

    n_hamiltonians = _positive_integer(count, name="count")
    _, integral_gradients = _spectral_integrals_and_gradients(
        curvatures, n_hamiltonians, sign=sign
    )
    return np.array(
        [
            _hierarchy_hamiltonian_weights(order) @ integral_gradients[:order]
            for order in range(1, n_hamiltonians + 1)
        ],
        dtype=float,
    )


def _spectral_primitive_coefficients(
    curvatures: ArrayLike, count: int, *, sign: int
) -> FloatArray:
    """Return the zero-normalised projector primitives ``Q_0, ...``."""

    kappa = _vector(curvatures, name="curvatures")
    n_coefficients = _positive_integer(count, name="count")
    boundary_sign = _sign(sign)
    max_degree = n_coefficients + 1
    base_monodromy = _normalised_twisted_monodromy_series(
        kappa, max_degree, sign=boundary_sign
    )
    trace = np.trace(base_monodromy, axis1=1, axis2=2)
    multiplier = _floquet_multiplier_series(trace, kappa.size, sign=boundary_sign)
    inverse_multiplier = _scalar_series_inverse(multiplier)
    small_multiplier = np.zeros_like(multiplier)
    for degree in range(kappa.size, max_degree + 1):
        small_multiplier[degree] = (
            boundary_sign * inverse_multiplier[degree - kappa.size]
        )
    inverse_gap = _scalar_series_inverse(multiplier - small_multiplier)

    primitives = np.zeros((n_coefficients, kappa.size), dtype=float)
    identity = np.eye(2)
    for base_index in range(kappa.size):
        numerator = _normalised_twisted_monodromy_series(
            kappa,
            max_degree,
            sign=boundary_sign,
            base_index=base_index,
        )
        numerator -= small_multiplier[:, None, None] * identity
        projector_11 = np.array(
            [
                sum(
                    numerator[index, 0, 0] * inverse_gap[degree - index]
                    for index in range(degree + 1)
                )
                for degree in range(max_degree + 1)
            ]
        )
        for degree in range(n_coefficients):
            lower = projector_11[degree - 1] if degree >= 2 else 0.0
            primitives[degree, base_index] = 2.0 * (projector_11[degree + 1] - lower)
    return primitives


def _mkdv1_field(curvatures: ArrayLike, *, sign: int) -> FloatArray:
    kappa = _vector(curvatures, name="curvatures")
    return (
        0.5
        * curvature_weights(kappa)
        * (twisted_shift(kappa, 1, sign=sign) - twisted_shift(kappa, -1, sign=sign))
    )


def _mkdv2_field(curvatures: ArrayLike, *, sign: int) -> FloatArray:
    kappa = _vector(curvatures, name="curvatures")
    d = curvature_weights(kappa)
    forward = twisted_shift(d, 1) * (kappa + twisted_shift(kappa, 2, sign=sign))
    backward = twisted_shift(d, -1) * (twisted_shift(kappa, -2, sign=sign) + kappa)
    return 0.5 * d * (forward - backward)


def mkdv_hierarchy_field(
    curvatures: ArrayLike, order: int, *, sign: int = 1
) -> FloatArray:
    """Return the ``order``-th positive semi-discrete mKdV field.

    Orders one and two use their explicit local formulas.  Higher orders are
    generated from the logarithmic twisted-monodromy Hamiltonians and the
    common Poisson operator, which is equivalent to the zero-normalised
    recursion operator in the manuscript.
    """

    hierarchy_order = _positive_integer(order, name="order")
    boundary_sign = _sign(sign)
    if hierarchy_order == 1:
        return _mkdv1_field(curvatures, sign=boundary_sign)
    if hierarchy_order == 2:
        return _mkdv2_field(curvatures, sign=boundary_sign)
    gradients = hierarchy_hamiltonian_gradients(
        curvatures, hierarchy_order, sign=boundary_sign
    )
    return poisson_operator(curvatures, gradients[-1], sign=boundary_sign)


def mkdv1_field(curvatures: ArrayLike, *, sign: int = 1) -> FloatArray:
    """First semi-discrete mKdV vector field."""

    return mkdv_hierarchy_field(curvatures, 1, sign=sign)


def mkdv2_field(curvatures: ArrayLike, *, sign: int = 1) -> FloatArray:
    """Second commuting semi-discrete mKdV vector field."""

    return mkdv_hierarchy_field(curvatures, 2, sign=sign)


def hierarchy_orbit_jacobian(
    curvatures: ArrayLike,
    count: int | None = None,
    *,
    sign: int = 1,
) -> FloatArray:
    """Return the Jacobian of the hierarchy multi-time orbit map.

    Column ``j - 1`` is ``X^(j)(kappa)``.  Equivalently, this is the
    derivative at the origin of the map obtained by composing the hierarchy
    flows with times ``t_1, ..., t_count``.

    By default ``count=N``.  This reaches the Cayley--Hamilton upper bound
    for the Krylov span of a recursion operator on the ``N``-dimensional
    finite curvature space, so no later hierarchy field can increase the
    pointwise span.
    """

    kappa = _vector(curvatures, name="curvatures")
    n_fields = kappa.size if count is None else _positive_integer(count, name="count")
    boundary_sign = _sign(sign)
    return np.column_stack(
        [
            mkdv_hierarchy_field(kappa, order, sign=boundary_sign)
            for order in range(1, n_fields + 1)
        ]
    )


def sine_gordon_potential(curvatures: ArrayLike, *, sign: int = -1) -> FloatArray:
    """Return the unique anti-periodic potential ``u``.

    The output contains one period ``u[0], ..., u[N-1]`` and satisfies
    ``phi[n] = u[n-1] - u[n]`` under anti-periodic indexing.
    """

    if _sign(sign) != -1:
        raise ValueError("the sine-Gordon potential requires sign=-1")
    phi = curvature_angles(curvatures)
    # sum(phi[1], ..., phi[N]) with phi[N] = -phi[0]
    u0 = 0.5 * (np.sum(phi[1:]) - phi[0])
    u = np.empty_like(phi)
    u[0] = u0
    if phi.size > 1:
        u[1:] = u0 - np.cumsum(phi[1:])
    return u


def sine_gordon_field(
    curvatures: ArrayLike,
    torsion_angle: float,
    *,
    sign: int = -1,
) -> FloatArray:
    """Semi-discrete sine--Gordon vector field in curvature coordinates."""

    kappa = _vector(curvatures, name="curvatures")
    mu = _torsion_angle(torsion_angle)
    u = sine_gordon_potential(kappa, sign=sign)
    angle_velocity = -(1.0 - np.cos(mu)) * (
        np.sin(u) + np.sin(twisted_shift(u, -1, sign=-1))
    )
    return curvature_weights(kappa) * angle_velocity


def _parse_flow(flow: FlowSpec) -> tuple[str, int | None, str]:
    if isinstance(flow, bool):
        raise TypeError("flow must be 'sine-gordon', 'mkdvN', or a positive integer")
    if isinstance(flow, (int, np.integer)):
        order = _positive_integer(int(flow), name="flow")
        return "mkdv", order, f"mkdv{order}"
    if not isinstance(flow, str):
        raise TypeError("flow must be 'sine-gordon', 'mkdvN', or a positive integer")
    normalised = flow.strip().lower().replace("_", "-")
    if normalised in {"sine-gordon", "sinegordon", "sg"}:
        return "sine-gordon", None, "sine-gordon"
    compact = normalised.replace("-", "")
    if compact.startswith("mkdv") and compact[4:].isdigit():
        order = _positive_integer(int(compact[4:]), name="flow order")
        return "mkdv", order, f"mkdv{order}"
    raise ValueError(f"unknown flow {flow!r}")


def _hierarchy_lift_coefficients(
    curvatures: FloatArray,
    torsion_angle: float,
    order: int,
    *,
    sign: int,
) -> LiftCoefficients:
    """Construct the Doliwa--Santini lift of a positive hierarchy flow."""

    kappa = _vector(curvatures, name="curvatures")
    hierarchy_order = _positive_integer(order, name="order")
    boundary_sign = _sign(sign)
    mu = _torsion_angle(torsion_angle)
    cosine = np.cos(mu)
    sine = np.sin(mu)
    spectral_x = 2.0 * cosine
    spectral_x_derivative = -2.0 * sine

    # The paper uses q_k = z_{n+1}.  The arrays below are indexed by k=n.
    q_values = twisted_shift(kappa / 2.0, 1, sign=boundary_sign)
    varpi = np.zeros_like(q_values)
    varpi_derivative = np.zeros_like(q_values)
    coefficient_v = np.full_like(q_values, 2.0 * spectral_x ** (order - 1))
    if hierarchy_order == 1:
        coefficient_v_derivative = np.zeros_like(q_values)
    else:
        coefficient_v_derivative = np.full_like(
            q_values,
            2.0
            * (hierarchy_order - 1)
            * spectral_x ** (hierarchy_order - 2)
            * spectral_x_derivative,
        )

    if hierarchy_order > 1:
        primitive_coefficients = _spectral_primitive_coefficients(
            2.0 * q_values,
            hierarchy_order - 1,
            sign=boundary_sign,
        )
        for field_order in range(1, hierarchy_order):
            power = hierarchy_order - 1 - field_order
            field = mkdv_hierarchy_field(
                2.0 * q_values, field_order, sign=boundary_sign
            )
            primitive = (
                _hierarchy_hamiltonian_weights(field_order)
                @ primitive_coefficients[:field_order]
            )
            factor = spectral_x**power
            varpi += 2.0 * factor * field
            coefficient_v += 2.0 * factor * primitive
            if power:
                factor_derivative = (
                    power * spectral_x ** (power - 1) * spectral_x_derivative
                )
                varpi_derivative += 2.0 * factor_derivative * field
                coefficient_v_derivative += 2.0 * factor_derivative * primitive

    next_v = twisted_shift(coefficient_v, 1)
    next_v_derivative = twisted_shift(coefficient_v_derivative, 1)
    next_u = varpi - q_values * next_v
    next_u_derivative = varpi_derivative - q_values * next_v_derivative
    coefficient_u = twisted_shift(next_u, -1, sign=boundary_sign)
    coefficient_u_derivative = twisted_shift(next_u_derivative, -1, sign=boundary_sign)

    q_weight = 1.0 + q_values**2
    coefficient_a = (
        2.0 * q_values / q_weight * next_v
        + (1.0 - q_values**2) / q_weight * next_u
        - cosine * coefficient_u
    )
    coefficient_a_derivative = (
        2.0 * q_values / q_weight * next_v_derivative
        + (1.0 - q_values**2) / q_weight * next_u_derivative
        + sine * coefficient_u
        - cosine * coefficient_u_derivative
    )

    scaled_u = sine * coefficient_u
    scaled_v = sine * coefficient_v
    scaled_u_derivative = cosine * coefficient_u + sine * coefficient_u_derivative
    scaled_v_derivative = cosine * coefficient_v + sine * coefficient_v_derivative
    conjugation = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

    angular = np.empty((kappa.size, 3), dtype=float)
    vertex = np.empty((kappa.size, 3), dtype=float)
    for index in range(kappa.size):
        spherical_matrix = np.array(
            [
                [0.0, scaled_v[index], scaled_u[index]],
                [-scaled_v[index], 0.0, coefficient_a[index]],
                [-scaled_u[index], -coefficient_a[index], 0.0],
            ]
        )
        spherical_derivative = np.array(
            [
                [
                    0.0,
                    scaled_v_derivative[index],
                    scaled_u_derivative[index],
                ],
                [
                    -scaled_v_derivative[index],
                    0.0,
                    coefficient_a_derivative[index],
                ],
                [
                    -scaled_u_derivative[index],
                    -coefficient_a_derivative[index],
                    0.0,
                ],
            ]
        )
        angular_matrix = 0.5 * conjugation @ spherical_matrix.T @ conjugation.T
        vertex_matrix = 0.5 * conjugation @ spherical_derivative.T @ conjugation.T
        angular[index] = (
            angular_matrix[2, 1],
            angular_matrix[0, 2],
            angular_matrix[1, 0],
        )
        vertex[index] = (
            vertex_matrix[2, 1],
            vertex_matrix[0, 2],
            vertex_matrix[1, 0],
        )
    return LiftCoefficients(vertex=vertex, angular=angular)


def lift_coefficients(
    curvatures: ArrayLike,
    torsion_angle: float,
    *,
    flow: FlowSpec = "mkdv1",
    sign: int = 1,
) -> LiftCoefficients:
    """Return the vertex and angular velocities in moving-frame coordinates."""

    kappa = _vector(curvatures, name="curvatures")
    boundary_sign = _sign(sign)
    mu = _torsion_angle(torsion_angle)
    flow_kind, order, _ = _parse_flow(flow)

    if flow_kind == "mkdv":
        assert order is not None
        return _hierarchy_lift_coefficients(kappa, mu, order, sign=boundary_sign)

    if flow_kind == "sine-gordon":
        if boundary_sign != -1:
            raise ValueError("the sine-Gordon lift is defined only for sign=-1")
        u = sine_gordon_potential(kappa)
        cosine = np.cos(mu)
        sine = np.sin(mu)
        vertex = np.column_stack((np.cos(u), np.sin(u), np.zeros_like(u)))
        angular = -np.column_stack(
            (
                sine * np.cos(u),
                sine * np.sin(u),
                (1.0 - cosine) * np.sin(u),
            )
        )
        return LiftCoefficients(vertex, angular)

    raise AssertionError("unreachable flow kind")


def lifted_velocities(
    configuration: FramedPolygon, *, flow: FlowSpec = "mkdv1"
) -> tuple[FloatArray, FloatArray]:
    """Return world-coordinate vertex and binormal velocities.

    Both arrays include a terminal value consistent with the frame monodromy.
    """

    coefficients = lift_coefficients(
        configuration.curvatures,
        configuration.torsion_angle,
        flow=flow,
        sign=configuration.sign,
    )
    frames = configuration.frames[:-1]
    vertex_velocity = np.einsum("nij,nj->ni", frames, coefficients.vertex)
    tangent = frames[:, :, 0]
    normal = frames[:, :, 1]
    angular = coefficients.angular
    binormal_velocity = angular[:, 1, None] * tangent - angular[:, 0, None] * normal
    return (
        np.vstack((vertex_velocity, vertex_velocity[0])),
        np.vstack((binormal_velocity, configuration.sign * binormal_velocity[0])),
    )


def first_hamiltonian(curvatures: ArrayLike) -> float:
    """Return ``E1 = sum(log(1 + kappa[n]**2 / 4))``."""

    return float(np.sum(np.log(curvature_weights(curvatures))))


def second_hamiltonian(curvatures: ArrayLike, *, sign: int = 1) -> float:
    """Return ``E2 = 1/2 sum(kappa[n] * kappa[n+1])``."""

    kappa = _vector(curvatures, name="curvatures")
    return float(0.5 * np.dot(kappa, twisted_shift(kappa, 1, sign=sign)))


def poisson_operator(
    curvatures: ArrayLike, covector: ArrayLike, *, sign: int = 1
) -> FloatArray:
    """Apply ``P = D (S - S^-1) D`` to a covector."""

    kappa = _vector(curvatures, name="curvatures")
    gradient = _vector(covector, name="covector")
    if gradient.shape != kappa.shape:
        raise ValueError("covector must have the same shape as curvatures")
    d = curvature_weights(kappa)
    weighted = d * gradient
    return d * (
        twisted_shift(weighted, 1, sign=sign) - twisted_shift(weighted, -1, sign=sign)
    )


def poisson_bracket(
    curvatures: ArrayLike,
    gradient_f: ArrayLike,
    gradient_g: ArrayLike,
    *,
    sign: int = 1,
) -> float:
    """Evaluate the Poisson bracket from two coordinate gradients."""

    kappa = _vector(curvatures, name="curvatures")
    grad_f = _vector(gradient_f, name="gradient_f")
    if grad_f.shape != kappa.shape:
        raise ValueError("gradient_f must have the same shape as curvatures")
    return float(
        np.dot(
            grad_f,
            poisson_operator(kappa, gradient_g, sign=sign),
        )
    )


def variational_u(curvatures: ArrayLike, *, sign: int = 1) -> FloatArray:
    """Return ``U[n] = D[n] kappa[n-1] kappa[n+1] - kappa[n]**2``."""

    kappa = _vector(curvatures, name="curvatures")
    return (
        curvature_weights(kappa)
        * twisted_shift(kappa, -1, sign=sign)
        * twisted_shift(kappa, 1, sign=sign)
        - kappa**2
    )


def recurrence_multiplier(curvatures: ArrayLike, *, sign: int = 1) -> float:
    """Least-squares multiplier for the variational three-term recurrence."""

    kappa = _vector(curvatures, name="curvatures")
    denominator = float(np.dot(kappa, kappa))
    if denominator == 0.0:
        raise ValueError("the zero curvature sequence has no unique multiplier")
    recurrence_left = curvature_weights(kappa) * (
        twisted_shift(kappa, -1, sign=sign) + twisted_shift(kappa, 1, sign=sign)
    )
    return float(np.dot(kappa, recurrence_left) / denominator)


def critical_torsion_cosine(curvatures: ArrayLike, *, sign: int = 1) -> float:
    """Recover the torsion cosine on the Lagrange-critical locus.

    This is the formula
    ``c = sum(kappa[n-1]/kappa[n] + kappa[n]/kappa[n-1]) / (2*N)``.
    It is meaningful when all curvatures are nonzero and the variational
    recurrence holds.
    """

    kappa = _vector(curvatures, name="curvatures")
    previous = twisted_shift(kappa, -1, sign=sign)
    if np.any(kappa == 0.0):
        raise ValueError("critical torsion recovery requires nonzero curvatures")
    return float(np.mean(previous / kappa + kappa / previous) / 2.0)


def critical_multiplier(
    curvatures: ArrayLike, torsion_cosine: float, *, sign: int = 1
) -> float:
    """Return ``lambda = 2*c + E2/N`` on the Lagrange-critical locus."""

    kappa = _vector(curvatures, name="curvatures")
    cosine = float(torsion_cosine)
    if not np.isfinite(cosine) or not -1.0 < cosine < 1.0:
        raise ValueError("torsion_cosine must lie strictly between -1 and 1")
    return 2.0 * cosine + second_hamiltonian(kappa, sign=sign) / kappa.size


def variational_recurrence_residual(
    curvatures: ArrayLike,
    multiplier: float | None = None,
    *,
    sign: int = 1,
) -> FloatArray:
    """Residual of ``D[n](kappa[n-1]+kappa[n+1])=lambda*kappa[n]``."""

    kappa = _vector(curvatures, name="curvatures")
    lam = recurrence_multiplier(kappa, sign=sign) if multiplier is None else multiplier
    return (
        curvature_weights(kappa)
        * (twisted_shift(kappa, -1, sign=sign) + twisted_shift(kappa, 1, sign=sign))
        - float(lam) * kappa
    )


def qrt_invariant(
    curvatures: ArrayLike, multiplier: float, *, sign: int = 1
) -> FloatArray:
    """Return the biquadratic invariant ``J[n]`` along the recurrence."""

    kappa = _vector(curvatures, name="curvatures")
    d = curvature_weights(kappa)
    return (
        twisted_shift(d, -1) * d
        - 0.25 * float(multiplier) * twisted_shift(kappa, -1, sign=sign) * kappa
    )


def curvature_flow(
    name: FlowSpec,
    *,
    sign: int,
    torsion_angle: float | None = None,
) -> Callable[[float, FloatArray], FloatArray]:
    """Create a ``solve_ivp``-compatible curvature vector field."""

    boundary_sign = _sign(sign)
    flow_kind, order, _ = _parse_flow(name)
    if flow_kind == "mkdv":
        assert order is not None
        return lambda _time, kappa: mkdv_hierarchy_field(
            kappa, order, sign=boundary_sign
        )
    if flow_kind == "sine-gordon":
        if boundary_sign != -1:
            raise ValueError("the sine-Gordon flow is defined only for sign=-1")
        if torsion_angle is None:
            raise ValueError("torsion_angle is required for the sine-Gordon flow")
        return lambda _time, kappa: sine_gordon_field(kappa, torsion_angle, sign=-1)
    raise AssertionError("unreachable flow kind")


@dataclass(frozen=True, slots=True)
class IntegrableEvolution:
    """Numerical solution of an integrable curvature flow."""

    times: FloatArray
    curvatures: FloatArray
    flow: str
    torsion_angle: float
    sign: int
    initial_frame: FloatArray

    @property
    def first_hamiltonian(self) -> FloatArray:
        """Values of the conserved first Hamiltonian."""

        return np.array([first_hamiltonian(row) for row in self.curvatures])

    @property
    def second_hamiltonian(self) -> FloatArray:
        """Values of the conserved second Hamiltonian."""

        return np.array(
            [second_hamiltonian(row, sign=self.sign) for row in self.curvatures]
        )

    def spectral_integrals(self, count: int) -> FloatArray:
        """Return histories of ``I_1, ..., I_count`` by sampled time."""

        return np.array(
            [spectral_integrals(row, count, sign=self.sign) for row in self.curvatures]
        )

    def hierarchy_hamiltonians(self, count: int) -> FloatArray:
        """Return histories of ``E_1, ..., E_count`` by sampled time."""

        return np.array(
            [
                hierarchy_hamiltonians(row, count, sign=self.sign)
                for row in self.curvatures
            ]
        )

    def configuration(self, index: int) -> FramedPolygon:
        """Reconstruct one sampled configuration."""

        return reconstruct_framed_polygon(
            self.curvatures[index],
            self.torsion_angle,
            sign=self.sign,
            initial_frame=self.initial_frame,
        )

    def configurations(self) -> list[FramedPolygon]:
        """Reconstruct all sampled configurations."""

        return [self.configuration(index) for index in range(self.times.size)]


def integrate_curvature_flow(
    curvatures: ArrayLike,
    torsion_angle: float,
    times: ArrayLike,
    *,
    flow: FlowSpec = "mkdv1",
    sign: int = 1,
    initial_frame: ArrayLike | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> IntegrableEvolution:
    """Integrate an integrable deformation and sample it at ``times``."""

    kappa_0 = _vector(curvatures, name="curvatures")
    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1 or sample_times.size < 2:
        raise ValueError("times must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(sample_times)):
        raise ValueError("times must contain only finite values")
    if np.any(np.diff(sample_times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    boundary_sign = _sign(sign)
    frame_0 = _frame(initial_frame)

    mu = _torsion_angle(torsion_angle)
    _, _, canonical_flow = _parse_flow(flow)
    field = curvature_flow(flow, sign=boundary_sign, torsion_angle=mu)
    solution = solve_ivp(
        field,
        (float(sample_times[0]), float(sample_times[-1])),
        kappa_0,
        t_eval=sample_times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"curvature integration failed: {solution.message}")
    return IntegrableEvolution(
        times=np.asarray(solution.t, dtype=float),
        curvatures=np.asarray(solution.y.T, dtype=float),
        flow=canonical_flow,
        torsion_angle=mu,
        sign=boundary_sign,
        initial_frame=frame_0.copy(),
    )


__all__ = [
    "FlowName",
    "FlowSpec",
    "FramedPolygon",
    "IntegrableEvolution",
    "LiftCoefficients",
    "SpectralCurveDiagnostics",
    "cayley_curvatures",
    "critical_multiplier",
    "critical_torsion_cosine",
    "curvature_angles",
    "curvature_flow",
    "curvature_weights",
    "first_hamiltonian",
    "framed_polygon_from_binormals",
    "hierarchy_hamiltonian_gradients",
    "hierarchy_hamiltonians",
    "hierarchy_orbit_jacobian",
    "integrate_curvature_flow",
    "lift_coefficients",
    "lifted_velocities",
    "mkdv1_field",
    "mkdv2_field",
    "mkdv_hierarchy_field",
    "poisson_bracket",
    "poisson_operator",
    "qrt_invariant",
    "reconstruct_framed_polygon",
    "recurrence_multiplier",
    "rotation_1",
    "rotation_3",
    "second_hamiltonian",
    "sine_gordon_field",
    "sine_gordon_potential",
    "spectral_integral_gradients",
    "spectral_integrals",
    "spectral_curve_diagnostics",
    "twisted_trace_polynomial",
    "twisted_shift",
    "variational_recurrence_residual",
    "variational_u",
]
