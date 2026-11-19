"""Jacobi elliptic torsion-critical families of Kaleidocycles.

This module implements the explicit critical configurations studied in
*Integrable deformations of Kaleidocycles* (Sections 6 and 7):

``kappa[n] = A cn(omega + n*delta; k)`` with ``delta = 2 K(k) m / N``,
``A = 2 k sn(delta) / dn(delta)`` and multiplier
``lambda = 2 cn(delta) / dn(delta)**2``.

Every such sequence solves the stationary recurrence
``D[n] (kappa[n-1] + kappa[n+1]) = lambda kappa[n]`` with boundary sign
``sigma = (-1)**m``.  Its torsion cosine is the explicit branch
``c_m(k) = cn(delta) + sn(delta)/dn(delta) Z(delta; k)``, and it closes up
exactly when the closing angle ``Theta(k)`` of the frame monodromy reaches
``-2*pi*j`` (Proposition "elliptic closure count").  The principal family
``m = 3, j = 1`` realises the global minimum ``mu_N`` of the torsion angle
over anti-oriented rings.

The module also provides the quadratic forms behind the index theorem: the
Jacobi operator of the recurrence and the second variation of the torsion on
the closed design space, together with the continuum constants
``k_inf`` and ``C_inf`` of the continuum-limit theorem.

Moduli are the elliptic modulus ``k`` (not the parameter ``k**2``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipe, ellipj, ellipk, elliprd, elliprf

from .integrable import FramedPolygon, reconstruct_framed_polygon

FloatArray = NDArray[np.float64]

__all__ = [
    "EllipticFamily",
    "TorsionHessian",
    "branch_continuum_constants",
    "closed_elliptic_family",
    "closing_angle_curve",
    "closing_moduli",
    "continuum_closing_density",
    "continuum_constants",
    "elliptic_family",
    "elliptic_torsion_cosine",
    "fixed_torsion_jacobian",
    "jacobi_operator",
    "jacobi_zeta",
    "minimum_torsion_angle",
    "planar_modulus",
    "project_to_design_space",
    "principal_family",
    "quadratic_inertia",
    "right_angle_family",
    "torsion_hessian",
    "torsion_limit_modulus",
    "torus_knot_binormals",
]


# ---------------------------------------------------------------------------
# Elliptic helpers
# ---------------------------------------------------------------------------


def _check_modulus(modulus: float) -> float:
    k = float(modulus)
    if not np.isfinite(k) or not 0.0 < k < 1.0:
        raise ValueError("modulus must lie strictly between 0 and 1")
    return k


def jacobi_zeta(u: ArrayLike, modulus: float) -> FloatArray | float:
    """Jacobi zeta function ``Z(u; k) = E(am u; k) - E(k) u / K(k)``.

    The incomplete integral is evaluated with Carlson's symmetric forms after
    reducing ``u`` to ``(-K, K]`` (``Z`` is odd with period ``2K``).  This
    avoids sporadic wrong values of ``scipy.special.ellipeinc``.
    """

    k = _check_modulus(modulus)
    parameter = k * k
    quarter = ellipk(parameter)
    values = np.asarray(u, dtype=float)
    reduced = values - 2.0 * quarter * np.round(values / (2.0 * quarter))
    sn, cn, dn, _ = ellipj(reduced, parameter)
    cn2 = cn * cn
    dn2 = dn * dn
    incomplete = sn * elliprf(cn2, dn2, 1.0) - parameter / 3.0 * sn**3 * elliprd(
        cn2, dn2, 1.0
    )
    zeta = incomplete - ellipe(parameter) / quarter * reduced
    if np.ndim(zeta) == 0:
        return float(zeta)
    return np.asarray(zeta, dtype=float)


def _rotation_angle_about(rotation: FloatArray, axis: FloatArray) -> float:
    """Signed angle in ``(-pi, pi]`` of a rotation fixing ``axis``."""

    unit = axis / np.linalg.norm(axis)
    helper = np.eye(3)[int(np.argmin(np.abs(unit)))]
    first = np.cross(unit, helper)
    first /= np.linalg.norm(first)
    image = rotation @ first
    return float(np.arctan2(unit @ np.cross(first, image), first @ image))


# ---------------------------------------------------------------------------
# Elliptic families
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EllipticFamily:
    """The cn family ``kappa[n] = A cn(omega + n delta; k)``.

    Attributes follow the paper: ``winding`` is the integer ``m`` with
    ``N delta = 2 K m``, ``sign = (-1)**m`` is the monodromy sign and
    ``torsion_cosine`` is the branch ``c_m(k)``.  The family is a closed
    Kaleidocycle only at a closing modulus; see :func:`closing_moduli`.
    """

    n_hinges: int
    winding: int
    modulus: float
    sign: int
    quarter_period: float
    step: float
    amplitude: float
    multiplier: float
    torsion_cosine: float

    @property
    def phase_period(self) -> float:
        """Period ``4K`` of the phase ``omega``."""

        return 4.0 * self.quarter_period

    @property
    def torsion_angle(self) -> float:
        """Torsion angle ``mu = arccos(c)``; requires ``|c| < 1``."""

        if not -1.0 < self.torsion_cosine < 1.0:
            raise ValueError(
                f"torsion cosine {self.torsion_cosine:.6g} is outside (-1, 1)"
            )
        return float(np.arccos(self.torsion_cosine))

    def curvatures(self, phase: float = 0.0) -> FloatArray:
        """Cayley curvatures ``kappa[0..N-1]`` at elliptic phase ``omega``."""

        arguments = float(phase) + self.step * np.arange(self.n_hinges)
        _, cn, _, _ = ellipj(arguments, self.modulus**2)
        return self.amplitude * cn

    def configuration(
        self,
        phase: float = 0.0,
        *,
        initial_frame: ArrayLike | None = None,
    ) -> FramedPolygon:
        """Reconstruct frames and vertices with ``F[0] = initial_frame``."""

        return reconstruct_framed_polygon(
            self.curvatures(phase),
            self.torsion_angle,
            sign=self.sign,
            initial_frame=initial_frame,
        )

    def binormals(self, phase: float = 0.0) -> FloatArray:
        """Hinge directions ``B[0..N]`` (terminal value included)."""

        return self.configuration(phase).binormals

    def multiplier_vector(self, phase: float = 0.0) -> FloatArray:
        """Conserved multiplier vector ``M`` of the multiplier lemma.

        It is computed from the frame ``F[0]`` of :meth:`configuration`
        (the identity), so it is expressed in that reconstruction.
        """

        kappa = self.curvatures(phase)
        c = self.torsion_cosine
        s = float(np.sqrt(1.0 - c * c))
        rho = -1.0 / (2.0 * self.n_hinges)
        k0, k1 = kappa[0], kappa[1]
        return rho * np.array(
            [(self.multiplier - 2.0 * c - k0 * k1 / 2.0) / s, (c * k0 - k1) / s, k0]
        )

    def closing_rotation(self, phase: float = 0.0) -> FloatArray:
        """Frame monodromy defect ``R = F[N] H_sigma^{-1}`` with ``F[0] = I``."""

        frames = self.configuration(phase).frames
        return frames[-1] @ np.diag([1.0, self.sign, self.sign])

    def closing_angle(self, phase: float = 0.0) -> float:
        """Principal value of the rotation angle of ``R`` about ``M``.

        ``R`` fixes the multiplier vector, and the family closes exactly when
        this angle vanishes.  Use :func:`closing_angle_curve` for the
        continuous lift ``Theta`` with ``Theta(0+) = 0``.
        """

        return _rotation_angle_about(
            self.closing_rotation(phase), self.multiplier_vector(phase)
        )


def elliptic_family(n_hinges: int, winding: int, modulus: float) -> EllipticFamily:
    """Construct the cn family with ``N`` hinges, winding ``m`` and modulus ``k``."""

    n = int(n_hinges)
    m = int(winding)
    if n < 3:
        raise ValueError("n_hinges must be at least 3")
    if not 0 < m < n:
        raise ValueError("winding must satisfy 0 < m < n_hinges")
    k = _check_modulus(modulus)
    parameter = k * k
    quarter = float(ellipk(parameter))
    step = 2.0 * quarter * m / n
    sn, cn, dn, _ = ellipj(step, parameter)
    amplitude = 2.0 * k * sn / dn
    multiplier = 2.0 * cn / dn**2
    cosine = cn + sn / dn * jacobi_zeta(step, k)
    return EllipticFamily(
        n_hinges=n,
        winding=m,
        modulus=k,
        sign=1 if m % 2 == 0 else -1,
        quarter_period=quarter,
        step=float(step),
        amplitude=float(amplitude),
        multiplier=float(multiplier),
        torsion_cosine=float(cosine),
    )


def elliptic_torsion_cosine(n_hinges: int, winding: int, modulus: float) -> float:
    """Torsion branch ``c_m(k) = cn(delta) + sn(delta)/dn(delta) Z(delta; k)``."""

    return elliptic_family(n_hinges, winding, modulus).torsion_cosine


def torsion_limit_modulus(n_hinges: int, winding: int, *, cosine: float = 1.0) -> float:
    """Modulus at which the increasing torsion branch reaches ``cosine``.

    For ``0 < m < N/2`` the branch increases from ``cos(pi m / N)`` to
    infinity, so the default returns the endpoint ``q_m^*`` of the
    admissible interval (``c_m = 1``).
    """

    m = int(winding)
    if not 0 < 2 * m < int(n_hinges):
        raise ValueError("torsion_limit_modulus requires 0 < m < N/2")

    def gap(k: float) -> float:
        return elliptic_torsion_cosine(n_hinges, winding, k) - cosine

    return float(brentq(gap, 1e-12, 1.0 - 1e-15, xtol=1e-15, rtol=1e-15))


def closing_angle_curve(
    n_hinges: int,
    winding: int,
    moduli: ArrayLike | None = None,
    *,
    samples: int = 240,
    phase: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return moduli and the continuous closing angle ``Theta`` on them.

    The default grid covers ``(0, q_m^*)`` up to ``c_m = 1 - 1e-8``.  The lift
    is normalised by ``Theta(0+) = 0``; by the paper it decreases strictly to
    ``-m*pi``.  On the default grid, intervals whose principal values jump by
    more than ``pi/2`` are bisected before unwrapping.
    """

    if moduli is None:
        upper = torsion_limit_modulus(n_hinges, winding, cosine=1.0 - 1e-8)
        grid = np.linspace(1e-4, upper, int(samples))
    else:
        grid = np.asarray(moduli, dtype=float)
        if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
            raise ValueError("moduli must be a strictly increasing 1-D array")

    def principal_values(values: FloatArray) -> FloatArray:
        return np.array(
            [elliptic_family(n_hinges, winding, k).closing_angle(phase) for k in values]
        )

    angles = principal_values(grid)
    for _ in range(6):
        jumps = np.angle(np.exp(1j * np.diff(angles)))
        coarse = np.nonzero(np.abs(jumps) > np.pi / 2)[0]
        if coarse.size == 0 or moduli is not None:
            break
        midpoints = 0.5 * (grid[coarse] + grid[coarse + 1])
        grid = np.sort(np.concatenate((grid, midpoints)))
        angles = principal_values(grid)
    jumps = np.angle(np.exp(1j * np.diff(angles)))
    theta = angles[0] + np.concatenate(([0.0], np.cumsum(jumps)))
    # Normalise the lift so that Theta(0+) = 0.
    theta -= 2.0 * np.pi * np.round(theta[0] / (2.0 * np.pi))
    return grid, theta


@lru_cache(maxsize=256)
def closing_moduli(n_hinges: int, winding: int) -> tuple[float, ...]:
    """Closing moduli ``k_{N,m,j}``, ``j = 1, ..., floor((m-1)/2)``.

    Each is the unique solution of ``Theta(k) = -2*pi*j``.  Even ``m`` gives
    oriented rings, odd ``m`` anti-oriented rings.
    """

    n = int(n_hinges)
    m = int(winding)
    if not 0 < 2 * m < n:
        raise ValueError("closing_moduli requires 0 < m < N/2")
    levels = range(1, (m - 1) // 2 + 1)
    if not levels:
        return ()
    ks, theta = closing_angle_curve(n, m)

    def principal(k: float) -> float:
        return elliptic_family(n, m, k).closing_angle()

    roots = []
    for level in levels:
        target = -2.0 * np.pi * level
        crossing = np.nonzero((theta[:-1] - target) * (theta[1:] - target) <= 0.0)[0]
        if crossing.size == 0:
            raise RuntimeError(f"closing level {level} not bracketed for N={n}, m={m}")
        index = int(crossing[0])
        roots.append(
            float(brentq(principal, ks[index], ks[index + 1], xtol=1e-15, rtol=1e-15))
        )
    return tuple(roots)


def closed_elliptic_family(
    n_hinges: int, winding: int = 3, level: int = 1
) -> EllipticFamily:
    """Closed cn family at closing level ``Theta = -2*pi*level``."""

    roots = closing_moduli(int(n_hinges), int(winding))
    if not 1 <= int(level) <= len(roots):
        raise ValueError(
            f"level must lie in 1..{len(roots)} for N={n_hinges}, m={winding}"
        )
    return elliptic_family(n_hinges, winding, roots[int(level) - 1])


def principal_family(n_hinges: int) -> EllipticFamily:
    """The principal anti-oriented family (``m = 3``), defined for ``N >= 7``."""

    if int(n_hinges) < 7:
        raise ValueError("the principal family exists for N >= 7")
    return closed_elliptic_family(n_hinges, 3, 1)


def minimum_torsion_angle(n_hinges: int) -> float:
    """Global minimum ``mu_N`` of the torsion angle of anti-oriented rings."""

    return principal_family(n_hinges).torsion_angle


def right_angle_family(n_hinges: int, level: int = 1) -> EllipticFamily:
    """Right-angle critical circle (``c = 0``) with closing index ``level``.

    These are the resonant circles of the paper: the cn family with
    ``m = N/2`` and ``k = sin(2 pi j / N)``, so ``kappa[n+2] = -kappa[n]`` and
    ``(1 + kappa[0]**2/4)(1 + kappa[1]**2/4) = sec(2 pi j / N)**2``.  They exist
    for even ``N`` with ``1 <= j < N/4`` and have ``sigma = (-1)**(N/2)``.
    ``N = 6, j = 1`` is the classical hexagonal Kaleidocycle.
    """

    n = int(n_hinges)
    j = int(level)
    if n < 6 or n % 2:
        raise ValueError("right-angle circles need an even n_hinges >= 6")
    if not 1 <= j < n / 4:
        raise ValueError(f"level must satisfy 1 <= level < N/4 for N={n}")
    return elliptic_family(n, n // 2, float(np.sin(2.0 * np.pi * j / n)))


def torus_knot_binormals(cells: int, turns: int, *, mirror: bool = False) -> FloatArray:
    """Explicit rotationally symmetric ring with ``N = 3 * cells`` hinges.

    This is the construction of the quadratic-linking proposition: with
    ``alpha = 2 pi turns / cells`` the binormal triple ``(u, v, w)`` is
    repeated by rotations through ``alpha``.  The ring is oriented, has
    torsion cosine ``-h / (1 + 3h)`` with ``h = cos((pi - alpha)/3)``, all
    curvatures positive and normalised self-linking ``cells * turns``.  Its
    centre line is the ``(turns, cells)`` torus knot.  ``mirror=True``
    applies ``B[n] -> (-1)**n B[n]``, giving monodromy ``(-1)**cells`` and
    ``SL = 3 cells / 2 - cells * turns``.

    Returns the ``N + 1`` binormals with the terminal value included.
    """

    m = int(cells)
    k = int(turns)
    if m < 3 or not 1 <= k < m / 2 or np.gcd(m, k) != 1:
        raise ValueError("requires cells >= 3, 1 <= turns < cells/2, gcd = 1")
    alpha = 2.0 * np.pi * k / m
    beta = (np.pi - alpha) / 3.0
    h = np.cos(beta)
    d = (1.0 + h) * (1.0 + 3.0 * h)
    r = np.sqrt((1.0 + 4.0 * h) / d)
    z = -np.sqrt(3.0) * h / np.sqrt(d)
    t = np.sqrt(3.0) * (1.0 + 2.0 * h) / (2.0 * np.sqrt(d))
    triple = np.array(
        [
            [r * h, r * np.sin(beta), z],
            [r / 2.0, 0.0, t],
            [r * h, -r * np.sin(beta), z],
        ]
    )
    blocks = []
    for index in range(m):
        cosine, sine = np.cos(index * alpha), np.sin(index * alpha)
        rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        blocks.append(triple @ rotation.T)
    binormals = np.vstack([*blocks, triple[:1]])
    if mirror:
        binormals = binormals * ((-1.0) ** np.arange(binormals.shape[0]))[:, None]
    return binormals


# ---------------------------------------------------------------------------
# Quadratic forms of the index theorem
# ---------------------------------------------------------------------------


def quadratic_inertia(
    matrix: ArrayLike, *, tol: float | None = None
) -> tuple[int, int, int]:
    """Inertia ``(positive, negative, zero)`` of a symmetric matrix."""

    sym = np.asarray(matrix, dtype=float)
    sym = 0.5 * (sym + sym.T)
    eigenvalues = np.linalg.eigvalsh(sym)
    if eigenvalues.size == 0:
        return (0, 0, 0)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    cutoff = 1e-8 * scale if tol is None else float(tol)
    positive = int(np.sum(eigenvalues > cutoff))
    negative = int(np.sum(eigenvalues < -cutoff))
    return (positive, negative, int(eigenvalues.size - positive - negative))


def _twisted_shift_matrix(size: int, sign: int) -> FloatArray:
    shift = np.zeros((size, size))
    shift[np.arange(size - 1), np.arange(1, size)] = 1.0
    shift[size - 1, 0] = float(sign)
    return shift


def jacobi_operator(
    curvatures: ArrayLike, multiplier: float, *, sign: int = 1
) -> FloatArray:
    """Jacobi operator ``Hess(E2 - lambda E1)`` on ``P_N^sigma``.

    ``J = (S + S^-1)/2 - lambda diag((1 - kappa**2/4) / (2 D**2))`` with the
    twisted shift ``S``.  At a solution of the recurrence its kernel contains
    the first mKdV field.
    """

    kappa = np.asarray(curvatures, dtype=float)
    if kappa.ndim != 1:
        raise ValueError("curvatures must be one-dimensional")
    shift = _twisted_shift_matrix(kappa.size, sign)
    weights = 1.0 + kappa**2 / 4.0
    diagonal = float(multiplier) * (1.0 - kappa**2 / 4.0) / (2.0 * weights**2)
    return 0.5 * (shift + shift.T) - np.diag(diagonal)


def _split_binormals(binormals: ArrayLike, sign: int) -> FloatArray:
    vectors = np.asarray(binormals, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] < 4:
        raise ValueError("binormals must have shape (N + 1, 3) with N >= 3")
    if not np.allclose(vectors[-1], sign * vectors[0], atol=1e-6):
        raise ValueError("terminal binormal does not match the monodromy sign")
    return vectors[:-1]


def _design_constraints(
    vectors: FloatArray, sign: int
) -> tuple[FloatArray, FloatArray]:
    """Residuals and Jacobian of the ``2N + 3`` constraints in ``(B, c)``."""

    n = vectors.shape[0]
    following = np.roll(vectors, -1, axis=0)
    following[-1] *= sign
    cosine = float(np.mean(np.einsum("ij,ij->i", vectors, following)))
    residual = np.concatenate(
        (
            np.einsum("ij,ij->i", vectors, vectors) - 1.0,
            np.einsum("ij,ij->i", vectors, following) - cosine,
            np.cross(vectors, following).sum(axis=0),
        )
    )
    jac = np.zeros((2 * n + 3, 3 * n + 1))
    for i in range(n):
        j = (i + 1) % n
        factor = sign if i == n - 1 else 1.0
        jac[i, 3 * i : 3 * i + 3] = 2.0 * vectors[i]
        jac[n + i, 3 * i : 3 * i + 3] += following[i]
        jac[n + i, 3 * j : 3 * j + 3] += factor * vectors[i]
        jac[n + i, -1] = -1.0
        # d(B_i x B_{i+1}) = -[B_{i+1}]_x dB_i + [B_i]_x dB_{i+1}
        jac[2 * n : 2 * n + 3, 3 * i : 3 * i + 3] += -_skew(following[i])
        jac[2 * n : 2 * n + 3, 3 * j : 3 * j + 3] += factor * _skew(vectors[i])
    return residual, jac


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fixed_torsion_jacobian(binormals: ArrayLike, *, sign: int) -> FloatArray:
    """Jacobian ``J_c`` of the ``2N + 3`` binormal constraints at fixed ``c``.

    By the rank-loss corollary it has rank ``2N + 2`` exactly at torsion
    critical closed configurations, and ``2N + 3`` otherwise.
    """

    vectors = _split_binormals(binormals, sign)
    _, jac = _design_constraints(vectors, sign)
    return jac[:, :-1]


def project_to_design_space(
    binormals: ArrayLike,
    *,
    sign: int,
    torsion_cosine: float | None = None,
    tol: float = 1e-13,
    max_iter: int = 50,
) -> FloatArray:
    """Project hinge directions onto the closed design space by Gauss-Newton.

    Solves the ``2N + 3`` equations ``|B_n| = 1``, ``B_n . B_{n+1} = c`` and
    ``C(B) = 0`` with minimum-norm Newton steps.  The torsion cosine ``c`` is
    an unknown unless ``torsion_cosine`` is given, in which case the result
    lies in the fixed-torsion fibre.  Returns an ``(N + 1, 3)`` array with the
    terminal value ``sign * B_0``.
    """

    boundary = int(sign)
    vectors = np.array(_split_binormals(binormals, boundary), dtype=float)
    n = vectors.shape[0]
    following = np.roll(vectors, -1, axis=0)
    following[-1] *= boundary
    cosine = (
        float(np.mean(np.einsum("ij,ij->i", vectors, following)))
        if torsion_cosine is None
        else float(torsion_cosine)
    )
    for _ in range(int(max_iter)):
        following = np.roll(vectors, -1, axis=0)
        following[-1] *= boundary
        residual = np.concatenate(
            (
                np.einsum("ij,ij->i", vectors, vectors) - 1.0,
                np.einsum("ij,ij->i", vectors, following) - cosine,
                np.cross(vectors, following).sum(axis=0),
            )
        )
        if np.linalg.norm(residual) < tol:
            break
        _, jac = _design_constraints(vectors, boundary)
        if torsion_cosine is not None:
            jac = jac[:, :-1]
        step, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        vectors = vectors + step[: 3 * n].reshape(n, 3)
        if torsion_cosine is None:
            cosine += float(step[-1])
    else:
        raise RuntimeError("projection onto the design space did not converge")
    return np.vstack((vectors, boundary * vectors[0]))


@dataclass(frozen=True, slots=True)
class TorsionHessian:
    """Second variation of the torsion on the closed design space.

    ``inertia`` is that of ``Hess(mu)`` modulo rotations, ordered as
    (positive, negative, zero); the index theorem predicts
    ``(N - m - 3, m - 3, 1)`` on the cn family of winding ``m``.
    ``design_inertia`` is the inertia on the full tangent space of the closed
    design space before removing the three rotations.  ``eigenvectors`` holds
    the corresponding orthonormal tangent directions in the ambient
    coordinates ``(B_0, ..., B_{N-1}, c)``, one column per eigenvalue (in
    increasing order); positive eigenvalues are directions in which the
    torsion angle increases.
    """

    inertia: tuple[int, int, int]
    design_inertia: tuple[int, int, int]
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    stationarity_residual: float
    constraint_residual: float
    design_dimension: int
    torsion_multipliers: FloatArray
    unit_multipliers: FloatArray
    closure_multiplier: FloatArray


def torsion_hessian(
    binormals: ArrayLike, *, sign: int, tol: float | None = None
) -> TorsionHessian:
    """Hessian of the torsion angle at a torsion-critical closed configuration.

    The Lagrange multipliers of ``S = -c + sum a_n (B_n.B_{n+1} - c) +
    sum b_n (|B_n|^2 - 1) + <M, C(B)>`` are obtained by least squares.  A
    small ``stationarity_residual`` certifies that the configuration is
    torsion critical; then ``Hess(mu) = G / sin(mu)`` on the tangent space,
    where ``G`` is the Hessian of ``S``.
    """

    boundary = int(sign)
    vectors = _split_binormals(binormals, boundary)
    n = vectors.shape[0]
    residual, jac = _design_constraints(vectors, boundary)
    target = np.zeros(3 * n + 1)
    target[-1] = 1.0
    multipliers, *_ = np.linalg.lstsq(jac.T, target, rcond=None)
    stationarity = float(np.linalg.norm(jac.T @ multipliers - target))
    unit = multipliers[:n]
    torsion = multipliers[n : 2 * n]
    closure = multipliers[2 * n :]

    hessian = np.zeros((3 * n + 1, 3 * n + 1))
    identity = np.eye(3)
    coupling = -_skew(closure)
    for i in range(n):
        j = (i + 1) % n
        factor = boundary if i == n - 1 else 1.0
        hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += 2.0 * unit[i] * identity
        block = factor * (torsion[i] * identity + coupling)
        hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
        hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += block.T

    _, singular, vt = np.linalg.svd(jac)
    rank = int(np.sum(singular > 1e-9 * max(singular[0], 1.0)))
    tangent = vt[rank:].T
    reduced = tangent.T @ hessian @ tangent
    design = quadratic_inertia(reduced, tol=tol)
    eigenvalues, eigenbasis = np.linalg.eigh(0.5 * (reduced + reduced.T))
    positive, negative, zero = design
    return TorsionHessian(
        inertia=(positive, negative, max(zero - 3, 0)),
        design_inertia=design,
        eigenvalues=eigenvalues,
        eigenvectors=tangent @ eigenbasis,
        stationarity_residual=stationarity,
        constraint_residual=float(np.linalg.norm(residual)),
        design_dimension=int(tangent.shape[1]),
        torsion_multipliers=torsion,
        unit_multipliers=unit,
        closure_multiplier=closure,
    )


# ---------------------------------------------------------------------------
# Continuum limit
# ---------------------------------------------------------------------------


def _continuum_parts(k: float) -> tuple[float, float, float, float]:
    parameter = k * k
    quarter = float(ellipk(parameter))
    a = 1.0 - float(ellipe(parameter)) / quarter
    t = float(np.sqrt(max(1.0 - 2.0 * a, 0.0)))
    d = float(np.sqrt(a * a + parameter * t * t))
    return quarter, a, t, d


def continuum_closing_density(modulus: float) -> float:
    """Limiting closing-angle density ``g(k) = 6 K t d / (k (1 - k^2))``.

    ``g(0) = 3*pi``; for winding ``m`` the density is ``(m/3) g``.
    """

    k = float(modulus)
    if k == 0.0:
        return 3.0 * np.pi
    quarter, _, t, d = _continuum_parts(k)
    return 6.0 * quarter * t * d / (k * (1.0 - k * k))


def planar_modulus() -> float:
    """The modulus ``k_p`` with ``2 E(k_p) = K(k_p)``."""

    return float(
        brentq(lambda k: 2.0 * ellipe(k * k) - ellipk(k * k), 0.5, 0.999999, xtol=1e-15)
    )


def _closing_integral(k: float) -> float:
    return float(quad(continuum_closing_density, 0.0, k, epsabs=1e-13, epsrel=1e-13)[0])


@lru_cache(maxsize=64)
def branch_continuum_constants(winding: int = 3, level: int = 1) -> tuple[float, float]:
    """Limits ``(k_{m,r}, C_{m,r})`` of the branch with winding ``m``.

    ``k_{m,r}`` solves ``(m/3) L(k) = 2*pi*r`` with ``L(k) = int_0^k g``, and
    ``mu_{N,m,r} = C_{m,r}/N + O(N^-3)`` with ``C_{m,r} = 2 m K t``.
    """

    m = int(winding)
    r = int(level)
    if m < 3 or not 1 <= r < m / 2:
        raise ValueError("requires m >= 3 and 1 <= level < m/2")
    upper = planar_modulus()
    target = 2.0 * np.pi * r * 3.0 / m
    k_limit = float(
        brentq(lambda k: _closing_integral(k) - target, 1e-6, upper - 1e-12, xtol=1e-14)
    )
    quarter, _, t, _ = _continuum_parts(k_limit)
    return k_limit, 2.0 * m * quarter * t


def continuum_constants() -> tuple[float, float]:
    """Limits ``(k_inf, C_inf)`` of the principal family.

    ``k_N -> k_inf`` and ``N mu_N -> C_inf``; the paper reports
    ``k_inf = 0.6309247559...`` and ``C_inf = 8.0934956856...``.
    """

    return branch_continuum_constants(3, 1)
