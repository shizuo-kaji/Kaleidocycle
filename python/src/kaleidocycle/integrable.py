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

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]
FlowName = Literal["mkdv1", "mkdv2", "sine-gordon"]


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
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )


def rotation_3(angle: float) -> FloatArray:
    """Rotation about the third coordinate axis."""

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


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
    shifted_angles = curvature_angles(
        twisted_shift(kappa, 1, sign=boundary_sign)
    )
    torsion_rotation = rotation_1(mu)

    for n in range(n_vertices):
        vertices[n + 1] = vertices[n] + frames[n, :, 0]
        frames[n + 1] = (
            frames[n] @ torsion_rotation @ rotation_3(shifted_angles[n])
        )

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
    sine = np.einsum(
        "ij,ij->i", tangents, np.cross(vectors[:-1], previous_tangents)
    )
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
    frames[-1] = initial_frame @ np.diag(
        [1.0, boundary_sign, boundary_sign]
    )
    return FramedPolygon(kappa, mu, boundary_sign, frames, vertices)


def _twisted_shift_rows(
    values: ArrayLike, offset: int, *, sign: int = 1
) -> FloatArray:
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


def mkdv1_field(curvatures: ArrayLike, *, sign: int = 1) -> FloatArray:
    """First semi-discrete mKdV vector field."""

    kappa = _vector(curvatures, name="curvatures")
    return 0.5 * curvature_weights(kappa) * (
        twisted_shift(kappa, 1, sign=sign)
        - twisted_shift(kappa, -1, sign=sign)
    )


def mkdv2_field(curvatures: ArrayLike, *, sign: int = 1) -> FloatArray:
    """Second commuting semi-discrete mKdV vector field."""

    kappa = _vector(curvatures, name="curvatures")
    d = curvature_weights(kappa)
    forward = twisted_shift(d, 1) * (
        kappa + twisted_shift(kappa, 2, sign=sign)
    )
    backward = twisted_shift(d, -1) * (
        twisted_shift(kappa, -2, sign=sign) + kappa
    )
    return 0.5 * d * (forward - backward)


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


def lift_coefficients(
    curvatures: ArrayLike,
    torsion_angle: float,
    *,
    flow: FlowName = "mkdv1",
    sign: int = 1,
) -> LiftCoefficients:
    """Return the vertex and angular velocities in moving-frame coordinates."""

    kappa = _vector(curvatures, name="curvatures")
    boundary_sign = _sign(sign)
    mu = _torsion_angle(torsion_angle)
    cosine = np.cos(mu)
    sine = np.sin(mu)
    d = curvature_weights(kappa)
    next_kappa = twisted_shift(kappa, 1, sign=boundary_sign)
    previous_kappa = twisted_shift(kappa, -1, sign=boundary_sign)

    if flow == "mkdv1":
        vertex = np.column_stack(
            (
                np.full_like(kappa, cosine),
                -0.5 * cosine * kappa,
                -0.5 * sine * kappa,
            )
        )
        angular = np.column_stack(
            (
                np.full_like(kappa, sine),
                -0.5 * sine * kappa,
                0.5 * (next_kappa + cosine * kappa),
            )
        )
        return LiftCoefficients(vertex, angular)

    if flow == "mkdv2":
        cos_2mu = np.cos(2.0 * mu)
        sin_2mu = np.sin(2.0 * mu)
        shared = (
            (1.0 - kappa**2 / 4.0) * next_kappa - d * previous_kappa
        )
        p = 0.5 * cosine * kappa * next_kappa + 2.0 * cos_2mu
        q = 0.5 * cosine * shared - cos_2mu * kappa
        r = -0.5 * sine * d * (next_kappa + previous_kappa) - sin_2mu * kappa
        xi = 0.5 * sine * kappa * next_kappa + sin_2mu
        eta = 0.5 * sine * shared - 0.5 * sin_2mu * kappa
        zeta = 0.5 * (
            twisted_shift(d, 1)
            * (kappa + twisted_shift(kappa, 2, sign=boundary_sign))
            + cosine * d * (previous_kappa + next_kappa)
        ) - sine**2 * kappa
        return LiftCoefficients(
            np.column_stack((p, q, r)),
            np.column_stack((xi, eta, zeta)),
        )

    if flow == "sine-gordon":
        if boundary_sign != -1:
            raise ValueError("the sine-Gordon lift is defined only for sign=-1")
        u = sine_gordon_potential(kappa)
        vertex = np.column_stack((np.cos(u), np.sin(u), np.zeros_like(u)))
        angular = -np.column_stack(
            (
                sine * np.cos(u),
                sine * np.sin(u),
                (1.0 - cosine) * np.sin(u),
            )
        )
        return LiftCoefficients(vertex, angular)

    raise ValueError(f"unknown flow {flow!r}")


def lifted_velocities(
    configuration: FramedPolygon, *, flow: FlowName = "mkdv1"
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
    return float(
        0.5 * np.dot(kappa, twisted_shift(kappa, 1, sign=sign))
    )


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
        twisted_shift(weighted, 1, sign=sign)
        - twisted_shift(weighted, -1, sign=sign)
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
        twisted_shift(kappa, -1, sign=sign)
        + twisted_shift(kappa, 1, sign=sign)
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
    return curvature_weights(kappa) * (
        twisted_shift(kappa, -1, sign=sign)
        + twisted_shift(kappa, 1, sign=sign)
    ) - float(lam) * kappa


def qrt_invariant(
    curvatures: ArrayLike, multiplier: float, *, sign: int = 1
) -> FloatArray:
    """Return the biquadratic invariant ``J[n]`` along the recurrence."""

    kappa = _vector(curvatures, name="curvatures")
    d = curvature_weights(kappa)
    return (
        twisted_shift(d, -1)
        * d
        - 0.25
        * float(multiplier)
        * twisted_shift(kappa, -1, sign=sign)
        * kappa
    )


def curvature_flow(
    name: FlowName,
    *,
    sign: int,
    torsion_angle: float | None = None,
) -> Callable[[float, FloatArray], FloatArray]:
    """Create a ``solve_ivp``-compatible curvature vector field."""

    boundary_sign = _sign(sign)
    if name == "mkdv1":
        return lambda _time, kappa: mkdv1_field(kappa, sign=boundary_sign)
    if name == "mkdv2":
        return lambda _time, kappa: mkdv2_field(kappa, sign=boundary_sign)
    if name == "sine-gordon":
        if boundary_sign != -1:
            raise ValueError("the sine-Gordon flow is defined only for sign=-1")
        if torsion_angle is None:
            raise ValueError("torsion_angle is required for the sine-Gordon flow")
        return lambda _time, kappa: sine_gordon_field(
            kappa, torsion_angle, sign=-1
        )
    raise ValueError(f"unknown flow {name!r}")


@dataclass(frozen=True, slots=True)
class IntegrableEvolution:
    """Numerical solution of an integrable curvature flow."""

    times: FloatArray
    curvatures: FloatArray
    flow: FlowName
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
    flow: FlowName = "mkdv1",
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
        flow=flow,
        torsion_angle=mu,
        sign=boundary_sign,
        initial_frame=frame_0.copy(),
    )


__all__ = [
    "FlowName",
    "FramedPolygon",
    "IntegrableEvolution",
    "LiftCoefficients",
    "cayley_curvatures",
    "critical_multiplier",
    "critical_torsion_cosine",
    "curvature_angles",
    "curvature_flow",
    "curvature_weights",
    "first_hamiltonian",
    "framed_polygon_from_binormals",
    "integrate_curvature_flow",
    "lift_coefficients",
    "lifted_velocities",
    "mkdv1_field",
    "mkdv2_field",
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
    "twisted_shift",
    "variational_recurrence_residual",
    "variational_u",
]
