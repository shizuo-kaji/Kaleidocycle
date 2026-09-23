"""A transverse self-crossing under the first semi-discrete mKdV flow.

The stored K12 example is anti-oriented, with torsion angle 6/5 radians.
Time zero denotes the certified contact. Numerical trajectories are separate
from the interval existence proof in :mod:`kaleidocycle.collision_certificate`.
Writhe here uses the standard Gauss normalisation (a crossing jumps by two).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .integrable import (
    FramedPolygon,
    cayley_curvatures,
    first_hamiltonian,
    lifted_velocities,
    mkdv1_field,
    reconstruct_framed_polygon,
)
from .samples import default_sample_directory

FloatArray = NDArray[np.float64]


def load_crossing_data(path: str | Path | None = None) -> dict[str, Any]:
    """Read the exact decimal centre, parameters and certificate matrix."""
    if path is None:
        path = (
            default_sample_directory().parent
            / "counterexamples"
            / "mkdv_self_crossing_k12.json"
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reconstruct_crossing() -> FramedPolygon:
    """Return the floating-point representative of the certified contact."""
    data = load_crossing_data()
    angles = np.asarray(data["center"][: data["n"]], dtype=float)
    return reconstruct_framed_polygon(
        cayley_curvatures(angles), float(data["torsion_angle"]), sign=data["sign"]
    )


def closest_segment_points(
    start_a: ArrayLike, end_a: ArrayLike, start_b: ArrayLike, end_b: ArrayLike
) -> tuple[float, float, float]:
    """Return distance and segment fractions, including endpoint minima."""
    a, b, c, d = np.asarray([start_a, end_a, start_b, end_b], dtype=float)
    edge_a, edge_b, delta = b - a, d - c, a - c
    aa, bb, ab = edge_a @ edge_a, edge_b @ edge_b, edge_a @ edge_b
    if min(aa, bb) <= 0:
        raise ValueError("segments must have positive length")
    ar, br = edge_a @ delta, edge_b @ delta
    candidates = []
    for u in (0.0, 1.0):
        candidates.append((u, float(np.clip((br + u * ab) / bb, 0, 1))))
    for v in (0.0, 1.0):
        candidates.append((float(np.clip((v * ab - ar) / aa, 0, 1)), v))
    det = aa * bb - ab * ab
    if det > 1e-14 * aa * bb:
        u = (ab * br - bb * ar) / det
        v = (aa * br - ab * ar) / det
        if 0 <= u <= 1 and 0 <= v <= 1:
            candidates.append((float(u), float(v)))
    return min(
        (float(np.linalg.norm(delta + u * edge_a - v * edge_b)), u, v)
        for u, v in candidates
    )


def segment_separations(vertices: ArrayLike) -> list[tuple[float, int, int]]:
    """Distances for all non-adjacent edges of a closed polygon."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("vertices must have shape (N+1, 3)")
    if not np.all(np.isfinite(points)) or not np.allclose(
        points[0], points[-1], atol=1e-8, rtol=0
    ):
        raise ValueError("polygon must be finite and closed")
    n = len(points) - 1
    return [
        (closest_segment_points(*points[i : i + 2], *points[j : j + 2])[0], i, j)
        for i in range(n)
        for j in range(i + 2, n)
        if (i, j) != (0, n - 1)
    ]


def gauss_writhe(vertices: ArrayLike, *, contact_tolerance: float = 1e-10) -> float:
    """Gauss writhe; return NaN at a contact instead of a spurious branch value.

    Each unordered edge pair contributes its oriented spherical quadrilateral
    area divided by 2*pi. This is the conventional writhe, unlike the historical
    pi-normalised ``geometry.writhe`` API. The terminal vertex is repeated.
    """
    points = np.asarray(vertices, dtype=float)
    pairs = segment_separations(points)
    if min(item[0] for item in pairs) <= contact_tolerance:
        return float("nan")

    def solid_angle(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
        a, b, c = (v / np.linalg.norm(v) for v in (a, b, c))
        return float(2 * np.arctan2(a @ np.cross(b, c), 1 + a @ b + b @ c + c @ a))

    total = 0.0
    for _, i, j in pairs:
        a, b = points[i] - points[j], points[i + 1] - points[j]
        c, d = points[i + 1] - points[j + 1], points[i] - points[j + 1]
        angle = solid_angle(a, b, c) + solid_angle(a, c, d)
        angle = (angle + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
        total -= angle / (2 * np.pi)
    return float(total)


def crossing_velocity(configuration: FramedPolygon | None = None) -> float:
    """Triple product of the two edge directions with their relative velocity."""
    configuration = reconstruct_crossing() if configuration is None else configuration
    data = load_crossing_data()
    i, j = data["edges"]
    u, v = np.asarray(data["center"][-2:], dtype=float)
    velocity, _ = lifted_velocities(configuration, flow="mkdv1")
    relative = (
        (1 - u) * velocity[i]
        + u * velocity[i + 1]
        - (1 - v) * velocity[j]
        - v * velocity[j + 1]
    )
    normal = np.cross(configuration.tangents[i], configuration.tangents[j])
    return float(normal @ relative)


@dataclass(frozen=True)
class CrossingEvolution:
    """Numerical samples with time measured from the certified contact."""

    times: FloatArray
    curvatures: FloatArray
    vertices: FloatArray
    min_distance: FloatArray
    pair_distance: FloatArray
    writhe: FloatArray
    linking: FloatArray
    closure: FloatArray
    monodromy: FloatArray
    energy: FloatArray
    twist: float


def sample_crossing(times: ArrayLike | None = None) -> CrossingEvolution:
    """Integrate from contact in both time directions using the package flow.

    The default interval is [-0.002, 0.002]. No constraint projection or
    collision-avoidance force is used. Geometry is reconstructed with F0=I,
    so its displayed motion is the geometric lift modulo rigid motion.
    """
    times = (
        np.linspace(-0.002, 0.002, 161) if times is None else np.asarray(times, float)
    )
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty finite vector")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    contact = reconstruct_crossing()
    data = load_crossing_data()
    kappa = np.tile(contact.curvatures, (len(times), 1))
    for mask in (times < 0, times > 0):
        if not mask.any():
            continue
        sample_times = times[mask]
        endpoint = sample_times[np.argmax(np.abs(sample_times))]
        solution = solve_ivp(
            lambda _, y: mkdv1_field(y, sign=contact.sign),
            (0, float(endpoint)),
            contact.curvatures,
            method="DOP853",
            dense_output=True,
            rtol=2e-13,
            atol=2e-14,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        kappa[mask] = solution.sol(sample_times).T
    polygons = [
        reconstruct_framed_polygon(k, contact.torsion_angle, sign=contact.sign)
        for k in kappa
    ]
    vertices = np.asarray([p.vertices for p in polygons])
    separations = [segment_separations(p) for p in vertices]
    i, j = data["edges"]
    wr = np.asarray([gauss_writhe(p) for p in vertices])
    # Zero is a certified contact, although double precision leaves a residual.
    wr[times == 0] = np.nan
    twist = len(contact.curvatures) * contact.torsion_angle / (2 * np.pi)
    return CrossingEvolution(
        times,
        kappa,
        vertices,
        np.asarray([min(s)[0] for s in separations]),
        np.asarray([next(d for d, a, b in s if (a, b) == (i, j)) for s in separations]),
        wr,
        wr + twist,
        np.asarray([np.linalg.norm(p.closure_residual) for p in polygons]),
        np.asarray([np.linalg.norm(p.monodromy_residual) for p in polygons]),
        np.asarray([first_hamiltonian(k) for k in kappa]),
        twist,
    )
