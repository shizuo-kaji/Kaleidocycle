"""Energy functionals translated from the legacy codebase."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

LOG2: Final[float] = math.log(2.0)


def _pairwise_vectors(vectors: NDArray[np.float64], wrap: bool) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return aligned pair arrays optionally wrapping the final vector."""

    if wrap:
        rolled = np.roll(vectors, -1, axis=0)
        return vectors, rolled
    return vectors[:-1], vectors[1:]


def _normalized_dot(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute dot products normalised by ‖a‖‖b‖ with clipping."""

    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    denom = norms_a * norms_b
    if np.any(denom == 0):
        raise ValueError("zero-length vector encountered in energy computation")
    ratio = np.einsum("ij,ij->i", a, b) / denom
    return np.clip(ratio, -1.0, 1.0)


def bending_energy(
    tangents: NDArray[np.float64],
    *,
    quadratic: bool = False,
) -> float:
    """Bobenko-Suris bending energy over the tangent vectors."""

    if tangents.ndim != 2 or tangents.shape[1] != 3:
        msg = f"expected (m, 3) tangent array, got shape {tangents.shape}"
        raise ValueError(msg)
    a, b = _pairwise_vectors(tangents, wrap=True)
    ratios = _normalized_dot(a, b)
    # avoid numerical error near ratio = -1
    ratios = np.clip(ratios, -1.0 + 1e-15, 1.0)
    # return squared angles or logarithmic form defined by Bobenko-Suris
    if quadratic:
        angles = np.arccos(ratios)
        return float(np.sum(angles**2))
    return float(np.sum(LOG2 - np.log1p(ratios)))


def torsion_energy(
    hinges: NDArray[np.float64],
    *,
    wrap: bool = False,
    quadratic: bool = False,
) -> float:
    """Torsion energy over hinge vectors."""

    if hinges.ndim != 2 or hinges.shape[1] != 3:
        msg = f"expected (n, 3) hinge array, got shape {hinges.shape}"
        raise ValueError(msg)
    a, b = _pairwise_vectors(hinges, wrap=wrap)
    ratios = _normalized_dot(a, b)
    if quadratic:
        angles = np.arccos(ratios)
        return float(np.sum(angles**2))
    return float(np.sum(LOG2 - np.log1p(ratios)))


def dipole_energy(
    hinges: NDArray[np.float64],
    curve: NDArray[np.float64],
) -> float:
    """Dipole energy between hinge directions and curve points."""

    if len(hinges) != len(curve):
        msg = "hinges and curve must have the same length (including repeated endpoint)"
        raise ValueError(msg)
    total = 0.0
    # exclude repeated last hinge/curve point to mirror Mathematica's behaviour
    limit = len(hinges) - 1
    for i in range(limit):
        for j in range(i + 1, limit):
            diff = curve[i] - curve[j]
            norm2 = float(np.dot(diff, diff))
            if norm2 == 0:
                continue
            dot_h = float(np.dot(hinges[i], hinges[j]))
            term = dot_h / (norm2 ** 1.5)
            proj = float(np.dot(hinges[i], diff) * np.dot(hinges[j], diff))
            term -= 3.0 * proj / (norm2 ** 2.5)
            total += term
    return total
