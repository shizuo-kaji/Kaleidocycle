"""Constraint helpers mirroring the Mathematica/Maple Setup function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .geometry import binormals_to_tangents, curvature_recursion, pairwise_curvature


@dataclass(slots=True)
class ConstraintConfig:
    """Configuration flags"""

    slide: float = 0.0  # slide term for closure constraint
    oriented: bool = False
    enforce_anchors: bool = False  # fix rigid motion by anchoring first two hinges
    constant_torsion: bool = True
    alignment: bool = True  # first and last hinge alignment
    closure: bool = True  # curve closure constraint
    full_alignment: bool = False
    # When True, alignment contributes the full 3-vector ``h[0] ∓ h[-1]``
    # instead of a single scalar norm. The scalar version has rank-0
    # Jacobian at the manifold, while the 3-vector version is rank-3,
    # so it is required for Kutzbach-Grübler-style DoF counting and
    # stationary-point analysis. Has no effect when ``alignment=False``.
    reference_torsion: float | None = (
        None  # reference value for constant torsion constraint
    )
    target_linking: float | None = None  # target linking number for linking constraint
    curvature_recursion: bool = (
        False  # curvature recursion constraint (u[i] - u[0] = 0)
    )


def enforce_terminal(
    hinges: NDArray[np.float64], oriented: bool
) -> NDArray[np.float64]:
    """Ensure the final hinge repeats the first hinge."""

    if len(hinges) == 0:
        return hinges
    hinges = np.array(hinges, dtype=float, copy=True)
    hinges[-1] = hinges[0] if oriented else -hinges[0]
    return hinges


def anchor_residuals(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Residuals for the anchored hinges used to kill rigid motion."""

    first = np.array([0.0, 0.0, 1.0])
    res = []
    res.extend((hinges[0] - first).tolist())
    res.append(hinges[1, 0])  # force x_2 = 0
    return np.asarray(res, dtype=float)


def unit_norm_residuals(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """‖h_i‖ - 1 for every hinge."""

    return np.sum(hinges**2, axis=1) - 1.0


def closure_residual(
    hinges: NDArray[np.float64],
    *,
    slide: float = 0.0,
) -> NDArray[np.float64]:
    """Sum of mid-axis vectors (plus optional slide term)."""

    tangents = binormals_to_tangents(hinges, normalize=False)
    ext = np.sum(tangents, axis=0)
    if slide != 0.0:
        ext = ext + slide * np.sum(hinges[:-1], axis=0)
    return ext


def alignment_residuals(hinges: NDArray[np.float64], oriented=True) -> float:
    """Norm of the first/last hinge alignment residual."""

    if oriented:
        residual = hinges[0] - hinges[-1]
    else:
        residual = hinges[0] + hinges[-1]
    return float(np.linalg.norm(residual))


def alignment_residuals_full(
    hinges: NDArray[np.float64], oriented: bool = True
) -> NDArray[np.float64]:
    """Vector form of the first/last hinge alignment residual.

    Returns the 3-component vector ``h[0] - h[-1]`` (oriented) or
    ``h[0] + h[-1]`` (non-oriented). Unlike :func:`alignment_residuals`,
    this version has full-rank (3) Jacobian at the manifold and is the
    correct constraint for Kutzbach-Grübler-style DoF analysis.
    """
    if oriented:
        return np.asarray(hinges[0] - hinges[-1], dtype=float)
    return np.asarray(hinges[0] + hinges[-1], dtype=float)


def constant_torsion_residuals(
    hinges: NDArray[np.float64], reference: float = None
) -> NDArray[np.float64]:
    """Enforce constant torsion angle: h_i · h_{i+1} = constant for all i.

    This is the InProd constraint from the Maple implementation.
    Returns residuals (h_i · h_{i+1}) - (h_1 · h_2) for i=1..N-1.
    """
    if len(hinges) < 2:
        return np.array([])

    # Compute dot products between consecutive hinges
    dot_products = np.sum(hinges[:-1] * hinges[1:], axis=1)

    # Residuals: each dot product should equal the reference
    if reference is None:
        return dot_products - dot_products[0]
    else:
        return dot_products - reference


def curvature_recursion_residuals(
    hinges: NDArray[np.float64], oriented: bool
) -> NDArray[np.float64]:
    """Enforce curvature recursion values to be constant: u[i] - u[0] = 0.

    u is the residual vector from geometry.curvature_recursion.
    """
    if len(hinges) < 3:
        return np.array([])

    tangents = binormals_to_tangents(hinges, normalize=True)
    curvatures = pairwise_curvature(hinges, tangents, oriented=oriented)
    u = curvature_recursion(curvatures, oriented=oriented)

    if len(u) == 0:
        return u

    return u[1:] - u[0]


def constraint_residuals(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
) -> Dict[str, NDArray[np.float64]]:
    """Return all constraint residual groups for the given hinge array."""

    # hinges = enforce_terminal(hinges, oriented=config.oriented)
    residuals: Dict[str, NDArray[np.float64]] = {}
    residuals["unit_norm"] = unit_norm_residuals(hinges[:-1])
    if config.closure:
        residuals["closure"] = closure_residual(hinges, slide=config.slide)
    if config.enforce_anchors:
        residuals["anchors"] = anchor_residuals(hinges)
    if config.constant_torsion:
        residuals["constant_torsion"] = constant_torsion_residuals(
            hinges, reference=config.reference_torsion
        )
    if config.alignment:
        if config.full_alignment:
            residuals["alignment"] = alignment_residuals_full(
                hinges, oriented=config.oriented
            )
        else:
            residuals["alignment"] = np.asarray(
                [alignment_residuals(hinges, oriented=config.oriented)], dtype=float
            )
    if config.curvature_recursion:
        residuals["curvature_recursion"] = curvature_recursion_residuals(
            hinges, oriented=config.oriented
        )
    return residuals


def constraint_penalty(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
) -> float:
    """Sum of squares penalty used by the Python solver."""

    residuals = constraint_residuals(hinges, config)
    return float(sum(float(np.sum(r**2)) for r in residuals.values()))
