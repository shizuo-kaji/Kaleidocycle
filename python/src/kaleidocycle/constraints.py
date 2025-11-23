"""Constraint helpers mirroring the Mathematica/Maple Setup function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .geometry import binormals_to_tangents


@dataclass(slots=True)
class ConstraintConfig:
    """Configuration flags"""

    slide: float = 0.0 # slide term for closure constraint
    oriented: bool = False
    enforce_anchors: bool = False # fix rigid motion by anchoring first two hinges
    constant_torsion: bool = True
    alignment: bool = True # first and last hinge alignment
    closure: bool = True # curve closure constraint
    reference_torsion: float | None = None # reference value for constant torsion constraint


def enforce_terminal(hinges: NDArray[np.float64], oriented: bool) -> NDArray[np.float64]:
    """Ensure the final hinge repeats the first hinge."""

    if len(hinges) == 0:
        return hinges
    hinges = np.array(hinges, dtype=float, copy=True)
    hinges[-1] = hinges[0] if oriented else -hinges[0]
    return hinges


def anchor_residuals(
    hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Residuals for the anchored hinges used to kill rigid motion."""

    first = np.array([0.0, 0.0, 1.0])
    res = []
    res.extend((hinges[0] - first).tolist())
    res.append(hinges[1, 0])  # force x_2 = 0
    return np.asarray(res, dtype=float)


def unit_norm_residuals(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """‖h_i‖ - 1 for every hinge."""

    return np.linalg.norm(hinges, axis=1) - 1.0


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

def alignment_residuals(hinges: NDArray[np.float64], oriented=True) -> NDArray[np.float64]:
        """Constraint: First and last hinge should match."""
        if oriented:
            return np.linalg.norm(hinges[0] - hinges[-1])
        else:
            return np.linalg.norm(hinges[0] + hinges[-1])


def constant_torsion_residuals(hinges: NDArray[np.float64], reference: float = None) -> NDArray[np.float64]:
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


def constraint_residuals(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
) -> Dict[str, NDArray[np.float64]]:
    """Return all constraint residual groups for the given hinge array."""

    #hinges = enforce_terminal(hinges, oriented=config.oriented)
    residuals: Dict[str, NDArray[np.float64]] = {}
    residuals["unit_norm"] = unit_norm_residuals(hinges[:-1])
    residuals["closure"] = closure_residual(hinges, slide=config.slide)
    if config.enforce_anchors:
        residuals["anchors"] = anchor_residuals(hinges)
    if config.constant_torsion:
        residuals["constant_torsion"] = constant_torsion_residuals(hinges, reference=config.reference_torsion)
    if config.alignment:
        residuals["alignment"] = alignment_residuals(hinges, oriented=config.oriented)
    return residuals


def constraint_penalty(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
) -> float:
    """Sum of squares penalty used by the Python solver."""

    residuals = constraint_residuals(hinges, config)
    return float(sum(float(np.sum(r**2)) for r in residuals.values()))
