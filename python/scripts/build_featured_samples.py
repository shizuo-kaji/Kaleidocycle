"""Build the featured critical and explicit samples shown in Kaleidocycle Studio.

Every configuration here is exact: a closed Jacobi elliptic (cn) family at a
closing modulus, a right-angle critical circle, or the explicit torus-knot
ring of the quadratic-linking proposition.  The script validates closure,
constant torsion and the stated invariants before exporting, annotates the
existing Möbius samples (which are mirror-principal rings), and rebuilds the
studio's ``file://`` fallback.

Run from the ``python/`` directory::

    python scripts/build_featured_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kaleidocycle import compute_linking_number
from kaleidocycle.elliptic import (
    EllipticFamily,
    closed_elliptic_family,
    continuum_constants,
    principal_family,
    right_angle_family,
    torus_knot_binormals,
)
from kaleidocycle.integrable import (
    first_hamiltonian,
    framed_polygon_from_binormals,
)
from kaleidocycle.samples import build_web_sample_fallback, export_sample

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = ROOT / "data" / "kaleidocycles"
WEB_FALLBACK = ROOT / "web" / "sample-fallback.js"
PAPER = "Integrable deformations of Kaleidocycles"


def _checked(binormals: np.ndarray, expected_self_linking: float) -> dict[str, Any]:
    """Validate an exact ring and return its measured invariants."""

    hinges = np.asarray(binormals, dtype=float)
    dots = np.einsum("ij,ij->i", hinges[:-1], hinges[1:])
    polygon = framed_polygon_from_binormals(hinges)
    closure = float(np.linalg.norm(polygon.vertices[-1] - polygon.vertices[0]))
    if np.ptp(dots) > 1e-10 or closure > 1e-10:
        raise RuntimeError(f"not an exact ring: ptp={np.ptp(dots):.2e}, {closure=}")
    # compute_linking_number returns 2 SL in the paper's normalisation.
    self_linking = compute_linking_number(hinges) / 2.0
    if abs(self_linking - expected_self_linking) > 1e-6:
        raise RuntimeError(f"SL={self_linking}, expected {expected_self_linking}")
    sign = 1 if np.allclose(hinges[-1], hinges[0]) else -1
    return {
        "oriented": sign == 1,
        "sigma": sign,
        "torsion_angle_over_pi": float(np.arccos(np.mean(dots)) / np.pi),
        "self_linking": float(round(2.0 * self_linking) / 2.0),
        "bending_energy": float(first_hamiltonian(polygon.curvatures)),
    }


def _family_parameters(family: EllipticFamily, level: int, phase: float) -> dict:
    return {
        "family": "cn",
        "N": family.n_hinges,
        "m": family.winding,
        "j": level,
        "k": family.modulus,
        "phase": phase,
        "phase_period": family.phase_period,
    }


def _export_family(
    name: str,
    family: EllipticFamily,
    *,
    level: int,
    phase: float,
    self_linking: float,
    label: str,
    description: str,
    extra: dict[str, Any] | None = None,
) -> None:
    binormals = family.binormals(phase)
    invariants = _checked(binormals, self_linking)
    n, m = family.n_hinges, family.winding
    metadata = {
        "kind": "critical",
        "label": label,
        "description": description,
        "source": f"scripts/build_featured_samples.py; {PAPER}",
        "family_parameters": _family_parameters(family, level, phase),
        "hessian_inertia": [n - m - 3, m - 3, 1] if n >= 7 else None,
        **invariants,
        **(extra or {}),
    }
    export_sample(binormals, name, directory=DATA_DIRECTORY, metadata=metadata)


def build_critical_samples() -> None:
    """Export the elliptic and right-angle critical circles."""

    six = right_angle_family(6, 1)
    _export_family(
        "classic_k6_right_angle",
        six,
        level=1,
        phase=0.0,
        self_linking=1.5,
        label="Classical K6 · right-angle circle",
        description=(
            "The hexagonal toy Kaleidocycle: six hinges, torsion exactly π/2, "
            "anti-oriented. It is the right-angle critical circle with m = 3, "
            "j = 1 and k = sin(π/3), so κₙ₊₂ = −κₙ. Its single "
            "turning motion is the first mKdV orbit; SL = 3/2 and "
            "E₁ = 3 log 4. With ribbon half-width 1/√2 the links are regular "
            "tetrahedra."
        ),
    )

    seven = principal_family(7)
    _export_family(
        "principal_k7_minimum",
        seven,
        level=1,
        phase=0.3,
        self_linking=1.5,
        label="Principal K7 · minimum torsion",
        description=(
            "The global minimum μ₇ = 0.40455π of the torsion angle over "
            "anti-oriented seven-hinge rings: the principal cn family "
            "(m = 3, j = 1), a Morse-Bott minimum with Hessian inertia "
            "(1, 0, 1). Its phase circle is a first mKdV orbit and it is "
            "also the global minimum of the bending energy E₁."
        ),
    )

    large = principal_family(48)
    k_limit, c_limit = continuum_constants()
    _export_family(
        "principal_k48_continuum",
        large,
        level=1,
        phase=0.3,
        self_linking=1.5,
        label="Principal K48 · continuum limit",
        description=(
            "The principal family with 48 hinges. As N grows, the modulus "
            f"tends to k∞ = {k_limit:.10f} and Nμ_N to "
            f"C∞ = {c_limit:.10f}, both with O(N⁻²) error; the centre "
            "line approaches a smooth critical rod."
        ),
        extra={
            "continuum_modulus_gap": float(large.modulus - k_limit),
            "continuum_constant_gap": float(48 * large.torsion_angle - c_limit),
        },
    )

    saddle = closed_elliptic_family(11, 4, 1)
    _export_family(
        "elliptic_k11_m4_saddle",
        saddle,
        level=1,
        phase=0.3,
        self_linking=2.0,
        label="Elliptic K11 · oriented saddle (m = 4)",
        description=(
            "An oriented cn critical circle with winding m = 4 and closing "
            "index j = 1. Its torsion Hessian has inertia (4, 1, 1): a "
            "saddle of index one. The once-round branch is an unknot "
            "with SL = m/2."
        ),
    )

    cinquefoil = closed_elliptic_family(11, 5, 2)
    _export_family(
        "elliptic_k11_m5_cinquefoil",
        cinquefoil,
        level=2,
        phase=0.3,
        self_linking=7.5,
        label="Elliptic K11 · (2,5) torus knot",
        description=(
            "The twice-around closing branch (j = 2) of the m = 5 cn family "
            "with eleven hinges. Numerically its centre line is the (2,5) "
            "torus knot (Alexander polynomial t⁴ − t³ + t² − t + 1), with "
            "SL = 15/2. Torsion Hessian inertia (3, 2, 1)."
        ),
        extra={"knot": "T(2,5)"},
    )

    torus37 = closed_elliptic_family(17, 7, 3)
    _export_family(
        "elliptic_k17_m7_torus_knot",
        torus37,
        level=3,
        phase=0.3,
        self_linking=17.5,
        label="Elliptic K17 · (3,7) torus knot",
        description=(
            "The j = 3 closing branch of the m = 7 cn family with 17 hinges. "
            "Numerically its centre line is the (3,7) torus knot, with "
            "SL = 35/2. Torsion Hessian inertia (7, 4, 1)."
        ),
        extra={"knot": "T(3,7)"},
    )

    eight = right_angle_family(8, 1)
    _export_family(
        "right_angle_k8_oriented",
        eight,
        level=1,
        phase=0.5 * eight.quarter_period,
        self_linking=2.0,
        label="Right-angle K8 · oriented energy minimum",
        description=(
            "The right-angle circle with eight hinges (m = 4, j = 1). It is "
            "the entire global minimum set of the bending energy, "
            "E₁ = 4 log 2, over oriented eight-hinge rings, yet its "
            "fixed-torsion fibre has local dimension two."
        ),
    )

    ten = right_angle_family(10, 1)
    _export_family(
        "right_angle_k10_energy",
        ten,
        level=1,
        phase=0.5 * ten.quarter_period,
        self_linking=2.5,
        label="Right-angle K10 · energy-minimising circle",
        description=(
            "Anti-oriented, ten hinges, torsion π/2 (m = 5, j = 1). A "
            "Morse-Bott local minimum of the bending energy with inertia "
            "(4, 0, 1), whose full motion fibre has dimension four: energy "
            "minimisation does not select one degree of freedom."
        ),
    )


def build_torus_knot_sample() -> None:
    """Export the explicit oriented K15 with SL = 10."""

    binormals = torus_knot_binormals(5, 2)
    invariants = _checked(binormals, 10.0)
    export_sample(
        binormals,
        "torus_knot_k15_sl10",
        directory=DATA_DIRECTORY,
        metadata={
            "kind": "knot",
            "label": "Torus-knot K15 · SL = 10",
            "description": (
                "Explicit oriented ring with fivefold rotational symmetry and "
                "all curvatures positive (cells = 5, turns = 2). Its centre "
                "line is the (2,5) torus knot and SL = 10: linking can grow "
                "quadratically in N. The mirror is anti-oriented with "
                "SL = −5/2."
            ),
            "source": f"scripts/build_featured_samples.py; {PAPER}",
            "family_parameters": {"family": "torus_knot", "cells": 5, "turns": 2},
            "knot": "T(2,5)",
            **invariants,
        },
    )


MIRROR_PRINCIPAL = {
    "mobius_k7_oriented": "seven",
    "mobius_k8_nonoriented": "eight",
    "mobius_k9_oriented": "nine",
}


def annotate_mirror_principal_samples() -> None:
    """Label the optimised Möbius samples as mirror-principal rings."""

    for name, word in MIRROR_PRINCIPAL.items():
        path = DATA_DIRECTORY / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        n = int(payload["n"])
        hinges = np.asarray(payload["hinges"], dtype=float)
        torsion = float(
            np.arccos(np.mean(np.einsum("ij,ij->i", hinges[:-1], hinges[1:])))
        )
        expected = np.pi - principal_family(n).torsion_angle
        if abs(torsion - expected) > 1e-8:
            raise RuntimeError(f"{name} is not the mirror principal ring")
        oriented = (
            "oriented" if payload["metadata"].get("oriented") else "anti-oriented"
        )
        payload["metadata"].update(
            {
                "label": f"Möbius K{n} · maximal torsion ({oriented})",
                "description": (
                    f"The optimised {word}-hinge Möbius Kaleidocycle. It is the "
                    "mirror of the principal family (B[n] ↦ (−1)ⁿB[n]), so "
                    f"its torsion π − μ_{n} = {torsion / np.pi:.5f}π is the "
                    "maximum for its monodromy and its Hessian inertia is "
                    f"(0, {n - 6}, 1)."
                ),
            }
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    build_critical_samples()
    build_torus_knot_sample()
    annotate_mirror_principal_samples()
    build_web_sample_fallback(WEB_FALLBACK, directory=DATA_DIRECTORY)
    print("Exported the featured samples and rebuilt web/sample-fallback.js.")


if __name__ == "__main__":
    main()
