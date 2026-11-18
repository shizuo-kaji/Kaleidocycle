"""Rebuild the certified example's shared sample and studio offline fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kaleidocycle.collision_certificate import certify_crossing
from kaleidocycle.collisions import load_crossing_data, sample_crossing
from kaleidocycle.integrable import reconstruct_framed_polygon
from kaleidocycle.samples import build_web_sample_fallback, export_sample

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Validate the exact contact before exporting the pre-contact sample."""
    certify_crossing()
    source = load_crossing_data()
    trajectory = sample_crossing(np.array([-0.002]))
    initial = reconstruct_framed_polygon(
        trajectory.curvatures[0], float(source["torsion_angle"]), sign=source["sign"]
    )
    export_sample(
        initial.binormals,
        "counterexample_k12_self_crossing",
        directory=ROOT / "data/kaleidocycles",
        metadata={
            "kind": "counterexample",
            "oriented": False,
            "label": "Self-crossing K12 · first mKdV flow",
            "time_from_contact": float(trajectory.times[0]),
            "contact_after_flow_time": 0.002,
            "description": (
                "Numerical pre-contact sample at t=-0.002; the certified local "
                "orbit crosses at t=0. See "
                "notebooks/SelfCrossing.ipynb."
            ),
            "source": "data/counterexamples/mkdv_self_crossing_k12.json",
        },
    )
    build_web_sample_fallback(
        ROOT / "web/sample-fallback.js", directory=ROOT / "data/kaleidocycles"
    )
    print("Certified contact; exported the pre-contact sample and studio fallback.")


if __name__ == "__main__":
    main()
