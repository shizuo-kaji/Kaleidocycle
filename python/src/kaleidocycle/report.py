"""Report generation for Kaleidocycle properties."""

from __future__ import annotations

import numpy as np

from .constraints import ConstraintConfig, constraint_residuals
from .energies import bending_energy, dipole_energy, torsion_energy
from .geometry import (
    binormals_to_tangents,
    compute_axis,
    compute_torsion,
    pairwise_cosines,
    pairwise_curvature,
    tangents_to_curve,
    total_twist_from_curve,
    writhe,
)


def format_report(
    hinges: np.ndarray | None = None,
    config: ConstraintConfig | None = None,
    *,
    kaleidocycle: 'Kaleidocycle | None' = None,
    precision: int = 6,
) -> str:
    """Generate a human-readable report of kaleidocycle properties.

    Can be called with either (hinges, config) or with a Kaleidocycle instance.

    Args:
        hinges: Hinge vectors array of shape (n+1, 3). Not needed if kaleidocycle is provided.
        config: Constraint configuration. Not needed if kaleidocycle is provided.
        kaleidocycle: Kaleidocycle instance with computed metadata.
        precision: Number of decimal places for float formatting.

    Returns:
        A formatted string containing the report.

    Example:
        >>> # Method 1: Direct from hinges
        >>> report = format_report(hinges, config)
        >>> # Method 2: From Kaleidocycle instance
        >>> kc = Kaleidocycle(hinges=hinges)
        >>> kc.compute(config=config)
        >>> report = format_report(kaleidocycle=kc)
    """
    # Import here to avoid circular dependency
    from .geometry import Kaleidocycle

    # Handle Kaleidocycle instance
    if kaleidocycle is not None:
        if hinges is not None:
            raise ValueError("Cannot specify both kaleidocycle and hinges")

        # Use metadata if available, otherwise compute on the fly
        if not kaleidocycle.metadata:
            if config is None:
                # Compute without constraints
                kaleidocycle.compute(['geometric', 'topological', 'energies'])
            else:
                kaleidocycle.compute(config=config)

        # Use the Kaleidocycle's data
        hinges = kaleidocycle.hinges
        # If config was provided, use it; otherwise try to get from metadata
        if config is None:
            config = kaleidocycle.metadata.get('constraints', {}).get('config')
        if config is None:
            # Create a default config based on orientation
            config = ConstraintConfig(oriented=kaleidocycle.oriented)

    # Validate inputs
    if hinges is None:
        raise ValueError("Must provide either hinges or kaleidocycle")
    if config is None:
        raise ValueError("Must provide either config or kaleidocycle with computed constraints")
    report_lines = ["Kaleidocycle Property Report"]
    report_lines.append("=" * len(report_lines[0]))

    n = len(hinges) - 1
    report_lines.append(f"Number of hinges (N): {n}")
    report_lines.append(f"Oriented: {config.oriented}")
    if config.slide != 0.0:
        report_lines.append(f"Slide factor: {config.slide}")
    residuals = constraint_residuals(hinges, config)
    report_lines.append(f"Penalty: {float(sum(float(np.sum(r**2)) for r in residuals.values())):.{precision}e}")

    # Format hinge vectors element-wise to avoid trying to use numeric format on numpy.ndarray
    first_str = ", ".join(f"{float(x):.{precision}f}" for x in np.asarray(hinges[0]))
    last_str = ", ".join(f"{float(x):.{precision}f}" for x in np.asarray(hinges[-1]))
    report_lines.append(f"First and last hinges: ({first_str}), ({last_str})")

    # --- Geometric Properties ---
    report_lines.append("\n--- Geometric Properties ---")
    tangents = binormals_to_tangents(hinges, normalize=True)
    curve = tangents_to_curve(tangents)
    cosines = pairwise_cosines(hinges)
    curvatures = pairwise_curvature(hinges, tangents)
    torsions = compute_torsion(hinges)

    report_lines.append(f"Mean pairwise cosine: {np.mean(cosines):.{precision}f}")
    report_lines.append(f"Std dev of cosines: {np.std(cosines):.{precision}e}")
    report_lines.append(f"Mean curvature: {np.mean(curvatures):.{precision}f}")
    report_lines.append(f"Mean torsion: {np.mean(torsions):.{precision}f}")

    try:
        axis = compute_axis(hinges, curvatures)
        axis_str = np.array2string(axis, precision=precision)
        report_lines.append(f"Computed axis: {axis_str}")
    except ValueError as e:
        report_lines.append(f"Computed axis: FAILED ({e})")

    # --- Topological Properties ---
    report_lines.append("\n--- Topological Properties ---")
    try:
        writhe_val = writhe(curve)
        report_lines.append(f"Writhe: {writhe_val:.{precision}f}")
    except ValueError as e:
        report_lines.append(f"Writhe: FAILED ({e})")

    try:
        twist_val = total_twist_from_curve(curve)
        report_lines.append(f"Total Twist (Tw/π): {twist_val:.{precision}f}")
    except ValueError as e:
        report_lines.append(f"Total Twist: FAILED ({e})")

    report_lines.append(f"Linking Number (Lk = Tw + Wr): {(writhe_val + twist_val):.{precision}f}")

    # --- Constraint Violations ---
    report_lines.append("\n--- Constraint Violations ---")
    report_lines.append(f"Configuration: {config}")
    total_penalty = 0.0
    for name, res_array in residuals.items():
        if res_array.size > 0:
            max_violation = np.max(np.abs(res_array))
            sum_sq = np.sum(res_array**2)
            total_penalty += sum_sq
            report_lines.append(
                f"  - {name}: "
                f"max_abs={max_violation:.{precision}e}, "
                f"sum_sq={sum_sq:.{precision}e}"
            )
    report_lines.append(f"Total Penalty (sum_sq): {total_penalty:.{precision}e}")

    # --- Energy ---
    report_lines.append("\n--- Energy ---")
    report_lines.append(f"Bending Energy: {bending_energy(tangents):.{precision}e}")
    report_lines.append(f"Dipole Energy: {dipole_energy(hinges, curve):.{precision}e}")
    report_lines.append(f"Torsion Energy: {torsion_energy(hinges):.{precision}e}")

    return "\n".join(report_lines)
