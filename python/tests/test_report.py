"""Tests for the report generation module."""

import numpy as np
import pytest

from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.geometry import random_hinges
from kaleidocycle.report import format_report


def test_format_report_smoke():
    """Smoke test for the format_report function."""
    # ARRANGE
    n = 6
    hinges = random_hinges(n, seed=42, oriented=True).as_array()
    config = ConstraintConfig(oriented=True)

    # ACT
    report = format_report(hinges, config)

    # ASSERT
    assert isinstance(report, str)
    assert "Kaleidocycle Property Report" in report
    assert f"Number of hinges (N): {n}" in report
    assert "Geometric Properties" in report
    assert "Topological Properties" in report
    assert "Constraint Violations" in report
    assert "Mean pairwise cosine" in report
    assert "Writhe" in report
    assert "Total Twist" in report
    assert "Total Penalty" in report
    assert "unit_norm" in report
    assert "closure" in report
    assert "anchors" in report
    assert "constant_torsion" in report


def test_format_report_failing_axis():
    """Test that the report handles a failing axis computation."""
    # ARRANGE
    # Create a set of hinges where the first 3 are coplanar (linearly dependent)
    # This should cause the np.linalg.solve in compute_axis to fail.
    hinges = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],  # linear combination of the first two
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    config = ConstraintConfig(oriented=False)

    # ACT
    report = format_report(hinges, config)

    # ASSERT
    assert "Computed axis: FAILED (singular system: cannot determine unique axis)" in report


def test_format_report_precision():
    """Test the precision argument of format_report."""
    # ARRANGE
    n = 8
    hinges = random_hinges(n, seed=123).as_array()
    config = ConstraintConfig()

    # ACT
    report_p2 = format_report(hinges, config, precision=2)
    report_p8 = format_report(hinges, config, precision=8)

    # ASSERT
    # Find a floating point value and check its precision
    for line in report_p2.splitlines():
        if "Mean pairwise cosine" in line:
            val_str = line.split(":")[1].strip()
            assert len(val_str.split(".")[1]) == 2
            break

    for line in report_p8.splitlines():
        if "Mean pairwise cosine" in line:
            val_str = line.split(":")[1].strip()
            assert len(val_str.split(".")[1]) == 8
            break
