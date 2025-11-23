"""Tests for curve generation functions."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle.geometry import binormals_to_curve, random_hinges
from kaleidocycle.animation import generate_animation, generate_curve_animation


def test_binormals_to_curve_basic() -> None:
    """Test basic curve generation from binormals."""
    binormals = random_hinges(6, seed=42).as_array()

    curve = binormals_to_curve(binormals)

    # Should have n+1 points (same as binormals)
    assert curve.shape == (7, 3)

    # Should be 3D points
    assert curve.shape[1] == 3


def test_binormals_to_curve_centered() -> None:
    """Test that centered curve has zero mean."""
    binormals = random_hinges(6, seed=42).as_array()

    curve = binormals_to_curve(binormals, center=True)

    # Centroid should be at origin
    centroid = np.mean(curve, axis=0)
    np.testing.assert_allclose(centroid, 0.0, atol=1e-10)


def test_binormals_to_curve_not_centered() -> None:
    """Test that non-centered curve starts at origin."""
    binormals = random_hinges(6, seed=42).as_array()

    curve = binormals_to_curve(binormals, center=False)

    # First point should be at origin
    np.testing.assert_allclose(curve[0], 0.0, atol=1e-10)


def test_binormals_to_curve_scale() -> None:
    """Test curve scaling."""
    binormals = random_hinges(6, seed=42).as_array()

    curve1 = binormals_to_curve(binormals, scale=1.0, center=False)
    curve2 = binormals_to_curve(binormals, scale=2.0, center=False)

    # Curve2 should be twice the size
    np.testing.assert_allclose(curve2, curve1 * 2.0, atol=1e-10)


def test_generate_curve_animation_basic() -> None:
    """Test basic curve animation generation."""
    binormals = random_hinges(6, seed=42).as_array()

    # Generate binormal animation
    frames = generate_animation(binormals, num_frames=10, step_size=0.02)

    # Generate curve animation
    curves = generate_curve_animation(frames)

    # Should have same number of frames
    assert len(curves) == len(frames)

    # Each curve should have correct shape
    assert all(c.shape == (7, 3) for c in curves)


def test_generate_curve_animation_centered() -> None:
    """Test that curve animation can be centered."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation(binormals, num_frames=5, step_size=0.02)
    curves = generate_curve_animation(frames, center=True)

    # Each curve should be centered
    for curve in curves:
        centroid = np.mean(curve, axis=0)
        np.testing.assert_allclose(centroid, 0.0, atol=1e-10)


def test_generate_curve_animation_scale() -> None:
    """Test curve animation scaling."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation(binormals, num_frames=5, step_size=0.02)

    curves1 = generate_curve_animation(frames, scale=1.0, center=False)
    curves2 = generate_curve_animation(frames, scale=2.0, center=False)

    # Curves should scale proportionally
    for c1, c2 in zip(curves1, curves2):
        np.testing.assert_allclose(c2, c1 * 2.0, atol=1e-10)


def test_generate_curve_animation_empty() -> None:
    """Test curve animation with empty input."""
    curves = generate_curve_animation([])

    assert curves == []


def test_curve_animation_smooth() -> None:
    """Test that curve animation is smooth (gradual changes)."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation(binormals, num_frames=20, step_size=0.01)
    curves = generate_curve_animation(frames)

    # Compute curve-to-curve distances
    distances = []
    for i in range(1, len(curves)):
        dist = np.linalg.norm(curves[i] - curves[i-1])
        distances.append(dist)

    # All transitions should be reasonably smooth
    assert all(d < 1.0 for d in distances)

    # No huge jumps
    assert max(distances) < 0.5


def test_curve_from_different_sizes() -> None:
    """Test curve generation for different kaleidocycle sizes."""
    for n in [4, 6, 8, 10]:
        binormals = random_hinges(n, seed=42).as_array()
        curve = binormals_to_curve(binormals)

        # Should have n+1 points
        assert curve.shape == (n + 1, 3)
