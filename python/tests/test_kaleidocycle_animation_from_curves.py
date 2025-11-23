"""Tests for KaleidocycleAnimation.from_curves functionality."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import (
    Kaleidocycle,
    KaleidocycleAnimation,
    generate_animation,
)
from kaleidocycle.geometry import binormals_to_tangents, tangents_to_curve


def test_from_curves_basic():
    """Test basic from_curves functionality."""
    kc = Kaleidocycle(6, seed=42, oriented=True)
    frames = generate_animation(
        kc.hinges,
        num_frames=3,
        step_size=0.02,
        rule="sine-Gordon",
        oriented=kc.oriented,
    )

    # Generate curves from frames
    curves = []
    for frame in frames:
        tangents = binormals_to_tangents(frame, normalize=False)
        curve = tangents_to_curve(tangents, center=True)
        curves.append(curve)

    # Create animation from curves
    anim = KaleidocycleAnimation.from_curves(curves)

    assert anim.n_frames == 3
    assert anim.n_vertices == 7
    assert anim.frame_shape == (7, 3)
    assert anim.evolution_rule == "unknown"


def test_from_curves_with_evolution_rule():
    """Test from_curves with specified evolution rule."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=3, rule="sine-Gordon")

    curves = []
    for frame in frames:
        tangents = binormals_to_tangents(frame, normalize=False)
        curve = tangents_to_curve(tangents, center=True)
        curves.append(curve)

    anim = KaleidocycleAnimation.from_curves(
        curves,
        evolution_rule="sine-Gordon",
    )

    assert anim.evolution_rule == "sine-Gordon"


def test_from_curves_with_metadata():
    """Test from_curves with metadata."""
    kc = Kaleidocycle(6, seed=42)
    curve = kc.curve

    anim = KaleidocycleAnimation.from_curves(
        [curve, curve],
        metadata={"source": "test", "version": 1},
    )

    assert anim.metadata == {"source": "test", "version": 1}
    assert anim.n_frames == 2


def test_from_curves_maintains_unit_length():
    """Test that binormals from curves maintain unit length."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=3, rule="sine-Gordon")

    curves = []
    for frame in frames:
        tangents = binormals_to_tangents(frame, normalize=False)
        curve = tangents_to_curve(tangents, center=True)
        curves.append(curve)

    anim = KaleidocycleAnimation.from_curves(curves)

    # Check all binormals are unit length
    for frame in anim.frames:
        norms = np.linalg.norm(frame, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


def test_from_curves_empty_list():
    """Test that from_curves raises error on empty list."""
    with pytest.raises(ValueError, match="curves list cannot be empty"):
        KaleidocycleAnimation.from_curves([])


def test_from_curves_invalid_shape():
    """Test that from_curves raises error on invalid curve shape."""
    invalid_curve = np.array([[1, 2], [3, 4]])  # Shape (2, 2) instead of (n, 3)

    with pytest.raises(ValueError, match="must have shape"):
        KaleidocycleAnimation.from_curves([invalid_curve])


def test_from_curves_with_reference():
    """Test from_curves with custom reference binormal."""
    kc = Kaleidocycle(6, seed=42)
    curve = kc.curve

    reference = np.array([1.0, 0.0, 0.0])
    anim = KaleidocycleAnimation.from_curves(
        [curve, curve],
        reference=reference,
    )

    assert anim.n_frames == 2
    # First binormal should be influenced by reference
    assert anim.frames[0].shape == (7, 3)


def test_from_curves_compute_properties():
    """Test that properties can be computed on animation from curves."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=3, rule="sine-Gordon")

    curves = []
    for frame in frames:
        tangents = binormals_to_tangents(frame, normalize=False)
        curve = tangents_to_curve(tangents, center=True)
        curves.append(curve)

    anim = KaleidocycleAnimation.from_curves(curves)

    # Should be able to compute properties
    curvatures = anim.compute_vertex_property("curvature")
    assert curvatures.shape[0] == 3  # n_frames
    assert curvatures.shape[1] == 6  # n_vertices - 1


def test_from_curves_with_precomputed_properties():
    """Test from_curves with pre-computed properties."""
    kc = Kaleidocycle(6, seed=42)
    curve = kc.curve

    # Create dummy properties
    vertex_props = {"test_vertex": np.random.rand(2, 7)}
    scalar_props = {"test_scalar": np.array([1.0, 2.0])}

    anim = KaleidocycleAnimation.from_curves(
        [curve, curve],
        vertex_properties=vertex_props,
        scalar_properties=scalar_props,
    )

    assert "test_vertex" in anim.vertex_properties
    assert "test_scalar" in anim.scalar_properties
    assert anim.vertex_properties["test_vertex"].shape == (2, 7)
    assert len(anim.scalar_properties["test_scalar"]) == 2


def test_from_curves_multiple_frames():
    """Test from_curves with multiple frames."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=10, rule="sine-Gordon")

    curves = []
    for frame in frames:
        tangents = binormals_to_tangents(frame, normalize=False)
        curve = tangents_to_curve(tangents, center=True)
        curves.append(curve)

    anim = KaleidocycleAnimation.from_curves(curves)

    assert anim.n_frames == 10
    assert len(anim.frames) == 10
    assert all(f.shape == (7, 3) for f in anim.frames)
