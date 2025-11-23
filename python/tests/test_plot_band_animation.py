"""Tests for plot_band animation functionality."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import (
    Kaleidocycle,
    KaleidocycleAnimation,
    generate_animation,
    plot_band,
    random_hinges,
    binormals_to_tangents,
    tangents_to_curve,
)


def test_plot_band_static():
    """Test static plot_band (original behavior)."""
    hinges = random_hinges(6, seed=42).as_array()
    tangents = binormals_to_tangents(hinges, normalize=False)
    curve = tangents_to_curve(tangents)

    ax = plot_band(curve, hinges, title="Static Test")

    assert ax is not None
    assert ax.get_xlabel() == "X"


def test_plot_band_from_kaleidocycle_animation():
    """Test plot_band with KaleidocycleAnimation."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine_gordon")

    fig, ani = plot_band(animation=anim, interval=50)

    assert fig is not None
    assert ani is not None


def test_plot_band_from_list_of_binormals():
    """Test plot_band with list of binormals."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)

    fig, ani = plot_band(hinges=frames, interval=50)

    assert fig is not None
    assert ani is not None


def test_plot_band_animation_with_scalar_properties():
    """Test plot_band animation with scalar properties displayed."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine_gordon")

    # Compute some scalar properties
    anim.compute_scalar_property("penalty")

    fig, ani = plot_band(
        animation=anim,
        scalar_properties=["penalty"],
        title="Test Animation",
        interval=50,
    )

    assert fig is not None
    assert ani is not None


def test_plot_band_animation_with_multiple_properties():
    """Test plot_band animation with multiple scalar properties."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine_gordon")

    # Compute multiple scalar properties
    anim.compute_scalar_property("penalty")
    anim.compute_scalar_property("linking_number")

    fig, ani = plot_band(
        animation=anim,
        scalar_properties=["penalty", "linking_number"],
        title="Multi-property",
        interval=50,
    )

    assert fig is not None
    assert ani is not None


def test_plot_band_animation_show_curve():
    """Test plot_band animation with curve overlay."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)

    fig, ani = plot_band(hinges=frames, show_curve=True, interval=50)

    assert fig is not None
    assert ani is not None


def test_plot_band_animation_custom_styling():
    """Test plot_band animation with custom styling."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)

    fig, ani = plot_band(
        hinges=frames,
        width=0.2,
        facecolor="lightgreen",
        edgecolor="darkgreen",
        alpha=0.8,
        linewidth=1.0,
        interval=50,
        figsize=(12, 10),
    )

    assert fig is not None
    assert ani is not None


def test_plot_band_static_requires_both_curve_and_hinges():
    """Test that static plot requires both curve and hinges."""
    hinges = random_hinges(6, seed=42).as_array()

    with pytest.raises(ValueError, match="both curve and hinges must be provided"):
        plot_band(hinges=hinges)

    with pytest.raises(ValueError, match="both curve and hinges must be provided"):
        tangents = binormals_to_tangents(hinges, normalize=False)
        curve = tangents_to_curve(tangents)
        plot_band(curve=curve)


def test_plot_band_animation_with_untwist():
    """Test plot_band animation with untwist option."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)

    fig, ani = plot_band(hinges=frames, untwist=True, interval=50)

    assert fig is not None
    assert ani is not None


def test_plot_band_from_list_of_curves():
    """Test plot_band with list of curves."""
    kc = Kaleidocycle(6, seed=42)
    frames = generate_animation(kc.hinges, num_frames=5, step_size=0.01)

    # Compute curves from hinges
    curves = []
    for frame in frames:
        tangents = binormals_to_tangents(frame, normalize=False)
        curve = tangents_to_curve(tangents)
        curves.append(curve)

    fig, ani = plot_band(curve=curves, interval=50)

    assert fig is not None
    assert ani is not None
