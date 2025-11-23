"""Tests for the visualization module."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import (
    ConstraintConfig,
    Kaleidocycle,
    SolverOptions,
    tangents_to_curve,
    binormals_to_tangents,
    compute_tetrahedron_vertices,
    create_rotation_animation,
    optimize_cycle,
    paper_model,
    pairwise_cosines,
    pairwise_curvature,
    compute_torsion,
    plot_band,
    plot_curve,
    plot_energy_comparison,
    plot_hinges,
    plot_tetrahedron,
    plot_vertex_values,
    random_hinges,
)


def _create_valid_kaleidocycle(n: int, seed: int = 42):
    """Create a valid kaleidocycle with constant torsion for testing."""
    initial = random_hinges(n, seed=seed)
    config = ConstraintConfig(enforce_anchors=False)
    opts = SolverOptions(method="SLSQP", maxiter=1000, penalty_weight=100.0)
    result = optimize_cycle(initial.as_array(), config, objective="mean_cos", options=opts)
    return result.hinges


def _hinge_frame_to_legacy_sample(n: int, seed: int = 42):
    """Create a valid kaleidocycle and convert to Kaleidocycle for testing."""
    hinges = _create_valid_kaleidocycle(n, seed)
    return Kaleidocycle(hinges=hinges)


def test_plot_curve_basic():
    """Test basic curve plotting."""
    # Create simple test curve
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])

    # Should not raise an error
    ax = plot_curve(curve, title="Test")
    assert ax is not None


def test_plot_curve_invalid_shape():
    """Test that invalid curve shape raises error."""
    invalid = np.array([[0, 0], [1, 1]])  # Only 2D

    with pytest.raises(ValueError, match="expected.*3.*curve"):
        plot_curve(invalid)


def test_plot_hinges_basic():
    """Test hinge vector plotting."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    hinges = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    ax = plot_hinges(curve, hinges, title="Test Hinges")
    assert ax is not None


def test_kaleidocycle_plot_method():
    """Test plotting a valid kaleidocycle using the plot method."""
    hinges = _create_valid_kaleidocycle(6, seed=42)
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot()
    assert ax is not None


def test_plot_energy_comparison():
    """Test energy comparison plotting."""
    samples = {
        "sample1": _hinge_frame_to_legacy_sample(6, seed=42),
        "sample2": _hinge_frame_to_legacy_sample(8, seed=43),
    }

    fig = plot_energy_comparison(samples)
    assert fig is not None


def test_create_rotation_animation():
    """Test animation creation."""
    sample = _hinge_frame_to_legacy_sample(6, seed=42)

    fig, anim = create_rotation_animation(sample.curve, frames=10)
    assert fig is not None
    assert anim is not None


def test_compute_tetrahedron_vertices():
    """Test tetrahedral vertex computation."""
    # Simple test case
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    hinges = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    vertices, faces = compute_tetrahedron_vertices(curve, hinges, width=0.1)

    # Should have 2*n vertices
    assert vertices.shape == (6, 3)
    # Should have 4 faces per segment, 2 segments
    assert len(faces) == 8


def test_compute_tetrahedron_vertices_with_legacy():
    """Test vertex computation with valid kaleidocycle."""
    sample = _hinge_frame_to_legacy_sample(6, seed=42)

    vertices, faces = compute_tetrahedron_vertices(
        sample.curve, sample.hinges, width=0.15
    )

    n = sample.curve.shape[0]
    assert vertices.shape[0] == 2 * n
    assert vertices.shape[1] == 3
    assert len(faces) == 4 * (n - 1)


def test_plot_tetrahedron_basic():
    """Test tetrahedral structure plotting."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    hinges = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    ax = plot_tetrahedron(curve, hinges, width=0.1, title="Test Tetrahedron")
    assert ax is not None


def test_plot_tetrahedron_with_legacy():
    """Test tetrahedral plotting with valid kaleidocycle."""
    sample = _hinge_frame_to_legacy_sample(6, seed=42)

    ax = plot_tetrahedron(
        sample.curve,
        sample.hinges,
        width=0.15,
        facecolor="lightblue",
        edgecolor="navy",
        alpha=0.7,
        show_curve=True,
    )
    assert ax is not None


def test_plot_tetrahedron_invalid_shape():
    """Test that invalid curve shape raises error for tetrahedron."""
    invalid_curve = np.array([[0, 0], [1, 1]])
    hinges = np.array([[0, 0, 1], [0, 0, 1]])

    with pytest.raises(ValueError, match="expected.*3.*curve"):
        plot_tetrahedron(invalid_curve, hinges)


def test_plot_band_basic():
    """Test band structure plotting with quadrilaterals."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    hinges = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    ax = plot_band(curve, hinges, width=0.1, title="Test Band")
    assert ax is not None


def test_plot_band_with_legacy():
    """Test band plotting with valid kaleidocycle."""
    sample = _hinge_frame_to_legacy_sample(6, seed=42)

    ax = plot_band(
        sample.curve,
        sample.hinges,
        width=0.15,
        facecolor="lightcoral",
        edgecolor="darkred",
        alpha=0.7,
        show_curve=True,
    )
    assert ax is not None


def test_plot_band_untwist():
    """Test band plotting with untwist option."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    hinges = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    ax = plot_band(curve, hinges, untwist=True)
    assert ax is not None


def test_plot_band_invalid_shape():
    """Test that invalid curve shape raises error."""
    invalid_curve = np.array([[0, 0], [1, 1]])
    hinges = np.array([[0, 0, 1], [0, 0, 1]])

    with pytest.raises(ValueError, match="expected.*3.*curve"):
        plot_band(invalid_curve, hinges)


# Tests for paper_model function

def test_paper_model_basic():
    """Test basic paper model generation."""
    hinges = random_hinges(6, seed=42).as_array()

    ax = paper_model(hinges)
    assert ax is not None
    assert hasattr(ax, 'patches')


def test_paper_model_with_curve():
    """Test paper model with provided curve."""
    hinges = random_hinges(6, seed=42).as_array()
    tangents = binormals_to_tangents(hinges, normalize=False)
    curve = tangents_to_curve(tangents)

    ax = paper_model(hinges, curve=curve)
    assert ax is not None


def test_paper_model_with_custom_colors():
    """Test paper model with custom colors."""
    hinges = random_hinges(6, seed=42).as_array()

    ax = paper_model(
        hinges,
        facecolors=["red", "blue", "green", "yellow"],
        edgecolor="black",
        alpha=0.5,
    )
    assert ax is not None


def test_paper_model_with_title():
    """Test paper model with custom title."""
    hinges = random_hinges(6, seed=42).as_array()

    title = "My Paper Model"
    ax = paper_model(hinges, title=title)
    assert ax is not None
    assert ax.get_title() == title


def test_paper_model_with_width():
    """Test paper model with custom width."""
    hinges = random_hinges(6, seed=42).as_array()

    ax = paper_model(hinges, width=0.25)
    assert ax is not None


def test_paper_model_with_custom_axes():
    """Test paper model with provided axes."""
    import matplotlib.pyplot as plt

    hinges = random_hinges(6, seed=42).as_array()

    fig, custom_ax = plt.subplots()
    returned_ax = paper_model(hinges, ax=custom_ax)

    assert returned_ax is custom_ax
    plt.close(fig)


def test_paper_model_small_kaleidocycle():
    """Test paper model with smallest kaleidocycle (n=3)."""
    hinges = random_hinges(3, seed=42).as_array()

    ax = paper_model(hinges)
    assert ax is not None


def test_paper_model_large_kaleidocycle():
    """Test paper model with larger kaleidocycle."""
    hinges = random_hinges(12, seed=42).as_array()

    ax = paper_model(hinges)
    assert ax is not None


def test_paper_model_invalid_shape():
    """Test that invalid hinges shape raises error."""
    invalid_hinges = np.array([[0, 0], [1, 1]])  # Only 2D

    with pytest.raises(ValueError, match="expected.*hinges"):
        paper_model(invalid_hinges)


def test_paper_model_aspect_ratio():
    """Test that paper model has equal aspect ratio."""
    hinges = random_hinges(6, seed=42).as_array()

    ax = paper_model(hinges)
    # When aspect is set to "equal", matplotlib converts it to 1.0 (or "equal")
    # depending on the axes type
    aspect = ax.get_aspect()
    assert aspect == "equal" or aspect == 1.0


def test_paper_model_produces_patches():
    """Test that paper model produces polygon patches."""
    hinges = random_hinges(6, seed=42).as_array()

    ax = paper_model(hinges)

    # Should have multiple patches (triangles)
    patches = ax.patches
    assert len(patches) > 0

    # Each patch should be a matplotlib Polygon
    from matplotlib.patches import Polygon as MplPolygon
    for patch in patches:
        assert isinstance(patch, MplPolygon)


# Tests for plot_vertex_values function

def test_plot_vertex_values_static_curvature():
    """Test static plot of curvature values."""
    hinges = random_hinges(6, seed=42).as_array()
    tangents = binormals_to_tangents(hinges, normalize=False)
    curve = tangents_to_curve(tangents)

    # Compute curvature values
    curvature = pairwise_curvature(hinges)

    # Plot should work
    ax = plot_vertex_values(curvature, curve, title="Curvature")
    assert ax is not None


def test_plot_vertex_values_static_torsion():
    """Test static plot of torsion values."""
    hinges = random_hinges(6, seed=42).as_array()
    tangents = binormals_to_tangents(hinges, normalize=False)
    curve = tangents_to_curve(tangents)

    # Compute torsion values
    torsion = compute_torsion(hinges)

    # Plot should work
    ax = plot_vertex_values(torsion, curve, title="Torsion", cmap="coolwarm")
    assert ax is not None


def test_plot_vertex_values_static_custom_values():
    """Test static plot with custom vertex values."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    values = np.array([0.0, 0.5, 1.0, 0.75, 0.25])

    ax = plot_vertex_values(
        values, curve,
        title="Custom Values",
        cmap="plasma",
        colorbar_label="Value",
        vmin=0.0,
        vmax=1.0,
    )
    assert ax is not None


def test_plot_vertex_values_static_with_axes():
    """Test that plot_vertex_values accepts custom axes."""
    import matplotlib.pyplot as plt

    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    values = np.array([0.0, 0.5, 1.0])

    fig = plt.figure()
    custom_ax = fig.add_subplot(111, projection="3d")

    returned_ax = plot_vertex_values(values, curve, ax=custom_ax)
    assert returned_ax is custom_ax
    plt.close(fig)


def test_plot_vertex_values_animation():
    """Test animation creation for time-evolving values."""
    hinges = random_hinges(6, seed=42).as_array()
    tangents = binormals_to_tangents(hinges, normalize=False)
    curve = tangents_to_curve(tangents)

    # Create time-evolving values (10 frames, 6 vertices)
    n_frames = 10
    n_vertices = len(curve)
    values_evolution = np.random.rand(n_frames, n_vertices)

    # Should return figure and animation object
    result = plot_vertex_values(values_evolution, curve, title="Evolution")
    assert isinstance(result, tuple)
    assert len(result) == 2

    fig, anim = result
    assert fig is not None
    assert anim is not None


def test_plot_vertex_values_animation_list_of_lists():
    """Test animation with list of lists input."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

    # List of lists for time evolution
    values_evolution = [
        [0.0, 0.5, 1.0],
        [0.2, 0.6, 0.9],
        [0.4, 0.7, 0.8],
    ]

    fig, anim = plot_vertex_values(values_evolution, curve)
    assert fig is not None
    assert anim is not None


def test_plot_vertex_values_invalid_curve_shape():
    """Test that invalid curve shape raises error."""
    invalid_curve = np.array([[0, 0], [1, 1]])  # Only 2D
    values = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="expected.*3.*curve"):
        plot_vertex_values(values, invalid_curve)


def test_plot_vertex_values_mismatched_shapes():
    """Test that mismatched shapes raise error."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    values = np.array([0.0, 1.0])  # Wrong length (not n or n-1)

    with pytest.raises(ValueError, match="doesn't match"):
        plot_vertex_values(values, curve)


def test_plot_vertex_values_animation_mismatched_shapes():
    """Test that mismatched shapes in animation raise error."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    values = np.random.rand(5, 2)  # 5 frames, but 2 values (should be 4 or 3)

    with pytest.raises(ValueError, match="doesn't match"):
        plot_vertex_values(values, curve)


def test_plot_vertex_values_invalid_ndim():
    """Test that 3D or higher dimensional arrays raise error."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    values = np.random.rand(2, 3, 4)  # 3D array

    with pytest.raises(ValueError, match="expected 1D or 2D"):
        plot_vertex_values(values, curve)


def test_plot_vertex_values_no_colorbar():
    """Test plotting without colorbar."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    values = np.array([0.0, 0.5, 1.0])

    ax = plot_vertex_values(values, curve, show_colorbar=False)
    assert ax is not None


def test_plot_vertex_values_no_axes():
    """Test plotting without axes."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    values = np.array([0.0, 0.5, 1.0])

    ax = plot_vertex_values(values, curve, show_axes=False)
    assert ax is not None


# Tests for 2D line plot mode (curve=None)

def test_plot_vertex_values_2d_static():
    """Test 2D line plot without curve."""
    values = np.array([0.0, 0.5, 1.0, 0.75, 0.25])

    ax = plot_vertex_values(values, curve=None, title="Values vs Index")
    assert ax is not None


def test_plot_vertex_values_2d_curvature():
    """Test 2D line plot with curvature values."""
    hinges = random_hinges(6, seed=42).as_array()
    curvature = pairwise_curvature(hinges)

    # Should work without providing curve
    ax = plot_vertex_values(curvature, title="Curvature", cmap="plasma")
    assert ax is not None


def test_plot_vertex_values_2d_torsion():
    """Test 2D line plot with torsion values."""
    hinges = random_hinges(6, seed=42).as_array()
    torsion = compute_torsion(hinges)

    ax = plot_vertex_values(torsion, title="Torsion", cmap="coolwarm")
    assert ax is not None


def test_plot_vertex_values_2d_with_axes():
    """Test 2D line plot with custom axes."""
    import matplotlib.pyplot as plt

    values = np.array([0.0, 0.5, 1.0, 0.75])

    fig, custom_ax = plt.subplots()
    returned_ax = plot_vertex_values(values, ax=custom_ax)

    assert returned_ax is custom_ax
    plt.close(fig)


def test_plot_vertex_values_2d_animation():
    """Test 2D animation without curve."""
    n_frames = 10
    n_vertices = 8

    # Create time-evolving values
    values_evolution = np.random.rand(n_frames, n_vertices)

    fig, anim = plot_vertex_values(values_evolution, curve=None, title="Evolution")
    assert fig is not None
    assert anim is not None


def test_plot_vertex_values_2d_animation_wave():
    """Test 2D animation with wave pattern."""
    n_frames = 15
    n_vertices = 10

    # Create wave pattern
    time = np.linspace(0, 2*np.pi, n_frames)
    vertex_positions = np.linspace(0, 2*np.pi, n_vertices)
    values = np.sin(time[:, np.newaxis] + vertex_positions)

    fig, anim = plot_vertex_values(values, title="Traveling Wave")
    assert fig is not None
    assert anim is not None


def test_plot_vertex_values_2d_no_colorbar():
    """Test 2D line plot without colorbar."""
    values = np.array([0.0, 0.5, 1.0, 0.75])

    ax = plot_vertex_values(values, show_colorbar=False)
    assert ax is not None


def test_plot_vertex_values_2d_custom_range():
    """Test 2D line plot with custom vmin/vmax."""
    values = np.array([0.2, 0.5, 0.8, 0.6])

    ax = plot_vertex_values(values, vmin=0.0, vmax=1.0, colorbar_label="Normalized")
    assert ax is not None
