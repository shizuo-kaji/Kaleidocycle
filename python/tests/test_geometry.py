"""Tests for geometry functions including writhe, curvature, and axis computation."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import ConstraintConfig
from kaleidocycle.geometry import (
    Kaleidocycle,
    pairwise_curvature,
    compute_axis,
    binormals_to_tangents,
    compute_torsion,
    curve_to_binormals,
    curve_to_tangents,
    lwrithe,
    random_hinges,
    tangents_to_binormals,
    total_twist,
    total_twist_from_curve,
    writhe,
)


def test_lwrithe_planar_segment_is_zero() -> None:
    """Local writhe of coplanar segments should be zero."""
    # Four points forming a planar quadrilateral
    u = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    result = lwrithe(u)
    assert abs(result) < 1e-10


def test_lwrithe_tetrahedral_segment() -> None:
    """Test local writhe for a non-planar tetrahedral configuration."""
    # Four points forming a tetrahedron
    u = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)],
    ])
    result = lwrithe(u)
    # Should be non-zero for non-planar configuration
    assert abs(result) > 1e-6


def test_lwrithe_invalid_shape() -> None:
    """lwrithe should raise ValueError for wrong shape."""
    u = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match="expected shape"):
        lwrithe(u)


def test_writhe_simple_circle() -> None:
    """Writhe of a simple planar circle should be close to zero."""
    # Create a planar circle with many points
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    curve = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    result = writhe(curve)
    # Planar curve should have writhe ≈ 0
    assert abs(result) < 0.1


def test_writhe_helix() -> None:
    """Writhe of a helix should be non-zero."""
    # Create a helix
    t = np.linspace(0, 4*np.pi, 50)
    curve = np.column_stack([
        np.cos(t),
        np.sin(t),
        0.3 * t,
    ])

    result = writhe(curve)
    # Helix should have non-zero writhe
    assert abs(result) > 0.01


def test_writhe_too_few_points() -> None:
    """writhe should raise ValueError for curves with fewer than 4 points."""
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    with pytest.raises(ValueError, match="at least 4 points"):
        writhe(curve)


def test_pairwise_curvature_simple_configuration() -> None:
    """Test curvature computation for a simple binormal configuration."""
    # Create 4 binormals (n+1 where n=3)
    # Use unit vectors along z-axis with slight rotation
    B = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, np.sqrt(1 - 0.1**2)],
        [0.0, 0.1, np.sqrt(1 - 0.1**2)],
        [0.0, 0.0, 1.0],
    ])

    K = pairwise_curvature(B, signed=False)

    # Should return array of length n=3
    assert K.shape == (3,)
    # Curvatures should be non-negative when signed=False
    assert np.all(K >= 0)


def test_pairwise_curvature_with_tangents() -> None:
    """Test curvature computation with explicit tangent vectors."""
    # Create simple binormals and tangents
    B = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    # Tangents rotating in xy-plane
    T = np.array([
        [1.0, 0.0, 0.0],
        [np.cos(np.pi/4), np.sin(np.pi/4), 0.0],
    ])

    K = pairwise_curvature(B, tangents=T, signed=False)

    assert K.shape == (2,)
    # First curvature should be π/4
    assert abs(K[0] - np.pi/4) < 1e-10


def test_pairwise_curvature_invalid_shape() -> None:
    """pairwise_curvature should raise ValueError for wrong shape."""
    B = np.array([[0, 0, 1]])
    with pytest.raises(ValueError, match="at least 2 binormals"):
        pairwise_curvature(B)


def test_compute_axis_aligned_configuration() -> None:
    """Test axis computation for binormals with known axis."""
    # Create binormals NOT all coplanar (need 3D span)
    # Place them such that A = [0, 0, 2] is the desired axis
    B = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],  # extra binormal
    ])

    # Set K such that tan(K[i]/2) = A · B[i]
    # A = [0, 0, 2], so A·B = [0, 0, 2]
    # tan(K[0]/2) = 0 → K[0] = 0
    # tan(K[1]/2) = 0 → K[1] = 0
    # tan(K[2]/2) = 2 → K[2] = 2*arctan(2)
    K = np.array([0.0, 0.0, 2*np.arctan(2.0)])

    A = compute_axis(B, K)

    # Axis should match the expected form
    assert A.shape == (3,)
    # Check that A·B[i] = tan(K[i]/2) for i=0,1,2
    for i in range(3):
        expected = np.tan(K[i] / 2)
        actual = np.dot(A, B[i])
        assert abs(actual - expected) < 1e-10


def test_compute_axis_simple_case() -> None:
    """Test axis computation with a simple orthogonal configuration."""
    # Three orthogonal binormals
    B = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Set curvatures such that we want A = [1, 1, 1]
    # tan(K[i]/2) = A · B[i] = 1
    # So K[i] = 2*arctan(1) = π/2
    K = np.full(3, np.pi/2)

    A = compute_axis(B, K)

    # Check that A·B[i] = tan(K[i]/2) = 1
    for i in range(3):
        expected = np.tan(K[i] / 2)
        actual = np.dot(A, B[i])
        assert abs(actual - expected) < 1e-10


def test_compute_axis_too_few_binormals() -> None:
    """compute_axis should raise ValueError for fewer than 3 binormals."""
    B = np.array([[1, 0, 0], [0, 1, 0]])
    K = np.array([0.5])
    with pytest.raises(ValueError, match="at least 3 binormals"):
        compute_axis(B, K)


def test_compute_axis_singular_system() -> None:
    """compute_axis should raise ValueError for singular/degenerate configuration."""
    # Three collinear binormals (singular system)
    B = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    K = np.array([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="singular system"):
        compute_axis(B, K)


def test_pairwise_curvature_and_binormals_to_tangents_integration() -> None:
    """Integration test: compute curvature from hinges via mid-axes."""
    # Create simple hinge configuration
    hinges = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Compute mid-axes (tangents)
    T = binormals_to_tangents(hinges, normalize=False)
    assert T.shape == (3, 3)

    # Compute curvature from hinges (as binormals)
    K = pairwise_curvature(hinges, tangents=T, signed=False)
    assert K.shape == (3,)

    # All curvatures should be positive and reasonable
    assert np.all(K > 0)
    assert np.all(K < np.pi)


# Tests for topology functions (Tw, TwX, torsion, X2T, T2B)

def test_curve_to_tangents_simple() -> None:
    """Test tangent computation from curve points."""
    # Linear curve along x-axis
    curve = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])

    tangents = curve_to_tangents(curve, normalize=False)
    assert tangents.shape == (3, 3)

    # All tangents should point in x-direction with length 1
    expected = np.array([[1.0, 0.0, 0.0]] * 3)
    np.testing.assert_allclose(tangents, expected, atol=1e-10)


def test_curve_to_tangents_normalized() -> None:
    """Test normalized tangent computation."""
    curve = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
    ])

    tangents = curve_to_tangents(curve, normalize=True)
    assert tangents.shape == (2, 3)

    # Check all tangents are unit length
    norms = np.linalg.norm(tangents, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_curve_to_tangents_too_few_points() -> None:
    """curve_to_tangents should raise ValueError for single point."""
    curve = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="at least 2 points"):
        curve_to_tangents(curve)


def test_tangents_to_binormals_simple() -> None:
    """Test binormal computation from tangents."""
    # Tangents in xy-plane forming a square
    tangents = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])

    binormals = tangents_to_binormals(tangents)
    assert binormals.shape == (5, 3)

    # All binormals should point in z-direction (or -z)
    for i in range(5):
        assert abs(abs(binormals[i, 2]) - 1.0) < 1e-6
        assert abs(binormals[i, 0]) < 1e-6
        assert abs(binormals[i, 1]) < 1e-6


def test_tangents_to_binormals_with_reference() -> None:
    """Test binormal computation with custom reference."""
    tangents = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    reference = np.array([0.0, 0.0, -1.0])
    binormals = tangents_to_binormals(tangents, reference=reference)

    # First binormal should align with reference (negative z)
    assert binormals[0, 2] < 0


def test_curve_to_binormals_integration() -> None:
    """Integration test: curve -> tangents -> binormals."""
    # Create a planar curve
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    curve = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    binormals = curve_to_binormals(curve)
    assert binormals.shape == (8, 3)

    # For planar curve, all binormals should be perpendicular to plane
    for i in range(8):
        # Should be mostly in z-direction
        assert abs(binormals[i, 2]) > 0.5


def test_compute_torsion_constant() -> None:
    """Test torsion for binormals with constant angle."""
    # Create binormals rotating uniformly
    angles = np.linspace(0, np.pi, 5)
    binormals = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])

    torsion = compute_torsion(binormals)
    assert torsion.shape == (4,)

    # All torsion angles should be approximately equal
    expected_angle = np.pi / 4
    np.testing.assert_allclose(torsion, expected_angle, rtol=1e-5)


def test_compute_torsion_zero() -> None:
    """Test torsion for constant binormals (zero torsion)."""
    # All binormals pointing in same direction
    binormals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    torsion = compute_torsion(binormals)
    assert torsion.shape == (2,)

    # All torsion angles should be zero
    np.testing.assert_allclose(torsion, 0.0, atol=1e-10)


def test_compute_torsion_too_few_binormals() -> None:
    """compute_torsion should raise ValueError for single binormal."""
    binormals = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="at least 2 binormals"):
        compute_torsion(binormals)


def test_total_twist_zero() -> None:
    """Test total twist for constant binormals."""
    binormals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    tw = total_twist(binormals)
    assert abs(tw) < 1e-10


def test_total_twist_rotation() -> None:
    """Test total twist for rotating binormals."""
    # Binormals rotating through 2π (full turn)
    angles = np.linspace(0, 2*np.pi, 9)
    binormals = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])

    tw = total_twist(binormals)
    # Total twist should be approximately 2 (2π/π = 2)
    assert abs(tw - 2.0) < 0.1


def test_total_twist_half_turn() -> None:
    """Test total twist for half rotation."""
    # Binormals rotating through π
    angles = np.linspace(0, np.pi, 5)
    binormals = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])

    tw = total_twist(binormals)
    # Total twist should be approximately 1 (π/π = 1)
    assert abs(tw - 1.0) < 0.1


def test_total_twist_from_curve_planar() -> None:
    """Test total twist from a planar curve."""
    # Planar circle - should have minimal twist
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    curve = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    tw = total_twist_from_curve(curve)
    # Planar curve should have very small twist
    assert abs(tw) < 0.2


def test_total_twist_from_curve_helix() -> None:
    """Test total twist from a helix."""
    # Helix with significant twist
    t = np.linspace(0, 4*np.pi, 50)
    curve = np.column_stack([
        np.cos(t),
        np.sin(t),
        0.5 * t,
    ])

    tw = total_twist_from_curve(curve)
    # Helix should have significant twist
    assert abs(tw) > 0.5


def test_total_twist_from_curve_with_reference() -> None:
    """Test total twist with custom reference binormal."""
    theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
    curve = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    reference = np.array([0.0, 0.0, -1.0])
    tw = total_twist_from_curve(curve, reference=reference)

    # Should still compute successfully with different reference
    assert isinstance(tw, float)
    assert not np.isnan(tw)


# Tests for Kaleidocycle.is_feasible method

def test_is_feasible_random_hinges_strict() -> None:
    """Test that random hinges are not feasible with strict tolerance."""
    hinges = random_hinges(6, seed=42).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Random hinges should not satisfy constraints with strict tolerance
    assert not kc.is_feasible(tolerance=1e-4)


def test_is_feasible_random_hinges_loose() -> None:
    """Test that random hinges might be feasible with loose tolerance."""
    hinges = random_hinges(6, seed=42).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # With very loose tolerance, might pass
    # Just check the method runs without error
    result = kc.is_feasible(tolerance=100.0)
    assert isinstance(result, bool)


def test_is_feasible_perfect_hinges() -> None:
    """Test that perfectly aligned and normalized hinges are feasible."""
    # Create perfect hinges with constant torsion and proper alignment
    n = 6
    theta = np.linspace(0, 2*np.pi, n+1)
    hinges = np.column_stack([
        np.cos(theta),
        np.sin(theta),
        np.zeros_like(theta),
    ])
    # Normalize
    hinges = hinges / np.linalg.norm(hinges, axis=1, keepdims=True)
    # Make it aligned (first and last should be equal for oriented)
    hinges[-1] = hinges[0]

    kc = Kaleidocycle(hinges=hinges, oriented=True)

    # This configuration might not be perfectly feasible,
    # but should have relatively low penalty
    # Just check the method runs
    result = kc.is_feasible(tolerance=10.0)
    assert isinstance(result, bool)


def test_is_feasible_custom_config() -> None:
    """Test is_feasible with custom constraint configuration."""
    hinges = random_hinges(6, seed=123).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Custom config with only closure constraint
    config = ConstraintConfig(
        oriented=kc.oriented,
        alignment=False,
        constant_torsion=False,
        enforce_anchors=False,
    )

    # Check with custom config
    result = kc.is_feasible(tolerance=100.0, config=config)
    assert isinstance(result, bool)


def test_is_feasible_oriented_vs_nonoriented() -> None:
    """Test is_feasible with both oriented and non-oriented kaleidocycles."""
    # Oriented
    hinges_oriented = random_hinges(6, seed=456, oriented=True).as_array()
    kc_oriented = Kaleidocycle(hinges=hinges_oriented, oriented=True)
    result_oriented = kc_oriented.is_feasible(tolerance=1.0)
    assert isinstance(result_oriented, bool)

    # Non-oriented
    hinges_nonoriented = random_hinges(6, seed=456, oriented=False).as_array()
    kc_nonoriented = Kaleidocycle(hinges=hinges_nonoriented, oriented=False)
    result_nonoriented = kc_nonoriented.is_feasible(tolerance=1.0)
    assert isinstance(result_nonoriented, bool)


def test_is_feasible_different_tolerances() -> None:
    """Test that looser tolerance is more permissive."""
    hinges = random_hinges(6, seed=789).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Very strict tolerance should fail
    assert not kc.is_feasible(tolerance=1e-10)

    # Very loose tolerance should pass
    assert kc.is_feasible(tolerance=1e10)


def test_is_feasible_returns_bool() -> None:
    """Test that is_feasible always returns a boolean."""
    hinges = random_hinges(8, seed=111).as_array()
    kc = Kaleidocycle(hinges=hinges)

    result = kc.is_feasible()
    assert isinstance(result, bool)
    assert result in [True, False]


def test_is_feasible_default_config_matches_spec() -> None:
    """Test that default config includes alignment, constant_torsion, and closure."""
    hinges = random_hinges(6, seed=222).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Create the default config as specified
    default_config = ConstraintConfig(
        oriented=kc.oriented,
        alignment=True,
        constant_torsion=True,
        enforce_anchors=False,
        slide=0.0,
    )

    # Test with default (no config)
    result_default = kc.is_feasible(tolerance=1.0)

    # Test with explicit config
    result_explicit = kc.is_feasible(tolerance=1.0, config=default_config)

    # Should give the same result
    assert result_default == result_explicit


def test_is_feasible_small_kaleidocycle() -> None:
    """Test is_feasible with smallest possible kaleidocycle (n=3)."""
    hinges = random_hinges(3, seed=333).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Should run without error even for small kaleidocycle
    result = kc.is_feasible(tolerance=1.0)
    assert isinstance(result, bool)


def test_is_feasible_large_kaleidocycle() -> None:
    """Test is_feasible with larger kaleidocycle."""
    hinges = random_hinges(12, seed=444).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Should run without error for larger kaleidocycle
    result = kc.is_feasible(tolerance=1.0)
    assert isinstance(result, bool)


# Tests for Kaleidocycle.report method

def test_report_returns_string() -> None:
    """Test that report returns a string."""
    hinges = random_hinges(6, seed=42).as_array()
    kc = Kaleidocycle(hinges=hinges)

    report = kc.report()
    assert isinstance(report, str)
    assert len(report) > 0


def test_report_contains_expected_sections() -> None:
    """Test that report contains expected section headers."""
    hinges = random_hinges(6, seed=42).as_array()
    kc = Kaleidocycle(hinges=hinges)

    report = kc.report()

    # Check for expected sections
    assert "Kaleidocycle Property Report" in report
    assert "Geometric Properties" in report
    assert "Topological Properties" in report
    assert "Constraint Violations" in report
    assert "Energy" in report


def test_report_with_custom_config() -> None:
    """Test report with custom constraint configuration."""
    hinges = random_hinges(6, seed=123).as_array()
    kc = Kaleidocycle(hinges=hinges)

    config = ConstraintConfig(
        oriented=kc.oriented,
        alignment=True,
        constant_torsion=False,
    )

    report = kc.report(config=config)
    assert isinstance(report, str)
    assert len(report) > 0


def test_report_with_precision() -> None:
    """Test report with custom precision."""
    hinges = random_hinges(6, seed=456).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Default precision
    report_default = kc.report()
    assert isinstance(report_default, str)

    # Higher precision
    report_high = kc.report(precision=10)
    assert isinstance(report_high, str)

    # Lower precision
    report_low = kc.report(precision=2)
    assert isinstance(report_low, str)


def test_report_oriented_vs_nonoriented() -> None:
    """Test report for both oriented and non-oriented kaleidocycles."""
    # Oriented
    hinges_oriented = random_hinges(6, seed=789, oriented=True).as_array()
    kc_oriented = Kaleidocycle(hinges=hinges_oriented, oriented=True)
    report_oriented = kc_oriented.report()
    assert "Oriented: True" in report_oriented

    # Non-oriented
    hinges_nonoriented = random_hinges(6, seed=789, oriented=False).as_array()
    kc_nonoriented = Kaleidocycle(hinges=hinges_nonoriented, oriented=False)
    report_nonoriented = kc_nonoriented.report()
    assert "Oriented: False" in report_nonoriented


def test_report_after_compute() -> None:
    """Test that report works after calling compute."""
    hinges = random_hinges(6, seed=111).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Compute properties first
    config = ConstraintConfig(oriented=kc.oriented)
    kc.compute(config=config)

    # Report should use cached metadata
    report = kc.report()
    assert isinstance(report, str)
    assert len(report) > 0


# Tests for Kaleidocycle.plot method

def test_plot_returns_axes() -> None:
    """Test that plot returns matplotlib axes."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(6, seed=42).as_array()
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot()

    # Check that we get an axes object
    assert ax is not None
    # Check it's a 3D axes
    assert hasattr(ax, 'zaxis')


def test_plot_with_custom_colors() -> None:
    """Test plot with custom colors."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(6, seed=123).as_array()
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot(
        facecolor="red",
        edgecolor="black",
        alpha=0.5,
    )

    assert ax is not None


def test_plot_with_title() -> None:
    """Test plot with title."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(6, seed=456).as_array()
    kc = Kaleidocycle(hinges=hinges)

    title = "Test Kaleidocycle"
    ax = kc.plot(title=title)

    assert ax is not None
    assert ax.get_title() == title


def test_plot_show_curve() -> None:
    """Test plot with curve backbone shown."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(6, seed=789).as_array()
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot(show_curve=True)

    assert ax is not None


def test_plot_with_custom_axes() -> None:
    """Test plot with provided axes."""
    mpl = pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    hinges = random_hinges(6, seed=111).as_array()
    kc = Kaleidocycle(hinges=hinges)

    # Create custom axes
    fig = plt.figure()
    custom_ax = fig.add_subplot(111, projection='3d')

    # Plot on custom axes
    returned_ax = kc.plot(ax=custom_ax)

    # Should return the same axes
    assert returned_ax is custom_ax

    plt.close(fig)


def test_plot_with_width() -> None:
    """Test plot with custom width."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(6, seed=222).as_array()
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot(width=0.3)

    assert ax is not None


def test_plot_oriented_vs_nonoriented() -> None:
    """Test plot for both oriented and non-oriented kaleidocycles."""
    pytest.importorskip("matplotlib")

    # Oriented
    hinges_oriented = random_hinges(6, seed=333, oriented=True).as_array()
    kc_oriented = Kaleidocycle(hinges=hinges_oriented, oriented=True)
    ax_oriented = kc_oriented.plot()
    assert ax_oriented is not None

    # Non-oriented
    hinges_nonoriented = random_hinges(6, seed=333, oriented=False).as_array()
    kc_nonoriented = Kaleidocycle(hinges=hinges_nonoriented, oriented=False)
    ax_nonoriented = kc_nonoriented.plot()
    assert ax_nonoriented is not None


def test_plot_small_kaleidocycle() -> None:
    """Test plot with smallest kaleidocycle (n=3)."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(3, seed=444).as_array()
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot()
    assert ax is not None


def test_plot_large_kaleidocycle() -> None:
    """Test plot with larger kaleidocycle."""
    pytest.importorskip("matplotlib")

    hinges = random_hinges(12, seed=555).as_array()
    kc = Kaleidocycle(hinges=hinges)

    ax = kc.plot()
    assert ax is not None
