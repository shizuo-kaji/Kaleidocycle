"""Tests for Jacobi theta function utilities."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle.theta import (
    alpha2,
    delta1,
    delta3,
    eC,
    eF,
    eH,
    eK,
    eTorsion,
    eX,
    eY,
    eZ,
    generate_animation_theta,
    generate_theta_curve,
    logarithmic_derivative_1,
    logarithmic_derivative_3,
    R1,
    R3,
    theta1,
    theta2,
    theta3,
    theta4,
)


def test_theta_functions_basic() -> None:
    """Test basic theta function evaluation."""
    # Test at simple points
    x = 0.5
    y = 1.0

    # All theta functions should return finite complex values
    th1 = theta1(x, y)
    th2 = theta2(x, y)
    th3 = theta3(x, y)
    th4 = theta4(x, y)

    assert np.isfinite(th1)
    assert np.isfinite(th2)
    assert np.isfinite(th3)
    assert np.isfinite(th4)


def test_theta_functions_at_zero() -> None:
    """Test theta functions at x=0."""
    y = 1.0

    th1 = theta1(0, y)
    th2 = theta2(0, y)
    th3 = theta3(0, y)
    th4 = theta4(0, y)

    # theta1(0) = 0
    assert abs(th1) < 1e-10

    # Others should be nonzero
    assert abs(th2) > 1e-10
    assert abs(th3) > 1e-10
    assert abs(th4) > 1e-10


def test_logarithmic_derivatives() -> None:
    """Test logarithmic derivative computation."""
    r = 0.3
    y = 1.0

    d1 = logarithmic_derivative_1(r, y)
    d3 = logarithmic_derivative_3(r, y)

    # Should return finite complex values
    assert np.isfinite(d1)
    assert np.isfinite(d3)


def test_R_functions() -> None:
    """Test R1 and R3 ratio functions."""
    v = 0.1
    r = 0.3
    y = 1.0

    r1 = R1(v, r, y)
    r3 = R3(v, r, y)

    # Should return finite complex values
    assert np.isfinite(r1)
    assert np.isfinite(r3)

    # Should be nonzero
    assert abs(r1) > 1e-10
    assert abs(r3) > 1e-10


def test_delta_functions() -> None:
    """Test delta1 and delta3 functions."""
    v = 0.1
    r = 0.3
    y = 1.0

    d1 = delta1(v, r, y)
    d3 = delta3(v, r, y)

    # Should return finite complex values
    assert np.isfinite(d1)
    assert np.isfinite(d3)


def test_alpha2() -> None:
    """Test alpha2 parameter."""
    r = 0.3
    y = 1.0

    a2 = alpha2(r, y)

    # Should return finite complex value
    assert np.isfinite(a2)

    # Should be positive real (or nearly real)
    assert abs(np.imag(a2)) < 1e-6
    assert np.real(a2) > 0


def test_eC() -> None:
    """Test evolution coefficient C."""
    v = 0.1
    r = 0.3
    y = 1.0

    c = eC(v, r, y)

    # Should return finite complex value
    assert np.isfinite(c)


def test_eF_and_eH() -> None:
    """Test F and H functions."""
    v = 0.1
    z = 0.0
    r = 0.3
    y = 1.0
    j = 5
    t = 0.0

    f = eF(v, z, r, y, j, t)
    h = eH(v, z, r, y, j, t)

    # Should return finite complex values
    assert np.isfinite(f)
    assert np.isfinite(h)

    # F should be nonzero
    assert abs(f) > 1e-10


def test_eX_eY_eZ() -> None:
    """Test coordinate functions."""
    v = 0.1
    z = 0.0
    r = 0.3
    y = 1.0
    j = 5
    t = 0.0

    x = eX(v, z, r, y, j, t)
    y_coord = eY(v, z, r, y, j, t)
    z_coord = eZ(v, z, r, y, j, t)

    # Should return finite real values
    assert np.isfinite(x)
    assert np.isfinite(y_coord)
    assert np.isfinite(z_coord)

    # All should be real
    assert isinstance(x, (float, np.floating))
    assert isinstance(y_coord, (float, np.floating))
    assert isinstance(z_coord, (float, np.floating))


def test_generate_theta_curve_basic() -> None:
    """Test basic XYZ curve generation."""
    # Use reasonable parameters
    v = 0.07
    z = 0.0
    r = 0.30
    y = 0.92
    N = 10

    curve = generate_theta_curve(v, z, r, y, N, t=0.0)

    # Should have N+1 points
    assert curve.shape == (N + 1, 3)

    # All coordinates should be finite
    assert np.all(np.isfinite(curve))

    # Should be real values
    assert curve.dtype == np.float64


def test_generate_theta_curve_different_N() -> None:
    """Test XYZ curve generation for different N values."""
    v = 0.07
    z = 0.0
    r = 0.30
    y = 0.92

    for N in [6, 12, 20]:
        curve = generate_theta_curve(v, z, r, y, N, t=0.0)
        assert curve.shape == (N + 1, 3)
        assert np.all(np.isfinite(curve))


def test_generate_theta_curve_time_evolution() -> None:
    """Test that curve evolves with time parameter."""
    v = 0.07
    z = 0.0
    r = 0.30
    y = 0.92
    N = 10

    curve0 = generate_theta_curve(v, z, r, y, N, t=0.0)
    curve1 = generate_theta_curve(v, z, r, y, N, t=0.1)

    # Curves should be different
    assert not np.allclose(curve0, curve1)

    # But same shape
    assert curve0.shape == curve1.shape


def test_generate_theta_animation_basic() -> None:
    """Test XYZ animation generation."""
    v = 0.07
    z = 0.0
    r = 0.30
    y = 0.92
    N = 10
    num_frames = 5

    frames = generate_animation_theta(v, z, r, y, N, num_frames, t_step=0.05)

    # Should have correct number of frames
    assert len(frames) == num_frames

    # Each frame should have correct shape
    assert all(f.shape == (N + 1, 3) for f in frames)

    # All should be finite
    assert all(np.all(np.isfinite(f)) for f in frames)


def test_generate_theta_animation_evolution() -> None:
    """Test that animation shows evolution."""
    v = 0.07
    z = 0.0
    r = 0.30
    y = 0.92
    N = 10
    num_frames = 10

    frames = generate_animation_theta(v, z, r, y, N, num_frames, t_step=0.05)

    # Consecutive frames should be different
    for i in range(len(frames) - 1):
        assert not np.allclose(frames[i], frames[i + 1])


def test_eTorsion() -> None:
    """Test torsion computation."""
    v = 0.1
    z = 0.0
    r = 0.3
    y = 1.0
    t = 0.0

    torsion = eTorsion(v, z, r, y, t)

    # Should return finite real value
    assert np.isfinite(torsion)
    assert isinstance(torsion, (float, np.floating))


def test_eK() -> None:
    """Test curvature angle computation."""
    v = 0.1
    z = 0.0
    r = 0.3
    y = 1.0
    j = 5
    t = 0.0

    k = eK(v, z, r, y, j, t)

    # Should return finite real value
    assert np.isfinite(k)
    assert isinstance(k, (float, np.floating))


def test_known_parameters() -> None:
    """Test with known good parameters from Maple code."""
    # From Maple: vals := [v = 0.07227972073349694, r = 0.30353311936556515, y = 0.9155431292909612]
    v = 0.072279720733
    z = 0.0
    r = 0.303533119366
    y = 0.915543129291
    N = 38

    # Should not crash and should produce finite values
    curve = generate_theta_curve(v, z, r, y, N, t=0.0)

    assert curve.shape == (N + 1, 3)
    assert np.all(np.isfinite(curve))

    # Check some basic properties
    # Curve should have reasonable size
    extent = np.linalg.norm(curve.max(axis=0) - curve.min(axis=0))
    assert extent > 0.1
    assert extent < 1000  # Not too large
