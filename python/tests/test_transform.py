"""Tests for transformation operations on Kaleidocycles."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import (
    Kaleidocycle,
    involution,
    mirror,
    negative_twist,
    random_hinges,
    reverse,
    subdivide,
    transform_kaleidocycle,
)


def test_mirror_alternates_signs() -> None:
    """Test that mirror operation alternates signs correctly."""
    hinges = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0] / np.sqrt(3),
    ])

    mirrored = mirror(hinges)

    # Check shape is preserved
    assert mirrored.shape == hinges.shape

    # Check alternating sign pattern: (-1)^i
    for i in range(len(hinges)):
        expected_sign = (-1) ** i
        np.testing.assert_allclose(mirrored[i], expected_sign * hinges[i])


def test_mirror_preserves_norms() -> None:
    """Test that mirror preserves vector norms."""
    hinges = random_hinges(6, seed=42).as_array()
    mirrored = mirror(hinges)

    original_norms = np.linalg.norm(hinges, axis=1)
    mirrored_norms = np.linalg.norm(mirrored, axis=1)

    np.testing.assert_allclose(mirrored_norms, original_norms)


def test_mirror_invalid_shape() -> None:
    """Test that mirror raises error for invalid input shape."""
    with pytest.raises(ValueError, match="expected.*hinge array"):
        mirror(np.array([1.0, 2.0, 3.0]))


def test_involution_component_signs() -> None:
    """Test that involution applies correct signs to each component."""
    hinges = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    inv = involution(hinges)

    # Check shape is preserved
    assert inv.shape == hinges.shape

    # Check sign pattern for each component
    for i in range(len(hinges)):
        sign_xz = (-1) ** i
        sign_y = (-1) ** (i + 1)
        expected = np.array([
            sign_xz * hinges[i, 0],
            sign_y * hinges[i, 1],
            sign_xz * hinges[i, 2],
        ])
        np.testing.assert_allclose(inv[i], expected)


def test_involution_preserves_norms() -> None:
    """Test that involution preserves vector norms."""
    hinges = random_hinges(8, seed=123).as_array()
    inv = involution(hinges)

    original_norms = np.linalg.norm(hinges, axis=1)
    inv_norms = np.linalg.norm(inv, axis=1)

    np.testing.assert_allclose(inv_norms, original_norms)


def test_negative_twist_negates_y() -> None:
    """Test that negative_twist negates only y-component."""
    hinges = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    neg_twist = negative_twist(hinges)

    # Check shape is preserved
    assert neg_twist.shape == hinges.shape

    # Check that only y-component is negated
    np.testing.assert_allclose(neg_twist[:, 0], hinges[:, 0])
    np.testing.assert_allclose(neg_twist[:, 1], -hinges[:, 1])
    np.testing.assert_allclose(neg_twist[:, 2], hinges[:, 2])


def test_negative_twist_preserves_norms() -> None:
    """Test that negative_twist preserves vector norms."""
    hinges = random_hinges(10, seed=456).as_array()
    neg_twist = negative_twist(hinges)

    original_norms = np.linalg.norm(hinges, axis=1)
    neg_twist_norms = np.linalg.norm(neg_twist, axis=1)

    np.testing.assert_allclose(neg_twist_norms, original_norms)


def test_negative_twist_double_application() -> None:
    """Test that applying negative_twist twice returns to original."""
    hinges = random_hinges(6, seed=789).as_array()
    double_twisted = negative_twist(negative_twist(hinges))

    np.testing.assert_allclose(double_twisted, hinges)


def test_reverse_reverses_order() -> None:
    """Test that reverse reverses the order of hinges."""
    hinges = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    reversed_hinges = reverse(hinges)

    # Check shape is preserved
    assert reversed_hinges.shape == hinges.shape

    # Check order is reversed
    for i in range(len(hinges)):
        np.testing.assert_allclose(reversed_hinges[i], hinges[-(i + 1)])


def test_reverse_double_application() -> None:
    """Test that applying reverse twice returns to original."""
    hinges = random_hinges(8, seed=111).as_array()
    double_reversed = reverse(reverse(hinges))

    np.testing.assert_allclose(double_reversed, hinges)


def test_subdivide_increases_size() -> None:
    """Test that subdivide increases the number of hinges."""
    hinges = random_hinges(6, seed=222).as_array()
    n = len(hinges) - 1  # Number of segments

    subdivided = subdivide(hinges, divisions=2)

    # Check new size: (n) * divisions + 1
    expected_size = n * 2 + 1
    assert subdivided.shape == (expected_size, 3)


def test_subdivide_preserves_endpoints() -> None:
    """Test that subdivide preserves first hinge."""
    hinges = random_hinges(6, seed=333).as_array()

    subdivided = subdivide(hinges, divisions=3)

    # First hinge should be the same
    np.testing.assert_allclose(subdivided[0], hinges[0])


def test_subdivide_interpolates_correctly_lerp() -> None:
    """Test that subdivide interpolates correctly with LERP."""
    hinges = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    subdivided = subdivide(hinges, divisions=2, method='lerp')

    # Check that all vectors are unit length (normalized)
    norms = np.linalg.norm(subdivided, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_subdivide_interpolates_correctly_slerp() -> None:
    """Test that subdivide interpolates correctly with SLERP."""
    hinges = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    subdivided = subdivide(hinges, divisions=2, method='slerp')

    # Check that all vectors are unit length
    norms = np.linalg.norm(subdivided, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_subdivide_invalid_divisions() -> None:
    """Test that subdivide raises error for invalid divisions."""
    hinges = random_hinges(6, seed=444).as_array()

    with pytest.raises(ValueError, match="divisions must be at least 1"):
        subdivide(hinges, divisions=0)


def test_subdivide_invalid_method() -> None:
    """Test that subdivide raises error for invalid interpolation method."""
    hinges = random_hinges(6, seed=555).as_array()

    with pytest.raises(ValueError, match="Unknown interpolation method"):
        subdivide(hinges, divisions=2, method='invalid')


def test_subdivide_divisions_one() -> None:
    """Test that subdivide with divisions=1 returns the original hinges."""
    hinges = random_hinges(6, seed=666).as_array()

    subdivided = subdivide(hinges, divisions=1)

    # Should be the same size and equal to original
    assert subdivided.shape == hinges.shape
    np.testing.assert_allclose(subdivided, hinges, atol=1e-10)


def test_transform_kaleidocycle_mirror() -> None:
    """Test transform_kaleidocycle with mirror operation."""
    hinges = random_hinges(6, seed=777).as_array()
    kc = Kaleidocycle(hinges=hinges)

    kc_mirror = transform_kaleidocycle(kc, 'mirror')

    # Check that n is preserved
    assert kc_mirror.n == kc.n

    # Check that hinges are mirrored
    expected = mirror(hinges)
    np.testing.assert_allclose(kc_mirror.hinges, expected)


def test_transform_kaleidocycle_involution() -> None:
    """Test transform_kaleidocycle with involution operation."""
    hinges = random_hinges(8, seed=888).as_array()
    kc = Kaleidocycle(hinges=hinges)

    kc_inv = transform_kaleidocycle(kc, 'involution')

    assert kc_inv.n == kc.n
    expected = involution(hinges)
    np.testing.assert_allclose(kc_inv.hinges, expected)


def test_transform_kaleidocycle_negative_twist() -> None:
    """Test transform_kaleidocycle with negative_twist operation."""
    hinges = random_hinges(6, seed=999).as_array()
    kc = Kaleidocycle(hinges=hinges)

    kc_neg = transform_kaleidocycle(kc, 'negative_twist')

    assert kc_neg.n == kc.n
    expected = negative_twist(hinges)
    np.testing.assert_allclose(kc_neg.hinges, expected)


def test_transform_kaleidocycle_reverse() -> None:
    """Test transform_kaleidocycle with reverse operation."""
    hinges = random_hinges(6, seed=101).as_array()
    kc = Kaleidocycle(hinges=hinges)

    kc_rev = transform_kaleidocycle(kc, 'reverse')

    assert kc_rev.n == kc.n
    expected = reverse(hinges)
    np.testing.assert_allclose(kc_rev.hinges, expected)


def test_transform_kaleidocycle_subdivide() -> None:
    """Test transform_kaleidocycle with subdivide operation."""
    hinges = random_hinges(6, seed=202).as_array()
    kc = Kaleidocycle(hinges=hinges)

    kc_subdiv = transform_kaleidocycle(kc, 'subdivide', divisions=2)

    # Check that n is doubled
    assert kc_subdiv.n == kc.n * 2

    expected = subdivide(hinges, divisions=2)
    np.testing.assert_allclose(kc_subdiv.hinges, expected)


def test_transform_kaleidocycle_invalid_operation() -> None:
    """Test that transform_kaleidocycle raises error for invalid operation."""
    hinges = random_hinges(6, seed=303).as_array()
    kc = Kaleidocycle(hinges=hinges)

    with pytest.raises(ValueError, match="Invalid operation"):
        transform_kaleidocycle(kc, 'invalid_op')


def test_transform_kaleidocycle_preserves_orientation() -> None:
    """Test that transform_kaleidocycle preserves orientation flag."""
    hinges = random_hinges(6, seed=404, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    kc_mirror = transform_kaleidocycle(kc, 'mirror')

    assert kc_mirror.oriented == kc.oriented


def test_mirror_composition_with_reverse() -> None:
    """Test composition of mirror and reverse operations."""
    hinges = random_hinges(6, seed=505).as_array()

    # Apply mirror then reverse
    result1 = reverse(mirror(hinges))

    # Apply reverse then mirror
    result2 = mirror(reverse(hinges))

    # These should be different unless the hinges have special symmetry
    # Just check that both operations complete without error
    assert result1.shape == hinges.shape
    assert result2.shape == hinges.shape


def test_subdivide_then_mirror() -> None:
    """Test that subdivide followed by mirror produces valid hinges."""
    hinges = random_hinges(4, seed=606).as_array()

    subdivided = subdivide(hinges, divisions=2)
    mirrored = mirror(subdivided)

    # Check that result is valid
    assert mirrored.shape[0] == subdivided.shape[0]
    norms = np.linalg.norm(mirrored, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)
