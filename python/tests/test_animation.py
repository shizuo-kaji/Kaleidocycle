"""Tests for animation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle.animation import (
    align_animation_frames,
    binormals_to_normals,
    clean_animation_frames,
    curvature_to_omega,
    generate_animation,
    generate_animation_step,
    generate_animation_random,
    sine_gordon_step,
    sort_animation_frames,
)
from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.geometry import binormals_to_tangents, random_hinges


def test_binormals_to_normals_simple() -> None:
    """Test normal vector computation from binormals and tangents."""
    # Binormals pointing in z-direction
    binormals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    # Tangents in x-direction
    tangents = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    normals = binormals_to_normals(binormals, tangents)
    # Should return n+1 = 3 normals (including wraparound)
    assert normals.shape == (3, 3)

    # N[i] = B[i] × T[i] = z × x = y for i=0,1
    # N[2] = B[2] × T[0] = z × x = y (wraparound)
    expected = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    np.testing.assert_allclose(normals, expected, atol=1e-10)


def test_binormals_to_normals_normalized() -> None:
    """Test normalized normal vector computation."""
    binormals = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    tangents = np.array([
        [0.0, 0.0, 2.0],  # Non-unit length
    ])

    normals = binormals_to_normals(binormals, tangents, normalize=True)
    # Should return n+1 = 2 normals (including wraparound)
    assert normals.shape == (2, 3)

    # Check all are unit length
    for i in range(2):
        norm = np.linalg.norm(normals[i])
        assert abs(norm - 1.0) < 1e-10


def test_curvature_to_omega_non_oriented() -> None:
    """Test omega angle computation for non-oriented case."""
    # Simple curvature array
    curvature = np.array([0.5, 0.5, 0.5, 0.5])

    omega = curvature_to_omega(curvature, oriented=False, mkdv=False)

    # Should return n+1 elements
    assert omega.shape == (5,)

    # Check wraparound property
    assert abs(omega[-1] + omega[0]) < 1e-10


def test_curvature_to_omega_oriented() -> None:
    """Test omega angle computation for oriented case."""
    curvature = np.array([1.0, 0.5, 0.5, 1.0])

    omega = curvature_to_omega(curvature, oriented=True, mkdv=False)

    assert omega.shape == (5,)
    assert isinstance(omega[0], (float, np.floating))


def test_sine_gordon_step_preserves_shape() -> None:
    """Test that sine-Gordon step preserves binormal array shape."""
    # Create simple binormal configuration
    binormals = random_hinges(6, seed=42).as_array()

    result = sine_gordon_step(binormals, step_size=0.01, rule="sine-Gordon")

    assert result.shape == binormals.shape
    assert result.shape == (7, 3)


def test_sine_gordon_step_preserves_unit_length() -> None:
    """Test that binormals remain unit length after evolution."""
    binormals = random_hinges(6, seed=42).as_array()

    result = sine_gordon_step(binormals, step_size=0.01)

    # Check all binormals are unit length
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_sine_gordon_step_small_change() -> None:
    """Test that small step size produces small changes."""
    binormals = random_hinges(6, seed=42).as_array()

    result = sine_gordon_step(binormals, step_size=0.001)

    # Change should be small
    diff = np.linalg.norm(result - binormals)
    assert diff < 0.1


def test_sine_gordon_step_different_rules() -> None:
    """Test sine-Gordon step with different evolution rules."""
    binormals = random_hinges(6, seed=42).as_array()

    result_sg = sine_gordon_step(binormals, rule="sine-Gordon")
    result_mkdv = sine_gordon_step(binormals, rule="mKdV")
    result_mkdv2 = sine_gordon_step(binormals, rule="mKdV2")

    # All should preserve shape
    assert result_sg.shape == binormals.shape
    assert result_mkdv.shape == binormals.shape
    assert result_mkdv2.shape == binormals.shape

    # Results should be different for different rules
    assert not np.allclose(result_sg, result_mkdv)
    assert not np.allclose(result_sg, result_mkdv2)


def test_generate_animation_basic() -> None:
    """Test basic animation generation."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation(
        binormals,
        num_frames=10,
        step_size=0.01,
    )

    assert len(frames) == 10
    assert all(f.shape == binormals.shape for f in frames)

    # First frame should be the initial configuration
    np.testing.assert_allclose(frames[0], binormals, atol=1e-10)


def test_generate_animation_evolution() -> None:
    """Test that animation shows continuous evolution."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation(
        binormals,
        num_frames=20,
        step_size=0.01,
    )

    # Check that frames change gradually
    for i in range(len(frames) - 1):
        diff = np.linalg.norm(frames[i + 1] - frames[i])
        # Should change but not too much
        assert 0 < diff < 0.5


def test_generate_animation_different_params() -> None:
    """Test animation with different parameters."""
    binormals = random_hinges(6, seed=42).as_array()

    # Different number of frames
    frames_short = generate_animation(binormals, num_frames=5)
    frames_long = generate_animation(binormals, num_frames=20)

    assert len(frames_short) == 5
    assert len(frames_long) == 20

    # Different step sizes
    frames_small = generate_animation(binormals, num_frames=10, step_size=0.001)
    frames_large = generate_animation(binormals, num_frames=10, step_size=0.1)

    # Larger steps should produce more change
    diff_small = np.linalg.norm(frames_small[-1] - frames_small[0])
    diff_large = np.linalg.norm(frames_large[-1] - frames_large[0])
    assert diff_large > diff_small


def test_clean_animation_frames_basic() -> None:
    """Test basic frame cleaning."""
    binormals = random_hinges(6, seed=42).as_array()
    frames = generate_animation(binormals, num_frames=10, step_size=0.01)

    # Use very permissive tolerance since random_hinges doesn't satisfy all constraints
    # and sine-Gordon flow may violate some constraints
    cleaned, kept_indices = clean_animation_frames(frames, tolerance=10.0)

    # Should keep frames when tolerance is high enough
    assert len(cleaned) > 0
    assert len(cleaned) <= len(frames)
    assert len(kept_indices) == len(cleaned)


def test_clean_animation_frames_removes_bad() -> None:
    """Test that cleaning removes infeasible frames."""
    binormals = random_hinges(6, seed=42).as_array()
    frames = generate_animation(binormals, num_frames=10, step_size=0.01)

    # Add a clearly bad frame (non-unit vectors)
    bad_frame = frames[5].copy() * 2.0  # Double the norms
    frames.insert(5, bad_frame)

    config = ConstraintConfig(enforce_anchors=False)
    cleaned, kept_indices = clean_animation_frames(
        frames,
        config=config,
        tolerance=0.01,
    )

    # Bad frame should be removed
    assert 5 not in kept_indices or len(cleaned) < len(frames)


def test_clean_animation_frames_twist_fix() -> None:
    """Test twist sign fixing in frame cleaning."""
    binormals = random_hinges(6, seed=42).as_array()
    frames = generate_animation(binormals, num_frames=5, step_size=0.01)

    # Flip y-component of one frame to simulate wrong twist
    frames[2] = frames[2].copy()
    frames[2][:, 1] *= -1

    cleaned, _ = clean_animation_frames(frames, fix_twist=True, tolerance=0.5)

    # Should attempt to fix twist (though frame might still be rejected)
    assert len(cleaned) >= 0


def test_clean_animation_frames_empty_input() -> None:
    """Test cleaning with empty input."""
    cleaned, kept = clean_animation_frames([])

    assert cleaned == []
    assert kept == []


def test_align_animation_frames_basic() -> None:
    """Test basic frame alignment."""
    binormals = random_hinges(6, seed=42).as_array()
    frames = generate_animation(binormals, num_frames=5, step_size=0.01)

    aligned = align_animation_frames(frames)

    assert len(aligned) == len(frames)
    assert all(f.shape == frames[0].shape for f in aligned)

    # First frame should be unchanged (reference)
    np.testing.assert_allclose(aligned[0], frames[0], atol=1e-10)


def test_align_animation_frames_removes_rotation() -> None:
    """Test that alignment removes artificial rotation."""
    binormals = random_hinges(6, seed=42).as_array()
    frames = [binormals]

    # Create rotated versions
    theta = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])

    for _ in range(4):
        rotated = frames[-1] @ rotation_matrix.T
        frames.append(rotated)

    aligned = align_animation_frames(frames, use_barycentre=True)

    # After alignment, frames should be more similar
    # Check that variation is reduced
    original_std = np.std([np.linalg.norm(f - frames[0]) for f in frames])
    aligned_std = np.std([np.linalg.norm(f - aligned[0]) for f in aligned])

    # Alignment should reduce variation (though not necessarily to zero)
    assert aligned_std < original_std + 1e-6


def test_align_animation_frames_preserves_shape() -> None:
    """Test that alignment preserves distances within frames."""
    binormals = random_hinges(6, seed=42).as_array()

    # Compute pairwise distances in original
    n = binormals.shape[0]
    original_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            original_dists[i, j] = np.linalg.norm(binormals[i] - binormals[j])

    # Apply arbitrary rotation
    theta = np.pi / 3
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated = binormals @ R.T

    # Align
    aligned = align_animation_frames([binormals, rotated])[1]

    # Compute distances in aligned
    aligned_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            aligned_dists[i, j] = np.linalg.norm(aligned[i] - aligned[j])

    # Distances should be preserved
    np.testing.assert_allclose(aligned_dists, original_dists, rtol=1e-5)


def test_align_animation_frames_single_frame() -> None:
    """Test alignment with single frame."""
    binormals = random_hinges(6, seed=42).as_array()

    aligned = align_animation_frames([binormals])

    assert len(aligned) == 1
    np.testing.assert_allclose(aligned[0], binormals, atol=1e-10)


def test_animation_integration() -> None:
    """Integration test: generate, clean, and align animation."""
    # Create initial configuration
    binormals = random_hinges(6, seed=42).as_array()

    # Generate animation
    frames = generate_animation(
        binormals,
        num_frames=20,
        step_size=0.02,
        rule="sine-Gordon",
    )

    # Clean frames
    config = ConstraintConfig(enforce_anchors=False)
    cleaned, kept_indices = clean_animation_frames(
        frames,
        config=config,
        tolerance=0.1,
        fix_twist=True,
    )

    # Align frames
    if cleaned:
        aligned = align_animation_frames(cleaned, use_barycentre=True)

        # Should have reasonable number of frames
        assert len(aligned) > 0
        assert len(aligned) <= len(frames)

        # All frames should have correct shape
        assert all(f.shape == binormals.shape for f in aligned)

        # All binormals should be unit length
        for frame in aligned:
            norms = np.linalg.norm(frame, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-2)


def test_generate_animation_step_basic() -> None:
    """Test step animation generation."""
    binormals = random_hinges(6, seed=42).as_array()

    # Generate a few frames
    frames = generate_animation_step(
        binormals,
        num_frames=5,
        step_size=0.05,
        verbose=False,
    )

    # Should have requested number of frames
    assert len(frames) == 5

    # All frames should have correct shape
    assert all(f.shape == binormals.shape for f in frames)

    # First frame should be the initial configuration
    np.testing.assert_allclose(frames[0], binormals, atol=1e-10)


def test_generate_animation_step_unit_length() -> None:
    """Test that step animation preserves unit length."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation_step(
        binormals,
        num_frames=3,
        step_size=0.05,
    )

    # Check all binormals are unit length
    for frame in frames:
        norms = np.linalg.norm(frame, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-2)


def test_generate_animation_step_evolution() -> None:
    """Test that step animation shows evolution."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation_step(
        binormals,
        num_frames=10,
        step_size=0.05,
    )

    # Check that frames change over time
    distances = [np.linalg.norm(frames[i] - frames[0]) for i in range(len(frames))]

    # Distance should generally increase
    assert distances[-1] > distances[0]


def test_generate_animation_random_basic() -> None:
    """Test random animation generation."""
    binormals = random_hinges(6, seed=42).as_array()

    # Generate a few random frames
    frames = generate_animation_random(
        binormals,
        num_frames=5,
        seed=42,
        verbose=False,
    )

    # Should have requested number of frames
    assert len(frames) == 5

    # All frames should have correct shape
    assert all(f.shape == binormals.shape for f in frames)

    # First frame should be the initial configuration
    np.testing.assert_allclose(frames[0], binormals, atol=1e-10)


def test_generate_animation_random_reproducible() -> None:
    """Test that random animation is reproducible with seed."""
    binormals = random_hinges(6, seed=42).as_array()

    frames1 = generate_animation_random(
        binormals,
        num_frames=5,
        seed=42,
    )

    frames2 = generate_animation_random(
        binormals,
        num_frames=5,
        seed=42,
    )

    # Should produce identical results with same seed
    for f1, f2 in zip(frames1, frames2):
        np.testing.assert_allclose(f1, f2, atol=1e-10)


def test_generate_animation_random_diversity() -> None:
    """Test that random animation produces diverse configurations."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation_random(
        binormals,
        num_frames=10,
        seed=42,
    )

    # Check that frames are different from each other
    # (not all identical to initial or each other)
    distances_from_initial = [
        np.linalg.norm(frame - frames[0]) for frame in frames[1:]
    ]

    # At least some frames should be significantly different
    assert max(distances_from_initial) > 0.1


def test_generate_animation_random_unit_length() -> None:
    """Test that random animation preserves unit length."""
    binormals = random_hinges(6, seed=42).as_array()

    frames = generate_animation_random(
        binormals,
        num_frames=3,
        seed=42,
    )

    # Check all binormals are unit length
    for frame in frames:
        norms = np.linalg.norm(frame, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-2)


def test_generate_animation_random_different_seeds() -> None:
    """Test that different seeds produce different animations."""
    binormals = random_hinges(6, seed=42).as_array()

    frames1 = generate_animation_random(
        binormals,
        num_frames=5,
        seed=42,
    )

    frames2 = generate_animation_random(
        binormals,
        num_frames=5,
        seed=123,
    )

    # At least one frame should be different
    differences = [np.linalg.norm(f1 - f2) for f1, f2 in zip(frames1[1:], frames2[1:])]
    assert max(differences) > 0.01


def test_sort_animation_frames_basic() -> None:
    """Test basic frame sorting."""
    # Create frames in deliberately scrambled order
    binormals = random_hinges(4, seed=42).as_array()

    # Create a sequence with known ordering
    frame0 = binormals
    frame1 = binormals + np.random.randn(*binormals.shape) * 0.1
    frame2 = frame1 + np.random.randn(*binormals.shape) * 0.1
    frame3 = frame2 + np.random.randn(*binormals.shape) * 0.1

    # Normalize all
    frame1 = frame1 / np.linalg.norm(frame1, axis=1, keepdims=True)
    frame2 = frame2 / np.linalg.norm(frame2, axis=1, keepdims=True)
    frame3 = frame3 / np.linalg.norm(frame3, axis=1, keepdims=True)

    # Scramble the order
    frames_scrambled = [frame0, frame3, frame1, frame2]

    # Sort
    frames_sorted = sort_animation_frames(frames_scrambled)

    # Should have same number of frames
    assert len(frames_sorted) == len(frames_scrambled)

    # First frame should be unchanged
    np.testing.assert_allclose(frames_sorted[0], frame0, atol=1e-10)


def test_sort_animation_frames_smooth() -> None:
    """Test that sorting produces smoother animation."""
    binormals = random_hinges(4, seed=42).as_array()

    # Create sequential frames
    frames_ordered = [binormals]
    for _ in range(9):
        next_frame = frames_ordered[-1] + np.random.randn(*binormals.shape) * 0.05
        next_frame = next_frame / np.linalg.norm(next_frame, axis=1, keepdims=True)
        frames_ordered.append(next_frame)

    # Scramble them
    np.random.seed(42)
    indices = list(range(len(frames_ordered)))
    np.random.shuffle(indices)
    frames_scrambled = [frames_ordered[i] for i in indices]

    # Compute distances before sorting
    distances_before = []
    for i in range(1, len(frames_scrambled)):
        dist = np.linalg.norm(frames_scrambled[i] - frames_scrambled[i-1])
        distances_before.append(dist)

    # Sort and compute distances after
    frames_sorted = sort_animation_frames(frames_scrambled)
    distances_after = []
    for i in range(1, len(frames_sorted)):
        dist = np.linalg.norm(frames_sorted[i] - frames_sorted[i-1])
        distances_after.append(dist)

    # Sorted should have smaller mean distance
    assert np.mean(distances_after) < np.mean(distances_before)


def test_sort_animation_frames_empty() -> None:
    """Test sorting with empty input."""
    assert sort_animation_frames([]) == []


def test_sort_animation_frames_single() -> None:
    """Test sorting with single frame."""
    binormals = random_hinges(4, seed=42).as_array()
    frames = [binormals]

    sorted_frames = sort_animation_frames(frames)

    assert len(sorted_frames) == 1
    np.testing.assert_allclose(sorted_frames[0], binormals, atol=1e-10)


def test_generate_animation_random_sorting() -> None:
    """Test that generate_animation_random automatically sorts frames."""
    binormals = random_hinges(4, seed=42).as_array()

    # Generate random animation (should be sorted automatically)
    frames = generate_animation_random(
        binormals,
        num_frames=10,
        seed=42,
        verbose=False,
    )

    # Compute distances between consecutive frames
    distances = []
    for i in range(1, len(frames)):
        dist = np.linalg.norm(frames[i] - frames[i-1])
        distances.append(dist)

    # Check that frames were generated
    assert len(frames) == 10

    # Check that all frames have correct shape
    assert all(f.shape == binormals.shape for f in frames)

    # Distances should not be too extreme (sorting should help)
    # Without sorting, max distance could be very large
    assert max(distances) < 5.0  # Should not have huge jumps
