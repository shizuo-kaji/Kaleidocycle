"""Tests for unified generate_animation with all rule modes."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import (
    Kaleidocycle,
    generate_animation,
    ConstraintConfig,
    constraint_penalty,
)


def test_generate_animation_sine_gordon():
    """Test generate_animation with sine-Gordon rule."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=10,
        step_size=0.02,
        rule="sine-Gordon",
        oriented=kc.oriented,
    )

    assert len(frames) == 10
    assert all(f.shape == (7, 3) for f in frames)

    # Check unit length preservation
    for frame in frames:
        norms = np.linalg.norm(frame, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


def test_generate_animation_mkdv():
    """Test generate_animation with mKdV rule."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=10,
        step_size=0.02,
        rule="mKdV",
        oriented=kc.oriented,
    )

    assert len(frames) == 10
    assert all(f.shape == (7, 3) for f in frames)


def test_generate_animation_step():
    """Test generate_animation with step rule."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=5,
        step_size=0.05,
        rule="step",
    )

    assert len(frames) == 5
    assert all(f.shape == (7, 3) for f in frames)

    # Check all frames maintain unit length
    for frame in frames:
        norms = np.linalg.norm(frame, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


def test_generate_animation_step_with_config():
    """Test generate_animation step mode with custom config."""
    kc = Kaleidocycle(6, seed=42)
    config = ConstraintConfig(
        oriented=kc.oriented,
        constant_torsion=True,
        enforce_anchors=False,
    )

    frames = generate_animation(
        kc.hinges,
        num_frames=5,
        step_size=0.05,
        rule="step",
        config=config,
    )

    assert len(frames) == 5

    # Check constraint satisfaction improves
    penalties = [constraint_penalty(f, config) for f in frames]
    # All should be reasonably low
    assert all(p < 0.1 for p in penalties)


def test_generate_animation_random():
    """Test generate_animation with random rule."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=5,
        rule="random",
        seed=42,
    )

    assert len(frames) == 5
    assert all(f.shape == (7, 3) for f in frames)

    # Check diversity (frames should be different)
    for i in range(len(frames) - 1):
        dist = np.linalg.norm(frames[i] - frames[i + 1])
        assert dist > 0.01  # Should be noticeably different


def test_generate_animation_random_with_objective():
    """Test generate_animation random mode with custom objective."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=5,
        rule="random",
        objective="bending",
        seed=42,
    )

    assert len(frames) == 5
    assert all(f.shape == (7, 3) for f in frames)


def test_generate_animation_random_reproducible():
    """Test that random mode is reproducible with seed."""
    kc = Kaleidocycle(6, seed=42)

    frames1 = generate_animation(
        kc.hinges,
        num_frames=5,
        rule="random",
        seed=42,
    )

    frames2 = generate_animation(
        kc.hinges,
        num_frames=5,
        rule="random",
        seed=42,
    )

    # Should produce identical results
    for f1, f2 in zip(frames1, frames2):
        np.testing.assert_array_almost_equal(f1, f2, decimal=5)


def test_generate_animation_random_different_seeds():
    """Test that random mode produces different results with different seeds."""
    kc = Kaleidocycle(6, seed=42)

    frames1 = generate_animation(
        kc.hinges,
        num_frames=5,
        rule="random",
        seed=42,
    )

    frames2 = generate_animation(
        kc.hinges,
        num_frames=5,
        rule="random",
        seed=123,
    )

    # Should produce different results
    assert not np.allclose(frames1[1], frames2[1])


def test_generate_animation_step_verbose(capsys):
    """Test generate_animation step mode with verbose output."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=3,
        step_size=0.05,
        rule="step",
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Generating 3 frames" in captured.out
    assert len(frames) == 3


def test_generate_animation_random_verbose(capsys):
    """Test generate_animation random mode with verbose output."""
    kc = Kaleidocycle(6, seed=42)

    frames = generate_animation(
        kc.hinges,
        num_frames=3,
        rule="random",
        seed=42,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Generating 3 random frames" in captured.out
    assert len(frames) == 3


def test_generate_animation_all_modes_produce_frames():
    """Test that all modes successfully produce frames."""
    kc = Kaleidocycle(6, seed=42)

    modes = ["sine-Gordon", "mKdV", "mKdV2", "step", "random"]

    for mode in modes:
        frames = generate_animation(
            kc.hinges,
            num_frames=3,
            step_size=0.05 if mode == "step" else 0.02,
            rule=mode,
            seed=42 if mode == "random" else None,
        )

        assert len(frames) == 3, f"Mode {mode} failed"
        assert all(f.shape == (7, 3) for f in frames), f"Mode {mode} has wrong shape"


def test_generate_animation_step_maintains_constraints():
    """Test that step mode maintains constraint satisfaction."""
    kc = Kaleidocycle(6, seed=42)
    config = ConstraintConfig(
        oriented=kc.oriented,
        constant_torsion=True,
    )

    frames = generate_animation(
        kc.hinges,
        num_frames=5,
        step_size=0.05,
        rule="step",
        config=config,
    )

    # All frames should satisfy constraints reasonably well
    for i, frame in enumerate(frames):
        penalty = constraint_penalty(frame, config)
        assert penalty < 0.1, f"Frame {i} has high penalty: {penalty}"


def test_generate_animation_preserves_initial_frame():
    """Test that all modes preserve the initial frame."""
    kc = Kaleidocycle(6, seed=42)

    for mode in ["sine-Gordon", "step", "random"]:
        frames = generate_animation(
            kc.hinges,
            num_frames=3,
            step_size=0.05 if mode == "step" else 0.02,
            rule=mode,
            seed=42 if mode == "random" else None,
        )

        # First frame should be the input (or very close for random)
        if mode != "random":
            np.testing.assert_array_almost_equal(
                frames[0], kc.hinges, decimal=10,
                err_msg=f"Mode {mode} didn't preserve initial frame"
            )
