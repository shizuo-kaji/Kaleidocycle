"""Tests for KaleidocycleAnimation class."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle import (
    KaleidocycleAnimation,
    random_hinges,
    generate_animation,
    pairwise_curvature,
    compute_torsion,
)


def test_animation_creation_basic():
    """Test basic animation creation."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(10)]

    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="random",
    )

    assert anim.n_frames == 10
    assert anim.n_vertices == 7  # 6 tetrahedra + 1 (closure)
    assert anim.frame_shape == (7, 3)
    assert anim.evolution_rule == "random"


def test_animation_with_properties():
    """Test creating animation with pre-computed properties."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]

    # Compute some properties
    curvatures = np.array([pairwise_curvature(f) for f in frames])
    energies = np.random.rand(5)

    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="step",
        vertex_properties={"curvature": curvatures},
        scalar_properties={"energy": energies},
    )

    assert "curvature" in anim.vertex_properties
    assert "energy" in anim.scalar_properties
    assert anim.vertex_properties["curvature"].shape == (5, 6)
    assert anim.scalar_properties["energy"].shape == (5,)


def test_animation_empty_frames():
    """Test that empty frames list raises error."""
    with pytest.raises(ValueError, match="cannot be empty"):
        KaleidocycleAnimation(frames=[])


def test_animation_inconsistent_shapes():
    """Test that inconsistent frame shapes raise error."""
    frames = [
        random_hinges(6, seed=0).as_array(),
        random_hinges(8, seed=1).as_array(),  # Different size
    ]

    with pytest.raises(ValueError, match="same shape"):
        KaleidocycleAnimation(frames=frames)


def test_animation_invalid_vertex_property():
    """Test that invalid vertex property raises error."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]

    # Wrong number of frames
    bad_prop = np.random.rand(3, 7)  # Should be (5, 7)

    with pytest.raises(ValueError, match="has 3 frames"):
        KaleidocycleAnimation(
            frames=frames,
            vertex_properties={"bad": bad_prop},
        )


def test_animation_invalid_scalar_property():
    """Test that invalid scalar property raises error."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]

    # Wrong number of values
    bad_prop = np.random.rand(3)  # Should be (5,)

    with pytest.raises(ValueError, match="has 3 values"):
        KaleidocycleAnimation(
            frames=frames,
            scalar_properties={"bad": bad_prop},
        )


def test_add_vertex_property():
    """Test adding vertex property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    # Add property
    curvatures = np.array([pairwise_curvature(f) for f in frames])
    anim.add_vertex_property("curvature", curvatures)

    assert "curvature" in anim.vertex_properties
    assert anim.vertex_properties["curvature"].shape == (5, 6)


def test_add_vertex_property_overwrite():
    """Test that adding existing property requires overwrite=True."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    prop = np.random.rand(5, 7)
    anim.add_vertex_property("test", prop)

    # Should fail without overwrite
    with pytest.raises(ValueError, match="already exists"):
        anim.add_vertex_property("test", prop)

    # Should succeed with overwrite
    anim.add_vertex_property("test", prop, overwrite=True)


def test_add_scalar_property():
    """Test adding scalar property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    energies = np.random.rand(5)
    anim.add_scalar_property("energy", energies)

    assert "energy" in anim.scalar_properties
    assert anim.scalar_properties["energy"].shape == (5,)


def test_compute_vertex_property_curvature():
    """Test computing curvature property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    curvatures = anim.compute_vertex_property("curvature")

    assert "curvature" in anim.vertex_properties
    assert curvatures.shape == (5, 6)
    np.testing.assert_array_equal(curvatures, anim.vertex_properties["curvature"])


def test_compute_vertex_property_torsion():
    """Test computing torsion property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    torsions = anim.compute_vertex_property("torsion")

    assert "torsion" in anim.vertex_properties
    assert torsions.shape == (5, 6)


def test_compute_vertex_property_dot_products():
    """Test computing dot products property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    dots = anim.compute_vertex_property("dot_products")

    assert "dot_products" in anim.vertex_properties
    assert dots.shape == (5, 6)  # n-1 pairwise dot products


def test_compute_vertex_property_invalid():
    """Test that invalid property name raises error."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    with pytest.raises(ValueError, match="unknown property"):
        anim.compute_vertex_property("invalid")


def test_compute_vertex_property_no_overwrite():
    """Test that compute doesn't overwrite by default."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    # Compute once
    curvatures1 = anim.compute_vertex_property("curvature")

    # Should return cached value
    curvatures2 = anim.compute_vertex_property("curvature")

    np.testing.assert_array_equal(curvatures1, curvatures2)


def test_compute_scalar_property_with_func():
    """Test computing scalar property with custom function."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    # Custom function: count hinges
    energies = anim.compute_scalar_property(
        "energy",
        lambda h: float(np.sum(h**2)),
    )

    assert "energy" in anim.scalar_properties
    assert energies.shape == (5,)


def test_compute_scalar_property_penalty():
    """Test computing built-in penalty property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    penalties = anim.compute_scalar_property("penalty")

    assert "penalty" in anim.scalar_properties
    assert penalties.shape == (5,)


def test_compute_scalar_property_linking_number():
    """Test computing built-in linking number property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    lks = anim.compute_scalar_property("linking_number")

    assert "linking_number" in anim.scalar_properties
    assert lks.shape == (5,)


def test_compute_scalar_property_bending_energy():
    """Test computing built-in bending energy property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    energies = anim.compute_scalar_property("bending_energy")

    assert "bending_energy" in anim.scalar_properties
    assert energies.shape == (5,)
    assert np.all(energies >= 0)  # Energy should be non-negative

    # Test alias
    energies2 = anim.compute_scalar_property("bending", overwrite=True)
    assert "bending" in anim.scalar_properties
    assert energies2.shape == (5,)


def test_compute_scalar_property_mean_torsion():
    """Test computing built-in mean torsion property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    mean_torsions = anim.compute_scalar_property("mean_torsion")

    assert "mean_torsion" in anim.scalar_properties
    assert mean_torsions.shape == (5,)
    assert np.all(mean_torsions >= 0)  # Torsion angles are non-negative


def test_compute_scalar_property_mean_curvature():
    """Test computing built-in mean curvature property."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    mean_curvatures = anim.compute_scalar_property("mean_curvature")

    assert "mean_curvature" in anim.scalar_properties
    assert mean_curvatures.shape == (5,)
    # Mean curvature can be positive or negative (signed angles)


def test_compute_scalar_property_invalid_builtin():
    """Test that invalid built-in property name raises error."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    with pytest.raises(ValueError, match="no built-in computation"):
        anim.compute_scalar_property("invalid")


def test_get_frame():
    """Test getting individual frames."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    frame0 = anim.get_frame(0)
    frame_last = anim.get_frame(-1)

    np.testing.assert_array_equal(frame0, frames[0])
    np.testing.assert_array_equal(frame_last, frames[-1])


def test_get_curves():
    """Test computing curves for all frames."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    curves = anim.get_curves()

    assert len(curves) == 5
    assert all(curve.shape == (7, 3) for curve in curves)


def test_slice_animation():
    """Test slicing animation."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(10)]
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="test")

    # Add some properties
    anim.compute_vertex_property("curvature")
    anim.compute_scalar_property("penalty")

    # Slice
    sliced = anim.slice(2, 7)

    assert sliced.n_frames == 5
    assert len(sliced.frames) == 5
    assert "curvature" in sliced.vertex_properties
    assert "penalty" in sliced.scalar_properties
    assert sliced.vertex_properties["curvature"].shape == (5, 6)
    assert sliced.scalar_properties["penalty"].shape == (5,)


def test_slice_animation_with_step():
    """Test slicing animation with step."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(10)]
    anim = KaleidocycleAnimation(frames=frames)

    sliced = anim.slice(0, 10, 2)

    assert sliced.n_frames == 5
    np.testing.assert_array_equal(sliced.frames[0], frames[0])
    np.testing.assert_array_equal(sliced.frames[1], frames[2])


def test_len():
    """Test __len__ method."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(7)]
    anim = KaleidocycleAnimation(frames=frames)

    assert len(anim) == 7


def test_getitem_index():
    """Test __getitem__ with index."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    frame2 = anim[2]
    np.testing.assert_array_equal(frame2, frames[2])


def test_getitem_slice():
    """Test __getitem__ with slice."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    sliced_frames = anim[1:4]
    assert len(sliced_frames) == 3


def test_repr():
    """Test string representation."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="test",
    )

    anim.compute_vertex_property("curvature")
    anim.compute_scalar_property("penalty")

    repr_str = repr(anim)

    assert "KaleidocycleAnimation" in repr_str
    assert "n_frames=5" in repr_str
    assert "n_vertices=7" in repr_str
    assert "evolution_rule='test'" in repr_str
    assert "curvature" in repr_str
    assert "penalty" in repr_str


def test_metadata():
    """Test metadata storage."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]

    metadata = {
        "dt": 0.01,
        "method": "sine_gordon",
        "initial_seed": 42,
    }

    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="sine_gordon",
        metadata=metadata,
    )

    assert anim.metadata["dt"] == 0.01
    assert anim.metadata["method"] == "sine_gordon"


def test_integration_with_generate_animation():
    """Test creating animation from generate_animation."""
    initial = random_hinges(6, seed=42).as_array()

    # Generate frames using existing function
    frames = generate_animation(initial, num_frames=10, step_size=0.01)

    # Create animation object
    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="sine_gordon",
        metadata={"step_size": 0.01, "num_frames": 10},
    )

    assert anim.n_frames == 10
    assert anim.evolution_rule == "sine_gordon"

    # Compute properties
    anim.compute_vertex_property("curvature")
    anim.compute_scalar_property("penalty")

    assert "curvature" in anim.vertex_properties
    assert "penalty" in anim.scalar_properties


def test_n_property():
    """Test n property (number of tetrahedra)."""
    frames = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim = KaleidocycleAnimation(frames=frames)

    assert anim.n == 6
    assert anim.n == anim.n_vertices - 1


def test_oriented_property():
    """Test oriented property."""
    # Create oriented kaleidocycle
    frames_oriented = [random_hinges(6, seed=i, oriented=True).as_array() for i in range(5)]
    anim_oriented = KaleidocycleAnimation(frames=frames_oriented)

    # Create non-oriented kaleidocycle
    frames_non_oriented = [random_hinges(6, seed=i, oriented=False).as_array() for i in range(5)]
    anim_non_oriented = KaleidocycleAnimation(frames=frames_non_oriented)

    assert anim_oriented.oriented is True
    assert anim_non_oriented.oriented is False


def test_is_closed_property():
    """Test is_closed property."""
    # Create animation from optimized kaleidocycles (should satisfy closure)
    from kaleidocycle import Kaleidocycle

    frames_closed = [Kaleidocycle(6, seed=i).hinges for i in range(5)]
    anim_closed = KaleidocycleAnimation(frames=frames_closed)

    # Optimized kaleidocycles should satisfy closure constraint
    assert anim_closed.is_closed is True

    # Create animation from random hinges (likely won't satisfy closure)
    frames_random = [random_hinges(6, seed=i).as_array() for i in range(3)]
    anim_random = KaleidocycleAnimation(frames=frames_random)

    # Random hinges typically don't satisfy closure constraint
    # Just check that the property returns a boolean
    result = anim_random.is_closed
    assert isinstance(result, bool)


def test_is_aligned_property():
    """Test is_aligned property."""
    # Create animation from optimized kaleidocycles (should satisfy alignment)
    from kaleidocycle import Kaleidocycle

    # Oriented kaleidocycles should satisfy alignment constraint
    frames_oriented = [Kaleidocycle(6, oriented=True, seed=i).hinges for i in range(3)]
    anim_oriented = KaleidocycleAnimation(frames=frames_oriented)
    assert anim_oriented.is_aligned is True

    # Non-oriented kaleidocycles should also satisfy alignment constraint
    frames_non_oriented = [Kaleidocycle(6, oriented=False, seed=i).hinges for i in range(3)]
    anim_non_oriented = KaleidocycleAnimation(frames=frames_non_oriented)
    assert anim_non_oriented.is_aligned is True

    # Create animation from random hinges (may not satisfy alignment)
    frames_random = [random_hinges(6, seed=i).as_array() for i in range(3)]
    anim_random = KaleidocycleAnimation(frames=frames_random)

    # Random hinges may or may not satisfy alignment - just check it's a boolean
    result = anim_random.is_aligned
    assert isinstance(result, bool)


def test_is_unit_norm_property():
    """Test is_unit_norm property."""
    # Create frames with unit norm
    frames_unit = [random_hinges(6, seed=i).as_array() for i in range(5)]
    anim_unit = KaleidocycleAnimation(frames=frames_unit)

    assert anim_unit.is_unit_norm is True

    # Create frames with non-unit norm
    frames_non_unit = [random_hinges(6, seed=i).as_array() * 2.0 for i in range(5)]
    anim_non_unit = KaleidocycleAnimation(frames=frames_non_unit)

    assert anim_non_unit.is_unit_norm is False


def test_constant_torsion_property():
    """Test constant_torsion property."""
    # Create kaleidocycle with approximately constant torsion
    # (this is typical for well-constructed kaleidocycles)
    frames = [random_hinges(6, seed=42).as_array() for i in range(3)]
    anim = KaleidocycleAnimation(frames=frames)

    # The constant_torsion property returns either a float or None
    result = anim.constant_torsion
    assert result is None or isinstance(result, float)

    # If it's a float, it should be positive (torsion angles are non-negative)
    if result is not None:
        assert result >= 0
