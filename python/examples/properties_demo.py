"""Demo: Kaleidocycle and KaleidocycleAnimation Properties

This example demonstrates the new properties added to both Kaleidocycle
and KaleidocycleAnimation classes:
- n: number of tetrahedra
- oriented: whether the kaleidocycle is oriented
- is_closed: whether closure constraint is satisfied (sum of tangents = 0)
- is_aligned: whether alignment constraint is satisfied (first/last hinge match)
- is_unit_norm: whether all vectors have unit norm
- constant_torsion: constant torsion value (or None if not constant)

For animations, is_closed and is_aligned check if ALL frames satisfy the constraints.

Usage:
    python examples/properties_demo.py
"""

from kaleidocycle import (
    Kaleidocycle,
    KaleidocycleAnimation,
    generate_animation,
    random_hinges,
)


def demo_kaleidocycle_properties():
    """Demo properties of a Kaleidocycle."""
    print("=" * 60)
    print("Kaleidocycle Properties Demo")
    print("=" * 60)

    # Create oriented kaleidocycle
    print("\n1. Creating oriented kaleidocycle with n=8 tetrahedra...")
    kc_oriented = Kaleidocycle(8, oriented=True, seed=42)

    print(f"   n (number of tetrahedra): {kc_oriented.n}")
    print(f"   oriented: {kc_oriented.oriented}")
    print(f"   is_closed: {kc_oriented.is_closed}")
    print(f"   is_aligned: {kc_oriented.is_aligned}")
    print(f"   is_unit_norm: {kc_oriented.is_unit_norm}")
    print(f"   constant_torsion: {kc_oriented.constant_torsion}")

    # Create non-oriented kaleidocycle
    print("\n2. Creating non-oriented kaleidocycle with n=7 tetrahedra...")
    kc_non_oriented = Kaleidocycle(7, oriented=False, seed=42)

    print(f"   n: {kc_non_oriented.n}")
    print(f"   oriented: {kc_non_oriented.oriented}")
    print(f"   is_closed: {kc_non_oriented.is_closed}")
    print(f"   is_aligned: {kc_non_oriented.is_aligned}")
    print(f"   is_unit_norm: {kc_non_oriented.is_unit_norm}")
    print(f"   constant_torsion: {kc_non_oriented.constant_torsion}")

    # Create from random hinges (not optimized)
    print("\n3. Creating kaleidocycle from random hinges (not optimized)...")
    hinges = random_hinges(6, seed=123).as_array()
    kc_random = Kaleidocycle(hinges=hinges)

    print(f"   n: {kc_random.n}")
    print(f"   oriented: {kc_random.oriented}")
    print(f"   is_closed: {kc_random.is_closed}")
    print(f"   is_aligned: {kc_random.is_aligned}")
    print(f"   is_unit_norm: {kc_random.is_unit_norm}")
    print(f"   constant_torsion: {kc_random.constant_torsion}")


def demo_animation_properties():
    """Demo properties of a KaleidocycleAnimation."""
    print("\n" + "=" * 60)
    print("KaleidocycleAnimation Properties Demo")
    print("=" * 60)

    # Create animation using sine-Gordon flow
    print("\n1. Creating animation with sine-Gordon flow...")
    hinges = random_hinges(8, seed=42, oriented=True).as_array()
    frames = generate_animation(
        hinges,
        num_frames=20,
        step_size=0.02,
        rule="sine-Gordon",
        oriented=True,
    )
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine-Gordon")

    print(f"   n (number of tetrahedra): {anim.n}")
    print(f"   n_frames: {anim.n_frames}")
    print(f"   oriented: {anim.oriented}")
    print(f"   is_closed (first frame = last frame): {anim.is_closed}")
    print(f"   is_aligned (consecutive frames aligned): {anim.is_aligned}")
    print(f"   is_unit_norm: {anim.is_unit_norm}")
    print(f"   constant_torsion: {anim.constant_torsion}")

    # Create closed animation (loop)
    print("\n2. Creating closed animation loop...")
    frames_loop = [random_hinges(6, seed=42).as_array() for i in range(10)]
    frames_loop.append(frames_loop[0].copy())  # Close the loop
    anim_closed = KaleidocycleAnimation(frames=frames_loop, evolution_rule="manual")

    print(f"   n: {anim_closed.n}")
    print(f"   n_frames: {anim_closed.n_frames}")
    print(f"   oriented: {anim_closed.oriented}")
    print(f"   is_closed: {anim_closed.is_closed}")  # Should be True
    print(f"   is_aligned: {anim_closed.is_aligned}")
    print(f"   is_unit_norm: {anim_closed.is_unit_norm}")


def demo_property_usage():
    """Demo using properties for validation and analysis."""
    print("\n" + "=" * 60)
    print("Using Properties for Validation")
    print("=" * 60)

    # Create kaleidocycle
    kc = Kaleidocycle(6, seed=42)

    print("\n1. Validating kaleidocycle quality:")
    print(f"   ✓ Has {kc.n} tetrahedra")
    print(f"   ✓ Is {'oriented' if kc.oriented else 'non-oriented'}")
    print(f"   ✓ Closure constraint satisfied: {kc.is_closed}")
    print(f"   ✓ Alignment constraint satisfied: {kc.is_aligned}")
    print(f"   ✓ Has unit norm vectors: {kc.is_unit_norm}")

    if kc.constant_torsion is not None:
        print(f"   ✓ Has constant torsion: {kc.constant_torsion:.4f} radians")
    else:
        print(f"   ✗ Does not have constant torsion")

    # Show constraint residuals
    from kaleidocycle.constraints import closure_residual, alignment_residuals
    import numpy as np

    closure_res = closure_residual(kc.hinges, slide=0.0)
    alignment_res = alignment_residuals(kc.hinges, oriented=kc.oriented)

    print(f"\n   Constraint residuals:")
    print(f"   - Closure residual norm: {np.linalg.norm(closure_res):.6e}")
    print(f"   - Alignment residual: {alignment_res:.6e}")

    # Create animation
    frames = generate_animation(kc.hinges, num_frames=15, rule="sine-Gordon")
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine-Gordon")

    print("\n2. Validating animation quality:")
    print(f"   ✓ Animation has {anim.n_frames} frames")
    print(f"   ✓ All frames satisfy closure: {anim.is_closed}")
    print(f"   ✓ All frames satisfy alignment: {anim.is_aligned}")
    print(f"   ✓ All frames have unit norm: {anim.is_unit_norm}")


def main():
    """Run all demos."""
    demo_kaleidocycle_properties()
    demo_animation_properties()
    demo_property_usage()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
