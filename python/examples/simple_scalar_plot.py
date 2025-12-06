"""Simple example: Plot scalar properties of a kaleidocycle animation

This is a minimal example showing how to:
1. Create a kaleidocycle animation
2. Compute scalar properties
3. Plot them

Usage:
    python examples/simple_scalar_plot.py
"""

import matplotlib.pyplot as plt
import numpy as np

from kaleidocycle import KaleidocycleAnimation, generate_animation, random_hinges


def main():
    # Step 1: Create initial kaleidocycle
    print("Creating kaleidocycle with n=8 tetrahedra...")
    n = 8
    hinges = random_hinges(n, seed=42, oriented=True).as_array()

    # Step 2: Generate animation using sine-Gordon evolution
    print(f"Generating animation with 30 frames...")
    frames = generate_animation(
        hinges,
        num_frames=30,
        step_size=0.02,
        rule="sine-Gordon",
        oriented=True,
    )

    # Step 3: Create animation object
    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="sine-Gordon",
    )
    print(f"Created animation with {anim.n_frames} frames")

    # Step 4: Compute scalar properties
    print("\nComputing scalar properties...")
    anim.compute_scalar_property("bending_energy")
    anim.compute_scalar_property("mean_torsion")
    anim.compute_scalar_property("mean_curvature")

    print(f"Computed properties: {list(anim.scalar_properties.keys())}")
    print(f"  - Bending energy range: [{np.min(anim.scalar_properties['bending_energy']):.4f}, "
          f"{np.max(anim.scalar_properties['bending_energy']):.4f}]")
    print(f"  - Mean torsion range: [{np.min(anim.scalar_properties['mean_torsion']):.4f}, "
          f"{np.max(anim.scalar_properties['mean_torsion']):.4f}]")
    print(f"  - Mean curvature range: [{np.min(anim.scalar_properties['mean_curvature']):.4f}, "
          f"{np.max(anim.scalar_properties['mean_curvature']):.4f}]")

    # Step 5: Plot the properties
    print("\nCreating plot...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Plot bending energy
    axes[0].plot(anim.scalar_properties["bending_energy"], 'b-o', linewidth=2, markersize=4)
    axes[0].set_ylabel("Bending Energy", fontsize=12)
    axes[0].set_title("Scalar Properties Evolution", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot mean torsion
    axes[1].plot(anim.scalar_properties["mean_torsion"], 'r-o', linewidth=2, markersize=4)
    axes[1].set_ylabel("Mean Torsion", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Plot mean curvature
    axes[2].plot(anim.scalar_properties["mean_curvature"], 'g-o', linewidth=2, markersize=4)
    axes[2].set_xlabel("Frame", fontsize=12)
    axes[2].set_ylabel("Mean Curvature", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/simple_scalar_plot.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/simple_scalar_plot.png")

    # Optionally show the plot
    # plt.show()


if __name__ == "__main__":
    main()
