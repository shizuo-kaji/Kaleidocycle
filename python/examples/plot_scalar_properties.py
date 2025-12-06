"""Example: Plotting Scalar Properties of Kaleidocycle Animations

This example demonstrates how to:
1. Generate kaleidocycle animations using different evolution rules
2. Compute scalar properties (bending energy, mean torsion, mean curvature)
3. Visualize how these properties evolve over time

Requirements:
    - kaleidocycle package
    - matplotlib

Usage:
    python examples/plot_scalar_properties.py
"""

import matplotlib.pyplot as plt
import numpy as np

from kaleidocycle import (
    KaleidocycleAnimation,
    generate_animation,
    random_hinges,
)


def plot_scalar_property_evolution(anim: KaleidocycleAnimation, property_names: list[str]) -> None:
    """Plot the evolution of scalar properties over animation frames.

    Parameters
    ----------
    anim : KaleidocycleAnimation
        Animation object with computed scalar properties
    property_names : list[str]
        Names of scalar properties to plot
    """
    n_props = len(property_names)
    fig, axes = plt.subplots(n_props, 1, figsize=(10, 3 * n_props))

    # Handle single subplot case
    if n_props == 1:
        axes = [axes]

    for ax, prop_name in zip(axes, property_names):
        if prop_name not in anim.scalar_properties:
            print(f"Warning: Property '{prop_name}' not found in animation")
            continue

        values = anim.scalar_properties[prop_name]
        frames = np.arange(len(values))

        ax.plot(frames, values, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel(prop_name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{prop_name.replace("_", " ").title()} Evolution', fontsize=14)

    plt.tight_layout()
    return fig


def example_sine_gordon_evolution():
    """Example 1: Sine-Gordon evolution with scalar properties."""
    print("=" * 60)
    print("Example 1: Sine-Gordon Evolution")
    print("=" * 60)

    # Create initial kaleidocycle
    n = 8
    hinges = random_hinges(n, seed=42, oriented=True).as_array()

    # Generate animation using sine-Gordon flow
    frames = generate_animation(
        hinges,
        num_frames=50,
        step_size=0.02,
        rule="sine-Gordon",
        oriented=True,
    )

    # Create animation object
    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="sine-Gordon",
    )

    print(f"Generated {anim.n_frames} frames with {anim.n_vertices} vertices each")

    # Compute scalar properties
    print("\nComputing scalar properties...")
    anim.compute_scalar_property("bending_energy")
    anim.compute_scalar_property("mean_torsion")
    anim.compute_scalar_property("mean_curvature")

    print(f"Scalar properties computed: {list(anim.scalar_properties.keys())}")

    # Plot evolution
    fig = plot_scalar_property_evolution(
        anim,
        ["bending_energy", "mean_torsion", "mean_curvature"]
    )
    plt.suptitle(f"Sine-Gordon Evolution (n={n}, oriented=True)", fontsize=16, y=1.00)
    plt.savefig("examples/sine_gordon_scalar_properties.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/sine_gordon_scalar_properties.png")

    return anim


def example_step_evolution():
    """Example 2: Step-based evolution with scalar properties."""
    print("\n" + "=" * 60)
    print("Example 2: Step-Based Evolution")
    print("=" * 60)

    # Create initial kaleidocycle
    n = 6
    hinges = random_hinges(n, seed=123, oriented=False).as_array()

    # Generate animation using step method
    frames = generate_animation(
        hinges,
        num_frames=30,
        step_size=0.05,
        rule="step",
        oriented=False,
        verbose=False,
    )

    # Create animation object
    anim = KaleidocycleAnimation(
        frames=frames,
        evolution_rule="step",
    )

    print(f"Generated {anim.n_frames} frames with {anim.n_vertices} vertices each")

    # Compute scalar properties
    print("\nComputing scalar properties...")
    anim.compute_scalar_property("bending_energy")
    anim.compute_scalar_property("mean_torsion")
    anim.compute_scalar_property("mean_curvature")

    # Plot evolution
    fig = plot_scalar_property_evolution(
        anim,
        ["bending_energy", "mean_torsion", "mean_curvature"]
    )
    plt.suptitle(f"Step Evolution (n={n}, oriented=False)", fontsize=16, y=1.00)
    plt.savefig("examples/step_scalar_properties.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/step_scalar_properties.png")

    return anim


def example_compare_properties():
    """Example 3: Compare properties across different kaleidocycles."""
    print("\n" + "=" * 60)
    print("Example 3: Comparing Properties Across Different n")
    print("=" * 60)

    results = {}

    for n in [6, 8, 10]:
        print(f"\nGenerating animation for n={n}...")
        hinges = random_hinges(n, seed=42, oriented=True).as_array()

        frames = generate_animation(
            hinges,
            num_frames=30,
            step_size=0.02,
            rule="sine-Gordon",
            oriented=True,
        )

        anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine-Gordon")
        anim.compute_scalar_property("bending_energy")

        results[n] = anim.scalar_properties["bending_energy"]

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for n, energies in results.items():
        frames = np.arange(len(energies))
        ax.plot(frames, energies, linewidth=2, marker='o', markersize=3, label=f'n={n}')

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Bending Energy', fontsize=12)
    ax.set_title('Bending Energy Evolution: Comparison Across n', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/compare_bending_energy.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/compare_bending_energy.png")


def example_custom_property():
    """Example 4: Adding custom scalar property."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Scalar Property")
    print("=" * 60)

    # Create animation
    n = 8
    hinges = random_hinges(n, seed=42, oriented=True).as_array()
    frames = generate_animation(hinges, num_frames=30, rule="sine-Gordon")
    anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine-Gordon")

    # Define custom property: maximum curvature
    def max_curvature(hinges):
        from kaleidocycle.geometry import pairwise_curvature, binormals_to_tangents
        tangents = binormals_to_tangents(hinges, normalize=True)
        curvature = pairwise_curvature(hinges, tangents)
        return float(np.max(np.abs(curvature)))

    # Compute custom property
    print("\nComputing custom property (max curvature)...")
    anim.compute_scalar_property("max_curvature", func=max_curvature)

    # Also compute mean curvature for comparison
    anim.compute_scalar_property("mean_curvature")

    # Plot both
    fig = plot_scalar_property_evolution(
        anim,
        ["mean_curvature", "max_curvature"]
    )
    plt.suptitle(f"Custom Property: Max Curvature (n={n})", fontsize=16, y=1.00)
    plt.savefig("examples/custom_property.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/custom_property.png")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Kaleidocycle Scalar Properties Examples")
    print("=" * 60)

    # Run examples
    anim1 = example_sine_gordon_evolution()
    anim2 = example_step_evolution()
    example_compare_properties()
    example_custom_property()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - examples/sine_gordon_scalar_properties.png")
    print("  - examples/step_scalar_properties.png")
    print("  - examples/compare_bending_energy.png")
    print("  - examples/custom_property.png")

    # Optionally show plots
    # plt.show()


if __name__ == "__main__":
    main()
