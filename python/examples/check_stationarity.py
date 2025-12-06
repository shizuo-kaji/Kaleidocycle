"""Example: Checking if a Kaleidocycle is at a stationary point.

This example demonstrates how to use the `is_stationary()` method to verify
if a Kaleidocycle configuration represents a stationary point (critical point)
of a given energy function under constraints.
"""

from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig, constraint_penalty


def main():
    """Demonstrate stationarity checking for Kaleidocycles."""

    print("=" * 70)
    print("Checking Stationarity of Kaleidocycle Configurations")
    print("=" * 70)

    # Example 1: Check an optimized kaleidocycle
    print("\n" + "=" * 70)
    print("Example 1: Optimized Kaleidocycle")
    print("=" * 70)

    # Create an optimized kaleidocycle
    kc = Kaleidocycle(n=8, oriented=True, seed=42)

    # Check feasibility first
    config = ConstraintConfig(oriented=True, constant_torsion=True)
    print(f"\nConfiguration: n=8, oriented=True")
    print(f"Constraint penalty: {constraint_penalty(kc.hinges, config):.2e}")
    print(f"Is feasible: {kc.is_feasible(config=config)}")

    # Check stationarity for bending energy
    result = kc.is_stationary('bending', tolerance=1e-4, config=config)

    print(f"\nStationarity check for 'bending' energy:")
    print(f"  Is stationary (tol=1e-4): {result['is_stationary']}")
    print(f"  Projected gradient norm: {result['projected_gradient_norm']:.6e}")
    print(f"  Full gradient norm: {result['gradient_norm']:.6e}")
    print(f"  Constraint penalty: {result['constraint_penalty']:.6e}")

    print(f"\nDiagnostic information:")
    print(f"  Energy function: {result['details']['energy']}")
    print(f"  Number of constraints: {result['details']['n_constraints']}")
    print(f"  Number of variables: {result['details']['n_variables']}")
    print(f"  Constraint rank: {result['details']['constraint_rank']}")

    # Example 2: Check mean_cos energy
    print("\n" + "=" * 70)
    print("Example 2: Different Energy Function (mean_cos)")
    print("=" * 70)

    kc2 = Kaleidocycle(n=7, oriented=False, seed=123)
    config2 = ConstraintConfig(oriented=False, constant_torsion=True)

    result2 = kc2.is_stationary('mean_cos', tolerance=1e-4, config=config2)

    print(f"\nConfiguration: n=7, oriented=False")
    print(f"Stationarity check for 'mean_cos' energy:")
    print(f"  Is stationary (tol=1e-4): {result2['is_stationary']}")
    print(f"  Projected gradient norm: {result2['projected_gradient_norm']:.6e}")
    print(f"  Full gradient norm: {result2['gradient_norm']:.6e}")

    # Example 3: Effect of tolerance
    print("\n" + "=" * 70)
    print("Example 3: Effect of Tolerance")
    print("=" * 70)

    tolerances = [1e-6, 1e-4, 1e-2, 1.0]
    print(f"\nConfiguration: n=6, oriented=True")
    print(f"Energy: bending\n")

    kc3 = Kaleidocycle(n=6, oriented=True, seed=456)
    config3 = ConstraintConfig(oriented=True, constant_torsion=True)

    print(f"{'Tolerance':<12} {'Is Stationary':<16} {'Projected Grad Norm':<20}")
    print("-" * 50)

    for tol in tolerances:
        result3 = kc3.is_stationary('bending', tolerance=tol, config=config3)
        print(f"{tol:<12.0e} {str(result3['is_stationary']):<16} "
              f"{result3['projected_gradient_norm']:<20.6e}")

    # Example 4: Custom constraint configuration
    print("\n" + "=" * 70)
    print("Example 4: Custom Constraint Configuration")
    print("=" * 70)

    kc4 = Kaleidocycle(n=6, oriented=True, seed=789)

    # Without constant torsion constraint
    config_no_torsion = ConstraintConfig(
        oriented=True,
        constant_torsion=False,
        alignment=True,
    )

    result_no_torsion = kc4.is_stationary(
        'bending', tolerance=1e-4, config=config_no_torsion
    )

    # With constant torsion constraint
    config_with_torsion = ConstraintConfig(
        oriented=True,
        constant_torsion=True,
        alignment=True,
    )

    result_with_torsion = kc4.is_stationary(
        'bending', tolerance=1e-4, config=config_with_torsion
    )

    print(f"\nConfiguration: n=6, oriented=True")
    print(f"Energy: bending\n")

    print(f"Without constant torsion constraint:")
    print(f"  Number of constraints: {result_no_torsion['details']['n_constraints']}")
    print(f"  Projected gradient norm: {result_no_torsion['projected_gradient_norm']:.6e}")

    print(f"\nWith constant torsion constraint:")
    print(f"  Number of constraints: {result_with_torsion['details']['n_constraints']}")
    print(f"  Projected gradient norm: {result_with_torsion['projected_gradient_norm']:.6e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The `is_stationary()` method checks whether a Kaleidocycle configuration
is at a stationary point (critical point) of an energy function under
constraints. Key features:

1. Supports multiple energy functions:
   - 'bending': Bobenko-Suris bending energy
   - 'mean_cos': Mean cosine (torsion)

2. Uses numerical differentiation (finite differences) to compute gradients

3. Projects energy gradient onto constraint tangent space using nullspace
   projection

4. Returns detailed diagnostic information including:
   - Whether configuration is stationary (within tolerance)
   - Projected gradient norm
   - Full gradient norm
   - Constraint penalty
   - Constraint rank and dimensions

5. Configurable tolerance and constraint settings

Note: Due to numerical optimization precision, even "optimized" Kaleidocycles
may not be perfect stationary points. The projected gradient norm provides
a quantitative measure of proximity to stationarity.
    """)


if __name__ == "__main__":
    main()
