"""Example: Compare NumPy and JAX backends.

This example demonstrates:
1. NumPy backend: scipy with finite difference gradients
2. JAX backend (penalty): scipy with JAX autodiff gradients
3. JAX backend (constrained): scipy with JAX autodiff gradients

Both backends use scipy.optimize.minimize, but:
- NumPy: Gradients computed via finite differences (approximate)
- JAX: Gradients computed via automatic differentiation (exact)
"""

from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig
from kaleidocycle.solvers import SolverOptions, optimize_cycle

# Create a kaleidocycle
kc = Kaleidocycle(n=6, oriented=True, seed=42)
config = ConstraintConfig(oriented=True, constant_torsion=True)

print("=" * 60)
print("Backend Comparison: NumPy vs JAX (both using scipy)")
print("=" * 60)
print()

# 1. NumPy backend with constrained optimization
print("1. NumPy Backend (scipy, finite differences):")
options_numpy = SolverOptions(maxiter=200, use_constraint_solver=True)
result_numpy = optimize_cycle(
    kc.hinges,
    config,
    objective='bending',
    options=options_numpy,
    backend='numpy'
)
print(f"   Backend: {result_numpy.backend_name}")
print(f"   Success: {result_numpy.success}")
print(f"   Energy: {result_numpy.energy:.6f}")
print(f"   Penalty: {result_numpy.penalty:.6e}")
print()

# 2. JAX backend with penalty method
try:
    print("2. JAX Backend (scipy + JAX autodiff, penalty method):")
    options_jax_penalty = SolverOptions(
        maxiter=200,
        penalty_weight=100.0,
        use_constraint_solver=False
    )
    result_jax_penalty = optimize_cycle(
        kc.hinges,
        config,
        objective='bending',
        options=options_jax_penalty,
        backend='jax'
    )
    print(f"   Backend: {result_jax_penalty.backend_name}")
    print(f"   Success: {result_jax_penalty.success}")
    print(f"   Energy: {result_jax_penalty.energy:.6f}")
    print(f"   Penalty: {result_jax_penalty.penalty:.6e}")
    print()

    # 3. JAX backend with constrained optimization
    print("3. JAX Backend (scipy + JAX autodiff, constrained):")
    options_jax_constrained = SolverOptions(
        maxiter=200,
        use_constraint_solver=True
    )
    result_jax_constrained = optimize_cycle(
        kc.hinges,
        config,
        objective='bending',
        options=options_jax_constrained,
        backend='jax'
    )
    print(f"   Backend: {result_jax_constrained.backend_name}")
    print(f"   Success: {result_jax_constrained.success}")
    print(f"   Energy: {result_jax_constrained.energy:.6f}")
    print(f"   Penalty: {result_jax_constrained.penalty:.6e}")
    print()

    # Compare results
    print("=" * 60)
    print("Comparison:")
    print("=" * 60)
    print(f"NumPy satisfies constraints: {result_numpy.penalty < 1e-3}")
    print(f"JAX penalty method satisfies constraints: {result_jax_penalty.penalty < 1e-2}")
    print(f"JAX constrained satisfies constraints: {result_jax_constrained.penalty < 1e-2}")
    print()
    print(f"Energy (NumPy): {result_numpy.energy:.6f}")
    print(f"Energy (JAX penalty): {result_jax_penalty.energy:.6f}")
    print(f"Energy (JAX constrained): {result_jax_constrained.energy:.6f}")

except ImportError as e:
    print(f"  JAX not available: {e}")
    print("  Install with: pip install kaleidocycle[jax]")
