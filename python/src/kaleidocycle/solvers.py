"""Penalty-based optimization wrappers for Kaleidocycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from .constraints import ConstraintConfig, constraint_penalty, enforce_terminal, constant_torsion_residuals, closure_residual, curvature_recursion_residuals
from .energies import bending_energy, dipole_energy, torsion_energy
from .geometry import (
    tangents_to_curve,
    binormals_to_tangents,
    mean_cosine,
    writhe,
    total_twist,
    compute_linking_number,
)

import warnings


ObjectiveFunc = Callable[[NDArray[np.float64]], float]


def _flatten(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    return hinges.ravel()


def _reshape(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=float)
    if arr.size % 3 != 0:
        raise ValueError("flat hinge vector must be divisible by 3")
    return arr.reshape(-1, 3)


def _get_objective(name: str, target=0, oriented=True) -> ObjectiveFunc:
    if name == "bending":
        return lambda hinges: bending_energy(np.cross(hinges[:-1], hinges[1:]))
    if name == "torsion":
        return lambda hinges: torsion_energy(hinges, wrap=True)
    if name == "dipole":
        return lambda hinges: dipole_energy(hinges, tangents_to_curve(binormals_to_tangents(hinges, normalize=False)))
    if name == "mean_cos":
        return lambda hinges: mean_cosine(hinges, wrap=False)
    if name == "neg_mean_cos":
        return lambda hinges: -mean_cosine(hinges, wrap=False)
    if name == "target_mean_cos":
        return lambda hinges: (mean_cosine(hinges, wrap=False) - target) ** 2
    if name == "curvature_recursion":
        return lambda hinges: (curvature_recursion_residuals(hinges, oriented=oriented)**2).sum()
    raise ValueError(f"unknown objective '{name}'")


@dataclass
class SolverOptions:
    penalty_weight: float = 100.0
    method: str = "BFGS"
    maxiter: int = 500
    use_constraint_solver: bool = True
    constraint_method: str = "trust-constr"  # Method to use when use_constraint_solver=True (trust-constr or SLSQP)


@dataclass
class OptimizationSummary:
    hinges: NDArray[np.float64]
    energy: float
    penalty: float
    _scipy_result: Optional[OptimizeResult] = None

    @property
    def result(self) -> OptimizeResult | None:
        """Underlying SciPy optimization result."""

        return self._scipy_result

    @property
    def backend_name(self) -> str:
        """Name of the backend used for optimization."""

        return "scipy" if self._scipy_result is not None else "unknown"

    @property
    def success(self) -> bool:
        return bool(
            self._scipy_result is not None and self._scipy_result.success
        )


def _build_constraint_dicts(config: ConstraintConfig) -> list[dict]:
    """Build scipy constraint dictionaries from ConstraintConfig.

    Args:
        config: Constraint configuration

    Returns:
        List of constraint dictionaries for scipy.optimize.minimize
    """
    from .constraints import (
        closure_residual,
        anchor_residuals,
        constant_torsion_residuals,
    )

    constraints = []

    # Unit norm constraint: all hinges except the last should be unit vectors
    # (last hinge is determined by first via alignment constraint)
    def unit_norm_constraint(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        hinges = _reshape(flat)
        norms_sq = np.sum(hinges[:-1] ** 2, axis=1)
        return norms_sq - 1.0

    constraints.append({
        "type": "eq",
        "fun": unit_norm_constraint,
    })

    # Closure constraint: sum of tangents should be zero
    if config.closure:
        def closure_constraint(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            hinges = _reshape(flat)
            # Don't call enforce_terminal here - let alignment constraint handle it
            return closure_residual(hinges, slide=config.slide)

        constraints.append({
            "type": "eq",
            "fun": closure_constraint,
        })

    # Alignment constraint: first and last hinge should match (or oppose)
    # This is essential for periodicity
    if config.alignment:
        def alignment_constraint(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            hinges = _reshape(flat)
            if config.oriented:
                # h[0] = h[-1]
                return hinges[0] - hinges[-1]
            else:
                # h[0] = -h[-1]
                return hinges[0] + hinges[-1]

        constraints.append({
            "type": "eq",
            "fun": alignment_constraint,
        })

    # Anchor constraints: fix first hinge and part of second
    if config.enforce_anchors:
        def anchor_constraint(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            hinges = _reshape(flat)
            return anchor_residuals(hinges)

        constraints.append({
            "type": "eq",
            "fun": anchor_constraint,
        })

    # Constant torsion constraint: all dot products should be equal
    if config.constant_torsion:
        def constant_torsion_constraint(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            hinges = _reshape(flat)
            return constant_torsion_residuals(hinges, reference=config.reference_torsion)

        constraints.append({
            "type": "eq",
            "fun": constant_torsion_constraint,
        })

    return constraints


def _optimize_cycle_jax_scipy(
    initial_hinges: NDArray[np.float64],
    config: ConstraintConfig,
    objective_name: str | ObjectiveFunc,
    objective_fn: ObjectiveFunc,
    options: SolverOptions,
) -> OptimizationSummary:
    """Optimize using scipy with JAX autodiff for gradients/Hessians.

    Args:
        initial_hinges: Initial hinge configuration, shape (N+1, 3)
        config: Constraint configuration
        objective_name: Name of objective function (for error messages)
        objective_fn: NumPy objective function
        options: Solver options

    Returns:
        OptimizationSummary with scipy results but JAX-computed gradients
    """
    import jax
    import jax.numpy as jnp

    # Define JAX-compatible objective function
    if isinstance(objective_name, str):
        import math
        LOG2 = math.log(2.0)

        if objective_name == 'bending':
            def jax_objective_fn(h):
                T = jnp.cross(h[:-1], h[1:])
                norms = jnp.linalg.norm(T, axis=1, keepdims=True)
                T = T / norms
                a = T
                b = jnp.roll(T, -1, axis=0)
                norms_a = jnp.linalg.norm(a, axis=1)
                norms_b = jnp.linalg.norm(b, axis=1)
                dots = jnp.einsum("ij,ij->i", a, b)
                ratios = jnp.clip(dots / (norms_a * norms_b), -1.0 + 1e-15, 1.0)
                return jnp.sum(LOG2 - jnp.log1p(ratios))
        elif objective_name == 'mean_cos':
            def jax_objective_fn(h):
                a = h[:-1]
                b = h[1:]
                #norms = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1)
                dots = jnp.einsum("ij,ij->i", a, b)
                #cosines = jnp.clip(dots / norms, -1.0, 1.0)
                return jnp.mean(dots)
        elif objective_name == 'neg_mean_cos':
            def jax_objective_fn(h):
                a = h[:-1]
                b = h[1:]
                #norms = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1)
                dots = jnp.einsum("ij,ij->i", a, b)
                #cosines = jnp.clip(dots / norms, -1.0, 1.0)
                return -jnp.mean(dots)
        elif objective_name == 'curvature_recursion':
            def jax_binormals_to_tangents(hinges, normalize=True):
                T = jnp.cross(hinges[:-1], hinges[1:])
                if normalize:
                    norms = jnp.linalg.norm(T, axis=1, keepdims=True)
                    T = T / norms
                return T

            def jax_pairwise_curvature(binormals, tangents, signed=True):
                # Replicate geometry.pairwise_curvature
                B = binormals
                T = tangents
                T_prev = jnp.roll(T, 1, axis=0)
                B_used = B[:-1]
                cross_prod = jnp.cross(B_used, T_prev)
                dots = jnp.sum(cross_prod * T, axis=1)
                s = jnp.sign(dots)
                s = jnp.where(s == 0, 1.0, s)
                if not signed:
                    s = jnp.ones_like(s)
                cos_dots = jnp.sum(T_prev * T, axis=1)
                cos_dots = jnp.clip(cos_dots, -1.0, 1.0)
                K = s * jnp.arccos(cos_dots)
                return K

            def jax_curvature_recursion(curvatures, oriented=True):
                # Replicate geometry.curvature_recursion
                K = curvatures
                n = len(K)
                s = jnp.ones(n)
                if not oriented:
                    s = s.at[0].set(-1.0)
                    s = s.at[-1].set(-1.0)
                K_plus = jnp.roll(K, -1)
                K_minus = jnp.roll(K, 1)
                tan_i = jnp.tan(K / 2.0)
                tan_plus = jnp.tan(K_plus / 2.0)
                tan_minus = jnp.tan(K_minus / 2.0)
                term1 = s * tan_plus * tan_minus
                term2 = tan_i**2
                term3 = term1 * term2 # s * ... * tan_i^2
                result = term1 - term2 + term3
                return result

            def jax_objective_fn(h):
                tangents = jax_binormals_to_tangents(h)
                curvatures = jax_pairwise_curvature(h, tangents)
                u = jax_curvature_recursion(curvatures, oriented=config.oriented)
                residuals = u[1:] - u[0]
                return jnp.sum(residuals**2)
        else:
            raise ValueError(f"JAX backend does not support objective '{objective_name}'")
    else:
        # Custom objective function - assume it's JAX-compatible
        jax_objective_fn = objective_name

    # Define JAX-compatible constraint penalty
    def jax_constraint_penalty(h):
        """Compute constraint penalty using JAX operations."""
        penalty = 0.0

        # Unit norm penalty
        norms_sq = jnp.sum(h[:-1] ** 2, axis=1)
        penalty = penalty + jnp.sum((norms_sq - 1.0) ** 2)

        # Closure penalty
        T = jnp.cross(h[:-1], h[1:])
        ext = jnp.sum(T, axis=0)
        if config.slide != 0.0:
            ext = ext + config.slide * jnp.sum(h[:-1], axis=0)
        penalty = penalty + jnp.sum(ext ** 2)

        # Constant torsion penalty
        if config.constant_torsion:
            dot_products = jnp.sum(h[:-1] * h[1:], axis=1)
            if config.reference_torsion is None:
                torsion_residuals = dot_products - dot_products[0]
            else:
                torsion_residuals = dot_products - config.reference_torsion
            penalty = penalty + jnp.sum(torsion_residuals ** 2)

        return penalty

    if options.use_constraint_solver:
        # Use constraint-based optimization with JAX autodiff for constraint Jacobians
        def energy_func(flat: NDArray[np.float64]) -> float:
            hinges_jax = jnp.array(flat.reshape(-1, 3))
            return float(jax_objective_fn(hinges_jax))

        def energy_grad(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            hinges_jax = jnp.array(flat.reshape(-1, 3))
            grad_fn = jax.grad(lambda h: jax_objective_fn(h))
            grad_hinges = grad_fn(hinges_jax)
            return np.asarray(grad_hinges.flatten(), dtype=float)

        def energy_hess(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            hinges_jax = jnp.array(flat.reshape(-1, 3))
            hess_fn = jax.hessian(lambda h_flat: jax_objective_fn(h_flat.reshape(-1, 3)))
            hess_flat = hess_fn(hinges_jax.flatten())
            return np.asarray(hess_flat, dtype=float)

        constraints = _build_constraint_dicts(config)

        result = minimize(
            energy_func,
            _flatten(initial_hinges),
            method=options.constraint_method,
            jac=energy_grad,
            hess=energy_hess,
            constraints=constraints,
            options={"maxiter": options.maxiter, "disp": False},
        )
        final_hinges = _reshape(result.x)

    else:
        # Use penalty-based optimization with JAX autodiff
        def loss_with_penalty(h):
            """Combined loss function for penalty method."""
            energy = jax_objective_fn(h)
            penalty = jax_constraint_penalty(h)
            return energy + options.penalty_weight * penalty

        def loss(flat: NDArray[np.float64]) -> float:
            # Enforce terminal alignment before computing loss
            hinges = enforce_terminal(_reshape(flat), oriented=config.oriented)
            hinges_jax = jnp.array(hinges)
            return float(loss_with_penalty(hinges_jax))

        def loss_grad(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            # Enforce terminal alignment
            hinges = enforce_terminal(_reshape(flat), oriented=config.oriented)
            hinges_jax = jnp.array(hinges)

            # Compute gradient using JAX autodiff
            grad_fn = jax.grad(lambda h: loss_with_penalty(h))
            grad_hinges = grad_fn(hinges_jax)

            return np.asarray(grad_hinges.flatten(), dtype=float)

        def loss_hess(flat: NDArray[np.float64]) -> NDArray[np.float64]:
            # Enforce terminal alignment
            hinges = enforce_terminal(_reshape(flat), oriented=config.oriented)
            hinges_jax = jnp.array(hinges)

            # Compute Hessian using JAX autodiff
            hess_fn = jax.hessian(lambda h_flat: loss_with_penalty(h_flat.reshape(-1, 3)))
            hess_flat = hess_fn(hinges_jax.flatten())

            return np.asarray(hess_flat, dtype=float)

        # Only pass Hessian to methods that support it
        minimize_kwargs = {
            "fun": loss,
            "x0": _flatten(initial_hinges),
            "method": options.method,
            "jac": loss_grad,
            "options": {"maxiter": options.maxiter, "disp": False},
        }

        # Methods that support Hessian: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact
        if options.method in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact']:
            minimize_kwargs["hess"] = loss_hess

        result = minimize(**minimize_kwargs)
        final_hinges = enforce_terminal(_reshape(result.x), oriented=config.oriented)

    return OptimizationSummary(
        hinges=final_hinges,
        energy=objective_fn(final_hinges),
        penalty=constraint_penalty(final_hinges, config),
        _scipy_result=result,
    )


def optimize_cycle(
    initial_hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    objective: str | ObjectiveFunc = "mean_cos",
    options: SolverOptions | None = None,
    backend: Optional[str] = None,
) -> OptimizationSummary:
    """Minimize an objective with constraints.

    Args:
        initial_hinges: Initial hinge configuration, shape (N+1, 3)
        config: Constraint configuration
        objective: Objective function to minimize (energy functional name or callable)
        options: Solver options (method, penalty weight, constraint solver flag, etc.)
        backend: Backend to use ('numpy' or 'jax'). If None, uses current global backend.
                 Both backends use scipy.optimize.minimize.
                 JAX backend uses automatic differentiation for exact gradients/Hessians.
                 NumPy backend uses finite differences for gradients.

    Returns:
        OptimizationSummary with optimized configuration and diagnostics

    Note:
        NumPy backend:
            - Uses scipy.optimize.minimize
            - Gradients computed via finite differences (approximate)
            - When options.use_constraint_solver=False: penalty method with BFGS
            - When options.use_constraint_solver=True: constrained optimization with trust-constr

        JAX backend:
            - Uses scipy.optimize.minimize (same as NumPy)
            - Gradients/Hessians computed via JAX autodiff (exact, machine precision)
            - When options.use_constraint_solver=False: penalty method with BFGS
            - When options.use_constraint_solver=True: constrained optimization with trust-constr
            - Provides 10-100x faster gradient computation with exact derivatives

    Examples:
        >>> # NumPy backend (default) - finite differences
        >>> result = optimize_cycle(hinges, config, objective='bending')

        >>> # JAX backend - autodiff for exact gradients
        >>> result = optimize_cycle(hinges, config, objective='bending', backend='jax')
    """

    # warnings for inconsistent config
    n = len(initial_hinges) - 1
    if isinstance(objective, str):
        if objective == "mean_cos":
            if (config.oriented and (n % 2 == 0)) or (not config.oriented and (n % 2 == 1)):
                warnings.warn(
                    "The objective 'mean_cos' is meaningless with the current configuration "
                    "(always -1.0). Consider using a different objective.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if objective == "neg_mean_cos":
            if config.oriented:
                warnings.warn(
                    "The objective 'neg_mean_cos' is meaningless with the current configuration "
                    "(always 1.0). Consider using a different objective.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    opts = options or SolverOptions()

    # Determine objective function
    if isinstance(objective, str):
        objective_fn = _get_objective(objective)
    else:
        objective_fn = objective

    # Backend dispatch: use JAX autodiff with scipy if JAX backend requested
    from .backends import get_backend
    backend_obj = get_backend(backend)

    if backend_obj.name == 'jax':
        # Use scipy optimizer with JAX autodiff for exact gradients/Hessians
        return _optimize_cycle_jax_scipy(initial_hinges, config, objective, objective_fn, opts)

    # NumPy backend: use scipy optimization
    if opts.use_constraint_solver:
        # Use constraint-based optimization
        def energy_func(flat: NDArray[np.float64]) -> float:
            hinges = _reshape(flat)
            return objective_fn(hinges)

        constraints = _build_constraint_dicts(config)

        result = minimize(
            energy_func,
            _flatten(initial_hinges),
            method=opts.constraint_method,
            constraints=constraints,
            options={"maxiter": opts.maxiter, "disp": False},
        )
        final_hinges = _reshape(result.x)

    else:
        # Use penalty-based optimization
        def loss(flat: NDArray[np.float64]) -> float:
            hinges = enforce_terminal(_reshape(flat), oriented=config.oriented)
            energy = objective_fn(hinges)
            penalty = constraint_penalty(hinges, config)
            return float(energy + opts.penalty_weight * penalty)

        result = minimize(
            loss,
            _flatten(initial_hinges),
            method=opts.method,
            options={"maxiter": opts.maxiter, "disp": False},
        )
        final_hinges = enforce_terminal(_reshape(result.x), oriented=config.oriented)

    return OptimizationSummary(
        hinges=final_hinges,
        energy=objective_fn(final_hinges),
        penalty=constraint_penalty(final_hinges, config),
        _scipy_result=result,
    )


def optimize_with_linking_constraint(
    initial_hinges: NDArray[np.float64],
    config: ConstraintConfig,
    objective: str | ObjectiveFunc = "bending",
    options: SolverOptions | None = None,
) -> OptimizationSummary:
    """Optimize energy while constraining linking number to a target value.

    Uses scipy's constrained optimization to minimize an energy functional
    while maintaining the linking number Lk = Tw + Wr at a specified value.

    Two-fold optimization strategy:
    --------------------------------
    Phase 1: Find feasible configuration satisfying Lk and closure
        - Minimizes: constant_torsion_residuals² + linking_constraint² + closure_residuals²
        - Subject to: unit_norm, alignment (no constant torsion yet)
        - Goal: Satisfy the linking number constraint and basic closure
        - Constant torsion is minimized but not enforced as hard constraint

    Phase 2: Refine with constant torsion as hard constraint
        - Minimizes: energy_func (e.g., bending energy)
        - Subject to: unit_norm, closure, alignment, constant_torsion
        - Starting from Phase 1 solution
        - Linking constraint dropped (closure + constant torsion implies fixed Lk)
        - Optimizes actual energy functional while maintaining all constraints

    This strategy is necessary because:
    - Direct optimization with all constraints (including Lk) is often ill-conditioned
    - Phase 1 finds a topologically valid configuration
    - Phase 2 refines it to minimize physical energy

    Args:
        initial_hinges: Initial binormal configuration, shape (N+1, 3)
        target_linking: Target linking number in units of π
        config: Constraint configuration (constant torsion, closure, etc.)
        objective: Energy functional to minimize ("bending", "torsion", "dipole", or callable)
        options: Solver options (method, maxiter, etc.)
        linking_tolerance: Tolerance for linking number constraint (in units of π)

    Returns:
        OptimizationSummary with optimized configuration and diagnostics

    Example:
        >>> # Minimize bending energy while maintaining Lk = 2π
        >>> config = ConstraintConfig(oriented=True, constant_torsion=True)
        >>> result = optimize_with_linking_constraint(
        ...     initial_hinges, target_linking=2.0, config=config
        ... )
        >>> print(f"Final Lk: {compute_linking_number(result.hinges):.3f}π")

    Note:
        The linking number is a topological invariant that cannot be changed
        by continuous deformations. This function explores configurations
        with different topologies by allowing the optimization to find
        local minima that satisfy the linking constraint.
    """
    opts = options or SolverOptions()

    if isinstance(objective, str):
        objective_fn = _get_objective(objective)
    else:
        objective_fn = objective

    def energy_func(flat: NDArray[np.float64]) -> float:
        """Objective function to minimize."""
        hinges = _reshape(flat)
        return objective_fn(hinges)

    def linking_constraint(flat: NDArray[np.float64]) -> float:
        """Constraint: Lk - target_linking should be zero."""
        hinges = _reshape(flat)
        lk = compute_linking_number(hinges)
        return lk - config.target_linking

    # check consistency of orientation and linking target
    if (config.oriented and (int(config.target_linking) % 2) != 0) or (not config.oriented and (int(config.target_linking) % 2) != 1):
        warnings.warn(
            "The parity of orientation and target_linking is inconsistent. "
            "The target_linking may be unattainable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Phase 1: Build constraints WITHOUT constant torsion
    # Must have closure=True for Lk to be well-defined and integer-valued
    config_phase1 = ConstraintConfig(
        slide=config.slide,
        oriented=config.oriented,
        enforce_anchors=config.enforce_anchors,
        constant_torsion=False,  # this will be dealt with softly in objective
        closure=True,  # MUST be closed for Lk to be integer-valued
        alignment=config.alignment,
        reference_torsion=config.reference_torsion,
    )
    constraints_phase1 = _build_constraint_dicts(config_phase1)
    constraints_phase1.append({
        "type": "eq",
        "fun": linking_constraint,  # Hard constraint: Lk = target_linking
    })

    # Phase 1 objective: minimize constant torsion violations
    # (linking number and closure are enforced as hard constraints above)
    def constant_torsion_residuals_flat(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Constant torsion residuals for soft minimization in phase 1."""
        hinges = _reshape(flat)
        return constant_torsion_residuals(hinges, reference=config.reference_torsion)

    def phase1_objective(flat: NDArray[np.float64]) -> float:
        """Phase 1: Minimize constant torsion violations while satisfying Lk = target and closure."""
        torsion_error = np.sum(constant_torsion_residuals_flat(flat)**2)
        return torsion_error

    # Run Phase 1: Find feasible configuration
    # Use trust-constr for better robustness with constraints
    result_phase1 = minimize(
        phase1_objective,
        _flatten(initial_hinges),
        method="trust-constr",
        constraints=constraints_phase1,
        options={"maxiter": opts.maxiter, "verbose": 0},
    )

    intermediate_hinges = _reshape(result_phase1.x)
    intermediate_penalty = constraint_penalty(intermediate_hinges, config)

    # Check Phase 1 success
    if not result_phase1.success or intermediate_penalty > 1e-4:
        print("Phase 1 Failed: Feasibility search without hard constant torsion:")
        print(f"  Status: {result_phase1.message}")
        print(f"  Constraint penalty: {intermediate_penalty:.3e}")
        print(f"  Linking error: {linking_constraint(_flatten(intermediate_hinges)):.3e}")

    print(f"Phase 1 completed: Linking = {compute_linking_number(intermediate_hinges):.6f}π, Penalty = {intermediate_penalty:.3e}")

    # Phase 2: Build constraints WITH constant torsion
    # Use the full config for phase 2
    constraints_phase2 = _build_constraint_dicts(config)

    # Run Phase 2: Optimize energy with all constraints (including constant torsion)
    # Note: Linking constraint is dropped - closure + constant torsion implies fixed Lk
    # Use trust-constr for better constraint handling
    result_phase2 = minimize(
        energy_func,
        _flatten(intermediate_hinges),
        method="trust-constr",
        constraints=constraints_phase2,
        options={"maxiter": opts.maxiter, "verbose": 1 if opts.maxiter > 100 else 0},
    )

    final_hinges = _reshape(result_phase2.x)

    return OptimizationSummary(
        hinges=final_hinges,
        energy=objective_fn(final_hinges),
        penalty=constraint_penalty(final_hinges, config),
        _scipy_result=result_phase2,
    )


def moore_penrose_inverse(
    A: NDArray[np.float64],
    eps: float = 1e-15,
) -> NDArray[np.float64]:
    """Compute Moore-Penrose pseudoinverse using SVD with thresholding.

    Computes the pseudoinverse A^+ by inverting only singular values
    larger than eps * max(singular values), setting others to zero.

    Args:
        A: Matrix to invert, shape (m, n)
        eps: Relative threshold for singular values (default: 1e-15)

    Returns:
        Pseudoinverse A^+, shape (n, m)

    References:
        Corresponds to MPinv function in Maple code (line 121)

    Note:
        This is similar to numpy.linalg.pinv but with explicit control
        over the singular value threshold.
    """
    A_arr = np.asarray(A, dtype=float)

    # Compute SVD: A = U * S * Vt
    U, s, Vt = np.linalg.svd(A_arr, full_matrices=False)

    # Threshold singular values
    s_max = np.max(s)
    threshold = eps * s_max

    # Invert singular values above threshold
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)

    # Compute pseudoinverse: A^+ = V * S^-1 * U^T
    return Vt.T @ np.diag(s_inv) @ U.T


def optimize_multi_seed(
    n: int,
    config: ConstraintConfig,
    *,
    seeds: list[int] | int | None = None,
    n_trials: int | None = None,
    objective: str | ObjectiveFunc = "mean_cos",
    options: SolverOptions | None = None,
    backend: Optional[str] = None,
    return_dataframe: bool = False,
) -> list | tuple[list, object]:
    """Run optimization with multiple random seeds and return list of Kaleidocycle objects.

    This function runs either optimize_cycle or optimize_with_linking_constraint
    multiple times with different random initial configurations, and returns the
    results as a list of Kaleidocycle objects. Optionally returns a pandas DataFrame
    with basic properties like mean_cos, bending energy, linking number, penalty, etc.

    Args:
        n: Number of tetrahedra in the kaleidocycle
        config: Constraint configuration
        seeds: Either a list of explicit seed values or a single integer specifying
               the number of trials. If None, n_trials must be specified.
        n_trials: Number of trials to run (alternative to specifying seeds).
                  If specified, random seeds will be generated.
        objective: Objective function to minimize (default: "mean_cos")
        options: Solver options (method, penalty weight, constraint solver flag, etc.)
        backend: Backend to use ('numpy' or 'jax'). If None, uses current global backend.
        return_dataframe: If True, return a tuple (results, dataframe) where dataframe
                         contains properties of each result. Requires pandas.

    Returns:
        If return_dataframe=False:
            List of Kaleidocycle objects, one per seed
        If return_dataframe=True:
            Tuple of (list of Kaleidocycle objects, pandas DataFrame with properties)

    Raises:
        ValueError: If both seeds and n_trials are None, or if both are specified
        ValueError: If optimizer is not recognized
        ValueError: If target_linking is None when using optimize_with_linking_constraint
        ImportError: If return_dataframe=True but pandas is not installed

    Examples:
        Run optimize_cycle with 5 random trials:
        >>> from kaleidocycle import ConstraintConfig
        >>> config = ConstraintConfig(oriented=True, constant_torsion=True)
        >>> results = optimize_multi_seed(9, config, n_trials=5)
        >>> len(results)
        5

        Run with explicit seeds:
        >>> results = optimize_multi_seed(9, config, seeds=[42, 123, 456])
        >>> len(results)
        3

        Get results with dataframe:
        >>> results, df = optimize_multi_seed(
        ...     9, config, n_trials=5, return_dataframe=True
        ... )
        >>> df.columns
        Index(['seed', 'mean_cos', 'bending_energy', 'linking_number', 'penalty', ...])

    """
    from .geometry import Kaleidocycle, random_hinges

    # Validate seed/n_trials arguments
    if seeds is None and n_trials is None:
        raise ValueError("Either 'seeds' or 'n_trials' must be specified")
    if seeds is not None and n_trials is not None:
        raise ValueError("Cannot specify both 'seeds' and 'n_trials'")

    # Generate list of seeds
    if seeds is not None:
        if isinstance(seeds, int):
            # seeds is the number of trials
            seed_list = list(range(seeds))
        else:
            # seeds is an explicit list
            seed_list = list(seeds)
    else:
        # n_trials is specified
        seed_list = list(range(n_trials))

    # Run optimization for each seed
    results: list[Kaleidocycle] = []

    for seed in seed_list:
        # Generate random initial configuration
        initial_hinges = random_hinges(n, seed=seed, oriented=config.oriented).as_array()

        # Run optimization
        if config.target_linking is None:
            opt_result = optimize_cycle(
                initial_hinges,
                config,
                objective=objective,
                options=options,
                backend=backend,
            )
        else:  # optimize_with_linking_constraint
            opt_result = optimize_with_linking_constraint(
                initial_hinges,
                config=config,
                objective=objective,
                options=options,
            )

        # Create Kaleidocycle object from optimized hinges
        kc = Kaleidocycle(hinges=opt_result.hinges)

        # Store the seed and optimization info in metadata
        kc.metadata['seed'] = seed
        kc.metadata['optimization'] = {
            'objective': objective if isinstance(objective, str) else 'custom',
            'energy': opt_result.energy,
            'penalty': opt_result.penalty,
            'success': opt_result.success,
            'backend': opt_result.backend_name,
        }
        if config.target_linking is not None:
            kc.metadata['optimization']['target_linking'] = config.target_linking

        results.append(kc)

    # Optionally create dataframe
    if return_dataframe:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for return_dataframe=True. "
                "Install with: pip install pandas"
            ) from e

        # Compute properties for all results
        data_rows = []
        for i, kc in enumerate(results):
            seed = kc.metadata.get('seed', i)
            opt_info = kc.metadata.get('optimization', {})

            # Compute basic properties
            row = {
                'seed': seed,
                'mean_cos': kc.mean_cosine,
                'objective': opt_info.get('objective'),
                'success': opt_info.get('success'),
                'backend': opt_info.get('backend'),
            }

            # Compute energies
            from .energies import bending_energy, dipole_energy, torsion_energy
            row['bending_energy'] = bending_energy(kc.tangents)
            row['dipole_energy'] = dipole_energy(kc.hinges, kc.curve)
            row['torsion_energy'] = torsion_energy(kc.hinges)

            # Compute topological properties
            try:
                row['writhe'] = writhe(kc.curve)
                row['twist'] = total_twist(kc.hinges)
                row['linking_number'] = compute_linking_number(kc.hinges)
            except Exception:
                row['writhe'] = None
                row['twist'] = None
                row['linking_number'] = None

            # Compute constraint penalty
            row['penalty'] = constraint_penalty(kc.hinges, config)

            # Add constant torsion if applicable
            const_torsion = kc.constant_torsion
            row['constant_torsion'] = const_torsion if const_torsion is not None else None

            # Add optimization energy and penalty
            row['opt_energy'] = opt_info.get('energy')
            row['opt_penalty'] = opt_info.get('penalty')

            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        return results, df

    return results


def optimize_curvature_torsion(
    n: int,
    *,
    oriented: bool = True,
    seed: int | None = None,
    initial_curvatures: NDArray[np.float64] | None = None,
    initial_torsion_angle: float | None = None,
    initial_binormal: NDArray[np.float64] | None = None,
    initial_tangent: NDArray[np.float64] | None = None,
    objective: str | ObjectiveFunc = "bending",
    options: SolverOptions | None = None,
    backend: Optional[str] = None,
) -> OptimizationSummary:
    """Optimize kaleidocycle by varying curvatures and constant torsion angle.

    This solver optimizes over the space of curvatures K[i] (n values) and
    constant torsion angle τ (1 value), subject to closure and alignment
    constraints. The hinges are reconstructed via from_curvatures_and_cos at
    each iteration.

    Args:
        n: Number of tetrahedra in the kaleidocycle
        oriented: Whether the kaleidocycle is oriented
        seed: Random seed for initial values
        initial_curvatures: Initial curvature values, shape (n,).
                           If None, uses random values in [-π, π].
        initial_torsion_angle: Initial torsion.
                              If None, uses random value
        initial_binormal: Initial binormal B[0] for reconstruction, shape (3,).
                         Defaults to [0, 0, 1] if not provided.
        initial_tangent: Initial tangent T[0] for reconstruction, shape (3,).
                        Defaults to [0, 1, 0] if not provided.
        objective: Objective function to minimize (energy functional name or callable)
        options: Solver options
        backend: Backend to use ('numpy' or 'jax'). If None, uses current global backend.

    Returns:
        OptimizationSummary with optimized configuration and diagnostics

    Example:
        >>> # Minimize bending energy over curvature space
        >>> result = optimize_curvature_torsion(
        ...     n=8,
        ...     oriented=True,
        ...     objective="bending"
        ... )
        >>> kc = Kaleidocycle(hinges=result.hinges)
    """
    from .geometry import from_curvatures_and_cos

    opts = options or SolverOptions()
    if seed is not None:
        np.random.seed(seed)

    # Set initial values
    if initial_curvatures is None:
        initial_curvatures = np.random.uniform(low=-np.pi, high=np.pi, size=(n,))
    else:
        initial_curvatures = np.asarray(initial_curvatures, dtype=float)
        if initial_curvatures.shape != (n,):
            raise ValueError(f"initial_curvatures must have shape ({n},), got {initial_curvatures.shape}")

    if initial_torsion_angle is None:
        # Random angle corresponding to cos ∈ [-1, 1]
        initial_torsion_angle = np.random.uniform(low=-np.pi, high=np.pi)

    # Flatten initial guess: [curvatures (n), torsion_angle (1)]
    x0 = np.append(initial_curvatures, initial_torsion_angle)

    def unpack_variables(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Extract curvatures and torsion angle from flat variable vector."""
        curvatures = x[:n]
        torsion_angle = float(x[n])
        return curvatures, torsion_angle

    def reconstruct_frames(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reconstruct binormals and tangents from curvatures and torsion angle."""
        curvatures, torsion_angle = unpack_variables(x)
        # Convert torsion angle to cosine for from_curvatures_and_cos
        cos_torsion = np.tanh(torsion_angle)
        B, T = from_curvatures_and_cos(
            curvatures,
            cos_torsion,
            initial_binormal=initial_binormal,
            initial_tangent=initial_tangent,
        )
        return B, T

    # Define objective function that works directly with curvatures and torsion
    def energy_func(x: NDArray[np.float64]) -> float:
        """Objective function to minimize, computed directly from curvatures and torsion."""
        curvatures, torsion_angle = unpack_variables(x)
        cos_torsion = np.tanh(torsion_angle)

        if isinstance(objective, str):
            # Objectives that don't require frame reconstruction
            if objective == "mean_cos":
                # Mean cosine = constant torsion cosine for constant torsion
                return cos_torsion

            elif objective == "neg_mean_cos":
                # Negative mean cosine
                return -cos_torsion

            elif objective == "target_mean_cos":
                # Target mean cosine objective
                target = getattr(objective, 'target', 0.0)
                return (cos_torsion - target) ** 2

            elif objective == "curvature_recursion":
                # Curvature recursion residuals - computed directly from curvatures
                from .geometry import curvature_recursion
                residuals = curvature_recursion(curvatures, oriented=oriented)
                return float(np.sum(residuals**2))

            # Objectives that require frame reconstruction
            elif objective == "bending":
                # Bending energy from tangents - requires T reconstruction
                from .energies import bending_energy
                return bending_energy(curvatures)

            elif objective == "dipole":
                # Dipole energy requires curve reconstruction
                B, T = reconstruct_frames(x)
                curve = tangents_to_curve(T, center=True)
                return dipole_energy(B, curve)

            else:
                # Fallback: reconstruct hinges and use standard objective
                B,_ = reconstruct_frames(x)
                objective_fn = _get_objective(objective, oriented=oriented)
                return objective_fn(B)
        else:
            # Custom objective function: reconstruct hinges
            B,_ = reconstruct_frames(x)
            return objective(B)

    # Build constraints
    def closure_constraint(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Constraint: sum of tangents should be zero."""
        B,_ = reconstruct_frames(x)
        return closure_residual(B, slide=0.0)

    def alignment_constraint(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Constraint: B[0] = ±B[-1] depending on orientation."""
        B,_ = reconstruct_frames(x)
        if oriented:
            return B[0] - B[-1]
        else:
            return B[0] + B[-1]

    constraints = [
        {"type": "eq", "fun": closure_constraint},
        {"type": "eq", "fun": alignment_constraint},
    ]

    # Run optimization
    result = minimize(
        energy_func,
        x0,
        method=opts.constraint_method,
        constraints=constraints,
        options={"maxiter": opts.maxiter, "disp": False},
    )

    # Extract final hinges
    final_hinges = reconstruct_frames(result.x)[0]

    # Compute constraint penalty
    from .constraints import ConstraintConfig
    config = ConstraintConfig(
        oriented=oriented,
        closure=True,
        alignment=True,
        constant_torsion=False,
    )
    penalty = constraint_penalty(final_hinges, config)

    return OptimizationSummary(
        hinges=final_hinges,
        energy=energy_func(result.x),
        penalty=penalty,
        _scipy_result=result,
    )


def newton_solve(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    *,
    max_iter: int = 2000,
    tol: float = 1e-8,
    max_step_factor: float = 0.1,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], bool, int]:
    """Newton's method for solving nonlinear systems with adaptive step size.

    Solves F(x) = 0 using Newton iteration with pseudoinverse and adaptive
    step size control for robustness.

    Args:
        residual_fn: Function computing residuals F(x), returns array of shape (m,)
        jacobian_fn: Function computing Jacobian dF/dx, returns array of shape (m, n)
        x0: Initial guess, shape (n,)
        max_iter: Maximum number of iterations (default: 2000)
        tol: Convergence tolerance on max(|F(x)|) (default: 1e-8)
        max_step_factor: Maximum step size factor (default: 0.1)
        verbose: If True, print convergence information

    Returns:
        Tuple of (solution, converged, num_iterations)
            - solution: Final solution x, shape (n,)
            - converged: Whether the method converged
            - num_iterations: Number of iterations performed

    References:
        Corresponds to newton function in Maple code (line 134)

    Algorithm:
        At each iteration:
        1. Compute J = dF/dx at current x
        2. Compute pseudoinverse J^+
        3. Compute Newton step: a = J^+ * F(x)
        4. Adaptive step size: dt = min(max_step_factor / max(|a|), 1)
        5. Update: x_new = x - dt * a
        6. Check convergence: max(|F(x)|) < tol
    """
    x = np.asarray(x0, dtype=float).copy()

    for iteration in range(1, max_iter + 1):
        # Evaluate residual and Jacobian at current point
        residual = residual_fn(x)
        jacobian = jacobian_fn(x)

        # Compute pseudoinverse of Jacobian
        J_pinv = moore_penrose_inverse(jacobian)

        # Compute Newton step
        a = J_pinv @ residual

        # Adaptive step size to prevent overshooting
        a_max = np.max(np.abs(a))
        if a_max > 1e-10:
            dt = min(max_step_factor / a_max, 1.0)
        else:
            dt = 1.0

        # Update solution
        x = x - dt * a

        # Check convergence
        residual_new = residual_fn(x)
        max_residual = np.max(np.abs(residual_new))

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: max residual = {max_residual:.6e}")

        if max_residual < tol:
            if verbose:
                print(f"Newton converged in {iteration} iterations: residual = {max_residual:.6e}")
            return x, True, iteration

    # Did not converge
    max_residual = np.max(np.abs(residual_fn(x)))
    if verbose:
        warnings.warn(
            f"Newton did not converge after {max_iter} iterations: residual = {max_residual:.6e}",
            RuntimeWarning,
            stacklevel=2,
        )

    return x, False, max_iter
