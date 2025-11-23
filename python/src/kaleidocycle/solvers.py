"""Penalty-based optimization wrappers for Kaleidocycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from .constraints import ConstraintConfig, constraint_penalty, enforce_terminal, constant_torsion_residuals, closure_residual
from .energies import bending_energy, dipole_energy, torsion_energy
from .geometry import (
    tangents_to_curve,
    binormals_to_tangents,
    mean_cosine,
    writhe,
    total_twist,
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


def _get_objective(name: str, target=0) -> ObjectiveFunc:
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
    scipy_result: OptimizeResult

    @property
    def success(self) -> bool:
        return bool(self.scipy_result.success)


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
        norms = np.linalg.norm(hinges[:-1], axis=1)
        return norms - 1.0

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


def optimize_cycle(
    initial_hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    objective: str | ObjectiveFunc = "mean_cos",
    options: SolverOptions | None = None,
) -> OptimizationSummary:
    """Minimize an objective with constraints.

    Args:
        initial_hinges: Initial hinge configuration, shape (N+1, 3)
        config: Constraint configuration
        objective: Objective function to minimize (energy functional name or callable)
        options: Solver options (method, penalty weight, constraint solver flag, etc.)

    Returns:
        OptimizationSummary with optimized configuration and diagnostics

    Note:
        When options.use_constraint_solver=False, uses penalty-based
        optimization with the specified method (default: BFGS).
        When options.use_constraint_solver=True, uses scipy's constrained
        optimization with the constraint_method (default: SLSQP).
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
    if isinstance(objective, str):
        objective_fn = _get_objective(objective)
    else:
        objective_fn = objective

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
        scipy_result=result,
    )


def compute_linking_number(hinges: NDArray[np.float64]) -> float:
    """Compute linking number Lk = Tw + Wr from binormals.

    Args:
        hinges: Binormal (hinge) vectors, shape (N+1, 3)

    Returns:
        Linking number in units of π (so Lk=1 means π linking)

    Note:
        Uses Călugăreanu-White-Fuller theorem: Lk = Tw + Wr
        - Tw (total twist) from binormals
        - Wr (writhe) from curve
    """
    tangents = binormals_to_tangents(hinges, normalize=True)
    curve = tangents_to_curve(tangents)

    tw = total_twist(hinges)
    wr = writhe(curve)

    return tw + wr


def optimize_with_linking_constraint(
    initial_hinges: NDArray[np.float64],
    target_linking: float,
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
        return lk - target_linking

    # check consistency of orientation and linking target
    if (config.oriented and (int(target_linking) % 2) != 0) or (not config.oriented and (int(target_linking) % 2) != 1):
        warnings.warn(
            "The parity of orientation and target_linking is inconsistent. "
            "The target_linking may be unattainable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Phase 1: Build constraints WITHOUT constant torsion
    # Create temporary config without constant torsion for phase 1
    config_phase1 = ConstraintConfig(
        slide=config.slide,
        oriented=config.oriented,
        enforce_anchors=config.enforce_anchors,
        constant_torsion=False,  # Explicitly exclude
        closure=False,  # Exclude closure constraint in phase 1
        alignment=config.alignment,
        reference_torsion=config.reference_torsion,
    )
    constraints_phase1 = _build_constraint_dicts(config_phase1)

    # Phase 1 objective: minimize constant torsion residuals + linking error
    def constant_torsion_residuals_flat(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Constant torsion residuals for soft minimization in phase 1."""
        hinges = _reshape(flat)
        return constant_torsion_residuals(hinges, reference=config.reference_torsion)

    def phase1_objective(flat: NDArray[np.float64]) -> float:
        """Phase 1: Minimize constant torsion violation + linking error."""
        torsion_error = np.sum(constant_torsion_residuals_flat(flat)**2)
        linking_error = linking_constraint(flat)**2
        closure_error = np.sum(closure_residual(_reshape(flat), slide=config.slide)**2)
        return torsion_error + linking_error + closure_error

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
        scipy_result=result_phase2,
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
