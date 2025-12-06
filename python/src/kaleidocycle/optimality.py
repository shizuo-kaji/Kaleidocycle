"""Optimality checking and gradient computation for Kaleidocycle configurations."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .constraints import ConstraintConfig, constraint_residuals
from .energies import bending_energy
from .geometry import binormals_to_tangents, mean_cosine


def compute_energy_gradient(
    hinges: NDArray[np.float64],
    energy: Literal['bending', 'mean_cos'],
    eps: float = 1e-8,
    backend: Optional[str] = None,
) -> NDArray[np.float64]:
    """Compute gradient of energy function.

    Uses either finite differences (NumPy backend) or automatic differentiation
    (JAX backend) depending on the backend parameter.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    energy : {'bending', 'mean_cos'}
        Energy function to compute gradient for:
        - 'bending': Bobenko-Suris bending energy
        - 'mean_cos': Mean cosine (torsion)
    eps : float, default=1e-8
        Step size for finite differences (only used for NumPy backend)
    backend : str, optional
        Backend to use ('numpy' or 'jax'). If None, uses current global backend.

    Returns
    -------
    np.ndarray
        Gradient array, shape (n+1, 3)

    Notes
    -----
    NumPy backend uses central finite differences: df/dx ≈ (f(x+eps) - f(x-eps)) / (2*eps)
    JAX backend uses automatic differentiation for exact gradients.

    Examples
    --------
    >>> from kaleidocycle import Kaleidocycle
    >>> kc = Kaleidocycle(n=6, oriented=True)
    >>> grad = compute_energy_gradient(kc.hinges, 'bending')
    >>> grad.shape
    (7, 3)

    Use JAX backend for exact gradients:
    >>> grad_jax = compute_energy_gradient(kc.hinges, 'bending', backend='jax')
    """
    from .backends import get_backend

    hinges_arr = np.asarray(hinges, dtype=float)
    if hinges_arr.ndim != 2 or hinges_arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {hinges_arr.shape}"
        raise ValueError(msg)

    # Get backend
    backend_obj = get_backend(backend)

    # Dispatch to backend-specific implementation
    if backend_obj.name == 'jax':
        return _compute_energy_gradient_jax(hinges_arr, energy, backend_obj)
    else:
        return _compute_energy_gradient_numpy(hinges_arr, energy, eps)


def _compute_energy_gradient_numpy(
    hinges: NDArray[np.float64],
    energy: Literal['bending', 'mean_cos'],
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute gradient using NumPy finite differences.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    energy : {'bending', 'mean_cos'}
        Energy function
    eps : float
        Step size for finite differences

    Returns
    -------
    np.ndarray
        Gradient array, shape (n+1, 3)
    """
    # Define energy function
    if energy == 'bending':
        def energy_func(h: NDArray[np.float64]) -> float:
            tangents = binormals_to_tangents(h, normalize=True)
            return bending_energy(tangents)
    elif energy == 'mean_cos':
        def energy_func(h: NDArray[np.float64]) -> float:
            return mean_cosine(h, wrap=False)
    else:
        msg = f"unknown energy type: {energy}"
        raise ValueError(msg)

    # Compute gradient via central differences
    grad = np.zeros_like(hinges)
    for i in range(hinges.shape[0]):
        for j in range(hinges.shape[1]):
            h_plus = hinges.copy()
            h_minus = hinges.copy()
            h_plus[i, j] += eps
            h_minus[i, j] -= eps

            e_plus = energy_func(h_plus)
            e_minus = energy_func(h_minus)

            grad[i, j] = (e_plus - e_minus) / (2 * eps)

    return grad


def _compute_energy_gradient_jax(
    hinges: NDArray[np.float64],
    energy: Literal['bending', 'mean_cos'],
    backend_obj,
) -> NDArray[np.float64]:
    """Compute gradient using JAX automatic differentiation.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    energy : {'bending', 'mean_cos'}
        Energy function
    backend_obj : JAXBackend
        JAX backend instance

    Returns
    -------
    np.ndarray
        Gradient array, shape (n+1, 3)
    """
    import jax.numpy as jnp
    import math

    LOG2 = math.log(2.0)

    # Convert to JAX array
    hinges_jax = backend_obj.asarray(hinges)

    # Define energy function based on type using JAX operations
    if energy == 'bending':
        def energy_func(h):
            # JAX-compatible binormals_to_tangents
            T = jnp.cross(h[:-1], h[1:])
            norms = jnp.linalg.norm(T, axis=1, keepdims=True)
            T = T / norms

            # JAX-compatible bending energy
            a = T
            b = jnp.roll(T, -1, axis=0)
            norms_a = jnp.linalg.norm(a, axis=1)
            norms_b = jnp.linalg.norm(b, axis=1)
            dots = jnp.einsum("ij,ij->i", a, b)
            ratios = jnp.clip(dots / (norms_a * norms_b), -1.0 + 1e-15, 1.0)
            return jnp.sum(LOG2 - jnp.log1p(ratios))

    elif energy == 'mean_cos' or energy == 'neg_mean_cos':
        def energy_func(h):
            # JAX-compatible mean_cosine
            a = h[:-1]
            b = h[1:]
            #norms = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1)
            dots = jnp.einsum("ij,ij->i", a, b)
            #cosines = jnp.clip(dots / norms, -1.0, 1.0)
            return jnp.mean(dots)
    else:
        msg = f"unknown energy type: {energy}"
        raise ValueError(msg)

    # Compute gradient using JAX autodiff
    grad_func = backend_obj.grad(energy_func)
    grad_jax = grad_func(hinges_jax)

    # Convert back to NumPy
    return backend_obj.to_numpy(grad_jax)


def compute_constraint_jacobian(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    eps: float = 1e-8,
    backend: Optional[str] = None,
) -> NDArray[np.float64]:
    """Compute Jacobian matrix of constraint residuals.

    Uses either finite differences (NumPy backend) or automatic differentiation
    (JAX backend) depending on the backend parameter.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    config : ConstraintConfig
        Configuration specifying which constraints to include
    eps : float, default=1e-8
        Step size for finite differences (only used for NumPy backend)
    backend : str, optional
        Backend to use ('numpy' or 'jax'). If None, uses current global backend.

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape (n_constraints, n_vars)
        where n_vars = (n+1) * 3 is the number of hinge components

    Notes
    -----
    The Jacobian J[i, j] = ∂r_i / ∂x_j where r is the constraint residual
    vector and x is the flattened hinge array.

    The constraints included depend on the config:
    - unit_norm: ||h_i|| = 1 for all i < n+1
    - closure: sum of tangent vectors = 0
    - alignment: h[0] = ±h[-1] (sign depends on oriented)
    - constant_torsion: h_i · h_{i+1} = constant

    Examples
    --------
    >>> from kaleidocycle import Kaleidocycle, ConstraintConfig
    >>> kc = Kaleidocycle(n=6, oriented=True)
    >>> config = ConstraintConfig(oriented=True, constant_torsion=True)
    >>> jac = compute_constraint_jacobian(kc.hinges, config)
    >>> jac.shape
    (15, 21)  # 6 unit_norm + 3 closure + 1 alignment + 5 constant_torsion, 7 hinges × 3
    """
    from .backends import get_backend

    hinges_arr = np.asarray(hinges, dtype=float)
    if hinges_arr.ndim != 2 or hinges_arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {hinges_arr.shape}"
        raise ValueError(msg)

    # Get backend
    backend_obj = get_backend(backend)

    # Dispatch to backend-specific implementation
    if backend_obj.name == 'jax':
        return _compute_constraint_jacobian_jax(hinges_arr, config, backend_obj)
    else:
        return _compute_constraint_jacobian_numpy(hinges_arr, config, eps)


def _flatten_residuals(res_dict):
    """Helper to flatten residual dictionary consistently."""
    result = []
    for key in sorted(res_dict.keys()):  # Sort for consistency
        r = res_dict[key]
        if np.isscalar(r):
            result.append(float(r))
        else:
            result.extend(r.flatten().tolist())
    return np.array(result, dtype=float)


def _compute_constraint_jacobian_numpy(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute Jacobian using NumPy finite differences.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    config : ConstraintConfig
        Constraint configuration
    eps : float
        Step size for finite differences

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape (n_constraints, n_vars)
    """
    # Get constraint residuals at current point
    residuals = constraint_residuals(hinges, config)
    r_flat = _flatten_residuals(residuals)
    n_constraints = len(r_flat)
    n_vars = hinges.size

    # Initialize Jacobian matrix
    jacobian = np.zeros((n_constraints, n_vars), dtype=float)

    # Compute each column via finite differences
    for idx in range(n_vars):
        i, j = np.unravel_index(idx, hinges.shape)
        h_plus = hinges.copy()
        h_minus = hinges.copy()
        h_plus[i, j] += eps
        h_minus[i, j] -= eps

        r_plus_dict = constraint_residuals(h_plus, config)
        r_minus_dict = constraint_residuals(h_minus, config)

        r_plus = _flatten_residuals(r_plus_dict)
        r_minus = _flatten_residuals(r_minus_dict)

        jacobian[:, idx] = (r_plus - r_minus) / (2 * eps)

    return jacobian


def _compute_constraint_jacobian_jax(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    backend_obj,
) -> NDArray[np.float64]:
    """Compute Jacobian using JAX automatic differentiation.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    config : ConstraintConfig
        Constraint configuration
    backend_obj : JAXBackend
        JAX backend instance

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape (n_constraints, n_vars)
    """
    import jax.numpy as jnp

    def constraint_residuals_flat(h_flat):
        """Flattened constraint residuals for JAX."""
        # Reshape to (n+1, 3)
        h = h_flat.reshape(-1, 3)

        # Manually compute constraints using JAX operations
        flat = []

        # Unit norm residuals (for all except last hinge)
        norms_sq = jnp.sum(h[:-1] ** 2, axis=1)

        flat.append(norms_sq - 1.0)

        # Closure residual
        T = jnp.cross(h[:-1], h[1:])  # tangents
        ext = jnp.sum(T, axis=0)
        if config.slide != 0.0:
            ext = ext + config.slide * jnp.sum(h[:-1], axis=0)
        flat.append(ext)

        # Alignment residual
        if config.alignment:
            if config.oriented:
                align_residual = h[0] - h[-1]
            else:
                align_residual = h[0] + h[-1]
            flat.append(align_residual)

        # Constant torsion residuals
        if config.constant_torsion:
            dot_products = jnp.sum(h[:-1] * h[1:], axis=1)
            if config.reference_torsion is None:
                torsion_residuals = dot_products - dot_products[0]
            else:
                torsion_residuals = dot_products - config.reference_torsion
            flat.append(torsion_residuals)

        # Concatenate all residuals
        return jnp.concatenate([r.flatten() if r.ndim > 0 else jnp.array([r]) for r in flat])

    # Compute Jacobian using JAX
    hinges_flat_jax = backend_obj.asarray(hinges.flatten())
    jac_func = backend_obj.jacobian(constraint_residuals_flat)
    jac_jax = jac_func(hinges_flat_jax)

    # Convert back to NumPy
    return backend_obj.to_numpy(jac_jax)


def project_gradient(
    gradient: NDArray[np.float64],
    jacobian: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project gradient onto nullspace of constraint Jacobian.

    Parameters
    ----------
    gradient : np.ndarray
        Gradient to project, shape (n+1, 3)
    jacobian : np.ndarray
        Constraint Jacobian matrix, shape (n_constraints, n_vars)

    Returns
    -------
    np.ndarray
        Projected gradient, shape (n+1, 3)

    Notes
    -----
    The projected gradient is computed as:
        g_proj = g - J^T (J J^T)^{-1} J g

    This projects g onto the nullspace of J, which is the tangent space
    of the constraint manifold.

    For numerical stability, we use the pseudoinverse and add small
    regularization to J J^T before inversion.

    At a stationary point of a constrained optimization problem, the
    projected gradient should be close to zero: ||g_proj|| ≈ 0

    Examples
    --------
    >>> grad = np.random.randn(7, 3)
    >>> jac = np.random.randn(10, 21)
    >>> proj_grad = project_gradient(grad, jac)
    >>> proj_grad.shape
    (7, 3)
    """
    grad_arr = np.asarray(gradient, dtype=float)
    jac_arr = np.asarray(jacobian, dtype=float)

    # Flatten gradient
    g_flat = grad_arr.flatten()

    # Check dimensions
    if jac_arr.shape[1] != len(g_flat):
        msg = (f"dimension mismatch: jacobian has {jac_arr.shape[1]} columns "
               f"but gradient has {len(g_flat)} elements")
        raise ValueError(msg)

    # Compute J J^T
    JJT = jac_arr @ jac_arr.T

    # Add small regularization for numerical stability
    # This helps when constraints are nearly degenerate
    reg = 1e-10 * np.eye(JJT.shape[0])
    JJT_reg = JJT + reg
    # Compute (J J^T)^{-1} using pseudoinverse for robustness
    try:
        JJT_inv = np.linalg.pinv(JJT)
    except np.linalg.LinAlgError:
        JJT_inv = np.linalg.inv(JJT_reg)

    # Project gradient: g_proj = g - J^T (J J^T)^{-1} J g
    Jg = jac_arr @ g_flat
    g_proj_flat = g_flat - jac_arr.T @ (JJT_inv @ Jg)

    # Reshape back to original shape
    return g_proj_flat.reshape(grad_arr.shape)


def check_stationarity(
    hinges: NDArray[np.float64],
    energy: Literal['bending', 'mean_cos'],
    config: ConstraintConfig,
    *,
    tolerance: float = 1e-6,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> dict:
    """Check if a kaleidocycle configuration is at a stationary point.

    Uses either finite differences (NumPy backend) or automatic differentiation
    (JAX backend) for gradient and Jacobian computation.

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3)
    energy : {'bending', 'mean_cos'}
        Energy function to check
    config : ConstraintConfig
        Constraint configuration
    tolerance : float, default=1e-6
        Tolerance for projected gradient norm
    finite_diff_step : float, default=1e-8
        Step size for numerical differentiation (only used for NumPy backend)
    backend : str, optional
        Backend to use ('numpy' or 'jax'). If None, uses current global backend.

    Returns
    -------
    dict
        Dictionary containing:
        - 'is_stationary': bool - Whether at stationary point
        - 'projected_gradient_norm': float - Norm of projected gradient
        - 'gradient_norm': float - Norm of full gradient (before projection)
        - 'constraint_penalty': float - Sum of squared constraint residuals
        - 'details': dict - Additional diagnostic information

    Notes
    -----
    A configuration is considered to be at a stationary point if the
    gradient of the energy function, when projected onto the tangent
    space of the constraint manifold, has norm less than tolerance.

    Mathematically, this checks the first-order KKT condition:
        ∇E(h) + Σ λ_i ∇g_i(h) = 0

    where E is the energy, g_i are the constraints, and λ_i are
    Lagrange multipliers.

    Examples
    --------
    >>> from kaleidocycle import Kaleidocycle, ConstraintConfig
    >>> kc = Kaleidocycle(n=6, oriented=True)  # Optimized kaleidocycle
    >>> config = ConstraintConfig(oriented=True, constant_torsion=True)
    >>> result = check_stationarity(kc.hinges, 'bending', config)
    >>> result['is_stationary']
    True

    Use JAX backend for exact gradients:
    >>> result = check_stationarity(kc.hinges, 'bending', config, backend='jax')
    """
    from .constraints import constraint_penalty

    hinges_arr = np.asarray(hinges, dtype=float)

    # Compute constraint penalty
    penalty = constraint_penalty(hinges_arr, config)

    # Compute energy gradient (with backend dispatch)
    grad = compute_energy_gradient(hinges_arr, energy, finite_diff_step, backend=backend)
    grad_norm = float(np.linalg.norm(grad))

    # Compute constraint Jacobian (with backend dispatch)
    jacobian = compute_constraint_jacobian(hinges_arr, config, finite_diff_step, backend=backend)

    # Project gradient onto constraint tangent space
    proj_grad = project_gradient(grad, jacobian)
    proj_grad_norm = float(np.linalg.norm(proj_grad))

    # Check stationarity
    is_stat = proj_grad_norm < tolerance

    return {
        'is_stationary': is_stat,
        'projected_gradient_norm': proj_grad_norm,
        'gradient_norm': grad_norm,
        'constraint_penalty': penalty,
        'details': {
            'energy': energy,
            'tolerance': tolerance,
            'finite_diff_step': finite_diff_step,
            'n_constraints': jacobian.shape[0],
            'n_variables': jacobian.shape[1],
            'constraint_rank': int(np.linalg.matrix_rank(jacobian)),
            'backend': backend or 'default',
        }
    }
