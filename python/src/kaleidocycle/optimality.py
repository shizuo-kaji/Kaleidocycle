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
    energy: Literal["bending", "mean_cos"],
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
    if backend_obj.name == "jax":
        return _compute_energy_gradient_jax(hinges_arr, energy, backend_obj)
    else:
        return _compute_energy_gradient_numpy(hinges_arr, energy, eps)


def _compute_energy_gradient_numpy(
    hinges: NDArray[np.float64],
    energy: Literal["bending", "mean_cos"],
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
    if energy == "bending":

        def energy_func(h: NDArray[np.float64]) -> float:
            tangents = binormals_to_tangents(h, normalize=True)
            return bending_energy(tangents)
    elif energy == "mean_cos":

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
    energy: Literal["bending", "mean_cos"],
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
    if energy == "bending":

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

    elif energy == "mean_cos" or energy == "neg_mean_cos":

        def energy_func(h):
            # JAX-compatible mean_cosine
            a = h[:-1]
            b = h[1:]
            norms = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1)
            dots = jnp.einsum("ij,ij->i", a, b)
            cosines = jnp.clip(dots / norms, -1.0, 1.0)
            return jnp.mean(cosines)
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
    if backend_obj.name == "jax":
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

        # Keep residual group order in sync with _flatten_residuals(), which
        # sorts the NumPy constraint_residuals() dictionary by key.

        if config.alignment:
            if config.oriented:
                align_vector = h[0] - h[-1]
            else:
                align_vector = h[0] + h[-1]
            if config.full_alignment:
                flat.append(align_vector)
            else:
                align_norm = jnp.linalg.norm(align_vector)
                align_residual = jnp.where(
                    align_norm < 1e-3,
                    0.0,
                    align_norm,
                )
                flat.append(align_residual)

        if config.closure:
            # Closure residual
            T = jnp.cross(h[:-1], h[1:])  # tangents
            ext = jnp.sum(T, axis=0)
            if config.slide != 0.0:
                ext = ext + config.slide * jnp.sum(h[:-1], axis=0)
            flat.append(ext)

        if config.constant_torsion:
            # Constant torsion residuals
            dot_products = jnp.sum(h[:-1] * h[1:], axis=1)
            if config.reference_torsion is None:
                torsion_residuals = dot_products - dot_products[0]
            else:
                torsion_residuals = dot_products - config.reference_torsion
            flat.append(torsion_residuals)

        # Unit norm residuals (for all except last hinge)
        norms_sq = jnp.sum(h[:-1] ** 2, axis=1)
        flat.append(norms_sq - 1.0)

        # Concatenate all residuals
        return jnp.concatenate(
            [r.flatten() if r.ndim > 0 else jnp.array([r]) for r in flat]
        )

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
        msg = (
            f"dimension mismatch: jacobian has {jac_arr.shape[1]} columns "
            f"but gradient has {len(g_flat)} elements"
        )
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


def _rigid_rotation_basis(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a basis for the infinitesimal rigid rotations acting on hinges.

    A global rotation about axis ω acts as δh_i = ω × h_i, which preserves
    every constraint in :class:`ConstraintConfig` (norms, dot products,
    alignment, closure). The three columns correspond to ω = e_x, e_y, e_z
    flattened in the same order as ``hinges.flatten()``.
    """

    h = np.asarray(hinges, dtype=float)
    n_pts = h.shape[0]
    basis = np.zeros((h.size, 3), dtype=float)
    axes = np.eye(3)
    for k in range(3):
        delta = np.cross(axes[k], h)  # shape (n_pts, 3)
        basis[:, k] = delta.reshape(-1)
    # Orthonormalize (drop dependent columns if hinges are colinear)
    q, r = np.linalg.qr(basis)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-12))
    return q[:, :rank]


def local_dof(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    tol: float | None = None,
    return_basis: bool = False,
    subtract_rigid: bool = True,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> dict:
    """Compute the local degrees of freedom of constraint-preserving motions.

    The constraint manifold M = {h : g(h) = 0} has tangent space ker(J)
    at the given configuration, where J is the constraint Jacobian. This
    routine returns dim(ker J), optionally subtracting the three rigid
    rotations that always preserve the constraints (and which are usually
    quotiented out in practical kinematic analyses).

    Parameters
    ----------
    hinges : np.ndarray
        Hinge (binormal) vectors, shape (n+1, 3).
    config : ConstraintConfig
        Constraint configuration to evaluate.
    tol : float, optional
        Singular-value cutoff used for rank determination. Defaults to
        ``max(J.shape) * eps(largest_sv)`` (the NumPy convention).
    return_basis : bool, default False
        If True, include an orthonormal basis of the (reduced) tangent
        space in the result under the ``"basis"`` key, shaped
        ``(n+1, 3, dof)``.
    subtract_rigid : bool, default True
        If True, project out the 3 infinitesimal global rotations from the
        nullspace before reporting the DoF. When ``config.enforce_anchors``
        is set the rigid rotations are already killed by the constraints,
        so subtraction is skipped automatically.
    finite_diff_step : float, default 1e-8
        Step size for NumPy-backend Jacobian computation.
    backend : str, optional
        Backend selector forwarded to :func:`compute_constraint_jacobian`.

    Returns
    -------
    dict
        Keys:
        - ``dof`` (int): local DoF after optional rigid subtraction.
        - ``raw_dof`` (int): nullspace dimension of the Jacobian.
        - ``rigid_dof`` (int): dimension of rigid rotations subtracted.
        - ``rank`` (int): numerical rank of the Jacobian.
        - ``n_constraints`` (int), ``n_variables`` (int).
        - ``singular_values`` (np.ndarray): singular values of J.
        - ``tol`` (float): tolerance actually used for rank cutoff.
        - ``basis`` (np.ndarray, optional): tangent-space basis with shape
          ``(n+1, 3, dof)`` when ``return_basis`` is True.

    Notes
    -----
    The DoF reported here is *infinitesimal* (a linear-algebra count at
    the current configuration). True finite mobility may differ if the
    Jacobian is rank-deficient at a singular point of the constraint
    variety.

    Examples
    --------
    >>> from kaleidocycle import Kaleidocycle, ConstraintConfig
    >>> kc = Kaleidocycle(n=8, oriented=True)
    >>> info = local_dof(kc.hinges, kc.config)
    >>> info["dof"]
    1
    """

    hinges_arr = np.asarray(hinges, dtype=float)
    if hinges_arr.ndim != 2 or hinges_arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {hinges_arr.shape}"
        raise ValueError(msg)

    jacobian = compute_constraint_jacobian(
        hinges_arr, config, finite_diff_step, backend=backend
    )

    n_constraints, n_vars = jacobian.shape

    # SVD-based rank / nullspace.
    U, s, Vt = np.linalg.svd(jacobian, full_matrices=True)
    if tol is None:
        max_sv = s[0] if s.size else 0.0
        tol_eff = max(jacobian.shape) * np.finfo(float).eps * max(max_sv, 1.0)
    else:
        tol_eff = float(tol)
    rank = int(np.sum(s > tol_eff))
    raw_dof = n_vars - rank

    # Nullspace basis from the trailing rows of Vt.
    null_basis = Vt[rank:].T  # shape (n_vars, raw_dof)

    rigid_dof = 0
    reduced_basis = null_basis
    if subtract_rigid and not config.enforce_anchors and raw_dof > 0:
        rigid = _rigid_rotation_basis(hinges_arr)  # (n_vars, k_rigid)
        if rigid.shape[1] > 0 and null_basis.shape[1] > 0:
            # How much of each rigid direction lies in the nullspace.
            coeffs = null_basis.T @ rigid  # (raw_dof, k_rigid)
            resid = rigid - null_basis @ coeffs
            in_null_mask = np.linalg.norm(resid, axis=0) < 1e-6
            rigid_in_null = rigid[:, in_null_mask]
            if rigid_in_null.shape[1] > 0:
                # Orthonormalize the rigid-in-null directions.
                u_r, s_r, _ = np.linalg.svd(rigid_in_null, full_matrices=False)
                k_rigid = int(np.sum(s_r > 1e-10 * (s_r[0] if s_r.size else 1.0)))
                q_r = u_r[:, :k_rigid]
                rigid_dof = k_rigid
                # Quotient out rigid: project null_basis orthogonally to q_r,
                # then use SVD (rank-revealing) to extract the surviving basis.
                projected = null_basis - q_r @ (q_r.T @ null_basis)
                u_n, s_n, _ = np.linalg.svd(projected, full_matrices=False)
                keep = int(
                    np.sum(s_n > 1e-8 * (s_n[0] if s_n.size else 1.0))
                )
                reduced_basis = u_n[:, :keep]

    dof = reduced_basis.shape[1]

    result = {
        "dof": int(dof),
        "raw_dof": int(raw_dof),
        "rigid_dof": int(rigid_dof),
        "rank": int(rank),
        "n_constraints": int(n_constraints),
        "n_variables": int(n_vars),
        "singular_values": s,
        "tol": float(tol_eff),
    }
    if return_basis:
        if dof > 0:
            result["basis"] = reduced_basis.reshape(
                hinges_arr.shape[0], 3, dof
            )
        else:
            result["basis"] = np.zeros((hinges_arr.shape[0], 3, 0))
    return result


def _newton_correct(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    tol: float,
    max_iter: int,
    finite_diff_step: float,
    backend: Optional[str],
) -> NDArray[np.float64] | None:
    """Project ``hinges`` onto the constraint manifold via damped Newton.

    Each step solves ``J · dh = -r`` in the minimum-norm sense, which is
    exactly the Gauss–Newton correction used by predictor–corrector
    continuation methods.

    Returns the corrected hinge array, or ``None`` if Newton fails to
    drive the residual below ``tol`` within ``max_iter`` iterations.
    """
    from .constraints import constraint_residuals

    h = np.asarray(hinges, dtype=float).copy()
    last_norm = np.inf
    for _ in range(max_iter):
        res = _flatten_residuals(constraint_residuals(h, config))
        res_norm = float(np.linalg.norm(res))
        if res_norm < tol:
            return h
        # Divergence guard.
        if res_norm > 10.0 * last_norm and last_norm < np.inf:
            return None
        last_norm = res_norm

        J = compute_constraint_jacobian(
            h, config, finite_diff_step, backend=backend
        )
        try:
            dh, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            return None
        h = h + dh.reshape(h.shape)

    final = _flatten_residuals(constraint_residuals(h, config))
    if float(np.linalg.norm(final)) < tol:
        return h
    return None


def finite_motion_dof(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    step_size: float = 1e-3,
    n_steps: int = 20,
    n_samples: int | None = None,
    correction_tol: float = 1e-8,
    max_newton_iter: int = 50,
    subtract_rigid: bool = True,
    rank_tol: float = 1e-3,
    nullspace_tol: float | None = None,
    seed: int | None = None,
    return_paths: bool = False,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> dict:
    """Estimate the dimension of finite (nonlinear) constraint-preserving motions.

    The linear analysis in :func:`local_dof` returns ``dim ker(J)``, which
    over-counts at singular points of the constraint variety (a tangent
    direction may fail to integrate to a path because of higher-order
    obstructions). This routine performs predictor–corrector continuation
    along sampled tangent directions and reports the rank of the resulting
    displacement matrix, which approximates ``dim_h M`` — the local
    dimension of the constraint variety at the given configuration.

    Method
    ------
    1. Compute the infinitesimal tangent basis ``V`` via
       :func:`local_dof` (optionally with rigid rotations removed).
    2. For each sample direction ``v = V α`` (the first ``k`` samples are
       the basis columns themselves; the rest are random unit
       combinations):

       - Predictor step ``h ← h + δ · v``.
       - Newton corrector projects back to ``{g = 0}``.
       - At each new point, re-project ``v`` onto the local tangent
         space so the path follows the variety.
       - Repeat for ``n_steps`` steps.

    3. Collect the successful endpoint displacements ``Δh = h_T − h_0``.
       The numerical rank of the displacement matrix is the finite DoF.

    Parameters
    ----------
    hinges, config
        Configuration and constraint specification.
    step_size : float
        Predictor step size ``δ``.
    n_steps : int
        Number of predictor-corrector steps per sample.
    n_samples : int, optional
        Number of tangent directions to try. Defaults to ``max(2k, 8)``
        where ``k`` is the infinitesimal DoF; the first ``k`` are the
        infinitesimal basis columns themselves.
    correction_tol : float
        Constraint residual tolerance for the Newton corrector.
    max_newton_iter : int
        Maximum Newton iterations per corrector step.
    subtract_rigid : bool, default True
        Forward to :func:`local_dof` — when True, global rigid rotations
        are not counted (they are always finite motions).
    rank_tol : float
        Relative singular-value cutoff for the displacement matrix.
    seed : int, optional
        RNG seed for reproducible random direction sampling.
    return_paths : bool
        If True, include the corrected path of each sample under
        ``"paths"``.
    finite_diff_step, backend
        Forwarded to :func:`compute_constraint_jacobian`.

    Returns
    -------
    dict
        Keys:
        - ``finite_dof`` (int): estimated finite-motion DoF.
        - ``infinitesimal_dof`` (int): from :func:`local_dof`.
        - ``rigid_dof`` (int): rigid rotations subtracted (if any).
        - ``n_samples`` (int), ``n_successful`` (int): sample counts.
        - ``displacement_singular_values`` (np.ndarray).
        - ``max_residual`` (float): worst constraint residual along all
          successful paths (sanity check).
        - ``step_size``, ``n_steps``: continuation parameters used.
        - ``paths`` (list[np.ndarray], optional).

    Notes
    -----
    The finite DoF is bounded above by the infinitesimal DoF; equality
    holds at smooth points of the variety. A strictly smaller finite DoF
    indicates a singular point (e.g. a bifurcation or branching) where
    some infinitesimal tangents do not integrate.

    Continuation can also fail simply because the chosen ``step_size``
    is too large for the local curvature, so a small finite DoF should
    be cross-checked by shrinking ``step_size``.

    Examples
    --------
    >>> from kaleidocycle import Kaleidocycle
    >>> kc = Kaleidocycle(n=8, oriented=True)
    >>> info = finite_motion_dof(kc.hinges, kc.config, seed=0)
    >>> info["finite_dof"]
    1
    """
    from .constraints import constraint_residuals

    hinges_arr = np.asarray(hinges, dtype=float)
    h0 = hinges_arr.copy()
    n_pts = h0.shape[0]
    n_vars = h0.size

    inf_info = local_dof(
        hinges_arr,
        config,
        tol=nullspace_tol,
        return_basis=True,
        subtract_rigid=subtract_rigid,
        finite_diff_step=finite_diff_step,
        backend=backend,
    )
    k = int(inf_info["dof"])

    if k == 0:
        return {
            "finite_dof": 0,
            "infinitesimal_dof": 0,
            "rigid_dof": int(inf_info["rigid_dof"]),
            "n_samples": 0,
            "n_successful": 0,
            "displacement_singular_values": np.array([]),
            "max_residual": 0.0,
            "step_size": step_size,
            "n_steps": n_steps,
            **({"paths": []} if return_paths else {}),
        }

    basis = inf_info["basis"].reshape(n_vars, k)

    if n_samples is None:
        n_samples = max(2 * k, 8)
    n_samples = max(n_samples, k)

    rng = np.random.default_rng(seed)

    displacements: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    n_successful = 0
    max_residual = 0.0

    for i in range(n_samples):
        if i < k:
            alpha = np.zeros(k)
            alpha[i] = 1.0
        else:
            alpha = rng.standard_normal(k)
            alpha /= np.linalg.norm(alpha) + 1e-15

        direction = basis @ alpha  # (n_vars,)
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-12:
            continue
        direction = direction / dir_norm

        h = h0.copy()
        path = [h.copy()] if return_paths else None

        for _step in range(n_steps):
            h_pred = h + step_size * direction.reshape(h.shape)
            h_corr = _newton_correct(
                h_pred,
                config,
                tol=correction_tol,
                max_iter=max_newton_iter,
                finite_diff_step=finite_diff_step,
                backend=backend,
            )
            if h_corr is None:
                break

            # Track worst residual for diagnostics.
            res = _flatten_residuals(constraint_residuals(h_corr, config))
            max_residual = max(max_residual, float(np.linalg.norm(res)))

            # Re-estimate tangent at the new point and reproject the
            # direction so we keep tracking the same branch. Use the
            # same subtract_rigid setting as the caller so rigid drift
            # is removed at the *current* configuration (rigid
            # directions are h-dependent: δh = ω × h).
            new_info = local_dof(
                h_corr,
                config,
                tol=nullspace_tol,
                return_basis=True,
                subtract_rigid=subtract_rigid,
                finite_diff_step=finite_diff_step,
                backend=backend,
            )
            new_basis = new_info["basis"].reshape(n_vars, new_info["dof"])
            if new_basis.shape[1] == 0:
                h = h_corr
                if path is not None:
                    path.append(h.copy())
                break
            new_dir = new_basis @ (new_basis.T @ direction)
            new_norm = np.linalg.norm(new_dir)
            if new_norm < 1e-10:
                break
            direction = new_dir / new_norm

            h = h_corr
            if path is not None:
                path.append(h.copy())

        if return_paths:
            paths.append(np.array(path))

        disp = (h - h0).reshape(-1)
        if np.linalg.norm(disp) > 10.0 * correction_tol:
            displacements.append(disp)
            n_successful += 1

    if displacements:
        D = np.array(displacements).T  # (n_vars, n_successful)
        if subtract_rigid:
            # Remove the rigid-rotation component of each displacement at
            # h0 so accumulated rotational drift along curved paths does
            # not inflate the displacement rank.
            R0 = _rigid_rotation_basis(h0)
            if R0.shape[1] > 0:
                D = D - R0 @ (R0.T @ D)
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        D_norm = D / np.where(norms > 1e-15, norms, 1.0)
        _, sigma, _ = np.linalg.svd(D_norm, full_matrices=False)
        s_max = sigma[0] if sigma.size else 0.0
        finite_dof = int(np.sum(sigma > rank_tol * max(s_max, 1.0)))
    else:
        sigma = np.array([])
        finite_dof = 0

    result = {
        "finite_dof": int(finite_dof),
        "infinitesimal_dof": int(inf_info["dof"]),
        "rigid_dof": int(inf_info["rigid_dof"]),
        "n_samples": int(n_samples),
        "n_successful": int(n_successful),
        "displacement_singular_values": sigma,
        "max_residual": float(max_residual),
        "step_size": float(step_size),
        "n_steps": int(n_steps),
    }
    if return_paths:
        result["paths"] = paths
    return result


def find_nearby_stationary(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    energy: Literal["bending", "mean_cos"] = "mean_cos",
    *,
    tol: float = 1e-10,
    maxfev: int = 2000,
    correction_tol: float = 1e-8,
    max_newton_iter: int = 100,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> dict:
    """Find the nearest critical point of ``energy`` under ``config``.

    Solves ``Π_{ker J(h)} ∇E(h) = 0`` for ``h`` via Powell's hybrid
    method (``scipy.optimize.root``). Unlike gradient descent, this
    converges at saddle points — which is essential for analytically
    constructed kaleidocycles (e.g. theta-function solutions) that are
    saddles of ``mean_cos`` rather than local minima.

    Parameters
    ----------
    hinges : np.ndarray
        Initial hinge configuration, shape ``(n+1, 3)``.
    config : ConstraintConfig
        Constraint configuration. For meaningful results use
        ``full_alignment=True``; otherwise the scalar alignment
        contributes a rank-0 row that leaves a permanent residual in
        the projected gradient.
    energy : {"bending", "mean_cos"}
        Energy whose stationary point is sought.
    tol : float, default 1e-10
        Termination tolerance for ``scipy.optimize.root``.
    maxfev : int, default 2000
        Maximum number of residual evaluations.
    correction_tol : float, default 1e-8
        Newton-corrector tolerance for projection onto the manifold.
    max_newton_iter : int, default 100
        Maximum Newton iterations inside the projector.

    Returns
    -------
    dict
        Keys: ``hinges`` (the stationary configuration),
        ``projected_gradient_norm`` (residual at convergence),
        ``n_eval`` (root-finder evaluations), ``success`` (root-finder
        flag), ``distance`` (Euclidean distance to the initial hinges).

    Notes
    -----
    The torsion *value* is left free unless ``config.reference_torsion``
    is set; only the constancy of the torsion is enforced. The location
    of the stationary determines the torsion value implicitly.
    """
    from scipy.optimize import root

    h0 = np.asarray(hinges, dtype=float)
    if h0.ndim != 2 or h0.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {h0.shape}"
        raise ValueError(msg)
    h0 = h0.copy()

    def F(h_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        h = h_flat.reshape(h0.shape)
        h_corr = _newton_correct(
            h, config,
            tol=correction_tol, max_iter=max_newton_iter,
            finite_diff_step=finite_diff_step, backend=backend,
        )
        if h_corr is None:
            return np.full(h_flat.size, 1.0)
        g = compute_energy_gradient(
            h_corr, energy, finite_diff_step, backend=backend
        ).flatten()
        J = compute_constraint_jacobian(
            h_corr, config, finite_diff_step, backend=backend
        )
        JJt = J @ J.T + 1e-12 * np.eye(J.shape[0])
        pg = g - J.T @ np.linalg.solve(JJt, J @ g)
        return pg

    sol = root(F, h0.flatten(), method="hybr",
               options={"xtol": tol, "maxfev": maxfev})

    h_final = sol.x.reshape(h0.shape)
    h_proj = _newton_correct(
        h_final, config,
        tol=correction_tol, max_iter=max_newton_iter,
        finite_diff_step=finite_diff_step, backend=backend,
    )
    if h_proj is not None:
        h_final = h_proj

    return {
        "hinges": h_final,
        "projected_gradient_norm": float(np.linalg.norm(sol.fun)),
        "n_eval": int(sol.nfev),
        "success": bool(sol.success),
        "distance": float(np.linalg.norm(h_final - h0)),
    }


def follow_motion(
    hinges: NDArray[np.float64],
    config: ConstraintConfig,
    *,
    direction_index: int = 0,
    step_size: float = 5e-4,
    n_steps: int = 80,
    bidirectional: bool = True,
    correction_tol: float = 1e-8,
    max_newton_iter: int = 50,
    subtract_rigid: bool = True,
    nullspace_tol: float | None = None,
    finite_diff_step: float = 1e-8,
    backend: Optional[str] = None,
) -> NDArray[np.float64]:
    """Predictor-corrector continuation along one tangent direction.

    Returns the array of corrected hinge frames produced by following the
    ``direction_index``-th column of the local tangent basis (from
    :func:`local_dof`) forward — and optionally backward — from
    ``hinges``. At every step the direction is re-projected onto the
    local tangent space at the current point.

    Parameters
    ----------
    hinges, config
        Starting configuration and constraint set.
    direction_index : int, default 0
        Which tangent basis column to follow (taken modulo the basis
        dimension).
    step_size : float, default 5e-4
        Predictor step length.
    n_steps : int, default 80
        Number of steps per side.
    bidirectional : bool, default True
        If True, walks both forward and backward and concatenates the
        result; the returned array contains ``2 * n_steps + 1`` frames
        ordered from the most-negative to the most-positive step.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_frames, n+1, 3)``.
    """
    h0 = np.asarray(hinges, dtype=float).copy()
    info = local_dof(
        h0, config, tol=nullspace_tol,
        return_basis=True, subtract_rigid=subtract_rigid,
        finite_diff_step=finite_diff_step, backend=backend,
    )
    if info["dof"] == 0:
        return np.array([h0])

    basis = info["basis"].reshape(h0.size, info["dof"])
    v0 = basis[:, direction_index % basis.shape[1]]
    v0 = v0 / np.linalg.norm(v0)

    def _walk(sign: float):
        h = h0.copy()
        direction = sign * v0
        out = []
        for _ in range(n_steps):
            h_pred = h + step_size * direction.reshape(h.shape)
            h_corr = _newton_correct(
                h_pred, config,
                tol=correction_tol, max_iter=max_newton_iter,
                finite_diff_step=finite_diff_step, backend=backend,
            )
            if h_corr is None:
                break
            new_info = local_dof(
                h_corr, config, tol=nullspace_tol,
                return_basis=True, subtract_rigid=subtract_rigid,
                finite_diff_step=finite_diff_step, backend=backend,
            )
            if new_info["dof"] == 0:
                out.append(h_corr.copy())
                break
            new_basis = new_info["basis"].reshape(h0.size, new_info["dof"])
            new_dir = new_basis @ (new_basis.T @ direction)
            new_norm = np.linalg.norm(new_dir)
            if new_norm < 1e-10:
                break
            direction = new_dir / new_norm
            h = h_corr
            out.append(h.copy())
        return out

    fwd = _walk(+1.0)
    if bidirectional:
        bwd = _walk(-1.0)
        frames = list(reversed(bwd)) + [h0.copy()] + fwd
    else:
        frames = [h0.copy()] + fwd
    return np.array(frames)


def check_stationarity(
    hinges: NDArray[np.float64],
    energy: Literal["bending", "mean_cos"],
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
    grad = compute_energy_gradient(
        hinges_arr, energy, finite_diff_step, backend=backend
    )
    grad_norm = float(np.linalg.norm(grad))

    # Compute constraint Jacobian (with backend dispatch)
    jacobian = compute_constraint_jacobian(
        hinges_arr, config, finite_diff_step, backend=backend
    )

    # Project gradient onto constraint tangent space
    proj_grad = project_gradient(grad, jacobian)
    proj_grad_norm = float(np.linalg.norm(proj_grad))

    # Check stationarity
    is_stat = proj_grad_norm < tolerance

    return {
        "is_stationary": is_stat,
        "projected_gradient_norm": proj_grad_norm,
        "gradient_norm": grad_norm,
        "constraint_penalty": penalty,
        "details": {
            "energy": energy,
            "tolerance": tolerance,
            "finite_diff_step": finite_diff_step,
            "n_constraints": jacobian.shape[0],
            "n_variables": jacobian.shape[1],
            "constraint_rank": int(np.linalg.matrix_rank(jacobian)),
            "backend": backend or "default",
        },
    }
