"""Geometry primitives and legacy helpers for the Kaleidocycle rewrite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(slots=True)
class HingeFrame:
    """Simple container for hinge directions."""

    vectors: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return a copy to decouple callers from the cached array."""

        return np.asarray(self.vectors, dtype=float).copy()


class Kaleidocycle:
    """Container for Kaleidocycle data with flexible initialization and property computation.

    The Kaleidocycle can be initialized from one of: hinges (binormals), curve, tangents, or normals.
    Properties can be computed on-demand using the compute() method.

    Attributes:
        n: Number of tetrahedra in the kaleidocycle
        hinges: Binormal (hinge) vectors, shape (n+1, 3)
        curve: 3D curve points, shape (n+1, 3)
        tangents: Tangent vectors, shape (n, 3)
        normals: Normal vectors, shape (n, 3)
        oriented: Whether the kaleidocycle is oriented
        metadata: Dictionary containing computed properties (energies, constraints, etc.)

    Example:
        >>> from kaleidocycle import Kaleidocycle, random_hinges, ConstraintConfig
        >>> hinges = random_hinges(6, seed=42, oriented=True).as_array()
        >>> kc = Kaleidocycle(hinges=hinges)
        >>> config = ConstraintConfig(oriented=True)
        >>> kc.compute(config=config)  # Compute all properties
        >>> print(kc.metadata['energies']['bending'])
    """

    def __init__(
        self,
        n: int | None = None,
        oriented: bool | None = None,
        *,
        hinges: np.ndarray | None = None,
        curve: np.ndarray | None = None,
        tangents: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        name: str | None = None,
        seed: int | None = None,
        solver_options: dict | None = None,
    ):
        """Initialize Kaleidocycle from one of: n (optimized), hinges, curve, tangents, or normals.

        Args:
            n: Number of tetrahedra. If provided alone, creates an optimized kaleidocycle
               using optimize_cycle with an appropriate objective.
            oriented: Whether the kaleidocycle is oriented.
                      - If n is provided: defaults to False
                      - If hinges/curve/tangents provided: inferred if None
            hinges: Binormal (hinge) vectors, shape (n+1, 3)
            curve: 3D curve points, shape (n+1, 3)
            tangents: Tangent vectors, shape (n, 3)
            normals: Normal vectors (currently not implemented), shape (n, 3)
            name: Stable name used by JSON catalogues and exports.
            seed: Random seed for initialization (only used when n is provided)
            solver_options: Optional dict with solver parameters when creating from n.
                           Special option: {"mode": "random_feasible"} creates a generic
                           feasible configuration via random initialization + Newton projection
                           instead of optimization.
                           Other options for random_feasible mode: max_iter, max_attempts, tol,
                           finite_diff_step, backend.
                           For optimized mode: maxiter, use_constraint_solver, etc.

        Raises:
            ValueError: If none or multiple initialization parameters are provided
            NotImplementedError: If normals initialization is requested

        Examples:
            Create optimized kaleidocycle from n:
            >>> kc = Kaleidocycle(9)  # Creates optimized kaleidocycle with 9 tetrahedra
            >>> kc = Kaleidocycle(8, oriented=False)  # Creates non-oriented with 8

            Create random feasible kaleidocycle (generic constraint manifold point):
            >>> kc = Kaleidocycle(10, oriented=False, solver_options={"mode": "random_feasible"}, seed=42)
            >>> kc.local_dof()['dof']  # Should be N - 6 for generic point
            4

            Create from existing data:
            >>> kc = Kaleidocycle(hinges=hinges_array)
            >>> kc = Kaleidocycle(curve=curve_array)
        """
        from .constraints import ConstraintConfig

        # Check that exactly one initialization parameter is provided
        init_params = sum(
            [
                n is not None,
                hinges is not None,
                curve is not None,
                tangents is not None,
                normals is not None,
            ]
        )

        if init_params == 0:
            raise ValueError(
                "Must provide one of: n, hinges, curve, tangents, or normals"
            )
        if init_params > 1:
            raise ValueError("Can only initialize from one parameter")

        # Initialize cached properties (computed lazily)
        self._curve: np.ndarray | None = None
        self._tangents: np.ndarray | None = None
        self._normals: np.ndarray | None = None
        self._curvatures: np.ndarray | None = None
        self._cosines: np.ndarray | None = None
        self._config: ConstraintConfig | None = None

        # Initialize metadata dictionary
        self.metadata: dict[str, any] = {}
        self.name = name

        # Initialize from n by creating optimized or random feasible kaleidocycle
        if n is not None:
            if not isinstance(n, int) or n < 3:
                raise ValueError(f"n must be an integer >= 3, got {n}")

            # Default oriented to False if not specified
            if oriented is None:
                oriented = False
            self.oriented = oriented

            # Check if random_feasible mode is requested
            mode = (
                solver_options.get("mode", "optimize") if solver_options else "optimize"
            )

            if mode == "random_feasible":
                # Create random feasible kaleidocycle via Newton projection
                self.hinges = self._create_random_feasible(
                    n, oriented, seed, solver_options
                )
                self.metadata: dict[str, any] = {"created_from": "random_feasible"}
            else:
                # Create optimized kaleidocycle
                self.hinges = self._create_optimized(
                    n, oriented, seed, solver_options, config=self.config
                )
                self.metadata: dict[str, any] = {"created_from": "optimize_cycle"}

            self.n = n
            self.oriented = oriented

            # Initialize cached properties
            self._curve: np.ndarray | None = None
            self._tangents: np.ndarray | None = None
            self._normals: np.ndarray | None = None
            self._curvatures: np.ndarray | None = None
            self._cosines: np.ndarray | None = None
            return

        # Initialize from the provided parameter
        if hinges is not None:
            self.hinges = np.asarray(hinges, dtype=float)
            if self.hinges.ndim != 2 or self.hinges.shape[1] != 3:
                raise ValueError(
                    f"hinges must have shape (n+1, 3), got {self.hinges.shape}"
                )

        elif curve is not None:
            curve_arr = np.asarray(curve, dtype=float)
            if curve_arr.ndim != 2 or curve_arr.shape[1] != 3:
                raise ValueError(
                    f"curve must have shape (n+1, 3), got {curve_arr.shape}"
                )
            self.hinges = curve_to_binormals(curve_arr)

        elif tangents is not None:
            tangents_arr = np.asarray(tangents, dtype=float)
            if tangents_arr.ndim != 2 or tangents_arr.shape[1] != 3:
                raise ValueError(
                    f"tangents must have shape (n, 3), got {tangents_arr.shape}"
                )
            self.hinges = tangents_to_binormals(tangents_arr)

        elif normals is not None:
            raise NotImplementedError(
                "Initialization from normals is not yet implemented. "
                "Please use hinges, curve, or tangents instead."
            )

        # Set basic properties
        self.n = len(self.hinges) - 1

        # Determine orientation if not provided
        if oriented is None:
            self.oriented = is_oriented(self.hinges)
        else:
            self.oriented = oriented

    @staticmethod
    def _create_random_feasible(
        n: int,
        oriented: bool,
        seed: int | None,
        solver_options: dict | None,
    ) -> np.ndarray:
        """Create a random feasible kaleidocycle via Newton projection.

        Generates a generic feasible configuration by randomly sampling unit hinges
        and Newton-projecting onto the constraint manifold. This produces a smooth
        point of the constraint variety without optimizing any specific energy.

        Args:
            n: Number of tetrahedra
            oriented: Whether kaleidocycle is oriented
            seed: Random seed for reproducibility
            solver_options: Dict with optional keys:
                - max_iter: Maximum Newton iterations per attempt (default 400)
                - max_attempts: Number of random initializations to try (default 8)
                - tol: Convergence tolerance (default 1e-9)
                - finite_diff_step: Finite difference step size (default 1e-8)
                - backend: "jax" or "numpy" (default "jax")

        Returns:
            Hinges array at a generic feasible configuration, shape (n+1, 3)

        Raises:
            RuntimeError: If projection fails after max_attempts
        """
        from .constraints import ConstraintConfig
        from .optimality import _newton_correct

        # Extract options
        opts = solver_options or {}
        max_iter = opts.get("max_iter", 400)
        max_attempts = opts.get("max_attempts", 8)
        tol = opts.get("tol", 1e-9)
        finite_diff_step = opts.get("finite_diff_step", 1e-8)
        backend = opts.get("backend", "jax")

        rng = np.random.default_rng(seed)

        # Initial random hinges
        raw = rng.standard_normal((n + 1, 3))
        raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        raw[-1] = raw[0] if oriented else -raw[0]

        # Configuration with full alignment
        cfg = ConstraintConfig(
            oriented=oriented,
            constant_torsion=True,
            alignment=True,
            closure=True,
            full_alignment=True,
        )

        # Try projection with multiple random initializations
        for attempt in range(max_attempts):
            h_proj = _newton_correct(
                raw,
                cfg,
                tol=tol,
                max_iter=max_iter,
                finite_diff_step=finite_diff_step,
                backend=backend,
            )
            if h_proj is not None:
                return h_proj

            # Generate new random initialization
            rng2 = np.random.default_rng(rng.integers(1 << 30))
            raw = rng2.standard_normal((n + 1, 3))
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            raw[-1] = raw[0] if oriented else -raw[0]

        raise RuntimeError(
            f"Could not project random hinges onto constraint manifold "
            f"after {max_attempts} attempts"
        )

    @staticmethod
    def _create_optimized(
        n: int,
        oriented: bool,
        seed: int | None,
        solver_options: dict | None,
        config: ConstraintConfig | None = None,
    ) -> np.ndarray:
        """Create an optimized kaleidocycle with n tetrahedra.

        Chooses an appropriate objective based on n and oriented parity:
        - mean_cos: Minimizes mean cosine (for configurations where it's not constant)
        - neg_mean_cos: Maximizes mean cosine (for non-oriented kaleidocycles)

        The objective is chosen such that:
        - For oriented with even n: use neg_mean_cos (mean_cos would be constant -1)
        - For oriented with odd n: use mean_cos
        - For non-oriented with even n: use mean_cos
        - For non-oriented with odd n: use neg_mean_cos (mean_cos would be constant -1)

        Args:
            n: Number of tetrahedra
            oriented: Whether kaleidocycle is oriented
            seed: Random seed for initial configuration
            solver_options: Optional solver parameters

        Returns:
            Optimized hinges array, shape (n+1, 3)
        """
        from .constraints import ConstraintConfig
        from .solvers import optimize_cycle, SolverOptions

        # Choose appropriate objective based on n and oriented parity
        # From the warnings in optimize_cycle:
        # - mean_cos is meaningless (always -1) when:
        #   (oriented and n%2==0) or (not oriented and n%2==1)
        # - neg_mean_cos is meaningless (always 1) when: oriented
        #
        # Objective selection:
        # - oriented + even n: Both mean_cos and neg_mean_cos are meaningless → use "bending"
        # - oriented + odd n: mean_cos is meaningful → use "mean_cos"
        # - non-oriented + even n: mean_cos is meaningful → use "mean_cos"
        # - non-oriented + odd n: mean_cos is meaningless → use "neg_mean_cos"
        if oriented:
            if n % 2 == 0:
                # Both cosine objectives are meaningless, use bending energy
                objective = "bending"
            else:
                objective = "mean_cos"
        else:
            if n % 2 == 1:
                objective = "neg_mean_cos"
            else:
                objective = "mean_cos"

        # Create initial random configuration
        initial = random_hinges(n, seed=seed, oriented=oriented).as_array()

        if solver_options is None:
            opts = SolverOptions()
        else:
            # Convert dict to SolverOptions
            opts = SolverOptions(**solver_options)

        # Run optimization
        result = optimize_cycle(
            initial,
            config,
            objective=objective,
            options=opts,
        )

        return result.hinges

    @property
    def config(self) -> ConstraintConfig:
        """Return a default ConstraintConfig for this kaleidocycle."""
        from .constraints import ConstraintConfig

        if self._config is None:
            self._config = ConstraintConfig(
                oriented=self.oriented,
                alignment=True,
                constant_torsion=True,
                enforce_anchors=False,
                slide=0.0,
            )
        return self._config

    @property
    def curve(self) -> np.ndarray:
        """Get curve points, computing if necessary."""
        if self._curve is None:
            tangents = binormals_to_tangents(self.hinges, normalize=False)
            self._curve = tangents_to_curve(tangents, center=True)
        return self._curve

    @property
    def binormals(self) -> np.ndarray:
        return self.hinges

    @property
    def tangents(self) -> np.ndarray:
        """Get tangent vectors, computing if necessary."""
        if self._tangents is None:
            self._tangents = binormals_to_tangents(self.hinges, normalize=True)
        return self._tangents

    @property
    def normals(self) -> np.ndarray:
        """Get normal vectors, computing if necessary."""
        if self._normals is None:
            # Normal = binormal × tangent
            T = self.tangents
            # For each hinge i, compute N[i] = B[i] × T[i-1] (with wraparound)
            n = len(T)
            N = np.zeros((n, 3))
            for i in range(n):
                N[i] = np.cross(self.hinges[i], T[(i - 1) % n])
                # Normalize
                norm = np.linalg.norm(N[i])
                if norm > 1e-10:
                    N[i] = N[i] / norm
            self._normals = N
        return self._normals

    @property
    def curvatures(self) -> np.ndarray:
        """Get tangent vectors, computing if necessary."""
        if self._curvatures is None:
            self._curvatures = pairwise_curvature(self.hinges, self.tangents)
        return self._curvatures

    @property
    def cosines(self) -> np.ndarray:
        """Get pairwise cosines, computing if necessary."""
        if self._cosines is None:
            self._cosines = pairwise_cosines(self.hinges)
        return self._cosines

    @property
    def mean_cosine(self) -> float:
        """Get mean cosine."""
        return float(np.mean(self.cosines))

    @property
    def is_closed(self) -> bool:
        """Check if the kaleidocycle satisfies the closure constraint.

        Checks if closure_residual (sum of tangents) is small.
        The closure constraint ensures that the tangent vectors sum to zero,
        forming a closed spatial polygon.

        Returns
        -------
        bool
            True if closure constraint is satisfied, False otherwise
        """
        from .constraints import closure_residual

        tolerance = 1e-3  # Tolerance for closure residual norm
        residual = closure_residual(self.hinges, slide=0.0)
        return bool(np.linalg.norm(residual) < tolerance)

    @property
    def is_aligned(self) -> bool:
        """Check if the kaleidocycle satisfies the alignment constraint.

        Checks if alignment_residuals (first and last hinge matching) is small.
        For oriented kaleidocycles, first and last hinges should be equal.
        For non-oriented, they should be opposite.

        Returns
        -------
        bool
            True if alignment constraint is satisfied, False otherwise
        """
        from .constraints import alignment_residuals

        tolerance = 1e-3  # Tolerance for alignment residual norm
        residual = alignment_residuals(self.hinges, oriented=self.oriented)
        return bool(residual < tolerance)

    @property
    def is_unit_norm(self) -> bool:
        """Check if all hinges have unit norm.

        Returns
        -------
        bool
            True if all hinges have norm approximately equal to 1, False otherwise
        """
        tolerance = 1e-6
        norms = np.linalg.norm(self.hinges, axis=1)
        return bool(np.allclose(norms, 1.0, rtol=tolerance, atol=tolerance))

    @property
    def constant_torsion(self) -> float | None:
        """Constant torsion value if torsion is constant, None otherwise.

        Computes the torsion and checks if it's constant (all torsion values
        are approximately equal). If constant, returns the average torsion
        value; otherwise returns None.

        Returns
        -------
        float | None
            Constant torsion value if torsion is constant, None otherwise
        """
        torsion = compute_torsion(self.hinges)

        # Check if all torsion values are approximately equal
        tolerance = 1e-4
        if np.std(torsion) < tolerance:
            return float(np.mean(torsion))
        else:
            return None

    def compute(
        self,
        props: list[str] | None = None,
        *,
        config: "ConstraintConfig | None" = None,
    ) -> None:
        """Compute specified properties and store in metadata.

        Args:
            props: List of property groups to compute. If None, compute all.
                   Available groups: 'geometric', 'topological', 'energies',
                   'constraints', 'objective', 'all'
            config: Constraint configuration needed for constraint calculations

        The metadata dictionary will be populated with:
            - 'geometric': mean_cosine, std_cosines, curvatures, torsions, axis
            - 'topological': writhe, twist, linking_number
            - 'energies': bending, dipole, torsion
            - 'constraints': residuals and penalties (requires config)
            - 'objective': objective function value (if applicable)

        Example:
            >>> kc.compute(['geometric', 'energies'])  # Compute specific properties
            >>> kc.compute()  # Compute all properties (requires config)
        """
        from .constraints import ConstraintConfig, constraint_residuals
        from .energies import bending_energy, dipole_energy, torsion_energy

        # Default to computing all properties
        if props is None:
            props = ["all"]

        compute_all = "all" in props

        # Compute geometric properties
        if compute_all or "geometric" in props:
            cosines = pairwise_cosines(self.hinges)
            curvatures = pairwise_curvature(self.hinges, self.tangents)
            torsions = compute_torsion(self.hinges)

            geometric = {
                "mean_cosine": float(np.mean(cosines)),
                "std_cosines": float(np.std(cosines)),
                "cosines": cosines,
                "curvatures": curvatures,
                "mean_curvature": float(np.mean(curvatures)),
                "torsions": torsions,
                "mean_torsion": float(np.mean(torsions)),
            }

            # Try to compute axis
            try:
                axis = compute_axis(self.hinges, curvatures)
                geometric["axis"] = axis
            except ValueError as e:
                geometric["axis"] = None
                geometric["axis_error"] = str(e)

            self.metadata["geometric"] = geometric

        # Compute topological properties
        if compute_all or "topological" in props:
            try:
                writhe_val = writhe(self.curve)
            except ValueError as e:
                writhe_val = None

            try:
                twist_val = total_twist_from_curve(self.curve)
            except ValueError as e:
                twist_val = None

            topological = {
                "writhe": writhe_val,
                "twist": twist_val,
            }

            if writhe_val is not None and twist_val is not None:
                topological["linking_number"] = writhe_val + twist_val
            else:
                topological["linking_number"] = None

            self.metadata["topological"] = topological

        # Compute energies
        if compute_all or "energies" in props:
            energies = {
                "bending": bending_energy(self.tangents),
                "dipole": dipole_energy(self.hinges, self.curve),
                "torsion": torsion_energy(self.hinges),
            }
            self.metadata["energies"] = energies

        # Compute constraint violations
        if compute_all or "constraints" in props:
            if config is None:
                if compute_all:
                    raise ValueError(
                        "config parameter is required to compute constraints"
                    )
                else:
                    # Skip constraints if not computing all
                    pass
            else:
                residuals = constraint_residuals(self.hinges, config)

                # Compute total penalty and max violations
                total_penalty = 0.0
                violations = {}
                for name, res_array in residuals.items():
                    if res_array.size > 0:
                        max_violation = float(np.max(np.abs(res_array)))
                        sum_sq = float(np.sum(res_array**2))
                        total_penalty += sum_sq
                        violations[name] = {
                            "max_abs": max_violation,
                            "sum_sq": sum_sq,
                            "residuals": res_array,
                        }

                self.metadata["constraints"] = {
                    "config": config,
                    "violations": violations,
                    "total_penalty": total_penalty,
                }

    def is_feasible(
        self,
        tolerance: float = 1e-4,
        config: "ConstraintConfig | None" = None,
    ) -> bool:
        """Check if the kaleidocycle satisfies constraints within tolerance.

        This method computes the constraint penalty (sum of squared residuals)
        and checks if it is less than the specified tolerance. If no config is
        provided, uses a default configuration with alignment, constant_torsion,
        and closure constraints enabled.

        Args:
            tolerance: Maximum allowed penalty (default 1e-4)
            config: Constraint configuration. If None, uses default with
                    alignment=True, constant_torsion=True, closure=True,
                    and enforce_anchors=False

        Returns:
            True if the constraint penalty is less than tolerance, False otherwise

        Example:
            >>> from kaleidocycle import Kaleidocycle, random_hinges
            >>> hinges = random_hinges(6, seed=42).as_array()
            >>> kc = Kaleidocycle(hinges=hinges)
            >>> kc.is_feasible()  # Check with default constraints
            False
            >>> kc.is_feasible(tolerance=1.0)  # Check with looser tolerance
            True
        """
        from .constraints import ConstraintConfig, constraint_penalty

        # Create default config if none provided
        if config is None:
            config = self.config
        # Compute constraint penalty
        penalty = constraint_penalty(self.hinges, config)

        return penalty < tolerance

    def is_stationary(
        self,
        energy: 'Literal["bending", "mean_cos"]' = "bending",
        *,
        tolerance: float = 1e-6,
        finite_diff_step: float = 1e-8,
        config: "ConstraintConfig | None" = None,
    ) -> dict:
        """Check if the kaleidocycle is at a stationary point for the given energy.

        This method checks whether the current kaleidocycle configuration represents
        a stationary point (critical point) of the specified energy function under
        the given constraints. A configuration is stationary if the gradient of the
        energy, when projected onto the tangent space of the constraint manifold,
        has norm less than the specified tolerance.

        Mathematically, this verifies the first-order Karush-Kuhn-Tucker (KKT)
        optimality condition for constrained optimization:
            ∇E(h) + Σ λ_i ∇g_i(h) = 0

        where E is the energy function, g_i are the constraint functions, and λ_i
        are the Lagrange multipliers.

        Args:
            energy: Energy function to check. Options:
                    - 'bending': Bobenko-Suris bending energy (function of tangents)
                    - 'mean_cos': Mean cosine, i.e., mean torsion (function of hinges)
            tolerance: Maximum allowed norm for the projected gradient.
                       Configurations with ||∇E_projected|| < tolerance are
                       considered stationary. Default is 1e-6.
            finite_diff_step: Step size for numerical differentiation via central
                              finite differences. Default is 1e-8.
            config: Constraint configuration. If None, uses the default configuration
                    for this kaleidocycle (with alignment=True, constant_torsion=True,
                    closure=True).

        Returns:
            Dictionary containing:
            - 'is_stationary': bool
                True if the configuration is at a stationary point
            - 'projected_gradient_norm': float
                Norm of the gradient projected onto the constraint tangent space.
                Small values indicate proximity to a stationary point.
            - 'gradient_norm': float
                Norm of the full gradient before projection.
            - 'constraint_penalty': float
                Sum of squared constraint residuals. Should be small for feasible
                configurations.
            - 'details': dict
                Additional diagnostic information including:
                - 'energy': Name of energy function used
                - 'tolerance': Tolerance value used
                - 'finite_diff_step': Step size used for finite differences
                - 'n_constraints': Number of constraint equations
                - 'n_variables': Number of variables (hinge components)
                - 'constraint_rank': Rank of constraint Jacobian matrix

        Notes:
            The method uses numerical differentiation to compute gradients, which
            may be sensitive to the choice of finite_diff_step. The default value
            of 1e-8 works well for most cases, but may need adjustment for
            ill-conditioned problems.

            For a configuration to be truly optimal (minimum), the second-order
            conditions (positive definiteness of the projected Hessian) would also
            need to be checked. This method only verifies the first-order condition.

        Examples:
            Check if an optimized kaleidocycle is at a stationary point:
            >>> from kaleidocycle import Kaleidocycle, ConstraintConfig
            >>> kc = Kaleidocycle(n=8, oriented=True)  # Creates optimized config
            >>> result = kc.is_stationary('bending')
            >>> result['is_stationary']
            True
            >>> result['projected_gradient_norm'] < 1e-5
            True

            Check with custom configuration:
            >>> config = ConstraintConfig(oriented=True, constant_torsion=False)
            >>> result = kc.is_stationary('mean_cos', tolerance=1e-5, config=config)
            >>> print(f"Stationary: {result['is_stationary']}")
            >>> print(f"Gradient norm: {result['projected_gradient_norm']:.2e}")
        """
        from typing import Literal
        from .optimality import check_stationarity

        # Use default config if not provided
        if config is None:
            config = self.config

        # Delegate to the optimality module
        return check_stationarity(
            self.hinges,
            energy,
            config,
            tolerance=tolerance,
            finite_diff_step=finite_diff_step,
        )

    def local_dof(
        self,
        config: "ConstraintConfig | None" = None,
        *,
        tol: float | None = None,
        return_basis: bool = False,
        subtract_rigid: bool = True,
        finite_diff_step: float = 1e-8,
        backend: str | None = None,
    ) -> dict:
        """Compute the local DoF of constraint-preserving motions.

        The constraint manifold M = {h : g(h) = 0} has tangent space
        ker(J) at the current hinge configuration, where J is the
        constraint Jacobian. This method returns dim(ker J) — by default
        with the three global rigid rotations quotiented out, since those
        always preserve every constraint and are usually treated as
        gauge.

        Args:
            config: Constraint configuration. If None, uses ``self.config``.
            tol: Singular-value cutoff for rank determination. None lets
                NumPy choose ``max(J.shape) * eps * max_sv``.
            return_basis: If True, return an orthonormal basis of the
                tangent space with shape ``(n+1, 3, dof)`` under the
                ``"basis"`` key.
            subtract_rigid: If True (default), subtract the dimension of
                global rigid rotations contained in the nullspace.
            finite_diff_step: Step size for the NumPy backend Jacobian.
            backend: Backend selector (``"numpy"`` or ``"jax"``).

        Returns:
            Dictionary with keys ``dof`` (post-quotient DoF), ``raw_dof``
            (nullspace dimension), ``rigid_dof`` (rigid directions
            removed), ``rank``, ``n_constraints``, ``n_variables``,
            ``singular_values``, ``tol``, and optionally ``basis``.

        Example:
            >>> from kaleidocycle import Kaleidocycle
            >>> kc = Kaleidocycle(n=8, oriented=True)
            >>> info = kc.local_dof()
            >>> info['dof']  # one-parameter family of motions
            1
        """
        from .optimality import local_dof

        if config is None:
            config = self.config

        return local_dof(
            self.hinges,
            config,
            tol=tol,
            return_basis=return_basis,
            subtract_rigid=subtract_rigid,
            finite_diff_step=finite_diff_step,
            backend=backend,
        )

    def finite_motion_dof(
        self,
        config: "ConstraintConfig | None" = None,
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
        backend: str | None = None,
    ) -> dict:
        """Estimate finite (nonlinear) motion DoF via continuation.

        Complements :meth:`local_dof`. Infinitesimal DoF counts the
        dimension of ``ker(J)`` (overcounts at singular points), while
        finite DoF measures the dimension of the constraint variety that
        is reachable by actual paths from the current configuration. See
        :func:`kaleidocycle.optimality.finite_motion_dof` for details on
        all keyword arguments.

        Args:
            config: Constraint configuration. If None, uses ``self.config``.
            step_size: Predictor step size.
            n_steps: Number of predictor-corrector steps per sample.
            n_samples: Number of tangent directions to try (defaults to
                ``max(2k, 8)`` where ``k`` is the infinitesimal DoF).
            correction_tol: Newton-corrector residual tolerance.
            max_newton_iter: Maximum Newton iterations per corrector.
            subtract_rigid: If True, exclude global rigid rotations.
            rank_tol: Relative SVD cutoff for the displacement matrix.
            seed: RNG seed for random direction sampling.
            return_paths: If True, include continuation paths.
            finite_diff_step: Step size for the NumPy-backend Jacobian.
            backend: Backend selector.

        Returns:
            Dictionary with ``finite_dof``, ``infinitesimal_dof``,
            ``rigid_dof``, sample counts, ``displacement_singular_values``,
            ``max_residual``, continuation parameters, and optionally
            ``paths``.

        Example:
            >>> from kaleidocycle import Kaleidocycle
            >>> kc = Kaleidocycle(n=8, oriented=True)
            >>> kc.finite_motion_dof(seed=0)["finite_dof"]
            1
        """
        from .optimality import finite_motion_dof

        if config is None:
            config = self.config

        return finite_motion_dof(
            self.hinges,
            config,
            step_size=step_size,
            n_steps=n_steps,
            n_samples=n_samples,
            correction_tol=correction_tol,
            max_newton_iter=max_newton_iter,
            subtract_rigid=subtract_rigid,
            rank_tol=rank_tol,
            nullspace_tol=nullspace_tol,
            seed=seed,
            return_paths=return_paths,
            finite_diff_step=finite_diff_step,
            backend=backend,
        )

    def find_nearby_stationary(
        self,
        energy: 'Literal["bending", "mean_cos"]' = "mean_cos",
        config: "ConstraintConfig | None" = None,
        *,
        tol: float = 1e-10,
        maxfev: int = 2000,
        correction_tol: float = 1e-8,
        max_newton_iter: int = 100,
        finite_diff_step: float = 1e-8,
        backend: str | None = None,
    ) -> "Kaleidocycle":
        """Return a new Kaleidocycle at the nearest critical point of ``energy``.

        Wraps :func:`kaleidocycle.optimality.find_nearby_stationary`. The
        returned ``Kaleidocycle`` carries a ``stationary_info`` attribute on
        its ``metadata`` dict with the residual norm, iteration count, and
        Euclidean distance from this configuration.

        For the projected gradient to actually vanish you must use a
        configuration with ``full_alignment=True``; the default scalar
        alignment leaves a permanent residual.
        """
        from .optimality import find_nearby_stationary

        if config is None:
            config = self.config

        info = find_nearby_stationary(
            self.hinges,
            config,
            energy,
            tol=tol,
            maxfev=maxfev,
            correction_tol=correction_tol,
            max_newton_iter=max_newton_iter,
            finite_diff_step=finite_diff_step,
            backend=backend,
        )
        new_kc = Kaleidocycle(hinges=info["hinges"], oriented=self.oriented)
        new_kc.metadata["stationary_info"] = {
            "projected_gradient_norm": info["projected_gradient_norm"],
            "n_eval": info["n_eval"],
            "success": info["success"],
            "distance": info["distance"],
            "energy": energy,
        }
        return new_kc

    def follow_motion(
        self,
        config: "ConstraintConfig | None" = None,
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
        backend: str | None = None,
    ) -> np.ndarray:
        """Continue along one tangent direction and return the path of hinge frames.

        Wraps :func:`kaleidocycle.optimality.follow_motion`. See that
        function for parameter semantics. Returns an array of shape
        ``(n_frames, n+1, 3)``.
        """
        from .optimality import follow_motion

        if config is None:
            config = self.config

        return follow_motion(
            self.hinges,
            config,
            direction_index=direction_index,
            step_size=step_size,
            n_steps=n_steps,
            bidirectional=bidirectional,
            correction_tol=correction_tol,
            max_newton_iter=max_newton_iter,
            subtract_rigid=subtract_rigid,
            nullspace_tol=nullspace_tol,
            finite_diff_step=finite_diff_step,
            backend=backend,
        )

    def report(
        self,
        config: "ConstraintConfig | None" = None,
        *,
        precision: int = 6,
    ) -> str:
        """Generate a human-readable report of kaleidocycle properties.

        This method wraps `format_report` to provide a convenient way to get
        a summary of geometric, topological, and constraint properties.

        Args:
            config: Constraint configuration. If None and constraints haven't been
                   computed, uses default configuration based on orientation.
            precision: Number of decimal places for float formatting (default 6)

        Returns:
            A formatted string containing the report

        Example:
            >>> from kaleidocycle import Kaleidocycle, random_hinges, ConstraintConfig
            >>> hinges = random_hinges(6, seed=42).as_array()
            >>> kc = Kaleidocycle(hinges=hinges)
            >>> print(kc.report())
            Kaleidocycle Property Report
            ============================
            ...
            >>> # With custom config
            >>> config = ConstraintConfig(oriented=True, constant_torsion=True)
            >>> print(kc.report(config=config))
        """
        from .report import format_report

        return format_report(kaleidocycle=self, config=config, precision=precision)

    def plot(
        self,
        ax: "Axes | None" = None,
        *,
        width: float = 0.15,
        facecolor: str = "lightblue",
        edgecolor: str = "darkblue",
        alpha: float = 0.7,
        linewidth: float = 0.5,
        title: str | None = None,
        show_curve: bool = False,
    ) -> "Axes":
        """Plot the band structure of the kaleidocycle.

        This method wraps `plot_band` to provide a convenient way to visualize
        the 3D structure of the kaleidocycle.

        Args:
            ax: Matplotlib 3D axes. If None, creates a new figure.
            width: Half-width of the tetrahedra along hinge directions (default 0.15)
            facecolor: Color for the faces (default "lightblue")
            edgecolor: Color for the edges (default "darkblue")
            alpha: Transparency of the faces (default 0.7)
            linewidth: Width of the edges (default 0.5)
            title: Optional title for the plot
            show_curve: If True, also plot the curve backbone (default False)

        Returns:
            Matplotlib 3D axes with the plot

        Example:
            >>> from kaleidocycle import Kaleidocycle, random_hinges
            >>> import matplotlib.pyplot as plt
            >>> hinges = random_hinges(6, seed=42).as_array()
            >>> kc = Kaleidocycle(hinges=hinges)
            >>> ax = kc.plot(title="My Kaleidocycle")
            >>> plt.show()
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from matplotlib.axes import Axes

        from .visualization import plot_band

        # Use cached curve property
        curve = self.curve

        # Default to plot_band (quadrilateral faces) for smoother appearance
        return plot_band(
            curve=curve,
            hinges=self.hinges,
            ax=ax,
            width=width,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            title=title,
            show_curve=show_curve,
        )

    def __repr__(self) -> str:
        """String representation of the Kaleidocycle."""
        return (
            f"Kaleidocycle(n={self.n}, oriented={self.oriented}, "
            f"computed_props={list(self.metadata.keys())})"
        )


def normalize_hinges(raw: Iterable[Iterable[float]]) -> HingeFrame:
    """Normalize a sequence of 3-vectors to unit length."""

    arr = np.asarray(list(raw), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n, 3) array, got shape {arr.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("hinge vector with zero length detected")
    return HingeFrame(arr / norms)


def binormals_to_tangents(
    binormals: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute tangent vectors from binormal vectors via cross product.

    Tangent vectors are computed as T_i = B_i × B_{i+1}, which gives the
    mid-axes of the kaleidocycle. These can optionally be normalized to
    unit length to obtain proper tangent vectors.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3)
        normalize: If True, normalize tangents to unit length (default True)

    Returns:
        Array of tangent vectors, shape (n, 3)

    References:
        Corresponds to B2T in Maple code (lines 304-305)
    """
    arr = np.asarray(binormals, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) binormal array, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.shape[0] < 2:
        raise ValueError("need at least two binormal vectors")

    T = np.cross(arr[:-1], arr[1:])

    if normalize:
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        T = T / norms

    return T


def is_oriented(hinges: np.ndarray) -> bool:
    """Check if the kaleidocycle is oriented based on hinge vectors.

    A kaleidocycle is considered oriented if the first and last hinge
    vectors are approximately equal (pointing in the same direction).

    Args:
        hinges: Array of hinge vectors, shape (n+1, 3)
    Returns:
        True if oriented, False otherwise
    """
    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.shape[0] < 2:
        raise ValueError("need at least two hinge vectors")

    first = arr[0] / np.linalg.norm(arr[0])
    last = arr[-1] / np.linalg.norm(arr[-1])
    cosine = np.clip(np.dot(first, last), -1.0, 1.0)

    return cosine > 0.9999  # Threshold for "approximately equal"


def tangents_to_curve(
    tangents: np.ndarray,
    *,
    scale: float = 1.0,
    center: bool = False,
) -> np.ndarray:
    """Convert tangent vectors to curve positions by accumulation.

    Args:
        tangents: Array of tangent vectors (mid-axes), shape (n, 3)
        scale: Scale factor for the curve (default 1.0)
        center: If True, center the curve at origin (default False)

    Returns:
        Array of 3D curve points, shape (n+1, 3)

    References:
        Corresponds to B2X accumulation in Maple code
    """
    segments = np.asarray(tangents, dtype=float) * scale
    points = np.zeros((segments.shape[0] + 1, 3), dtype=float)
    for i in range(segments.shape[0]):
        points[i + 1] = points[i] + segments[i]

    if center:
        centroid = np.mean(points, axis=0)
        points = points - centroid

    return points


def align_first_three(curve: np.ndarray) -> np.ndarray:
    """Rigidly align a curve so the first three vertices are fixed.

    After alignment:

    - ``curve[0]`` is the origin,
    - ``curve[1]`` lies on the positive ``x``-axis,
    - ``curve[2]`` sits in the ``xy``-plane with ``y >= 0``.

    This removes the entire 6-dim rigid-motion group (3 translation +
    3 rotation), making it convenient to visualize trajectories along
    a finite motion without rigid drift contaminating the picture.

    Parameters
    ----------
    curve : np.ndarray
        Array of shape ``(n_pts, 3)``.

    Returns
    -------
    np.ndarray
        Aligned copy of ``curve``, same shape.
    """
    c = np.asarray(curve, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or c.shape[0] < 3:
        raise ValueError(f"expected (n_pts>=3, 3) curve, got shape {c.shape}")
    c = c - c[0]

    v1 = c[1]
    n1 = float(np.linalg.norm(v1))
    if n1 < 1e-12:
        return c
    e = v1 / n1
    target = np.array([1.0, 0.0, 0.0])
    axis = np.cross(e, target)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(e, target))
    if sin_a < 1e-12:
        R1 = np.eye(3) if cos_a > 0 else np.diag([-1.0, -1.0, 1.0])
    else:
        axis = axis / sin_a
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        R1 = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    c = c @ R1.T

    y, z = c[2, 1], c[2, 2]
    r = float(np.hypot(y, z))
    if r >= 1e-12:
        cos_b = y / r
        sin_b = -z / r
        R2 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_b, -sin_b],
                [0.0, sin_b, cos_b],
            ]
        )
        c = c @ R2.T
    return c


def binormals_to_curve(
    binormals: np.ndarray,
    *,
    scale: float = 1.0,
    center: bool = True,
) -> np.ndarray:
    """Generate 3D curve from binormal vectors.

    Args:
        binormals: Array of binormal (hinge) vectors, shape (n+1, 3)
        scale: Scale factor for the curve (default 1.0)
        center: If True, center the curve at origin (default True)

    Returns:
        Array of 3D curve points, shape (n+1, 3)

    References:
        Equivalent to XYZ output in Maple, but computed from binormals
        rather than analytic theta functions

    Example:
        >>> binormals = random_hinges(6, seed=42).as_array()
        >>> curve = binormals_to_curve(binormals)
        >>> curve.shape
        (7, 3)
    """
    # Compute tangents from binormals (unnormalized mid-axes)
    tangents = binormals_to_tangents(binormals, normalize=False)

    # Accumulate to get curve points
    curve = tangents_to_curve(tangents, scale=scale, center=center)

    return curve


def random_hinges(
    n: int,
    *,
    seed: int | None = None,
    oriented: bool = False,
) -> HingeFrame:
    """Replicate the behaviour of ``RndH`` with deterministic seeding."""

    if n < 3:
        raise ValueError("need at least 3 hinges")
    rng = np.random.default_rng(seed)

    # First hinge matches Mathematica's hard-coded reference.
    hinges: list[np.ndarray] = [np.array([0.0, 0.0, 1.0], dtype=float)]

    # Second hinge only needs x=0; sample in the yz-plane and normalise.
    vec = rng.normal(size=2)
    vec /= np.linalg.norm(vec)
    hinges.append(np.array([0.0, vec[0], vec[1]], dtype=float))

    # Interior hinges come directly from random normals.
    for _ in range(n - 2):
        sample = rng.normal(size=3)
        sample /= np.linalg.norm(sample)
        hinges.append(sample)

    # Final hinge enforces orientation (mirrors Setup's handling of h[n+1]).
    last = np.array([0.0, 0.0, 1.0 if oriented else -1.0], dtype=float)
    hinges.append(last)
    return HingeFrame(np.vstack(hinges))


def pairwise_cosines(
    hinges: np.ndarray,
    *,
    wrap: bool = False,
) -> np.ndarray:
    """Normalised dot products between consecutive hinge vectors."""

    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)
    if wrap:
        a = arr
        b = np.roll(arr, -1, axis=0)
    else:
        a = arr[:-1]
        b = arr[1:]
    if a.size == 0:
        return np.array([], dtype=float)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero-length hinge vector encountered")
    dots = np.einsum("ij,ij->i", a, b)
    return np.clip(dots / norms, -1.0, 1.0)


def mean_cosine(
    hinges: np.ndarray,
    *,
    wrap: bool = False,
) -> float:
    """Average of the pairwise cosines."""

    cosines = pairwise_cosines(hinges, wrap=wrap)
    if cosines.size == 0:
        return 0.0
    return float(np.mean(cosines))


def alternating_layer_hinges(
    n: int,
    *,
    beta: float,
    phi: float,
    delta: float,
) -> HingeFrame:
    """Generate the alternating-layer ansatz for initial hinges."""

    if n % 2 != 0:
        raise ValueError("alternating-layer construction requires even n")
    vectors: list[list[float]] = []
    for i in range(n):
        layer = i % 2
        ang = i * delta + (layer * phi)
        z = (1 if layer == 0 else -1) * np.cos(beta)
        r = np.sin(beta)
        vectors.append([r * np.cos(ang), r * np.sin(ang), z])
    vectors.append(vectors[0])
    return HingeFrame(np.array(vectors, dtype=float))


def lwrithe(u: np.ndarray) -> float:
    """Compute local writhe contribution for four curve points using Levitt's formula.

    Computes the signed solid angle subtended by the geodesic quadrilateral
    formed by four consecutive curve segments.

    Args:
        u: Array of 4 points, shape (4, 3), representing [C[i], C[i+1], C[j], C[j+1]]

    Returns:
        Local writhe contribution (signed solid angle)

    References:
        Levitt, M. "Protein folding by restrained energy minimization and
        molecular dynamics." J. Mol. Biol. 170, 723-764 (1983).
    """
    if u.shape != (4, 3):
        msg = f"expected shape (4, 3), got {u.shape}"
        raise ValueError(msg)

    # Compute edge vectors
    r13 = u[2] - u[0]
    r14 = u[3] - u[0]
    r23 = u[2] - u[1]
    r24 = u[3] - u[1]

    # Compute normalized normals to the four triangular faces
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return np.zeros_like(v)
        return v / norm

    n1 = normalize(np.cross(r13, r14))
    n2 = normalize(np.cross(r14, r24))
    n3 = normalize(np.cross(r24, r23))
    n4 = normalize(np.cross(r23, r13))

    # Compute sign from triple product
    cross_prod = np.cross(u[3] - u[2], u[1] - u[0])
    sign_val = np.sign(np.dot(cross_prod, r13))
    # Note: Maple's sign(0) = 1, but NumPy's sign(0) = 0
    # Use 1.0 as default if sign is zero
    if sign_val == 0:
        sign_val = 1.0

    # Sum of angles (dihedral angles between adjacent faces)
    # Clamp to avoid numerical issues with arcsin
    angle_sum = (
        np.arcsin(np.clip(np.dot(n1, n2), -1.0, 1.0))
        + np.arcsin(np.clip(np.dot(n2, n3), -1.0, 1.0))
        + np.arcsin(np.clip(np.dot(n3, n4), -1.0, 1.0))
        + np.arcsin(np.clip(np.dot(n4, n1), -1.0, 1.0))
    )

    return float(sign_val * angle_sum)


def writhe(curve: np.ndarray) -> float:
    """Compute the writhe of a closed curve.

    The writhe measures the global entanglement of a closed space curve,
    computed as the sum of signed solid angles over all pairs of non-adjacent
    curve segments.

    Args:
        curve: Array of points defining the curve, shape (n, 3)

    Returns:
        Writhe value normalized by π

    References:
        Fuller, F. B. "The writhing number of a space curve."
        Proc. Natl. Acad. Sci. USA 68, 815-819 (1971).
    """
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n, 3) curve array, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.shape[0] < 4:
        raise ValueError("need at least 4 points to compute writhe")

    n = arr.shape[0]
    wr = 0.0

    # Sum over all pairs of non-adjacent segments
    # For curve with n points: j from 0 to n-4, i from j+2 to n-2
    for j in range(n - 3):
        for i in range(j + 2, n - 1):
            # Skip the wraparound case (last segment with first)
            # Maple skips when i=nops(C)-1 and j=1, which is i=n-2 and j=0 in Python
            if not (i == n - 2 and j == 0):
                segment = np.array([arr[i], arr[i + 1], arr[j], arr[j + 1]])
                wr += lwrithe(segment)

    return float(wr / np.pi)


def pairwise_curvature(
    binormals: np.ndarray,
    tangents: np.ndarray | None = None,
    *,
    signed: bool = True,
    oriented: bool = True,
) -> np.ndarray:
    """Compute pairwise curvature K from binormal vectors B (and optionally tangent vectors T).

    The discrete curvature is computed as the angle between consecutive tangent
    vectors, optionally with sign determined by the orientation of the Frenet frame.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3)
        tangents: Optional array of tangent vectors, shape (n, 3).
                  If None, computed from binormals via cross product.
        signed: If True, include sign based on orientation
        oriented: If True, the kaleidocycle is oriented (affects sign computation)

    Returns:
        Array of curvature values, shape (n,)

    References:
        Corresponds to B2K function in Maple code (line 305)
    """
    B = np.asarray(binormals, dtype=float)
    if B.ndim != 2 or B.shape[1] != 3:
        msg = f"expected (n+1, 3) binormal array, got shape {B.shape}"
        raise ValueError(msg)
    if B.shape[0] < 2:
        raise ValueError("need at least 2 binormals")

    # Compute tangents if not provided
    if tangents is None:
        T_raw = np.cross(B[:-1], B[1:])
        T = T_raw / np.linalg.norm(T_raw, axis=1, keepdims=True)
    else:
        T_arr = np.asarray(tangents, dtype=float)
        T = T_arr / np.linalg.norm(T_arr, axis=1, keepdims=True)

    n = T.shape[0]

    # Helper to handle modular indexing (1-indexed in Maple, 0-indexed here)
    def mod_n(i: int) -> int:
        return i % n

    # Compute signs if requested
    if signed:
        s = np.ones(n)
        for i in range(n):
            # sign((Cross[B[i], T[modN(i-1)]]).T[modN(i)])
            cross_prod = np.cross(B[i], T[mod_n(i - 1)])
            sign_val = np.sign(np.dot(cross_prod, T[mod_n(i)]))
            if sign_val == 0:
                sign_val = 1.0
            s[i] = sign_val
    else:
        s = np.ones(n)

    # Compute curvature as signed angle between consecutive tangents
    K = np.zeros(n)
    for i in range(n):
        cos_angle = np.clip(np.dot(T[mod_n(i - 1)], T[mod_n(i)]), -1.0, 1.0)
        K[i] = s[i] * np.arccos(cos_angle)

    return K


def compute_axis(binormals: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """Compute the axis vector A from binormals B and curvature K.

    Solves for the axis vector A such that:
        A · B[i] = tan(K[i]/2) for i = 0, 1, 2

    This determines the axis about which the kaleidocycle rotates.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3) where n >= 3
        curvature: Array of curvature values, shape (n,)

    Returns:
        Axis vector A, shape (3,)

    Raises:
        ValueError: If system is underdetermined or singular

    References:
        Corresponds to axis function in Maple code (line 295)
    """
    B = np.asarray(binormals, dtype=float)
    K = np.asarray(curvature, dtype=float)

    if B.ndim != 2 or B.shape[1] != 3:
        msg = f"expected (n+1, 3) binormal array, got shape {B.shape}"
        raise ValueError(msg)
    if B.shape[0] < 3:
        raise ValueError("need at least 3 binormals to determine axis")
    if K.shape[0] < 3:
        raise ValueError("need at least 3 curvature values")

    # Set up linear system: B[i] · A = tan(K[i]/2) for i = 0, 1, 2
    # This gives us 3 equations in 3 unknowns
    mat = B[:3, :]  # (3, 3) matrix
    rhs = np.tan(K[:3] / 2)  # (3,) vector

    # Solve the linear system
    try:
        axis = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError as e:
        msg = "singular system: cannot determine unique axis"
        raise ValueError(msg) from e

    return axis


def curve_to_tangents(
    curve: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Compute tangent vectors from curve points.

    Computes differences between consecutive points: T[i] = X[i+1] - X[i]

    Args:
        curve: Array of curve points, shape (n+1, 3)
        normalize: If True, normalize tangent vectors to unit length

    Returns:
        Array of tangent vectors, shape (n, 3)

    References:
        Corresponds to X2T function in Maple code (line 321)
    """
    X = np.asarray(curve, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        msg = f"expected (n+1, 3) curve array, got shape {X.shape}"
        raise ValueError(msg)
    if X.shape[0] < 2:
        raise ValueError("need at least 2 points to compute tangents")

    # Compute differences between consecutive points
    T = X[1:] - X[:-1]

    if normalize:
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        # Avoid division by zero
        if np.any(norms == 0):
            raise ValueError("zero-length tangent vector detected")
        T = T / norms

    return T


def tangents_to_binormals(
    tangents: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Compute binormal vectors from tangent vectors using Frenet frame.

    The binormals are computed as normalized cross products of consecutive
    tangents, with signs chosen to maintain continuity of the frame.

    Args:
        tangents: Array of tangent vectors, shape (n, 3)
        reference: Optional reference binormal for first frame, shape (3,).
                   Defaults to [0, 0, 1] if not provided.

    Returns:
        Array of binormal vectors, shape (n+1, 3)

    References:
        Corresponds to T2B function in Maple code (line 322)
    """
    T = np.asarray(tangents, dtype=float)
    if T.ndim != 2 or T.shape[1] != 3:
        msg = f"expected (n, 3) tangent array, got shape {T.shape}"
        raise ValueError(msg)
    if T.shape[0] < 2:
        raise ValueError("need at least 2 tangents to compute binormals")

    if reference is None:
        Bp0 = np.array([0.0, 0.0, 1.0])
    else:
        Bp0 = np.asarray(reference, dtype=float)
        if Bp0.shape != (3,):
            msg = f"reference binormal must have shape (3,), got {Bp0.shape}"
            raise ValueError(msg)

    n = T.shape[0]

    # Helper for modular indexing
    def mod_n(i: int) -> int:
        return i % n

    # Compute cross products: B[i] = T[i-1] × T[i] (with wraparound)
    B = np.zeros((n + 1, 3))
    for i in range(n + 1):
        B[i] = np.cross(T[mod_n(i - 1)], T[mod_n(i)])

    # Normalize all binormals
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    # Handle zero-length binormals (parallel tangents)
    for i in range(n + 1):
        if norms[i, 0] < 1e-10:
            # Parallel tangents - use previous binormal or reference
            if i > 0:
                B[i] = B[i - 1]
            else:
                B[i] = Bp0
        else:
            B[i] = B[i] / norms[i, 0]

    # Fix sign of first binormal to match reference
    if np.dot(B[0], Bp0) < 0:
        B[0] = -B[0]

    # Fix signs of subsequent binormals for continuity
    for i in range(1, n + 1):
        # sign((B[i-1] × B[i]) · T[i-1])
        cross = np.cross(B[i - 1], B[i])
        sign_val = np.sign(np.dot(cross, T[mod_n(i - 1)]))
        if sign_val == 0:
            sign_val = 1.0
        if sign_val < 0:
            B[i] = -B[i]

    return B


def curve_to_binormals(
    curve: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Compute binormal vectors directly from curve points.

    Convenience function that combines curve_to_tangents and tangents_to_binormals.

    Args:
        curve: Array of curve points, shape (n+1, 3)
        reference: Optional reference binormal for first frame, shape (3,)

    Returns:
        Array of binormal vectors, shape (n+1, 3)

    References:
        Corresponds to X2B function in Maple code (line 323)
    """
    T = curve_to_tangents(curve, normalize=False)
    return tangents_to_binormals(T, reference)


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


def compute_torsion(binormals: np.ndarray) -> np.ndarray:
    """Compute torsion angles between consecutive binormal vectors.

    The torsion angle is the angle of rotation about the tangent vector,
    measured as the angle between consecutive binormals.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3)

    Returns:
        Array of torsion angles in radians, shape (n,)

    References:
        Corresponds to torsion function in Maple code (line 294)
    """
    B = np.asarray(binormals, dtype=float)
    if B.ndim != 2 or B.shape[1] != 3:
        msg = f"expected (n+1, 3) binormal array, got shape {B.shape}"
        raise ValueError(msg)
    if B.shape[0] < 2:
        raise ValueError("need at least 2 binormals")

    # Compute angles between consecutive binormals
    n = B.shape[0] - 1
    torsion_angles = np.zeros(n)

    for i in range(n):
        # Compute dot product and clamp to [-1, 1]
        cos_angle = np.clip(np.dot(B[i], B[i + 1]), -1.0, 1.0)
        torsion_angles[i] = np.arccos(cos_angle)

    return torsion_angles


def total_twist(binormals: np.ndarray) -> float:
    """Compute total twist (sum of torsion angles) normalized by π.

    The total twist is the sum of all torsion angles around the closed curve,
    which measures the total rotation of the binormal frame.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3)

    Returns:
        Total twist normalized by π

    References:
        Corresponds to Tw function in Maple code (line 286)
    """
    torsion_angles = compute_torsion(binormals)
    return float(np.sum(torsion_angles) / np.pi)


def total_twist_from_curve(
    curve: np.ndarray,
    reference: np.ndarray | None = None,
) -> float:
    """Compute total twist directly from curve points.

    Convenience function that converts curve to binormals then computes total twist.

    Args:
        curve: Array of curve points, shape (n+1, 3)
        reference: Optional reference binormal for first frame, shape (3,)

    Returns:
        Total twist normalized by π

    References:
        Corresponds to TwX function in Maple code (line 290)
    """
    B = curve_to_binormals(curve, reference)
    return total_twist(B)


def _K2omega(
    K: np.ndarray,
    *,
    oriented: bool = True,
    mKdV: bool = False,
) -> np.ndarray:
    """Convert curvature K to omega angles for sine-Gordon/mKdV deformation.

    This is a helper function for cos_invariant that computes auxiliary angle
    variables (omega/phi) from curvature values. These angles are used in
    sine-Gordon and modified Korteweg-de Vries (mKdV) theory.

    Args:
        K: Array of curvature values, shape (n,)
        oriented: Whether the kaleidocycle is oriented
        mKdV: If True, use mKdV formula; if False, use sine-Gordon formula

    Returns:
        Array of omega angles, shape (n+1,)

    References:
        Corresponds to K2omega function in Maple code (line 383)
    """
    n = len(K)
    s = 1 if oriented else -1

    if mKdV:
        # mKdV case: -phi[i-1] - phi[i] = K[i]
        # This requires solving a linear system
        # For now, we'll use a simplified approach
        raise NotImplementedError("mKdV case not yet implemented")

    else:
        # sine-Gordon case
        if oriented:
            # phi[i] = phi[i-1] - K[i], with constraint sum(sin(phi[i])) = 0
            # We need to solve for the initial value p = phi[0]
            from scipy.optimize import fsolve

            def constraint(p):
                phi = np.zeros(n)
                phi[0] = p
                for i in range(1, n):
                    phi[i] = phi[i - 1] - K[i]
                return np.sum(np.sin(phi))

            # Solve for p using fsolve
            p_solution = fsolve(constraint, 0.0)[0]

            # Build phi array
            phi = np.zeros(n)
            phi[0] = p_solution
            for i in range(1, n):
                phi[i] = phi[i - 1] - K[i]

        else:
            # Non-oriented case: phi[i] = 0.5 * (sum(K[j] for j>i) - sum(K[j] for j<=i))
            phi = np.zeros(n)
            K_sum = np.sum(K)
            cumsum = 0.0
            for j in range(n):
                # sum from j+1 to n (exclusive end in Python)
                sum_right = K_sum - cumsum - K[j]
                # sum from 1 to j (inclusive, which is 0 to j in Python)
                sum_left = cumsum + K[j]
                phi[j] = 0.5 * (sum_right - sum_left)
                cumsum += K[j]

        # Return [phi[0], ..., phi[n-1], s*phi[0]]
        return np.append(phi, s * phi[0])


def cos_invariant(
    curvature: np.ndarray,
    *,
    oriented: bool = True,
    mKdV: bool = False,
) -> float:
    """Compute cosine-based invariant from curvature values.

    This function computes a conserved quantity (invariant) based on the sum
    of cosines of auxiliary angle variables derived from the curvature. This
    invariant is preserved under sine-Gordon or mKdV evolution.

    Args:
        curvature: Array of curvature values, shape (n,)
        oriented: Whether the kaleidocycle is oriented
        mKdV: If True, use mKdV formula; if False, use sine-Gordon formula (default)

    Returns:
        Sum of cosines of omega angles

    References:
        Corresponds to cos_invariant function in Maple code (line 194)

    Example:
        >>> K = pairwise_curvature(hinges, tangents)
        >>> inv = cos_invariant(K, oriented=True)
    """
    K = np.asarray(curvature, dtype=float)
    if K.ndim != 1:
        msg = f"expected 1D curvature array, got shape {K.shape}"
        raise ValueError(msg)

    # Compute omega angles
    omega = _K2omega(K, oriented=oriented, mKdV=mKdV)

    # Sum cosines (excluding the last element which is s*omega[0])
    n = len(K)
    return float(np.sum(np.cos(omega[:n])))


def from_curvatures_and_cos(
    curvatures: np.ndarray,
    cos_torsion: float,
    *,
    initial_binormal: np.ndarray | None = None,
    initial_tangent: np.ndarray | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create kaleidocycle hinges from curvatures and constant torsion cosine.

    Reconstructs binormal (hinge) vectors from a sequence of discrete curvatures
    and a constant torsion angle (specified via its cosine). Uses discrete Frenet
    frame evolution where T[i] = B[i] × B[i+1].

    Args:
        curvatures: Array of curvature values, shape (n,)
        cos_torsion: Cosine of the constant torsion angle between consecutive binormals
        initial_binormal: Optional initial binormal vector B[0], shape (3,).
                         Defaults to [0, 0, 1] if not provided.
        initial_tangent: Optional initial tangent vector T[0], shape (3,).
                        Must be perpendicular to initial_binormal.
                        Defaults to [0, 1, 0] if not provided.

    Returns:
        Tuple of (binormals, tangents):
        - binormals: Array of binormal vectors, shape (n+1, 3)
        - tangents: Array of tangent vectors, shape (n, 3)

    Example:
        >>> curvatures = np.array([0.5, 0.6, 0.5, 0.6, 0.5, 0.6])
        >>> cos_val = 0.8
        >>> B, T = from_curvatures_and_cos(curvatures, cos_val)
        >>> kc = Kaleidocycle(hinges=B)
    """
    K = np.asarray(curvatures, dtype=float)
    if K.ndim != 1:
        raise ValueError(f"curvatures must be 1D array, got shape {K.shape}")

    n = len(K)
    if n < 3:
        raise ValueError("need at least 3 curvatures")

    if not -1.0 <= cos_torsion <= 1.0:
        raise ValueError(f"cos_torsion must be in [-1, 1], got {cos_torsion}")

    sin_torsion = -np.sqrt(1 - cos_torsion**2)

    # Initialize arrays
    B = np.zeros((n + 1, 3))
    T = np.zeros((n, 3))

    # Set initial binormal B[0]
    if initial_binormal is None:
        B[0] = np.array([0.0, 0.0, 1.0])
    else:
        B[0] = np.asarray(initial_binormal, dtype=float)
        if B[0].shape != (3,):
            raise ValueError(f"initial_binormal must have shape (3,), got {B[0].shape}")
        B[0] = B[0] / np.linalg.norm(B[0])

    # Set initial tangent T[0] (perpendicular to B[0])
    if initial_tangent is None:
        T[0] = np.array([0.0, 1.0, 0.0])
        # Make perpendicular to B[0]
        T[0] = T[0] - np.dot(T[0], B[0]) * B[0]
        T[0] = T[0] / np.linalg.norm(T[0])
    else:
        T[0] = np.asarray(initial_tangent, dtype=float)
        if T[0].shape != (3,):
            raise ValueError(f"initial_tangent must have shape (3,), got {T[0].shape}")
        # Check perpendicularity
        if np.abs(np.dot(T[0], B[0])) > 1e-6:
            raise ValueError(
                "initial_tangent must be perpendicular to initial_binormal"
            )
        T[0] = T[0] / np.linalg.norm(T[0])

    # First step: compute B[1] from B[0] and T[0]
    # N[0] = B[0] × T[0]
    N0 = np.cross(B[0], T[0])
    N0 = N0 / np.linalg.norm(N0)

    # B[1] = cos(τ) * B[0] + sin(τ) * N[0]
    B[1] = cos_torsion * B[0] + sin_torsion * N0
    if normalize:
        B[1] = B[1] / np.linalg.norm(B[1])

    # Iterate to build the rest of the sequence
    for i in range(1, n):
        # Compute T[i] from T[i-1] and curvature K[i]
        # M = B[i] × T[i-1]
        M = np.cross(B[i], T[i - 1])
        if normalize:
            M = M / np.linalg.norm(M)

        # T[i] = cos(K[i]) * T[i-1] + sin(K[i]) * M
        T[i] = np.cos(K[i]) * T[i - 1] + np.sin(K[i]) * M
        if normalize:
            T[i] = T[i] / np.linalg.norm(T[i])

        # Compute B[i+1] from B[i] and T[i]
        # N[i] = B[i] × T[i]
        N = np.cross(B[i], T[i])
        if normalize:
            N = N / np.linalg.norm(N)

        # B[i+1] = cos(τ) * B[i] + sin(τ) * N[i]
        B[i + 1] = cos_torsion * B[i] + sin_torsion * N
        if normalize:
            B[i + 1] = B[i + 1] / np.linalg.norm(B[i + 1])

    return B, T


def curvature_recursion(
    curvature: np.ndarray,
    *,
    oriented: bool = True,
) -> np.ndarray:
    """Compute the curvature-recurrence quantity at every vertex.

    This legacy angle-coordinate API returns one quarter of
    ``integrable.variational_u``.  A configuration in the variational
    one-degree-of-freedom reduction has a *constant* result; the values do not
    generally vanish.  Use ``variational_recurrence_residual`` for a residual
    that is zero on the three-term recurrence.

    Args:
        curvature: Array of curvature values, shape (n,)
        oriented: Whether the kaleidocycle is oriented

    Returns:
        Array of recurrence values, shape (n,)

    References:
        Corresponds to curvature_recursion function in Maple code (line 232)

    Example:
        >>> K = pairwise_curvature(hinges, tangents)
        >>> residuals = curvature_recursion(K, oriented=True)
        >>> print(np.ptp(residuals))  # Small on the variational reduction
    """
    angles = np.asarray(curvature, dtype=float)
    if angles.ndim != 1:
        msg = f"expected 1D curvature array, got shape {angles.shape}"
        raise ValueError(msg)
    from .integrable import variational_u

    sign = 1 if oriented else -1
    # Keep the legacy diagnostic defined at an angle numerically equal to pi.
    # The canonical integrable chart intentionally rejects this nonregular
    # boundary, but reports should still be able to describe degenerate input.
    curvatures = 2.0 * np.tan(angles / 2.0)
    return 0.25 * variational_u(curvatures, sign=sign)


def curvature_recursion_from_tangents(
    tangents: np.ndarray,
    *,
    oriented: bool = True,
) -> np.ndarray:
    """Compute curvature recursion relation directly from tangent vectors.

    This is an alternative formulation of the curvature recursion that works
    directly with tangent vectors instead of curvature angles.

    Args:
        tangents: Array of tangent vectors, shape (n, 3)
        oriented: Whether the kaleidocycle is oriented

    Returns:
        Array of recursion residuals, shape (n,)

    References:
        Corresponds to curvature_recursion_T function in Maple code (line 233)

    Example:
        >>> T = binormals_to_tangents(hinges, normalize=True)
        >>> residuals = curvature_recursion_T(T, oriented=True)
    """
    T = np.asarray(tangents, dtype=float)
    if T.ndim != 2 or T.shape[1] != 3:
        msg = f"expected (n, 3) tangent array, got shape {T.shape}"
        raise ValueError(msg)

    n = len(T)

    # Normalize tangents to unit length
    T_norm = T / np.linalg.norm(T, axis=1, keepdims=True)

    # Helper for modular indexing
    def mod_n(i: int) -> int:
        return i % n

    # Compute tk[i] = (1 - T[i-1]·T[i]) / (1 + T[i-1]·T[i]) which is equivalent to tan^2(K[i]/2)
    tk = np.zeros(n)
    for i in range(n):
        dot_prod = np.dot(T_norm[mod_n(i - 1)], T_norm[mod_n(i)])
        # Clamp to avoid numerical issues
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        tk[i] = (1 - dot_prod) / (1 + dot_prod)

    # Initialize sign array
    s = np.ones(n)
    if not oriented:
        s[0] = -1

    # Compute recursion for each index
    result = np.zeros(n)
    for i in range(n):
        i_plus = (i + 1) % n
        i_minus = (i - 1) % n

        # -s[i]*sqrt(tk[i+1]*tk[i-1])*(1 + tk[i]) - tk[i]
        result[i] = -s[i] * np.sqrt(tk[i_plus] * tk[i_minus]) * (1 + tk[i]) - tk[i]

    return result


# Late import to avoid eager JSON dependency unless needed.
import json  # noqa: E402  (import at end to keep public API tidy)
