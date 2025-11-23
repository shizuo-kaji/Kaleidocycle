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
            seed: Random seed for initialization (only used when n is provided)
            solver_options: Optional dict with solver parameters when creating from n
                           (e.g., {"maxiter": 1000, "use_constraint_solver": True})

        Raises:
            ValueError: If none or multiple initialization parameters are provided
            NotImplementedError: If normals initialization is requested

        Examples:
            Create optimized kaleidocycle from n:
            >>> kc = Kaleidocycle(9)  # Creates oriented kaleidocycle with 9 tetrahedra
            >>> kc = Kaleidocycle(8, oriented=False)  # Creates non-oriented with 8

            Create from existing data:
            >>> kc = Kaleidocycle(hinges=hinges_array)
            >>> kc = Kaleidocycle(curve=curve_array)
        """
        from .constraints import ConstraintConfig

        # Check that exactly one initialization parameter is provided
        init_params = sum([
            n is not None,
            hinges is not None,
            curve is not None,
            tangents is not None,
            normals is not None,
        ])

        if init_params == 0:
            raise ValueError("Must provide one of: n, hinges, curve, tangents, or normals")
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

        # Initialize from n by creating optimized kaleidocycle
        if n is not None:
            if not isinstance(n, int) or n < 3:
                raise ValueError(f"n must be an integer >= 3, got {n}")

            # Default oriented to False if not specified
            if oriented is None:
                oriented = False
            self.oriented = oriented

            # Create optimized kaleidocycle
            self.hinges = self._create_optimized(n, oriented, seed, solver_options, config=self.config)
            self.n = n
            self.oriented = oriented

            # Initialize cached properties
            self._curve: np.ndarray | None = None
            self._tangents: np.ndarray | None = None
            self._normals: np.ndarray | None = None
            self._curvatures: np.ndarray | None = None
            self._cosines: np.ndarray | None = None
            self.metadata: dict[str, any] = {"created_from": "optimize_cycle"}
            return

        # Initialize from the provided parameter
        if hinges is not None:
            self.hinges = np.asarray(hinges, dtype=float)
            if self.hinges.ndim != 2 or self.hinges.shape[1] != 3:
                raise ValueError(f"hinges must have shape (n+1, 3), got {self.hinges.shape}")

        elif curve is not None:
            curve_arr = np.asarray(curve, dtype=float)
            if curve_arr.ndim != 2 or curve_arr.shape[1] != 3:
                raise ValueError(f"curve must have shape (n+1, 3), got {curve_arr.shape}")
            self.hinges = curve_to_binormals(curve_arr)

        elif tangents is not None:
            tangents_arr = np.asarray(tangents, dtype=float)
            if tangents_arr.ndim != 2 or tangents_arr.shape[1] != 3:
                raise ValueError(f"tangents must have shape (n, 3), got {tangents_arr.shape}")
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

    def compute(
        self,
        props: list[str] | None = None,
        *,
        config: 'ConstraintConfig | None' = None,
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
            props = ['all']

        compute_all = 'all' in props

        # Compute geometric properties
        if compute_all or 'geometric' in props:
            cosines = pairwise_cosines(self.hinges)
            curvatures = pairwise_curvature(self.hinges, self.tangents)
            torsions = compute_torsion(self.hinges)

            geometric = {
                'mean_cosine': float(np.mean(cosines)),
                'std_cosines': float(np.std(cosines)),
                'cosines': cosines,
                'curvatures': curvatures,
                'mean_curvature': float(np.mean(curvatures)),
                'torsions': torsions,
                'mean_torsion': float(np.mean(torsions)),
            }

            # Try to compute axis
            try:
                axis = compute_axis(self.hinges, curvatures)
                geometric['axis'] = axis
            except ValueError as e:
                geometric['axis'] = None
                geometric['axis_error'] = str(e)

            self.metadata['geometric'] = geometric

        # Compute topological properties
        if compute_all or 'topological' in props:
            try:
                writhe_val = writhe(self.curve)
            except ValueError as e:
                writhe_val = None

            try:
                twist_val = total_twist_from_curve(self.curve)
            except ValueError as e:
                twist_val = None

            topological = {
                'writhe': writhe_val,
                'twist': twist_val,
            }

            if writhe_val is not None and twist_val is not None:
                topological['linking_number'] = writhe_val + twist_val
            else:
                topological['linking_number'] = None

            self.metadata['topological'] = topological

        # Compute energies
        if compute_all or 'energies' in props:
            energies = {
                'bending': bending_energy(self.tangents),
                'dipole': dipole_energy(self.hinges, self.curve),
                'torsion': torsion_energy(self.hinges),
            }
            self.metadata['energies'] = energies

        # Compute constraint violations
        if compute_all or 'constraints' in props:
            if config is None:
                if compute_all:
                    raise ValueError("config parameter is required to compute constraints")
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
                            'max_abs': max_violation,
                            'sum_sq': sum_sq,
                            'residuals': res_array,
                        }

                self.metadata['constraints'] = {
                    'config': config,
                    'violations': violations,
                    'total_penalty': total_penalty,
                }

    def is_feasible(
        self,
        tolerance: float = 1e-4,
        config: 'ConstraintConfig | None' = None,
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
            config = ConstraintConfig(
                oriented=self.oriented,
                alignment=True,
                constant_torsion=True,
                enforce_anchors=False,  # Don't check anchors by default
                slide=0.0,
            )

        # Compute constraint penalty
        penalty = constraint_penalty(self.hinges, config)

        return penalty < tolerance

    def report(
        self,
        config: 'ConstraintConfig | None' = None,
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
        ax: 'Axes | None' = None,
        *,
        width: float = 0.15,
        facecolor: str = "lightblue",
        edgecolor: str = "darkblue",
        alpha: float = 0.7,
        linewidth: float = 0.5,
        title: str | None = None,
        show_curve: bool = False,
    ) -> 'Axes':
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
    """Generate the alternating-layer ansatz used for the synthetic legacy sample."""

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
        cos_angle = np.clip(
            np.dot(T[mod_n(i - 1)], T[mod_n(i)]), -1.0, 1.0
        )
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


def curvature_recursion(
    curvature: np.ndarray,
    *,
    oriented: bool = True,
) -> np.ndarray:
    """Compute curvature recursion relation residuals.

    This function evaluates the discrete curvature recursion formula, which
    expresses a constraint relating curvature values at consecutive indices.
    The result should be close to zero for valid kaleidocycle configurations.

    Args:
        curvature: Array of curvature values, shape (n,)
        oriented: Whether the kaleidocycle is oriented

    Returns:
        Array of recursion residuals, shape (n,)

    References:
        Corresponds to curvature_recursion function in Maple code (line 232)

    Example:
        >>> K = pairwise_curvature(hinges, tangents)
        >>> residuals = curvature_recursion(K, oriented=True)
        >>> print(np.max(np.abs(residuals)))  # Should be small for valid config
    """
    K = np.asarray(curvature, dtype=float)
    if K.ndim != 1:
        msg = f"expected 1D curvature array, got shape {K.shape}"
        raise ValueError(msg)

    n = len(K)

    # Initialize sign array
    s = np.ones(n)
    if not oriented:
        s[0] = -1
        s[-1] = -1

    # Compute recursion for each index
    # K[i+1], K[i-1], K[i] with wraparound
    result = np.zeros(n)
    for i in range(n):
        i_plus = (i + 1) % n
        i_minus = (i - 1) % n

        tan_i = np.tan(K[i] / 2)
        tan_plus = np.tan(K[i_plus] / 2)
        tan_minus = np.tan(K[i_minus] / 2)

        # s[i]*tan(K[i+1]/2)*tan(K[i-1]/2) - tan(K[i]/2)^2
        # + s[i]*tan(K[i+1]/2)*tan(K[i-1]/2)*tan(K[i]/2)^2
        result[i] = (
            s[i] * tan_plus * tan_minus
            - tan_i**2
            + s[i] * tan_plus * tan_minus * tan_i**2
        )

    return result


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
