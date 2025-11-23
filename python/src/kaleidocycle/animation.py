"""Animation utilities for kaleidocycle evolution and processing.

This module provides tools for generating and processing animation sequences
of kaleidocycle configurations, including sine-Gordon flow evolution,
frame cleaning, and rotation alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .constraints import constraint_residuals, constraint_penalty, ConstraintConfig
from .geometry import (
    pairwise_curvature,
    binormals_to_tangents,
    curve_to_tangents,
    tangents_to_binormals,
    compute_torsion,
    pairwise_cosines,
    tangents_to_curve,
)


@dataclass
class KaleidocycleAnimation:
    """Container for kaleidocycle animation sequences with properties.

    Stores a sequence of kaleidocycle configurations (hinge frames) along with
    associated properties that evolve over time. Properties can be vertex-associated
    (e.g., curvature, torsion) or scalar (e.g., total energy, linking number).

    Attributes
    ----------
    frames : list[NDArray[np.float64]]
        List of hinge arrays, each with shape (n+1, 3) representing a frame
    evolution_rule : str
        Description of how the animation was generated
        (e.g., "mKdV", "sine_gordon", "step", "random", "manual")
    vertex_properties : dict[str, NDArray[np.float64]]
        Dictionary of vertex-based properties evolving over time
        Each value is shape (n_frames, n_vertices)
        Examples: "curvature", "torsion", "dot_products"
    scalar_properties : dict[str, NDArray[np.float64]]
        Dictionary of scalar properties evolving over time
        Each value is shape (n_frames,)
        Examples: "energy", "penalty", "linking_number"
    metadata : dict
        Additional metadata about the animation

    Examples
    --------
    Create animation from generated frames:
    >>> from kaleidocycle import generate_animation, KaleidocycleAnimation
    >>> frames = generate_animation(initial_hinges, steps=20)
    >>> anim = KaleidocycleAnimation(
    ...     frames=frames,
    ...     evolution_rule="sine_gordon",
    ... )

    Create animation from curves:
    >>> curves = [curve_frame1, curve_frame2, curve_frame3]
    >>> anim = KaleidocycleAnimation.from_curves(
    ...     curves,
    ...     evolution_rule="manual",
    ... )

    Add properties:
    >>> anim.compute_vertex_property("curvature")
    >>> anim.compute_scalar_property("energy", energy_func)

    Access frames and properties:
    >>> frame_10 = anim.frames[10]
    >>> curvatures = anim.vertex_properties["curvature"]  # shape (n_frames, n_vertices)
    >>> energies = anim.scalar_properties["energy"]  # shape (n_frames,)

    Visualize:
    >>> from kaleidocycle import plot_vertex_values
    >>> fig, anim_obj = plot_vertex_values(
    ...     anim.vertex_properties["curvature"],
    ...     title="Curvature Evolution"
    ... )
    """

    frames: list[NDArray[np.float64]]
    evolution_rule: str = "unknown"
    vertex_properties: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    scalar_properties: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate animation data."""
        if not self.frames:
            raise ValueError("frames list cannot be empty")

        # Check all frames have same shape
        shapes = [frame.shape for frame in self.frames]
        if len(set(shapes)) > 1:
            raise ValueError(f"all frames must have same shape, got {set(shapes)}")

        # Validate property dimensions
        n_frames = len(self.frames)
        n_vertices = self.frames[0].shape[0]

        for name, values in self.vertex_properties.items():
            if values.shape[0] != n_frames:
                raise ValueError(
                    f"vertex property '{name}' has {values.shape[0]} frames, "
                    f"expected {n_frames}"
                )
            if values.shape[1] not in [n_vertices, n_vertices - 1]:
                raise ValueError(
                    f"vertex property '{name}' has {values.shape[1]} vertices, "
                    f"expected {n_vertices} or {n_vertices - 1}"
                )

        for name, values in self.scalar_properties.items():
            if len(values) != n_frames:
                raise ValueError(
                    f"scalar property '{name}' has {len(values)} values, "
                    f"expected {n_frames}"
                )

    @classmethod
    def from_curves(
        cls,
        curves: list[NDArray[np.float64]],
        *,
        evolution_rule: str = "unknown",
        reference: NDArray[np.float64] | None = None,
        vertex_properties: dict[str, NDArray[np.float64]] | None = None,
        scalar_properties: dict[str, NDArray[np.float64]] | None = None,
        metadata: dict | None = None,
    ) -> 'KaleidocycleAnimation':
        """Create animation from a list of curve arrays.

        Converts curves to binormals (hinge vectors) and creates a
        KaleidocycleAnimation instance.

        Parameters
        ----------
        curves : list[NDArray[np.float64]]
            List of curve arrays, each with shape (n+1, 3) or (n, 3)
        evolution_rule : str, optional
            Description of how the animation was generated (default "unknown")
        reference : NDArray[np.float64], optional
            Reference binormal for first frame, shape (3,)
            Defaults to [0, 0, 1] if not provided
        vertex_properties : dict, optional
            Pre-computed vertex properties
        scalar_properties : dict, optional
            Pre-computed scalar properties
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        KaleidocycleAnimation
            Animation instance with frames as binormals

        Examples
        --------
        >>> from kaleidocycle import KaleidocycleAnimation
        >>> curves = [curve1, curve2, curve3]  # Each shape (n+1, 3)
        >>> anim = KaleidocycleAnimation.from_curves(curves)
        >>> anim.n_frames
        3
        """
        from .geometry import curve_to_binormals

        if not curves:
            raise ValueError("curves list cannot be empty")

        # Convert each curve to binormals
        frames = []
        for i, curve in enumerate(curves):
            curve_arr = np.asarray(curve, dtype=float)
            if curve_arr.ndim != 2 or curve_arr.shape[1] != 3:
                raise ValueError(
                    f"curve {i} must have shape (n, 3) or (n+1, 3), "
                    f"got {curve_arr.shape}"
                )

            # Use reference for first frame, then use previous binormal
            if i == 0:
                binormals = curve_to_binormals(curve_arr, reference)
            else:
                # Use first binormal of previous frame as reference for continuity
                binormals = curve_to_binormals(curve_arr, frames[-1][0])

            frames.append(binormals)

        # Create instance
        return cls(
            frames=frames,
            evolution_rule=evolution_rule,
            vertex_properties=vertex_properties or {},
            scalar_properties=scalar_properties or {},
            metadata=metadata or {},
        )

    @property
    def n_frames(self) -> int:
        """Number of frames in the animation."""
        return len(self.frames)

    @property
    def n_vertices(self) -> int:
        """Number of vertices (hinges) per frame."""
        return self.frames[0].shape[0]

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Shape of each frame (n_vertices, 3)."""
        return self.frames[0].shape

    def add_vertex_property(
        self,
        name: str,
        values: NDArray[np.float64],
        *,
        overwrite: bool = False,
    ) -> None:
        """Add a vertex-based property to the animation.

        Parameters
        ----------
        name : str
            Property name
        values : NDArray[np.float64]
            Property values, shape (n_frames, n_vertices) or (n_frames, n_vertices-1)
        overwrite : bool
            If True, overwrite existing property with same name
        """
        if name in self.vertex_properties and not overwrite:
            raise ValueError(f"property '{name}' already exists (use overwrite=True)")

        values_array = np.asarray(values, dtype=float)
        if values_array.ndim != 2:
            raise ValueError(f"expected 2D array, got shape {values_array.shape}")
        if values_array.shape[0] != self.n_frames:
            raise ValueError(
                f"expected {self.n_frames} frames, got {values_array.shape[0]}"
            )

        self.vertex_properties[name] = values_array

    def add_scalar_property(
        self,
        name: str,
        values: NDArray[np.float64],
        *,
        overwrite: bool = False,
    ) -> None:
        """Add a scalar property to the animation.

        Parameters
        ----------
        name : str
            Property name
        values : NDArray[np.float64]
            Property values, shape (n_frames,)
        overwrite : bool
            If True, overwrite existing property with same name
        """
        if name in self.scalar_properties and not overwrite:
            raise ValueError(f"property '{name}' already exists (use overwrite=True)")

        values_array = np.asarray(values, dtype=float)
        if values_array.ndim != 1:
            raise ValueError(f"expected 1D array, got shape {values_array.shape}")
        if len(values_array) != self.n_frames:
            raise ValueError(
                f"expected {self.n_frames} frames, got {len(values_array)}"
            )

        self.scalar_properties[name] = values_array

    def compute_vertex_property(
        self,
        property_name: str,
        *,
        overwrite: bool = False,
    ) -> NDArray[np.float64]:
        """Compute a standard vertex property for all frames.

        Parameters
        ----------
        property_name : str
            Property to compute. Options:
            - "curvature": Discrete curvature
            - "torsion": Torsion angles
            - "dot_products": Pairwise dot products (constant torsion check)
        overwrite : bool
            If True, recompute even if property exists

        Returns
        -------
        values : NDArray[np.float64]
            Computed property values, shape (n_frames, n_vertices)
        """
        if property_name in self.vertex_properties and not overwrite:
            return self.vertex_properties[property_name]

        values = []
        for frame in self.frames:
            if property_name == "curvature":
                values.append(pairwise_curvature(frame))
            elif property_name == "torsion":
                values.append(compute_torsion(frame))
            elif property_name == "dot_products":
                values.append(pairwise_cosines(frame))
            else:
                raise ValueError(
                    f"unknown property '{property_name}'. "
                    f"Options: 'curvature', 'torsion', 'dot_products'"
                )

        values_array = np.array(values)
        self.add_vertex_property(property_name, values_array, overwrite=True)
        return values_array

    def compute_scalar_property(
        self,
        property_name: str,
        func: callable = None,
        *,
        overwrite: bool = False,
    ) -> NDArray[np.float64]:
        """Compute a scalar property for all frames.

        Parameters
        ----------
        property_name : str
            Property name for storage
        func : callable
            Function that takes hinges array and returns a scalar
            If None, uses built-in property computation
        overwrite : bool
            If True, recompute even if property exists

        Returns
        -------
        values : NDArray[np.float64]
            Computed property values, shape (n_frames,)
        """
        if property_name in self.scalar_properties and not overwrite:
            return self.scalar_properties[property_name]

        if func is None:
            # Try to compute built-in properties
            if property_name == "penalty":
                from .constraints import ConstraintConfig

                config = ConstraintConfig()
                func = lambda h: constraint_penalty(h, config)
            elif property_name == "linking_number":
                from .solvers import compute_linking_number

                func = compute_linking_number
            else:
                raise ValueError(
                    f"no built-in computation for '{property_name}', "
                    f"please provide func argument"
                )

        values = np.array([func(frame) for frame in self.frames])
        self.add_scalar_property(property_name, values, overwrite=True)
        return values

    def align(self) -> None:
        """Align all frames to a common reference orientation."""
        self.frames = align_animation_frames(self.frames)

    def get_frame(self, index: int) -> NDArray[np.float64]:
        """Get a specific frame by index.

        Parameters
        ----------
        index : int
            Frame index (supports negative indexing)

        Returns
        -------
        frame : NDArray[np.float64]
            Hinge array, shape (n_vertices, 3)
        """
        return self.frames[index]

    def get_curves(self) -> list[NDArray[np.float64]]:
        """Compute 3D curves for all frames.

        Returns
        -------
        curves : list[NDArray[np.float64]]
            List of curve arrays, each shape (n_vertices, 3)
        """
        curves = []
        for frame in self.frames:
            tangents = binormals_to_tangents(frame, normalize=False)
            curve = tangents_to_curve(tangents)
            curves.append(curve)
        return curves

    def slice(self, start: int = None, stop: int = None, step: int = None):
        """Create a new animation from a slice of frames.

        Parameters
        ----------
        start, stop, step : int, optional
            Slice parameters (same as list slicing)

        Returns
        -------
        sliced : KaleidocycleAnimation
            New animation with sliced frames and properties
        """
        slice_obj = slice(start, stop, step)
        sliced_frames = self.frames[slice_obj]

        sliced_vertex_props = {
            name: values[slice_obj] for name, values in self.vertex_properties.items()
        }
        sliced_scalar_props = {
            name: values[slice_obj] for name, values in self.scalar_properties.items()
        }

        return KaleidocycleAnimation(
            frames=sliced_frames,
            evolution_rule=f"{self.evolution_rule}_sliced[{start}:{stop}:{step}]",
            vertex_properties=sliced_vertex_props,
            scalar_properties=sliced_scalar_props,
            metadata=self.metadata.copy(),
        )

    def __len__(self) -> int:
        """Return number of frames."""
        return self.n_frames

    def __getitem__(self, index: int | slice) -> NDArray[np.float64]:
        """Access frames by index or slice."""
        return self.frames[index]

    def __repr__(self) -> str:
        """String representation."""
        vertex_props = list(self.vertex_properties.keys())
        scalar_props = list(self.scalar_properties.keys())

        return (
            f"KaleidocycleAnimation(\n"
            f"  n_frames={self.n_frames},\n"
            f"  n_vertices={self.n_vertices},\n"
            f"  evolution_rule='{self.evolution_rule}',\n"
            f"  vertex_properties={vertex_props},\n"
            f"  scalar_properties={scalar_props}\n"
            f")"
        )


def binormals_to_normals(
    binormals: NDArray[np.float64],
    tangents: NDArray[np.float64],
    *,
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Compute normal vectors from binormals and tangents.

    The normal vectors are computed as N[i] = B[i] × T[i] for i = 0..n-1,
    and N[n] = B[n] × T[0] (wraparound), forming the third component of the
    Frenet-Serret frame.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3)
        tangents: Array of tangent vectors, shape (n, 3)
        normalize: If True, normalize normal vectors to unit length

    Returns:
        Array of normal vectors, shape (n+1, 3) including wraparound element

    References:
        Corresponds to B2N function in Maple code (line 325)
        Maple: nv := zip(CrossProduct, B, [op(T), T[1]])
    """
    B = np.asarray(binormals, dtype=float)
    T = np.asarray(tangents, dtype=float)

    if B.ndim != 2 or B.shape[1] != 3:
        msg = f"expected (n+1, 3) binormal array, got shape {B.shape}"
        raise ValueError(msg)
    if T.ndim != 2 or T.shape[1] != 3:
        msg = f"expected (n, 3) tangent array, got shape {T.shape}"
        raise ValueError(msg)

    n = T.shape[0]
    normals = np.zeros((n + 1, 3))

    # Compute normals: N[i] = B[i] × T[i] for i = 0..n-1
    for i in range(n):
        normals[i] = np.cross(B[i], T[i])

    # Wraparound: N[n] = B[n] × T[0]
    normals[n] = np.cross(B[n], T[0])

    if normalize:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # Avoid division by zero
        normals = np.where(norms > 1e-10, normals / norms, normals)

    return normals


def curvature_to_omega(
    curvature: NDArray[np.float64],
    *,
    oriented: bool = True,
    mkdv: bool = False,
) -> NDArray[np.float64]:
    """Convert curvature angles to auxiliary omega angles.

    The omega angles are used in sine-Gordon and mKdV evolution equations.
    For non-oriented kaleidocycles and sine-Gordon flow, uses a simplified
    formula based on cumulative sums.

    Args:
        curvature: Array of curvature values, shape (n,)
        oriented: Whether the kaleidocycle is oriented
        mkdv: If True, use mKdV formulation; otherwise sine-Gordon

    Returns:
        Array of omega angles, shape (n+1,) for compatibility with binormals

    References:
        Corresponds to K2omega function in Maple code (line 383)

    Note:
        The full implementation of the oriented+sine-Gordon case requires
        solving a nonlinear equation, which is simplified here.
    """
    K = np.asarray(curvature, dtype=float)
    n = K.shape[0]

    if not oriented and not mkdv:
        # Non-oriented sine-Gordon: simplified formula
        # phi[j] = 0.5 * (sum(K[j+1:n]) - sum(K[0:j]))
        phi = np.zeros(n)
        for j in range(n):
            phi[j] = 0.5 * (np.sum(K[j + 1 :]) - np.sum(K[: j + 1]))

        # Add wraparound element
        return np.append(phi, -phi[0])

    elif oriented and not mkdv:
        # Oriented sine-Gordon: requires solving nonlinear system
        # Simplified: use recursive relation phi[i] = phi[i-1] - K[i]
        phi = np.zeros(n + 1)
        # Start with arbitrary value, should be adjusted to satisfy constraints
        phi[0] = 0.0
        for i in range(1, n + 1):
            phi[i] = phi[i - 1] - K[i - 1]

        # Adjust to satisfy periodicity (simplified)
        correction = phi[n] / n
        phi -= np.linspace(0, correction * n, n + 1)

        return phi

    else:
        # mKdV formulation: -phi[i-1] - phi[i] = K[i]
        # Not fully implemented - return simple approximation
        phi = -np.cumsum(K) / 2
        return np.append(phi, phi[0] if oriented else -phi[0])


def sine_gordon_step(
    binormals: NDArray[np.float64],
    step_size: float = 0.01,
    *,
    rule: Literal["sine-Gordon", "mKdV", "mKdV2"] = "sine-Gordon",
    oriented: bool = True,
) -> NDArray[np.float64]:
    """Perform one step of sine-Gordon or mKdV evolution on binormals.

    Uses 4th-order Runge-Kutta integration to evolve the binormal frame
    according to the specified integrable evolution equation.

    Args:
        binormals: Array of binormal vectors, shape (n+1, 3)
        step_size: Time step for evolution (h parameter)
        rule: Evolution rule - "sine-Gordon", "mKdV", or "mKdV2"
        oriented: Whether the kaleidocycle is oriented

    Returns:
        Array of evolved binormal vectors, shape (n+1, 3)

    References:
        Corresponds to sGmoveB function in Maple code (line 419)
    """
    B = np.asarray(binormals, dtype=float)

    def diffB(b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute time derivative of binormals."""
        # Compute normalized tangents from binormals
        T = binormals_to_tangents(b, normalize=True)

        # Compute curvature
        K = pairwise_curvature(b, tangents=T, signed=True, oriented=oriented)

        # Compute normals
        U = binormals_to_normals(b, T, normalize=True)

        n = T.shape[0]

        if rule == "mKdV2":
            # mKdV2 formulation: dB/dt = -tan(K/2) * T - U
            s = 1 if oriented else -1
            dB = np.zeros((n, 3))
            for i in range(n):
                dB[i] = -np.tan(K[i] / 2) * T[i] - U[i]

            # Add wraparound element
            return np.vstack([dB, s * dB[0]])

        else:
            # Sine-Gordon or mKdV: dB/dt = -cos(omega) * U + sin(omega) * T
            mkdv = rule == "mKdV"
            omega = curvature_to_omega(K, oriented=oriented, mkdv=mkdv)

            dB = np.zeros((n + 1, 3))
            for i in range(n + 1):
                # Use modular indexing for T (T has n elements)
                # U now has n+1 elements including wraparound, so use U[i] directly
                t_idx = i % n
                dB[i] = -np.cos(omega[i]) * U[i] + np.sin(omega[i]) * T[t_idx]

            return dB

    # 4th-order Runge-Kutta integration
    k1 = diffB(B)
    k2 = diffB(B + 0.5 * step_size * k1)
    k3 = diffB(B + 0.5 * step_size * k2)
    k4 = diffB(B + step_size * k3)

    B_new = B + (step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Renormalize binormals to maintain unit length
    B_new = B_new / np.linalg.norm(B_new, axis=1, keepdims=True)

    return B_new


def generate_animation(
    binormals: NDArray[np.float64],
    num_frames: int = 100,
    step_size: float = 0.02,
    *,
    rule: Literal["sine-Gordon", "mKdV", "mKdV2", "step", "random"] = "sine-Gordon",
    oriented: bool = True,
    config: ConstraintConfig | None = None,
    objective: str = "mean_cos",
    seed: int | None = None,
    verbose: bool = False,
) -> list[NDArray[np.float64]]:
    """Generate animation sequence using specified evolution rule.

    Creates a sequence of kaleidocycle configurations by applying one of several
    evolution methods: sine-Gordon flow, step-based optimization, or random sampling.

    Args:
        binormals: Initial binormal configuration, shape (n+1, 3)
        num_frames: Number of animation frames to generate
        step_size: Time step for sine-Gordon/mKdV (default 0.02) or target distance
                   for step method (default 0.05 recommended for step)
        rule: Evolution rule:
              - "sine-Gordon": Sine-Gordon integrable flow
              - "mKdV": Modified Korteweg-de Vries flow
              - "mKdV2": Alternative mKdV formulation
              - "step": Step through configuration space with constraint optimization
              - "random": Random sampling with spatial sorting
        oriented: Whether the kaleidocycle is oriented (used for sine-Gordon/mKdV)
        config: Constraint configuration (used for step and random methods)
        objective: Objective function for random method (e.g., "mean_cos", "bending")
        seed: Random seed for reproducibility (used for random method)
        verbose: If True, print progress information (used for step and random)

    Returns:
        List of binormal arrays, each with shape (n+1, 3)

    References:
        - sine-Gordon/mKdV: anim_dd function in Maple code (line 457)
        - step: anim_step function in Maple code (line 423)
        - random: anim_rnd function in Maple code (line 447)

    Examples:
        Sine-Gordon evolution:
        >>> frames = generate_animation(hinges, num_frames=50, rule="sine-Gordon")

        Step-based optimization:
        >>> frames = generate_animation(hinges, num_frames=20, step_size=0.05,
        ...                             rule="step", verbose=True)

        Random sampling:
        >>> frames = generate_animation(hinges, num_frames=30, rule="random",
        ...                             seed=42, objective="bending")
    """
    if rule == "step":
        # Use step-based optimization
        return generate_animation_step(
            binormals,
            num_frames=num_frames,
            step_size=step_size,
            config=config,
            verbose=verbose,
        )
    elif rule == "random":
        # Use random sampling with sorting
        return generate_animation_random(
            binormals,
            num_frames=num_frames,
            config=config,
            objective=objective,
            seed=seed,
            verbose=verbose,
        )
    else:
        # Use sine-Gordon or mKdV flow
        frames = [np.asarray(binormals, dtype=float)]

        for _ in range(1, num_frames):
            next_frame = sine_gordon_step(
                frames[-1],
                step_size=step_size,
                rule=rule,
                oriented=oriented,
            )
            frames.append(next_frame)

        return frames


def clean_animation_frames(
    frames: list[NDArray[np.float64]],
    config: ConstraintConfig | None = None,
    *,
    tolerance: float = 0.001,
    fix_twist: bool = True,
    check_torsion: bool = False,
    torsion_tolerance: float = 0.1,
) -> tuple[list[NDArray[np.float64]], list[int]]:
    """Filter animation frames to remove infeasible configurations.

    Removes frames that violate constraints or have inconsistent torsion.
    Optionally flips twist sign if the second binormal component is negative.

    Args:
        frames: List of binormal arrays to clean
        config: Constraint configuration for feasibility checking
        tolerance: Maximum constraint violation allowed
        fix_twist: If True, flip twist sign when B[1,1] < 0
        check_torsion: If True, check for torsion drift from first frame
        torsion_tolerance: Maximum torsion drift allowed (only if check_torsion=True)

    Returns:
        Tuple of (cleaned_frames, kept_indices)
            - cleaned_frames: List of feasible binormal arrays
            - kept_indices: Indices of frames that were kept

    References:
        Corresponds to clean_anim function in Maple code (line 477)

    Note:
        For sine-Gordon flow, torsion is expected to evolve, so check_torsion
        should typically be False.
    """
    if not frames:
        return [], []

    if config is None:
        config = ConstraintConfig(enforce_anchors=False)

    # Get reference torsion from first frame
    reference_torsion = float(np.dot(frames[0][0], frames[0][1]))

    cleaned = []
    kept_indices = []

    for i, frame in enumerate(frames):
        skip_frame = False

        # Optionally check torsion consistency
        if check_torsion:
            current_torsion = float(np.dot(frame[0], frame[1]))
            torsion_diff = abs(current_torsion - reference_torsion)

            if torsion_diff > torsion_tolerance:
                skip_frame = True

        # Check constraint feasibility
        if not skip_frame:
            residuals = constraint_residuals(frame, config)
            total_violation = sum(
                np.linalg.norm(r) for r in residuals.values()
            )

            if total_violation > tolerance:
                skip_frame = True

        if skip_frame:
            continue

        # Optionally fix twist orientation
        if fix_twist and frame.shape[0] > 1 and frame[1, 1] < 0:
            # Flip signs in y-component to fix twist
            frame_fixed = frame.copy()
            frame_fixed[:, 1] *= -1
            cleaned.append(frame_fixed)
        else:
            cleaned.append(frame)

        kept_indices.append(i)

    return cleaned, kept_indices


def align_animation_frames(
    frames: list[NDArray[np.float64]],
    *,
    use_barycentre: bool = True,
) -> list[NDArray[np.float64]]:
    """Align animation frames to remove rigid rotations using Procrustes analysis.

    Uses SVD-based Procrustes alignment to find optimal rotation matrices
    that align each frame to the first frame, removing spurious global rotations.

    Following Maple's fix_rot, this function converts binormals to curves first,
    computes rotation from curve alignment, then applies to binormals.

    Args:
        frames: List of binormal arrays to align
        use_barycentre: If True, center curves at origin before alignment

    Returns:
        List of aligned binormal arrays

    References:
        Corresponds to fix_rot function in Maple code (line 514)
        Uses Procrustes/Kabsch algorithm for optimal rotation
    """
    if not frames or len(frames) < 2:
        return frames

    # Convert binormals to curves (following Maple's approach)
    curves = []
    for frame in frames:
        # Compute tangents from binormals (unnormalized)
        tangents = binormals_to_tangents(frame, normalize=False)
        # Convert to curve points
        curve = tangents_to_curve(tangents, scale=1.0, center=use_barycentre)
        # Take first n points (excluding the wraparound closing point)
        curves.append(curve[:-1])

    aligned = []
    reference_curve = curves[0]

    # For each frame, find optimal rotation to align curves
    for i, (frame, curve) in enumerate(zip(frames, curves)):
        # Compute cross-covariance matrix from curves
        # In Maple: A = Xs[1] @ Transpose(Xs[i]) where Xs are 3xN (column vectors)
        # In Python with row vectors (Nx3): H = ref.T @ curve
        H = reference_curve.T @ curve

        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)

        # Ensure proper rotation (det = 1)
        # Maple checks: if Determinant(A) < 0 then u := Multiply(u, DiagonalMatrix([1, 1, -1]))
        # We match Maple by checking det(H) instead of det(R)
        if np.linalg.det(H) < 0:
            # Fix reflection by flipping sign of last column of U
            U[:, -1] *= -1

        # Compute optimal rotation
        R = U @ Vt

        # Apply rotation to binormals (not curves)
        # In Maple with column vectors: R @ b for each binormal b
        # In Python with row vectors: b @ R.T for each binormal b
        aligned_frame = frame @ R.T

        aligned.append(aligned_frame)

    return aligned


def generate_animation_step(
    binormals: NDArray[np.float64],
    num_frames: int = 100,
    step_size: float = 0.05,
    config: ConstraintConfig | None = None,
    *,
    verbose: bool = False,
) -> list[NDArray[np.float64]]:
    """Generate animation by stepping through configuration space with uniform distances.

    Creates animation by finding successive configurations that minimize constraint
    violations while maintaining a target distance from previous frames.

    Args:
        binormals: Initial binormal configuration, shape (n+1, 3)
        num_frames: Number of animation frames to generate
        step_size: Target step size in configuration space (as fraction of radius)
        config: Constraint configuration for feasibility checking
        verbose: If True, print progress information

    Returns:
        List of binormal arrays, each with shape (n+1, 3)

    References:
        Corresponds to anim_step function in Maple code (line 423)

    Algorithm:
        For each frame:
        1. Use BFGS optimization to minimize constraint violations
        2. Add distance penalty to maintain target_distance from previous frame
        3. Append solution to animation sequence

    Note:
        This generates a path through configuration space by taking
        uniform steps while respecting constraints. Uses BFGS optimization
        rather than Newton solver for improved stability.
    """
    from scipy.optimize import minimize

    if config is None:
        config = ConstraintConfig(enforce_anchors=False)

    B0 = np.asarray(binormals, dtype=float)
    n = B0.shape[0] - 1
    frames = [B0]

    # Target step size (as distance in configuration space)
    target_distance = step_size

    if verbose:
        print(f"Generating {num_frames} frames with step size {step_size:.4f}")

    for i in range(1, num_frames):
        if verbose and i % 10 == 0:
            print(f"Frame {i}/{num_frames}")

        # Reference frame for distance measurement
        reference_frame = frames[-1]

        # Initial guess: small perturbation from previous frame
        if i == 1:
            # Random perturbation for first step
            direction = np.random.randn(*frames[-1].shape)
            direction = direction / np.linalg.norm(direction)
            x0 = frames[-1] + direction * target_distance * 0.5
        else:
            # Extrapolate from velocity
            velocity = frames[-1] - frames[-2]
            x0 = frames[-1] + velocity

        # Normalize initial guess
        x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

        # Define objective: minimize constraint violations + distance penalty
        def objective(x_flat: NDArray[np.float64]) -> float:
            B = x_flat.reshape(-1, 3)

            # Constraint penalty
            penalty = constraint_penalty(B, config)

            # Distance penalty (soft constraint to maintain target distance)
            dist = np.linalg.norm(B - reference_frame)
            distance_penalty = 100.0 * (dist - target_distance) ** 2

            return penalty + distance_penalty

        try:
            # Optimize
            result = minimize(
                objective,
                x0.ravel(),
                method='BFGS',
                options={'maxiter': 200, 'disp': False},
            )

            # Reshape and normalize
            B_new = result.x.reshape(-1, 3)
            B_new = B_new / np.linalg.norm(B_new, axis=1, keepdims=True)

            actual_dist = np.linalg.norm(B_new - reference_frame)
            penalty_val = constraint_penalty(B_new, config)

            # Accept if reasonably close to target distance
            if abs(actual_dist - target_distance) < target_distance * 2.0:
                frames.append(B_new)
                if verbose and (i <= 3 or i % 10 == 0):
                    print(f"  Frame {i}: distance={actual_dist:.4f} "
                          f"(target={target_distance:.4f}), penalty={penalty_val:.6f}")
            else:
                # Take smaller step if full step failed
                direction = (B_new - reference_frame) / np.linalg.norm(B_new - reference_frame)
                B_new = reference_frame + direction * target_distance
                B_new = B_new / np.linalg.norm(B_new, axis=1, keepdims=True)
                frames.append(B_new)
                if verbose:
                    print(f"  Frame {i}: adjusted step, distance={target_distance:.4f}")

        except Exception as e:
            if verbose:
                print(f"  Frame {i}: error {e}, repeating previous frame")
            frames.append(frames[-1])

    return frames


def sort_animation_frames(
    frames: list[NDArray[np.float64]],
    *,
    verbose: bool = False,
) -> list[NDArray[np.float64]]:
    """Sort animation frames by spatial proximity using nearest neighbor algorithm.

    Reorders frames so that consecutive frames are spatially close, creating
    a smooth animation path. Uses greedy nearest-neighbor approach starting
    from the first frame.

    Args:
        frames: List of binormal arrays to sort
        verbose: If True, print progress information

    Returns:
        Sorted list of binormal arrays

    References:
        Corresponds to sort_anim function in Maple code (line 502)

    Algorithm:
        1. Start with first frame
        2. Find closest remaining frame to current frame
        3. Add it to sorted list and remove from remaining
        4. Repeat until all frames are sorted

    Note:
        This is particularly useful after generate_animation_random, which
        produces frames in random order.
    """
    if not frames or len(frames) <= 1:
        return frames

    if verbose:
        print(f"Sorting {len(frames)} animation frames by spatial proximity...")

    # Start with first frame
    sorted_frames = [frames[0]]
    remaining = list(frames[1:])

    # Greedily add nearest frame
    while remaining:
        current = sorted_frames[-1]

        # Find closest frame to current
        min_dist = float('inf')
        min_idx = 0
        for i, frame in enumerate(remaining):
            dist = np.linalg.norm(frame - current)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        # Add closest frame and remove from remaining
        sorted_frames.append(remaining[min_idx])
        remaining.pop(min_idx)

        if verbose and len(sorted_frames) % 10 == 0:
            print(f"  Sorted {len(sorted_frames)}/{len(frames)} frames")

    if verbose:
        print(f"  Sorting complete")

    return sorted_frames


def generate_animation_random(
    binormals: NDArray[np.float64],
    num_frames: int = 100,
    config: ConstraintConfig | None = None,
    *,
    objective: str = "mean_cos",
    seed: int | None = None,
    verbose: bool = False,
) -> list[NDArray[np.float64]]:
    """Generate animation by randomly sampling configuration space.

    Creates animation by repeatedly optimizing from random initial
    configurations, producing a diverse set of feasible structures.
    Frames are automatically sorted by spatial proximity for smooth animation.

    Args:
        binormals: Initial binormal configuration, shape (n+1, 3)
        num_frames: Number of animation frames to generate
        config: Constraint configuration for feasibility checking
        objective: Objective function name for optimization
        seed: Random seed for reproducibility
        verbose: If True, print progress information

    Returns:
        List of binormal arrays sorted by spatial proximity, each with shape (n+1, 3)

    References:
        Corresponds to anim_rnd function in Maple code (line 447)

    Algorithm:
        For each frame:
        1. Generate random initial configuration
        2. Optimize to satisfy constraints
        3. Append result to animation sequence
        4. Sort all frames by spatial proximity (sort_anim)

    Note:
        This explores the configuration space more broadly than
        generate_animation_step, useful for finding diverse
        feasible structures. The final sorting ensures smooth animation.
    """
    from .solvers import optimize_cycle, SolverOptions
    from .geometry import random_hinges

    if config is None:
        config = ConstraintConfig(enforce_anchors=False)

    if seed is not None:
        np.random.seed(seed)

    B0 = np.asarray(binormals, dtype=float)
    n = B0.shape[0] - 1
    frames = [B0]

    if verbose:
        print(f"Generating {num_frames} random frames")

    for i in range(1, num_frames):
        if verbose and i % 10 == 0:
            print(f"Frame {i}/{num_frames}")

        # Generate random initial configuration
        initial = random_hinges(n, seed=None if seed is None else seed + i).as_array()

        # Optimize from random start
        try:
            result = optimize_cycle(
                initial,
                config,
                objective=objective,
                options=SolverOptions(
                    penalty_weight=100.0,
                    method="Powell",
                    maxiter=500,
                ),
            )

            frames.append(result.hinges)

            if verbose and i <= 5:
                print(f"  Frame {i}: energy={result.energy:.6f}, penalty={result.penalty:.6f}")

        except Exception as e:
            if verbose:
                print(f"  Frame {i}: error {e}, using random hinges")
            frames.append(initial)

    # Sort frames by spatial proximity for smooth animation
    if verbose:
        print(f"\nSorting frames for smooth animation...")

    frames_sorted = sort_animation_frames(frames, verbose=verbose)

    return frames_sorted


def generate_curve_animation(
    frames: list[NDArray[np.float64]],
    *,
    scale: float = 1.0,
    center: bool = True,
) -> list[NDArray[np.float64]]:
    """Generate 3D curve animation from binormal animation frames.

    Converts a sequence of binormal configurations into corresponding
    3D curve coordinates, suitable for visualization and analysis.

    Args:
        frames: List of binormal arrays, each with shape (n+1, 3)
        scale: Scale factor for the curves (default 1.0)
        center: If True, center each curve at origin (default True)

    Returns:
        List of 3D curve arrays, each with shape (n+1, 3)

    References:
        Equivalent to generating XYZ animation in Maple (line 793),
        but computed from binormals rather than analytic solutions

    Example:
        >>> binormals = random_hinges(6, seed=42).as_array()
        >>> frames = generate_animation(binormals, num_frames=20)
        >>> curves = generate_curve_animation(frames)
        >>> len(curves)
        20
        >>> curves[0].shape
        (7, 3)

    Note:
        This is a practical alternative to Maple's XYZ function, which
        uses Jacobi theta functions for exact analytic solutions. This
        function works with any binormal sequence from optimization or
        sine-Gordon flow.
    """
    from .geometry import binormals_to_tangents, tangents_to_curve

    curves = []
    for frame in frames:
        # Compute tangents from binormals (unnormalized)
        tangents = binormals_to_tangents(frame, normalize=False)

        # Accumulate to get curve points
        curve = tangents_to_curve(tangents, scale=scale, center=center)

        curves.append(curve)

    return curves
