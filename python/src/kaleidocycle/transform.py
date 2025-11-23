"""Transformation operations for Kaleidocycles.

This module provides geometric transformations that create new Kaleidocycles (hinges)
from existing ones. These operations preserve the topological and geometric properties
of the kaleidocycle while producing related configurations.

References:
    Corresponds to transformation operations in Maple code (lines 162-173):
    mirror, involution, negativeTwist, reverse
"""

from __future__ import annotations

import numpy as np


def mirror(hinges: np.ndarray) -> np.ndarray:
    """Apply mirror transformation to hinges.

    The mirror operation alternates the sign of each binormal vector:
    B'[i] = (-1)^(i-1) * B[i]

    This transformation reflects the kaleidocycle, creating a mirror image
    configuration.

    Args:
        hinges: Array of hinge (binormal) vectors, shape (n+1, 3)

    Returns:
        Array of mirrored hinge vectors, shape (n+1, 3)

    References:
        Corresponds to mirror function in Maple code (line 162)

    Example:
        >>> from kaleidocycle import random_hinges, mirror
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> mirrored = mirror(hinges)
        >>> mirrored.shape
        (7, 3)
    """
    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)

    n = arr.shape[0]
    result = np.zeros_like(arr)

    for i in range(n):
        # Maple uses 1-based indexing, so i-1 in Maple corresponds to i in Python
        # For i=0: (-1)^(-1) = -1
        # For i=1: (-1)^(0) = 1
        # For i=2: (-1)^(1) = -1
        sign = (-1) ** i
        result[i] = sign * arr[i]

    return result


def involution(hinges: np.ndarray) -> np.ndarray:
    """Apply involution transformation to hinges.

    The involution operation alternates the signs of each component differently:
    B'[i] = [(-1)^(i-1)*x, (-1)^i*y, (-1)^(i-1)*z]

    This is a more complex reflection that affects each coordinate independently.

    Args:
        hinges: Array of hinge (binormal) vectors, shape (n+1, 3)

    Returns:
        Array of transformed hinge vectors, shape (n+1, 3)

    References:
        Corresponds to involution function in Maple code (line 165)

    Example:
        >>> from kaleidocycle import random_hinges, involution
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> inv = involution(hinges)
        >>> inv.shape
        (7, 3)
    """
    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)

    n = arr.shape[0]
    result = np.zeros_like(arr)

    for i in range(n):
        # In Maple (1-indexed): [(-1)^(i-1)*x, (-1)^i*y, (-1)^(i-1)*z]
        # In Python (0-indexed): [(-1)^i*x, (-1)^(i+1)*y, (-1)^i*z]
        sign_xz = (-1) ** i
        sign_y = (-1) ** (i + 1)
        result[i] = np.array([
            sign_xz * arr[i, 0],
            sign_y * arr[i, 1],
            sign_xz * arr[i, 2],
        ])

    return result


def negative_twist(hinges: np.ndarray) -> np.ndarray:
    """Apply negative twist transformation to hinges.

    The negative twist operation negates only the y-component of each vector:
    B'[i] = [x, -y, z]

    This transformation reverses the handedness of the twist.

    Args:
        hinges: Array of hinge (binormal) vectors, shape (n+1, 3)

    Returns:
        Array of transformed hinge vectors, shape (n+1, 3)

    References:
        Corresponds to negativeTwist function in Maple code (line 168)

    Example:
        >>> from kaleidocycle import random_hinges, negative_twist
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> neg_twist = negative_twist(hinges)
        >>> neg_twist.shape
        (7, 3)
    """
    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)

    result = arr.copy()
    result[:, 1] = -result[:, 1]

    return result


def subdivide(
    hinges: np.ndarray,
    divisions: int = 2,
    *,
    method: str = 'slerp',
) -> np.ndarray:
    """Subdivide a kaleidocycle by interpolating between hinge vectors.

    This operation creates a finer kaleidocycle by inserting interpolated
    binormal vectors between each pair of consecutive hinges. The interpolation
    can be done using linear interpolation (LERP) or spherical linear
    interpolation (SLERP). SLERP is recommended for unit vectors as it
    preserves the length and provides smooth rotation on the unit sphere.

    Args:
        hinges: Array of hinge (binormal) vectors, shape (n+1, 3)
        divisions: Number of subdivisions per segment (default 2)
        method: Interpolation method, either 'lerp' or 'slerp' (default 'slerp')

    Returns:
        Array of subdivided hinge vectors, shape ((n)*divisions+1, 3)

    References:
        Corresponds to subdiv function in Maple code (line 150).
        The Maple code uses LERP with a TODO note for SLERP.

    Example:
        >>> from kaleidocycle import random_hinges, subdivide
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> hinges.shape
        (7, 3)
        >>> subdivided = subdivide(hinges, divisions=2)
        >>> subdivided.shape
        (13, 3)
    """
    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)
    if divisions < 1:
        raise ValueError("divisions must be at least 1")

    n = arr.shape[0] - 1  # Number of segments
    result_size = n * divisions + 1
    result = np.zeros((result_size, 3))

    # First hinge stays the same
    result[0] = arr[0]

    # Interpolate between consecutive hinges
    for i in range(n):
        v0 = arr[i]
        v1 = arr[i + 1]

        # Normalize vectors (in case they aren't already)
        v0_norm = v0 / np.linalg.norm(v0)
        v1_norm = v1 / np.linalg.norm(v1)

        for j in range(1, divisions + 1):
            t = j / divisions
            idx = divisions * i + j

            if method == 'lerp':
                # Linear interpolation
                interpolated = (1 - t) * v0_norm + t * v1_norm
                # Normalize result
                interpolated = interpolated / np.linalg.norm(interpolated)
            elif method == 'slerp':
                # Spherical linear interpolation
                # Compute angle between vectors
                dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
                omega = np.arccos(dot)

                # Handle case where vectors are nearly parallel
                if abs(omega) < 1e-10:
                    interpolated = v0_norm
                else:
                    # SLERP formula
                    interpolated = (
                        np.sin((1 - t) * omega) / np.sin(omega) * v0_norm +
                        np.sin(t * omega) / np.sin(omega) * v1_norm
                    )
            else:
                raise ValueError(f"Unknown interpolation method '{method}'. Use 'lerp' or 'slerp'")

            result[idx] = interpolated

    return result


def reverse(hinges: np.ndarray) -> np.ndarray:
    """Reverse the order of hinge vectors.

    This transformation reverses the sequence of binormal vectors, effectively
    reversing the direction of traversal around the kaleidocycle.

    Args:
        hinges: Array of hinge (binormal) vectors, shape (n+1, 3)

    Returns:
        Array of reversed hinge vectors, shape (n+1, 3)

    References:
        Corresponds to reverse function in Maple code (line 171)

    Example:
        >>> from kaleidocycle import random_hinges, reverse
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> rev = reverse(hinges)
        >>> rev.shape
        (7, 3)
        >>> np.allclose(rev[0], hinges[-1])
        True
    """
    arr = np.asarray(hinges, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        msg = f"expected (n+1, 3) hinge array, got shape {arr.shape}"
        raise ValueError(msg)

    return arr[::-1].copy()


# Convenience function to work with Kaleidocycle objects
def transform_kaleidocycle(
    kc: 'Kaleidocycle',
    operation: str,
    **kwargs,
) -> 'Kaleidocycle':
    """Apply a transformation operation to a Kaleidocycle object.

    Args:
        kc: Kaleidocycle object to transform
        operation: Name of the transformation operation:
                   'mirror', 'involution', 'negative_twist', 'reverse', or 'subdivide'
        **kwargs: Additional arguments for the transformation operation
                  (e.g., divisions=2, method='slerp' for subdivide)

    Returns:
        New Kaleidocycle object with transformed hinges

    Raises:
        ValueError: If operation name is invalid

    Example:
        >>> from kaleidocycle import Kaleidocycle, random_hinges, transform_kaleidocycle
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> kc = Kaleidocycle(hinges=hinges)
        >>> kc_mirror = transform_kaleidocycle(kc, 'mirror')
        >>> kc_mirror.n
        6
        >>> kc_subdiv = transform_kaleidocycle(kc, 'subdivide', divisions=2)
        >>> kc_subdiv.n
        12
    """
    from .geometry import Kaleidocycle

    operations = {
        'mirror': mirror,
        'involution': involution,
        'negative_twist': negative_twist,
        'reverse': reverse,
        'subdivide': subdivide,
    }

    if operation not in operations:
        valid = ', '.join(operations.keys())
        msg = f"Invalid operation '{operation}'. Valid operations: {valid}"
        raise ValueError(msg)

    transformed_hinges = operations[operation](kc.hinges, **kwargs)
    return Kaleidocycle(hinges=transformed_hinges, oriented=kc.oriented)
