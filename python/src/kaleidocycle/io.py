"""Import/Export functions for Kaleidocycle configurations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

import numpy as np
from numpy.typing import NDArray

from .constraints import ConstraintConfig, constraint_residuals
from .geometry import tangents_to_curve, binormals_to_tangents, mean_cosine, pairwise_cosines

if TYPE_CHECKING:
    from .geometry import Kaleidocycle


def _convert_numpy_to_lists(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization.

    Args:
        obj: Any object that might contain numpy arrays

    Returns:
        The same object with all numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    else:
        return obj


def export_json(
    hinges: Union[NDArray[np.float64], 'Kaleidocycle'],
    filepath: str | Path,
    *,
    metadata: Dict[str, Any] | None = None,
    include_derived: bool = True,
    config: ConstraintConfig | None = None,
) -> None:
    """Export kaleidocycle configuration to JSON format.

    Args:
        hinges: Either a hinge vectors array of shape (n+1, 3) or a Kaleidocycle instance.
                If a Kaleidocycle instance is provided, its hinges and metadata are used.
        filepath: Path to save JSON file
        metadata: Optional metadata dictionary (oriented, seed, energy, etc.).
                  If hinges is a Kaleidocycle, this is merged with its metadata.
        include_derived: If True, include derived quantities (curve, tangents)
        config: Optional constraint config to compute penalties

    Example:
        >>> from kaleidocycle import Kaleidocycle, random_hinges, export_json
        >>> # Method 1: Export from array
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> export_json(hinges, "output.json")
        >>> # Method 2: Export from Kaleidocycle instance
        >>> kc = Kaleidocycle(hinges=hinges, oriented=True)
        >>> export_json(kc, "output.json")
    """
    from .geometry import Kaleidocycle

    filepath = Path(filepath)

    # Handle Kaleidocycle instance
    if isinstance(hinges, Kaleidocycle):
        kc = hinges
        hinges_array = kc.hinges

        # Merge metadata from Kaleidocycle with provided metadata
        kc_metadata = {
            "oriented": str(kc.oriented), # to prevent "TypeError: Object of type bool is not JSON serializable"
            "n": kc.n,
        }

        # Add computed metadata if available
        if kc.metadata:
            # Recursively convert any numpy arrays in metadata to lists for JSON serialization
            for key, value in kc.metadata.items():
                kc_metadata[key] = _convert_numpy_to_lists(value)

        # Merge with user-provided metadata (user metadata takes precedence)
        if metadata is not None:
            kc_metadata.update(metadata)

        metadata = kc_metadata
        hinges = hinges_array
    else:
        hinges = np.asarray(hinges, dtype=float)

    data: Dict[str, Any] = {
        "hinges": hinges.tolist(),
        "n": len(hinges) - 1,  # Number of tetrahedra
    }

    if metadata is not None:
        data["metadata"] = metadata

    # Compute and include cosine statistics
    cos_mean = float(mean_cosine(hinges, wrap=False))
    cosines = pairwise_cosines(hinges, wrap=False)
    cos_std = float(np.std(cosines))

    data["cos_mean"] = cos_mean
    data["cos_std"] = cos_std

    # Compute constraint penalties if config provided
    if config is not None:
        residuals = constraint_residuals(hinges, config)
        penalties = {}
        for name, residual in residuals.items():
            penalties[name] = float(np.sum(residual**2))
        penalties["total"] = sum(penalties.values())
        data["penalties"] = penalties

    if include_derived:
        tangents = binormals_to_tangents(hinges, normalize=False)
        curve = tangents_to_curve(tangents)
        data["tangents"] = tangents.tolist()
        data["curve"] = curve.tolist()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def import_json(filepath: str | Path) -> 'Kaleidocycle':
    """Import kaleidocycle configuration from JSON format.

    Args:
        filepath: Path to JSON file

    Returns:
        Kaleidocycle instance with:
            - hinges: numpy array of shape (n+1, 3)
            - n: number of tetrahedra
            - oriented: whether the kaleidocycle is oriented
            - metadata: dictionary containing cos_mean, cos_std, penalties,
                       tangents, curve (if present), and any custom metadata

    Example:
        >>> from kaleidocycle import import_json
        >>> kc = import_json("output.json")
        >>> print(kc.n)
        >>> print(kc.metadata['cos_mean'])
    """
    from .geometry import Kaleidocycle

    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract hinges
    hinges = np.array(data["hinges"], dtype=float)

    # Determine orientation from metadata if available, otherwise infer
    metadata = data.get("metadata", {})
    oriented = metadata.get("oriented", None)
    # conver string to bool if it's a string
    if isinstance(oriented, str):
        oriented = oriented.lower() == 'true'
    if oriented is None:
        from .geometry import is_oriented
        oriented = is_oriented(hinges)

    # Create Kaleidocycle instance
    kc = Kaleidocycle(hinges=hinges, oriented=oriented)

    # Store additional data in metadata (convert oriented back to boolean if needed)
    if metadata:
        # Update oriented in metadata dict to be boolean (in case it was string)
        metadata_copy = metadata.copy()
        metadata_copy["oriented"] = oriented
        kc.metadata.update(metadata_copy)

    if "cos_mean" in data:
        kc.metadata["cos_mean"] = data["cos_mean"]

    if "cos_std" in data:
        kc.metadata["cos_std"] = data["cos_std"]

    if "penalties" in data:
        kc.metadata["penalties"] = data["penalties"]

    if "tangents" in data:
        kc.metadata["tangents"] = np.array(data["tangents"], dtype=float)

    if "curve" in data:
        kc.metadata["curve"] = np.array(data["curve"], dtype=float)

    return kc


def export_csv(
    hinges: Union[NDArray[np.float64], 'Kaleidocycle'],
    filepath: str | Path,
    *,
    header: bool = True,
) -> None:
    """Export hinge vectors (binormals) to CSV format.

    CSV contains only the hinge direction vectors in a simple format:
    Each row contains [hx, hy, hz] for one hinge vector.
    Total size: (n+1) rows × 3 columns

    Args:
        hinges: Either a hinge vectors array of shape (n+1, 3) or a Kaleidocycle instance.
                If a Kaleidocycle instance is provided, its hinges are exported.
        filepath: Path to save CSV file
        header: If True, include column headers

    Example:
        >>> from kaleidocycle import Kaleidocycle, random_hinges, export_csv
        >>> # Method 1: Export from array
        >>> hinges = random_hinges(6, seed=42).as_array()
        >>> export_csv(hinges, "output.csv")
        >>> # Method 2: Export from Kaleidocycle instance
        >>> kc = Kaleidocycle(hinges=hinges)
        >>> export_csv(kc, "output.csv")
    """
    from .geometry import Kaleidocycle

    filepath = Path(filepath)

    # Handle Kaleidocycle instance
    if isinstance(hinges, Kaleidocycle):
        hinges = hinges.hinges
    else:
        hinges = np.asarray(hinges, dtype=float)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        if header:
            writer.writerow(['hx', 'hy', 'hz'])

        for hinge in hinges:
            writer.writerow([f"{val:.16e}" for val in hinge])


def import_csv(filepath: str | Path) -> NDArray[np.float64]:
    """Import hinge vectors (binormals) from CSV format.

    Args:
        filepath: Path to CSV file

    Returns:
        Hinge vectors array of shape (n+1, 3)
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        reader = csv.reader(f)

        # Check if first row is header
        first_row = next(reader)
        if first_row[0].lower() in ['hx', 'h_x', 'x']:
            # Skip header
            rows = list(reader)
        else:
            # First row is data
            rows = [first_row] + list(reader)

    hinges = np.array([[float(val) for val in row] for row in rows], dtype=float)

    if hinges.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (hx, hy, hz), got {hinges.shape[1]}")

    return hinges
