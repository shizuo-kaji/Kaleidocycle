"""Shared sample catalogue for notebooks, tests, and the static web viewer."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Kaleidocycle
from .io import export_json, import_json

_SAMPLE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_CATALOG_FILENAME = "catalog.json"


def default_sample_directory() -> Path:
    """Return the repository's canonical ``data/kaleidocycles`` directory.

    ``KALEIDOCYCLE_SAMPLE_DIR`` overrides discovery. Otherwise the current
    directory and the installed module location are searched upwards for the
    repository's ``pyproject.toml``. This works both from the repository root
    and from a notebook whose working directory is ``notebooks/``.
    """

    override = os.environ.get("KALEIDOCYCLE_SAMPLE_DIR")
    if override:
        return Path(override).expanduser().resolve()

    candidates = [Path.cwd(), Path(__file__).resolve()]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").is_file():
                return parent / "data" / "kaleidocycles"
    raise FileNotFoundError(
        "Could not locate data/kaleidocycles; set KALEIDOCYCLE_SAMPLE_DIR"
    )


def _sample_directory(directory: str | Path | None) -> Path:
    return default_sample_directory() if directory is None else Path(directory)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _orientation(payload: Mapping[str, Any]) -> bool:
    value = payload.get("metadata", {}).get("oriented")
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _catalogue_entry(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path} has no non-empty top-level name")
    metadata = payload.get("metadata", {})
    kind = metadata.get("kind")
    if kind is None:
        kind = "generic" if name.startswith("generic_") else "mobius"
    return {
        "name": name,
        "file": path.name,
        "n": int(payload.get("n", len(payload.get("hinges", [])) - 1)),
        "oriented": _orientation(payload),
        "kind": str(kind),
        "non_critical": bool(metadata.get("non_critical", False)),
    }


def write_sample_catalogue(
    directory: str | Path | None = None,
    *,
    default: str | None = None,
) -> dict[str, Any]:
    """Validate sample JSON files and write their deterministic catalogue."""

    sample_directory = _sample_directory(directory)
    sample_directory.mkdir(parents=True, exist_ok=True)
    payloads = [
        (path, _read_json(path))
        for path in sorted(sample_directory.glob("*.json"))
        if path.name != _CATALOG_FILENAME
    ]
    entries = [_catalogue_entry(path, payload) for path, payload in payloads]
    names = {entry["name"] for entry in entries}
    if len(names) != len(entries):
        raise ValueError("sample names must be unique")

    marked_defaults = [
        payload["name"]
        for _, payload in payloads
        if payload.get("metadata", {}).get("web_default")
    ]
    resolved_default = default or (marked_defaults[0] if marked_defaults else None)
    if resolved_default is None and entries:
        resolved_default = entries[0]["name"]
    if resolved_default is not None and resolved_default not in names:
        raise ValueError(f"unknown default sample {resolved_default!r}")

    catalogue = {
        "schema_version": 1,
        "default": resolved_default,
        "samples": entries,
    }
    output = sample_directory / _CATALOG_FILENAME
    output.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    return catalogue


def sample_catalogue(
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Read the canonical catalogue, generating it when it is absent."""

    sample_directory = _sample_directory(directory)
    path = sample_directory / _CATALOG_FILENAME
    if not path.is_file():
        return write_sample_catalogue(sample_directory)
    return _read_json(path)


def sample_path(name: str, directory: str | Path | None = None) -> Path:
    """Resolve a catalogue name to its canonical JSON path."""

    for entry in sample_catalogue(directory).get("samples", []):
        if entry.get("name") == name:
            return _sample_directory(directory) / entry["file"]
    raise KeyError(f"unknown kaleidocycle sample {name!r}")


def load_sample(name: str, directory: str | Path | None = None) -> Kaleidocycle:
    """Load one named sample from the shared catalogue."""

    return import_json(sample_path(name, directory))


def export_sample(
    hinges: ArrayLike | Kaleidocycle,
    name: str,
    *,
    directory: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    default: bool = False,
    include_derived: bool = True,
) -> Path:
    """Export a named canonical sample and refresh the catalogue.

    This is the promotion path for a notebook result that should become a web
    built-in. Ordinary exploratory output should continue to use
    :func:`kaleidocycle.export_json` in ``notebooks/output``.
    """

    if not _SAMPLE_NAME.fullmatch(name):
        raise ValueError(
            "sample name must contain only lower-case letters, digits, '_' or '-'"
        )
    sample_directory = _sample_directory(directory)
    sample_directory.mkdir(parents=True, exist_ok=True)
    merged_metadata = {"name": name, **dict(metadata or {})}
    if default:
        merged_metadata["web_default"] = True
    output = sample_directory / f"{name}.json"
    export_json(
        hinges,
        output,
        name=name,
        metadata=merged_metadata,
        include_derived=include_derived,
    )
    write_sample_catalogue(
        sample_directory,
        default=name if default else None,
    )
    return output


def build_web_sample_fallback(
    output: str | Path,
    *,
    directory: str | Path | None = None,
) -> Path:
    """Build the ``file://`` fallback from the same canonical JSON files."""

    sample_directory = _sample_directory(directory)
    catalogue = sample_catalogue(sample_directory)
    samples = {
        entry["name"]: _read_json(sample_directory / entry["file"])
        for entry in catalogue.get("samples", [])
    }
    bundle = {"catalogue": catalogue, "samples": samples}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "/* Generated from data/kaleidocycles by kaleidocycle.samples. */\n"
        f"window.KALEIDOCYCLE_SAMPLE_FALLBACK = {json.dumps(bundle)};\n",
        encoding="utf-8",
    )
    return target


def sample_shape_score(hinges: ArrayLike) -> dict[str, float]:
    """Score how clearly a generic sample reads as a three-dimensional ring."""

    from .integrable import framed_polygon_from_binormals

    polygon = framed_polygon_from_binormals(hinges)
    vertices = polygon.vertices[:-1]
    centred = vertices - np.mean(vertices, axis=0)
    radius_of_gyration = float(
        np.sqrt(np.mean(np.einsum("ij,ij->i", centred, centred)))
    )
    distances = [
        np.linalg.norm(centred[left] - centred[right])
        for left in range(vertices.shape[0])
        for right in range(left + 1, vertices.shape[0])
        if right - left not in (1, vertices.shape[0] - 1)
    ]
    minimum_separation = float(min(distances))
    covariance_eigenvalues = np.linalg.eigvalsh(
        centred.T @ centred / vertices.shape[0]
    )[::-1]
    second_axis_ratio = float(covariance_eigenvalues[1] / covariance_eigenvalues[0])
    third_axis_ratio = float(covariance_eigenvalues[2] / covariance_eigenvalues[0])
    maximum_curvature = float(np.max(np.abs(polygon.curvatures)))
    score = (
        1.4 * radius_of_gyration
        + 1.5 * minimum_separation
        + 0.6 * np.sqrt(max(second_axis_ratio, 0.0))
        + 1.2 * np.sqrt(max(third_axis_ratio, 0.0))
        - 0.1 * max(0.0, maximum_curvature - 5.0)
    )
    return {
        "score": float(score),
        "radius_of_gyration": radius_of_gyration,
        "minimum_separation": minimum_separation,
        "second_axis_ratio": second_axis_ratio,
        "third_axis_ratio": third_axis_ratio,
        "maximum_cayley_curvature": maximum_curvature,
    }


def select_presentable_generic(
    n: int,
    *,
    oriented: bool,
    seeds: Iterable[int] = range(100),
    solver_options: Mapping[str, Any] | None = None,
) -> Kaleidocycle:
    """Return the highest-scoring generic feasible cycle among several seeds."""

    options = {
        "mode": "random_feasible",
        "backend": "numpy",
        "max_iter": 400,
        "max_attempts": 4,
        **dict(solver_options or {}),
    }
    best: tuple[float, Kaleidocycle, dict[str, float], int] | None = None
    for seed in seeds:
        try:
            cycle = Kaleidocycle(
                n,
                oriented=oriented,
                seed=int(seed),
                solver_options=options,
            )
        except RuntimeError:
            continue
        quality = sample_shape_score(cycle.hinges)
        candidate = (quality["score"], cycle, quality, int(seed))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("no generic sample converged for the requested seeds")
    _, cycle, quality, seed = best
    cycle.metadata.update(
        {
            "created_from": "presentable_generic_search",
            "seed": seed,
            "shape_quality": quality,
        }
    )
    return cycle
