"""Tests for import/export functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kaleidocycle import (
    ConstraintConfig,
    Kaleidocycle,
    export_csv,
    export_json,
    import_csv,
    import_json,
    random_hinges,
)


def test_json_export_import(tmp_path: Path) -> None:
    """Test JSON export and import round-trip."""
    # Create test data
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    filepath = tmp_path / "test.json"

    metadata = {
        "n": 6,
        "oriented": True,
        "seed": 42,
        "test_value": 123.456,
    }

    # Export
    export_json(hinges, filepath, metadata=metadata, include_derived=True)

    # Import
    loaded = import_json(filepath)

    # Verify - loaded is now a Kaleidocycle instance
    assert loaded.n == 6
    assert np.allclose(loaded.hinges, hinges)
    assert loaded.metadata["seed"] == 42
    assert loaded.metadata["test_value"] == 123.456
    assert "curve" in loaded.metadata
    assert "tangents" in loaded.metadata
    # Check new fields
    assert "cos_mean" in loaded.metadata
    assert "cos_std" in loaded.metadata
    assert isinstance(loaded.metadata["cos_mean"], float)
    assert isinstance(loaded.metadata["cos_std"], float)


def test_json_export_minimal(tmp_path: Path) -> None:
    """Test JSON export without metadata or derived quantities."""
    hinges = random_hinges(4, seed=10, oriented=False).as_array()
    filepath = tmp_path / "minimal.json"

    # Export without extras
    export_json(hinges, filepath, metadata=None, include_derived=False)

    # Import
    loaded = import_json(filepath)

    # Verify - loaded is now a Kaleidocycle instance
    assert loaded.n == 4
    assert np.allclose(loaded.hinges, hinges)
    # metadata dict still exists but should not have extra fields beyond cos_mean/cos_std
    assert "curve" not in loaded.metadata
    assert "tangents" not in loaded.metadata


def test_csv_export_import(tmp_path: Path) -> None:
    """Test CSV export and import round-trip."""
    # Create test data
    hinges = random_hinges(8, seed=99, oriented=True).as_array()
    filepath = tmp_path / "test.csv"

    # Export
    export_csv(hinges, filepath, header=True)

    # Import
    loaded = import_csv(filepath)

    # Verify shape and values
    assert loaded.shape == hinges.shape
    assert np.allclose(loaded, hinges)


def test_csv_export_no_header(tmp_path: Path) -> None:
    """Test CSV export without header."""
    hinges = random_hinges(5, seed=20, oriented=False).as_array()
    filepath = tmp_path / "no_header.csv"

    # Export without header
    export_csv(hinges, filepath, header=False)

    # Import
    loaded = import_csv(filepath)

    # Verify
    assert loaded.shape == hinges.shape
    assert np.allclose(loaded, hinges)


def test_csv_import_invalid_columns(tmp_path: Path) -> None:
    """Test CSV import with wrong number of columns."""
    filepath = tmp_path / "invalid.csv"

    # Create invalid CSV with 4 columns instead of 3
    with open(filepath, 'w') as f:
        f.write("1.0,2.0,3.0,4.0\n")
        f.write("5.0,6.0,7.0,8.0\n")

    # Should raise ValueError
    with pytest.raises(ValueError, match="Expected 3 columns"):
        import_csv(filepath)


def test_json_csv_consistency(tmp_path: Path) -> None:
    """Test that JSON and CSV export the same hinges."""
    hinges = random_hinges(7, seed=77, oriented=True).as_array()

    json_path = tmp_path / "test.json"
    csv_path = tmp_path / "test.csv"

    # Export to both formats
    export_json(hinges, json_path)
    export_csv(hinges, csv_path)

    # Import from both
    loaded_json = import_json(json_path)
    loaded_csv = import_csv(csv_path)

    # Verify they match - loaded_json is now a Kaleidocycle instance
    assert np.allclose(loaded_json.hinges, loaded_csv)
    assert np.allclose(loaded_json.hinges, hinges)


def test_json_export_with_penalties(tmp_path: Path) -> None:
    """Test JSON export with constraint penalties."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    filepath = tmp_path / "with_penalties.json"

    config = ConstraintConfig(oriented=True, enforce_anchors=False, constant_torsion=True)

    # Export with config
    export_json(hinges, filepath, config=config, include_derived=True)

    # Import
    loaded = import_json(filepath)

    # Verify penalties are included - loaded is now a Kaleidocycle instance
    assert "penalties" in loaded.metadata
    assert "unit_norm" in loaded.metadata["penalties"]
    assert "closure" in loaded.metadata["penalties"]
    assert "constant_torsion" in loaded.metadata["penalties"]
    assert "total" in loaded.metadata["penalties"]

    # Verify all penalties are floats
    for name, value in loaded.metadata["penalties"].items():
        assert isinstance(value, float)

    # Verify total is sum of individual penalties
    individual_sum = sum(v for k, v in loaded.metadata["penalties"].items() if k != "total")
    assert np.isclose(loaded.metadata["penalties"]["total"], individual_sum)


# Tests for Kaleidocycle instance export

def test_json_export_kaleidocycle_instance(tmp_path: Path) -> None:
    """Test JSON export with Kaleidocycle instance."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    filepath = tmp_path / "kaleidocycle.json"

    # Export Kaleidocycle instance
    export_json(kc, filepath, include_derived=True)

    # Import
    loaded = import_json(filepath)

    # Verify hinges match - loaded is now a Kaleidocycle instance
    assert np.allclose(loaded.hinges, hinges)

    # Verify metadata from Kaleidocycle
    assert loaded.metadata["oriented"] == True
    assert loaded.metadata["n"] == 6

    # Verify derived quantities
    assert "curve" in loaded.metadata
    assert "tangents" in loaded.metadata


def test_json_export_kaleidocycle_with_metadata(tmp_path: Path) -> None:
    """Test JSON export with Kaleidocycle and additional metadata."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    # Add some metadata to the Kaleidocycle
    kc.metadata["custom_field"] = "test_value"
    kc.metadata["energy"] = 123.456

    filepath = tmp_path / "kaleidocycle_meta.json"

    # Export with additional metadata (should merge)
    extra_metadata = {"user_field": "user_value"}
    export_json(kc, filepath, metadata=extra_metadata)

    # Import
    loaded = import_json(filepath)

    # Verify all metadata is present - loaded is now a Kaleidocycle instance
    assert loaded.metadata["oriented"] == True
    assert loaded.metadata["n"] == 6
    assert loaded.metadata["custom_field"] == "test_value"
    assert loaded.metadata["energy"] == 123.456
    assert loaded.metadata["user_field"] == "user_value"


def test_json_export_kaleidocycle_metadata_override(tmp_path: Path) -> None:
    """Test that user metadata takes precedence over Kaleidocycle metadata."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    # Set some metadata in the Kaleidocycle
    kc.metadata["value"] = "original"

    filepath = tmp_path / "kaleidocycle_override.json"

    # Export with metadata that overrides
    override_metadata = {"value": "overridden"}
    export_json(kc, filepath, metadata=override_metadata)

    # Import
    loaded = import_json(filepath)

    # User metadata should take precedence - loaded is now a Kaleidocycle instance
    assert loaded.metadata["value"] == "overridden"


def test_csv_export_kaleidocycle_instance(tmp_path: Path) -> None:
    """Test CSV export with Kaleidocycle instance."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    filepath = tmp_path / "kaleidocycle.csv"

    # Export Kaleidocycle instance
    export_csv(kc, filepath, header=True)

    # Import
    loaded = import_csv(filepath)

    # Verify hinges match
    assert loaded.shape == hinges.shape
    assert np.allclose(loaded, hinges)


def test_csv_export_kaleidocycle_vs_array(tmp_path: Path) -> None:
    """Test that CSV export from Kaleidocycle matches array export."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    csv_from_kc = tmp_path / "from_kc.csv"
    csv_from_array = tmp_path / "from_array.csv"

    # Export both ways
    export_csv(kc, csv_from_kc)
    export_csv(hinges, csv_from_array)

    # Import both
    loaded_kc = import_csv(csv_from_kc)
    loaded_array = import_csv(csv_from_array)

    # Should be identical
    assert np.allclose(loaded_kc, loaded_array)
    assert np.allclose(loaded_kc, hinges)


def test_json_export_kaleidocycle_with_config(tmp_path: Path) -> None:
    """Test JSON export with Kaleidocycle and constraint config."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    filepath = tmp_path / "kaleidocycle_config.json"

    config = ConstraintConfig(oriented=True, constant_torsion=True)

    # Export with config to compute penalties
    export_json(kc, filepath, config=config, include_derived=True)

    # Import
    loaded = import_json(filepath)

    # Verify penalties are included - loaded is now a Kaleidocycle instance
    assert "penalties" in loaded.metadata
    assert "total" in loaded.metadata["penalties"]

    # Verify metadata from Kaleidocycle
    assert loaded.metadata["oriented"] == True


def test_json_export_kaleidocycle_computed_properties(tmp_path: Path) -> None:
    """Test that computed properties from Kaleidocycle are exported."""
    hinges = random_hinges(6, seed=42, oriented=True).as_array()
    kc = Kaleidocycle(hinges=hinges, oriented=True)

    # Compute some properties
    kc.compute(['geometric', 'topological'])

    filepath = tmp_path / "kaleidocycle_computed.json"

    # Export
    export_json(kc, filepath)

    # Import
    loaded = import_json(filepath)

    # Metadata should include computed properties - loaded is now a Kaleidocycle instance
    assert "geometric" in loaded.metadata
    assert "topological" in loaded.metadata
    # The metadata dict should have been populated by compute()
