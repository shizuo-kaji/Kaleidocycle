"""Tests for the shared notebook/web sample catalogue."""

from __future__ import annotations

import json
from pathlib import Path

from kaleidocycle import (
    Kaleidocycle,
    build_web_sample_fallback,
    export_sample,
    load_sample,
    sample_catalogue,
    sample_shape_score,
    select_presentable_generic,
)


def test_repository_catalogue_is_named_and_loadable() -> None:
    catalogue = sample_catalogue()

    assert catalogue["default"] == "generic_k15_noncritical"
    names = {entry["name"] for entry in catalogue["samples"]}
    assert {
        "generic_k8_nonoriented",
        "generic_k9_oriented",
        "generic_k15_noncritical",
    } <= names
    for name in names:
        assert load_sample(name).name == name


def test_export_sample_builds_catalogue_and_fallback(tmp_path: Path) -> None:
    cycle = Kaleidocycle(
        8,
        oriented=False,
        seed=0,
        solver_options={"mode": "random_feasible", "backend": "numpy"},
    )

    output = export_sample(
        cycle,
        "generic_k8_test",
        directory=tmp_path,
        metadata={"kind": "generic", "non_critical": True},
        default=True,
    )

    with output.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["name"] == "generic_k8_test"
    assert payload["metadata"]["web_default"] is True
    assert load_sample("generic_k8_test", tmp_path).name == "generic_k8_test"

    catalogue = sample_catalogue(tmp_path)
    assert catalogue["default"] == "generic_k8_test"
    assert catalogue["samples"][0]["file"] == "generic_k8_test.json"

    fallback = build_web_sample_fallback(
        tmp_path / "sample-fallback.js",
        directory=tmp_path,
    )
    text = fallback.read_text(encoding="utf-8")
    assert "window.KALEIDOCYCLE_SAMPLE_FALLBACK" in text
    assert '"generic_k8_test"' in text


def test_shape_selection_records_reproducible_quality() -> None:
    cycle = select_presentable_generic(8, oriented=False, seeds=range(3))

    assert cycle.metadata["created_from"] == "presentable_generic_search"
    assert cycle.metadata["seed"] in range(3)
    assert cycle.metadata["shape_quality"] == sample_shape_score(cycle.hinges)
