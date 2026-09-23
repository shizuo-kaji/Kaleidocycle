"""Geometric and interval checks for the first-flow crossing counterexample."""

import copy

import numpy as np
import pytest

from kaleidocycle.collision_certificate import certify_crossing
from kaleidocycle.collisions import (
    closest_segment_points,
    crossing_velocity,
    gauss_writhe,
    load_crossing_data,
    reconstruct_crossing,
    sample_crossing,
)


def test_certificate_and_corrupted_input() -> None:
    assert certify_crossing()["verified"]
    altered = copy.deepcopy(load_crossing_data())
    altered["center"][0] = "-1.6"
    with pytest.raises(ArithmeticError):
        certify_crossing(altered)


def test_closed_regular_contact_and_geometric_velocity() -> None:
    contact = reconstruct_crossing()
    assert np.linalg.norm(contact.closure_residual) < 1e-12
    assert np.linalg.norm(contact.monodromy_residual) < 1e-12
    assert np.max(np.abs(contact.angles)) < 2.72
    assert -3.283 < crossing_velocity(contact) < -3.281
    assert np.isnan(gauss_writhe(contact.vertices))
    assert np.allclose(np.linalg.norm(np.diff(contact.vertices, axis=0), axis=1), 1)


def test_flow_crossing_changes_writhe_without_losing_closure() -> None:
    result = sample_crossing([-0.002, -0.0001, 0, 0.0001, 0.002])
    assert result.min_distance[0] > 0.006
    assert result.min_distance[-1] > 0.006
    assert result.min_distance[2] < 1e-12
    assert np.isnan(result.writhe[2]) and np.isnan(result.linking[2])
    assert np.allclose(result.linking[:2], 4.5, atol=1e-10)
    assert np.allclose(result.linking[-2:], 2.5, atol=1e-10)
    assert result.writhe[3] - result.writhe[1] == pytest.approx(-2, abs=1e-10)
    assert result.closure.max() < 1e-10
    assert result.monodromy.max() < 1e-10
    assert np.ptp(result.energy) < 1e-10


def test_segment_distance_handles_interiors_endpoints_and_parallel_edges() -> None:
    assert (
        closest_segment_points([0, 0, 0], [1, 0, 0], [0.5, -1, 0], [0.5, 1, 0])[0]
        < 1e-15
    )
    assert closest_segment_points([0, 0, 0], [1, 0, 0], [2, 1, 0], [2, 2, 0])[
        0
    ] == pytest.approx(np.sqrt(2))
    assert closest_segment_points([0, 0, 0], [1, 0, 0], [0, 0, 2], [1, 0, 2])[0] == 2


def test_gauss_writhe_is_scale_and_orientation_independent() -> None:
    vertices = sample_crossing([-0.001]).vertices[0]
    expected = gauss_writhe(vertices)
    assert gauss_writhe(vertices[::-1]) == pytest.approx(expected)
    assert gauss_writhe(vertices * 3 + [4, 7, -2]) == pytest.approx(expected)
    assert gauss_writhe(vertices * [-1, 1, 1]) == pytest.approx(-expected)
