"""Tests for parametric 3D CAD model generation (kaleidocycle.cad)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from kaleidocycle import (
    ClearanceError,
    ClearanceWarning,
    Kaleidocycle,
    TwistedPlateConfig,
    build_hinge_pin,
    build_kaleidocycle_assembly,
    build_print_plate,
    build_twisted_plate,
    build_twisted_plate_parts,
    check_hinge_fit,
    check_kaleidocycle_clearance,
    export_print_plate,
    export_twisted_plate,
    trace_kaleidocycle_motion,
    twisted_plate_from_kaleidocycle,
)
from kaleidocycle.cad import (
    _compute_plate_positions,
    _plate_layout,
    _require_build123d,
    _write_3mf_archive,
)
from kaleidocycle.samples import load_sample

_NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
_BED = 256.0


def _overlap(a, b) -> float:
    """Intersection volume; build123d >= 0.9 returns a list of shapes."""
    common = a.intersect(b)
    if common is None:
        return 0.0
    if isinstance(common, (list, tuple)):
        return float(sum(s.volume for s in common))
    return float(common.volume)


def _require_cad():
    try:
        return _require_build123d()
    except ImportError as exc:
        pytest.skip(str(exc))


def _translation(transform: str | None) -> np.ndarray:
    """Return the translation of a 3MF affine ``transform`` (identity rotation)."""
    if not transform:
        return np.zeros(3)
    values = np.array([float(v) for v in transform.split()])
    assert values.shape == (12,)
    np.testing.assert_allclose(values[:9], np.eye(3).ravel())
    return values[9:]


def _validate_3mf(path) -> dict:
    """Check that ``path`` is a self-contained, slicer-neutral core 3MF.

    Verifies package layout, unique object ids, valid references, closed and
    outward-oriented meshes, and that every build item rests on ``z = 0``
    inside a 256 mm bed without overlapping other items.
    """
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        assert names == {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
        rels = ET.fromstring(zf.read("_rels/.rels"))
        targets = [r.get("Target") for r in rels]
        assert targets == ["/3D/3dmodel.model"]
        root = ET.fromstring(zf.read("3D/3dmodel.model"))

    assert root.get("unit") == "millimeter"
    metadata = {m.get("name"): m.text for m in root.findall("m:metadata", _NS)}
    # Bambu Studio treats "BambuStudio-*" files as its own projects and then
    # requires a complete project configuration.
    assert not metadata.get("Application", "").startswith("BambuStudio-")

    objects = root.findall("./m:resources/m:object", _NS)
    ids = [int(o.get("id")) for o in objects]
    assert len(ids) == len(set(ids)), f"duplicate object ids: {ids}"

    meshes: dict[int, np.ndarray] = {}
    components: dict[int, list[int]] = {}
    for obj in objects:
        oid = int(obj.get("id"))
        assert obj.get("type") == "model"
        mesh = obj.find("m:mesh", _NS)
        if mesh is not None:
            verts = np.array(
                [[float(v.get(a)) for a in "xyz"] for v in mesh.find("m:vertices", _NS)]
            )
            tris = np.array(
                [
                    [int(t.get(k)) for k in ("v1", "v2", "v3")]
                    for t in mesh.find("m:triangles", _NS)
                ]
            )
            assert len(tris) > 0 and tris.min() >= 0 and tris.max() < len(verts)
            edges = Counter()
            for a, b, c in tris:
                edges.update([(a, b), (b, c), (c, a)])
            assert all(n == 1 for n in edges.values()), "non-manifold edge"
            assert all((v, u) in edges for (u, v) in edges), "open mesh"
            p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
            volume = np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0
            assert volume > 0.0, "inward-facing mesh"
            meshes[oid] = verts
        else:
            comps = obj.findall("./m:components/m:component", _NS)
            assert comps, "object without mesh or components"
            components[oid] = []
            for comp in comps:
                ref = int(comp.get("objectid"))
                assert ref in meshes, f"component references unknown mesh {ref}"
                assert not np.any(_translation(comp.get("transform")))
                components[oid].append(ref)

    items = root.findall("./m:build/m:item", _NS)
    assert items
    footprints = []
    for item in items:
        ref = int(item.get("objectid"))
        mesh_ids = components.get(ref, [ref])
        assert all(mid in meshes for mid in mesh_ids)
        pts = np.vstack([meshes[mid] for mid in mesh_ids]) + _translation(
            item.get("transform")
        )
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        assert np.isclose(lo[2], 0.0, atol=1e-5)
        assert lo[0] >= 0.0 and lo[1] >= 0.0 and hi[0] <= _BED and hi[1] <= _BED
        footprints.append((lo[:2], hi[:2]))
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            (alo, ahi), (blo, bhi) = footprints[i], footprints[j]
            assert np.any(ahi < blo) or np.any(bhi < alo), "overlapping units"

    parts_per_item = [
        len(components.get(int(item.get("objectid")), [0])) for item in items
    ]
    return {
        "items": len(items),
        "meshes": len(meshes),
        "assemblies": len(components),
        "parts_per_item": parts_per_item,
    }


def _fake_box(x0, y0, z0, size, label):
    """A closed, outward-oriented cube that mimics the build123d API we use."""
    unit = [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    corners = np.array([x0, y0, z0]) + size * np.array(unit, dtype=float)
    # fmt: off
    tris = [
        (0, 2, 1), (1, 2, 3),  # bottom (-z)
        (4, 5, 6), (5, 7, 6),  # top (+z)
        (0, 1, 4), (1, 5, 4),  # front (-y)
        (2, 6, 3), (3, 6, 7),  # back (+y)
        (0, 4, 2), (2, 4, 6),  # left (-x)
        (1, 3, 5), (3, 7, 5),  # right (+x)
    ]
    # fmt: on
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    vec = lambda p: SimpleNamespace(X=p[0], Y=p[1], Z=p[2])  # noqa: E731
    bbox = SimpleNamespace(min=vec(lo), max=vec(hi))
    return SimpleNamespace(
        label=label,
        verts=[tuple(c) for c in corners],
        tris=tris,
        bounding_box=lambda: bbox,
    )


def _fake_mesh_shape(solid, tolerance, angular_tolerance):
    return solid.verts, solid.tris


_FAKE_BD = SimpleNamespace(Mesher=SimpleNamespace(_mesh_shape=_fake_mesh_shape))


class TestTwistedPlateConfig:
    """Tests for configuration and parameter validation (no build123d needed)."""

    def test_defaults(self):
        cfg = TwistedPlateConfig()
        cfg.validate()
        assert cfg.with_ring is False
        assert cfg.resolved_cut_slits is False
        assert np.isclose(cfg.torsion_degrees, 72.8)
        assert cfg.ribbon_width == 43.0
        assert cfg.edge_length == 29.0
        assert np.isclose(cfg.resolved_ring_diameter, 17.2)
        assert cfg.hinge_type == "pin"
        assert cfg.resolved_hinge_diameter == 7.0
        assert TwistedPlateConfig(hinge_type="cone").resolved_hinge_diameter == 6.0

    def test_default_hinge_fit_matches_reference_design(self):
        # Octagonal 4 mm pin, held at zero corner clearance in the bottom
        # knuckles and rotating with 0.2 mm clearance in the top knuckles.
        cfg = TwistedPlateConfig()
        assert cfg.pin_radius == 2.0
        assert np.isclose(cfg.pin_inradius, 2.0 * np.cos(np.pi / 8))
        assert cfg.fixed_bore_radius == cfg.pin_radius
        assert np.isclose(cfg.free_bore_radius, cfg.pin_radius + 0.2)
        assert cfg.pin_radius < cfg.bore_flat_height < cfg.free_bore_radius
        assert cfg.hinge_radius - cfg.free_bore_radius >= 0.8
        assert cfg.z_bottom_hinge - cfg.fixed_bore_radius >= 0.8
        assert cfg.hinge_gap == 0.2
        assert np.isclose(cfg.z_top_hinge - cfg.z_bottom_hinge, cfg.edge_length)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ribbon_width": 0.0},
            {"edge_length": 4.0},
            {"body_thickness": -1.0},
            {"hinge_diameter": 5.0},
            {"pin_diameter": 0.0},
            {"pin_clearance": -0.1},
            {"pin_fit_clearance": -1.5},
            {"pin_core_diameter": 3.0},
            {"pin_chamfer": 1.6},
            {"num_knuckles": 4},
            {"num_knuckles": 1},
            {"ribbon_width": 2.0, "hinge_gap": 0.5, "num_knuckles": 5},
            {"loft_slices": 3},
            {"with_ring": True, "ring_diameter": 4.0},
            {"with_ring": True, "ring_diameter": 45.0},
            {"with_ring": True, "edge_length": 9.0},
            {"with_ring": True, "body_thickness": 8.0, "ring_gap": 0.1},
            {"hinge_type": "hook"},
            {"hinge_type": "cone", "cone_diameter": 5.0},
            {"hinge_type": "cone", "cone_angle": 90.0},
            {"hinge_type": "cone", "snap_gap": 0.5},
            {"hinge_type": "cone", "num_knuckles": 11},
        ],
    )
    def test_invalid_configs_raise(self, kwargs):
        cfg = TwistedPlateConfig(**kwargs)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_pin_knuckle_intervals_split_the_gap(self):
        cfg = TwistedPlateConfig()
        bottom, top = cfg.knuckle_intervals()
        assert len(bottom) == 3 and len(top) == 2
        knuckles = sorted(bottom + top)
        assert knuckles[0][0] == -cfg.ribbon_width / 2
        assert knuckles[-1][1] == cfg.ribbon_width / 2
        gaps = [b[0] - a[1] for a, b in zip(knuckles[:-1], knuckles[1:], strict=True)]
        np.testing.assert_allclose(gaps, cfg.hinge_gap)

    def test_cone_hinge_matches_reference_layout(self):
        # NovemberKou's "Twisted Plate with Ring", 45 mm wide: 9 mm bottom
        # blocks, top knuckles at |t| in [6.75, 13.3], 60-degree cones of 4 mm
        # base (1.155 mm high) on the outer blocks, 6 mm knuckles.
        cfg = TwistedPlateConfig(hinge_type="cone", ribbon_width=45.0)
        cfg.validate()
        bottom, top = cfg.knuckle_intervals()
        np.testing.assert_allclose(
            bottom, [(-22.5, -13.5), (-4.5, 4.5), (13.5, 22.5)], atol=1e-9
        )
        np.testing.assert_allclose(top, [(-13.3, -6.75), (6.75, 13.3)], atol=1e-9)
        assert np.isclose(cfg.cone_height, 2.0 / np.sqrt(3.0))
        assert cfg.hinge_radius == 3.0
        assert np.isclose(cfg.snap_deflection, cfg.cone_height - cfg.hinge_gap)

    def test_from_kaleidocycle(self):
        kc = load_sample("mobius_k7_oriented")
        cfg = twisted_plate_from_kaleidocycle(
            kc, ribbon_width=50.0, edge_length=32.0
        )
        assert cfg.with_ring is False
        assert cfg.ribbon_width == 50.0
        assert cfg.edge_length == 32.0
        expected_torsion = float(np.arccos(np.clip(kc.hinges[0] @ kc.hinges[1], -1, 1)))
        assert np.isclose(cfg.torsion_angle, expected_torsion, atol=1e-6)

    def test_from_kaleidocycle_with_scale(self):
        kc = load_sample("mobius_k7_oriented")
        cfg = twisted_plate_from_kaleidocycle(kc, scale=30.0)
        seg_len = float(np.linalg.norm(np.cross(kc.hinges[0], kc.hinges[1])))
        assert np.isclose(cfg.edge_length, 30.0 * seg_len)

    def test_from_kaleidocycle_bad_segment(self):
        kc = load_sample("mobius_k7_oriented")
        with pytest.raises(ValueError):
            twisted_plate_from_kaleidocycle(kc, segment=7)

    def test_plate_positions_grid(self):
        positions = _compute_plate_positions(7, 45.0, 43.0, spacing=6.0)
        assert len(positions) == 7
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        assert min(xs) > 30.0 and max(xs) < 226.0
        assert min(ys) > 30.0 and max(ys) < 226.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = abs(positions[i][0] - positions[j][0])
                dy = abs(positions[i][1] - positions[j][1])
                assert dx >= 50.9 or dy >= 48.9

    @pytest.mark.parametrize("count", [1, 5, 7, 10])
    def test_plate_layout_with_pins(self, count):
        unit_w, unit_d, pin_len, pin_w = 45.0, 43.5, 45.0, 3.7
        units, pins = _plate_layout(
            count,
            unit_w,
            unit_d,
            pin_count=count,
            pin_length=pin_len,
            pin_width=pin_w,
            spacing=6.0,
        )
        assert len(units) == count and len(pins) == count
        boxes = [
            (x - unit_w / 2, y - unit_d / 2, x + unit_w / 2, y + unit_d / 2)
            for x, y in units
        ] + [
            (x - pin_len / 2, y - pin_w / 2, x + pin_len / 2, y + pin_w / 2)
            for x, y in pins
        ]
        for x0, y0, x1, y1 in boxes:
            assert x0 >= 0.0 and y0 >= 0.0 and x1 <= _BED and y1 <= _BED
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                gap_x = max(b[0] - a[2], a[0] - b[2])
                gap_y = max(b[1] - a[3], a[1] - b[3])
                assert max(gap_x, gap_y) >= 6.0 - 1e-9

    def test_plate_layout_without_pins_matches_grid(self):
        units, pins = _plate_layout(7, 45.0, 43.0, spacing=6.0)
        assert pins == []
        assert units == _compute_plate_positions(7, 45.0, 43.0, spacing=6.0)


class TestThreeMFWriter:
    """Structure of the 3MF writer, using fake solids (no build123d needed)."""

    def test_shared_meshes_unique_ids_and_grouped_parts(self, tmp_path):
        unit_a = {
            "plate": _fake_box(-5.0, -5.0, 2.0, 10.0, "plate"),
            "ring": _fake_box(-2.0, -2.0, 5.0, 3.0, "ring"),
        }
        unit_b = {"plate": _fake_box(0.0, 0.0, -3.0, 20.0, "plate")}
        entries = [("a", unit_a)] * 3 + [("b", unit_b)] * 2
        positions = _compute_plate_positions(5, 20.0, 20.0, spacing=6.0)
        out = _write_3mf_archive(
            _FAKE_BD,
            tmp_path / "fake.3mf",
            entries,
            positions,
            unit_name="twisted_plate",
        )
        info = _validate_3mf(out)
        assert info["items"] == 5
        assert info["meshes"] == 3  # plate+ring of A, plate of B, each written once
        assert info["assemblies"] == 1
        assert info["parts_per_item"] == [2, 2, 2, 1, 1]

        with zipfile.ZipFile(out) as zf:
            root = ET.fromstring(zf.read("3D/3dmodel.model"))
        names = {o.get("name") for o in root.findall("./m:resources/m:object", _NS)}
        assert names == {
            "twisted_plate_1",
            "twisted_plate_1_plate",
            "twisted_plate_1_ring",
            "twisted_plate_2",
        }

    def test_object_names_override(self, tmp_path):
        unit = {"plate": _fake_box(-5.0, -5.0, 0.0, 10.0, "plate")}
        pin = {"pin": _fake_box(-1.0, -1.0, 0.0, 2.0, "pin")}
        entries = [("u", unit)] * 2 + [("p", pin)] * 2
        positions = [(20.0, 20.0), (40.0, 20.0), (20.0, 40.0), (40.0, 40.0)]
        out = _write_3mf_archive(
            _FAKE_BD,
            tmp_path / "named.3mf",
            entries,
            positions,
            object_names={"u": "twisted_plate", "p": "hinge_pin"},
        )
        assert _validate_3mf(out)["items"] == 4
        with zipfile.ZipFile(out) as zf:
            root = ET.fromstring(zf.read("3D/3dmodel.model"))
        names = [o.get("name") for o in root.findall("./m:resources/m:object", _NS)]
        assert names == ["twisted_plate", "hinge_pin"]

    def test_length_mismatch_and_empty_input_raise(self, tmp_path):
        unit = {"plate": _fake_box(0.0, 0.0, 0.0, 1.0, "plate")}
        with pytest.raises(ValueError):
            _write_3mf_archive(_FAKE_BD, tmp_path / "x.3mf", [("a", unit)], [])
        with pytest.raises(ValueError):
            _write_3mf_archive(_FAKE_BD, tmp_path / "y.3mf", [], [])


@pytest.mark.cad
class TestTwistedPlateCAD:
    """Tests that build and export 3D B-Rep solids with build123d."""

    def test_build_plate_without_ring_default(self):
        _require_cad()
        parts = build_twisted_plate_parts()
        assert list(parts.keys()) == ["plate"]
        plate = parts["plate"]
        assert plate.is_valid()
        assert plate.label == "plate"
        assert plate.volume > 100.0

        cfg = TwistedPlateConfig()
        bbox = plate.bounding_box()
        assert np.isclose(bbox.min.Z, 0.0, atol=1e-4)
        assert np.isclose(bbox.max.Z, cfg.z_top_hinge + cfg.hinge_radius, atol=1e-4)

    def test_hinge_pin_geometry(self):
        _require_cad()
        cfg = TwistedPlateConfig()
        pin = build_hinge_pin(cfg)
        assert pin.is_valid()
        assert pin.label == "pin"
        bbox = pin.bounding_box()
        np.testing.assert_allclose(
            [bbox.size.X, bbox.size.Y, bbox.size.Z],
            [cfg.ribbon_width, 2 * cfg.pin_inradius, 2 * cfg.pin_inradius],
            atol=1e-4,
        )
        # Flat face down: the pin lies on the bed without support.
        bottom = [f for f in pin.faces() if np.isclose(f.center().Z, bbox.min.Z)]
        assert bottom and max(f.area for f in bottom) > 0.5 * cfg.ribbon_width
        # Octagonal tube with a hollow core and chamfered ends.
        area = 2 * np.sqrt(2) * (cfg.pin_radius**2 - (cfg.pin_core_diameter / 2) ** 2)
        assert 0.95 * area * cfg.ribbon_width < pin.volume < area * cfg.ribbon_width

    @pytest.mark.parametrize("roll_deg", [0.0, 22.5])
    def test_pin_fit_in_fixed_and_free_bores(self, roll_deg):
        """The pin is gripped by the bottom knuckles and turns in the top ones.

        A nominal pin touches the fixed bores only at its corners and clears
        the free bores; a pin 0.05 mm oversize already bites into the fixed
        bores (so a slightly fat print cannot wobble there) but still turns
        freely in the free bores.
        """
        bd = _require_cad()
        cfg = TwistedPlateConfig()
        plate = build_twisted_plate_parts(cfg)["plate"]

        def overlap(oversize: float) -> tuple[float, float]:
            pin = build_hinge_pin(
                cfg, pin_diameter=cfg.pin_diameter + 2 * oversize, pin_chamfer=0.0
            )
            pin = bd.Rotation(roll_deg, 0.0, 0.0) * pin
            in_fixed = pin.moved(bd.Location((0.0, 0.0, cfg.z_bottom_hinge)))
            in_free = bd.Rotation(0.0, 0.0, cfg.torsion_degrees) * pin.moved(
                bd.Location((0.0, 0.0, cfg.z_top_hinge))
            )
            return _overlap(plate, in_fixed), _overlap(plate, in_free)

        fixed, free = overlap(0.0)
        assert fixed < 1e-3 and free < 1e-6
        fixed, free = overlap(0.05)
        assert fixed > 0.5 and free < 1e-6

    def test_build_plate_with_ring(self):
        bd = _require_cad()
        parts = build_twisted_plate_parts(with_ring=True)
        assert set(parts.keys()) == {"plate", "ring"}
        plate = parts["plate"]
        ring = parts["ring"]
        assert plate.is_valid()
        assert ring.is_valid()
        assert plate.label == "plate"
        assert ring.label == "ring"

        assert _overlap(plate, ring) < 1e-6

        comp = build_twisted_plate(with_ring=True)
        assert isinstance(comp, bd.Compound)
        assert len(comp.solids()) == 2

    @pytest.mark.parametrize("angle_deg", [0.0, 45.0, -72.8, 90.0])
    def test_custom_parameters(self, angle_deg):
        _require_cad()
        plate = build_twisted_plate(
            torsion_angle=angle_deg,
            ribbon_width=36.0,
            edge_length=24.0,
            num_knuckles=3,
            with_ring=False,
            degrees=True,
        )
        assert plate.is_valid()
        assert plate.volume > 100.0

    def test_cone_hinge_unit(self, tmp_path):
        _require_cad()
        cfg = TwistedPlateConfig(hinge_type="cone")
        plate = build_twisted_plate_parts(cfg)["plate"]
        assert plate.is_valid()
        bottom, top = cfg.knuckle_intervals()
        h = cfg.cone_height
        alpha = np.radians(cfg.torsion_degrees)
        # The cones stand out of the outer bottom knuckles ...
        for x in (bottom[0][1] + 0.5 * h, bottom[-1][0] - 0.5 * h):
            assert plate.is_inside((x, 0.0, cfg.z_bottom_hinge))
        assert not plate.is_inside((bottom[0][1] + 1.5 * h, 0.0, cfg.z_bottom_hinge))
        # ... and the sockets are hollowed into the outer top knuckles.
        for t in (top[0][0] + 0.5 * h, top[-1][1] - 0.5 * h):
            point = (t * np.cos(alpha), t * np.sin(alpha), cfg.z_top_hinge)
            assert not plate.is_inside(point)
        t = top[0][0] + 1.5 * h
        assert plate.is_inside((t * np.cos(alpha), t * np.sin(alpha), cfg.z_top_hinge))

        report = check_hinge_fit(cfg)
        assert report.ok
        # Axial play of the cone in its socket, measured normal to the cone.
        expected = cfg.hinge_gap * np.sin(np.radians(cfg.cone_angle))
        assert report.hinge_clearance == pytest.approx(expected, abs=0.01)

        out = export_print_plate(tmp_path / "cone.3mf", cfg, count=3)
        info = _validate_3mf(out)
        assert info["items"] == 3  # no separate pins
        assert info["meshes"] == 1

    def test_pin_hinge_fit_check(self):
        _require_cad()
        assert check_hinge_fit().ok
        assert check_hinge_fit().hinge_clearance == pytest.approx(0.2, abs=0.01)
        tight = check_hinge_fit(pin_clearance=0.0)
        assert not tight.ok
        assert {issue.parts for issue in tight.issues} == {("pin", "top knuckles")}
        pressed = check_hinge_fit(pin_fit_clearance=-0.1)
        assert ("pin", "bottom knuckles") in {i.parts for i in pressed.issues}

    def test_export_stl_step_and_3mf(self, tmp_path):
        _require_cad()
        stl_path = export_twisted_plate(
            tmp_path / "plate_no_ring.stl",
            torsion_angle=60.0,
            ribbon_width=40.0,
            edge_length=26.0,
            degrees=True,
        )
        step_path = export_twisted_plate(
            tmp_path / "plate_with_ring.step",
            with_ring=True,
        )
        threemf_path = export_twisted_plate(
            tmp_path / "plate_with_ring.3mf",
            with_ring=True,
        )
        assert stl_path.exists() and stl_path.stat().st_size > 1000
        assert step_path.exists() and step_path.stat().st_size > 1000
        assert threemf_path.exists() and threemf_path.stat().st_size > 1000

        info = _validate_3mf(threemf_path)
        assert info["items"] == 2  # the unit and its hinge pin
        assert info["meshes"] == 3  # plate + ring + pin
        assert info["parts_per_item"] == [2, 1]

    def test_kaleidocycle_method_and_assembly(self, tmp_path):
        bd = _require_cad()
        kc = load_sample("mobius_k7_oriented")

        stl_path = kc.export_twisted_plate(
            tmp_path / "k7_unit.stl",
            ribbon_width=45.0,
            edge_length=29.0,
            check_clearance=False,
        )
        assert stl_path.exists() and stl_path.stat().st_size > 1000

        bare = build_kaleidocycle_assembly(
            kc,
            ribbon_width=45.0,
            edge_length=29.0,
            with_ring=False,
            with_pins=False,
        )
        assert isinstance(bare, bd.Compound)
        assert len(bare.solids()) == kc.n

        assembly = build_kaleidocycle_assembly(
            kc,
            ribbon_width=45.0,
            edge_length=29.0,
            with_ring=False,
        )
        solids = {s.label: s for s in assembly.children}
        assert len(solids) == 2 * kc.n
        # Pin i runs through link i (fixed bores) and link i-1 (free bores)
        # without cutting into either of them.
        for i in range(kc.n):
            pin = solids[f"pin_{i}"]
            for j in (i, (i - 1) % kc.n):
                assert _overlap(pin, solids[f"plate_{j}"]) < 1e-3

    def test_n_unit_print_plate_and_3mf(self, tmp_path):
        bd = _require_cad()
        kc = load_sample("mobius_k7_oriented")

        plate_compound = build_print_plate(
            kc,
            ribbon_width=45.0,
            edge_length=29.0,
            with_ring=False,
        )
        assert isinstance(plate_compound, bd.Compound)
        solids = plate_compound.solids()
        assert len(solids) == 2 * kc.n  # n units + n hinge pins
        labels = sorted(child.label for child in plate_compound.children)
        assert labels == sorted(
            [f"plate_{i}" for i in range(kc.n)] + [f"pin_{i}" for i in range(kc.n)]
        )
        for solid in solids:
            assert np.isclose(solid.bounding_box().min.Z, 0.0, atol=1e-4)

        k7_plate_3mf = kc.export_twisted_plate(
            tmp_path / "k7_plate.3mf",
            ribbon_width=45.0,
            edge_length=29.0,
            with_ring=True,
            check_clearance=False,
        )
        assert k7_plate_3mf.exists()

        info = _validate_3mf(k7_plate_3mf)
        assert info["items"] == 2 * kc.n
        assert info["parts_per_item"] == [2] * kc.n + [1] * kc.n
        assert info["meshes"] == 2 * info["assemblies"] + 1  # + one shared pin
        with zipfile.ZipFile(k7_plate_3mf) as zf:
            root = ET.fromstring(zf.read("3D/3dmodel.model"))
        names = {o.get("name") for o in root.findall("./m:resources/m:object", _NS)}
        assert "hinge_pin" in names

        custom_plate = export_print_plate(
            tmp_path / "custom_5_plate.3mf",
            TwistedPlateConfig(),
            count=5,
        )
        info = _validate_3mf(custom_plate)
        assert info["items"] == 10
        assert info["meshes"] == 2
        assert info["parts_per_item"] == [1] * 10

        no_pins = export_print_plate(
            tmp_path / "custom_5_plate_no_pins.3mf",
            TwistedPlateConfig(),
            count=5,
            with_pins=False,
        )
        info = _validate_3mf(no_pins)
        assert info["items"] == 5
        assert info["meshes"] == 1

        assembly_3mf = kc.export_twisted_plate(
            tmp_path / "k7_assembly.3mf", assembly=True, check_clearance=False
        )
        info = _validate_3mf(assembly_3mf)
        assert info["items"] == 1
        assert info["parts_per_item"] == [2 * kc.n]


@pytest.mark.cad
class TestClearanceCheck:
    """Automatic contact checks over the rotation of a kaleidocycle."""

    @pytest.fixture(scope="class")
    def k7_anti(self):
        return Kaleidocycle(7, oriented=False, seed=42)

    def test_trace_motion_keeps_constraints(self):
        kc = load_sample("mobius_k7_oriented")
        frames, description = trace_kaleidocycle_motion(kc)
        assert "flow" in description and len(frames) > 20
        cos0 = np.sum(frames[0][:-1] * frames[0][1:], axis=1)
        for frame in frames[:: max(1, len(frames) // 10)]:
            cos = np.sum(frame[:-1] * frame[1:], axis=1)
            np.testing.assert_allclose(cos, cos0, atol=1e-4)

    @pytest.mark.parametrize("hinge_type", ["pin", "cone"])
    def test_k7_clears_its_rotation_at_43mm(self, k7_anti, hinge_type):
        _require_cad()
        report = check_kaleidocycle_clearance(
            k7_anti, hinge_type=hinge_type, max_frames=80
        )
        assert report.ok, report.summary()
        assert report.closest_non_adjacent > report.min_clearance
        assert "flow" in report.motion

    def test_k7_collides_at_45mm(self, k7_anti):
        _require_cad()
        report = check_kaleidocycle_clearance(k7_anti, ribbon_width=45.0)
        assert not report.ok
        assert {issue.kind for issue in report.issues} == {"non-adjacent"}
        assert report.closest_non_adjacent < report.min_clearance

    def test_export_warns_or_raises_on_contact(self, tmp_path):
        _require_cad()
        # The oriented sample has 107-degree torsion: wide links run into
        # their neighbours' far ends during the rotation.
        kc = load_sample("mobius_k7_oriented")
        with pytest.warns(ClearanceWarning, match="adjacent"):
            kc.export_twisted_plate(tmp_path / "warn.3mf")
        assert (tmp_path / "warn.3mf").exists()
        with pytest.raises(ClearanceError):
            export_print_plate(tmp_path / "strict.3mf", kc, check_clearance="error")
        assert not (tmp_path / "strict.3mf").exists()
