"""Automatic contact checks for printable kaleidocycle models.

Two checks guard the parts produced by :mod:`kaleidocycle.cad`:

hinge fit (:func:`check_hinge_fit`)
    A link and a copy of it attached at its bottom hinge must not touch:
    knuckles, pins, cones and sockets keep their clearances; a pin may touch
    the bottom knuckles that hold it (a press fit) but not cut into them.  Needs only a
    :class:`~kaleidocycle.cad.TwistedPlateConfig`.

motion (:func:`check_kaleidocycle_clearance`)
    The closed ring of links is moved through one full cycle of its rotation,
    traced with the kaleidocycle's integrable (mKdV / sine-Gordon) flow, and
    every pair of links is checked for clearance.  Non-adjacent links must
    stay ``min_clearance`` apart; adjacent links are compared away from their
    shared hinge, whose fit the first check covers.  Surface samples screen
    all pairs quickly; near misses are then measured exactly on the B-rep.

The export functions of :mod:`kaleidocycle.cad` and
:meth:`Kaleidocycle.export_twisted_plate` run these checks automatically and
emit a :class:`ClearanceWarning` (or raise :class:`ClearanceError` with
``check_clearance="error"``) when a contact is found.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .animation import sine_gordon_step
from .cad import (
    TwistedPlateConfig,
    _config_key,
    _extract_hinges,
    _link_placements,
    _require_build123d,
    build_hinge_pin,
    build_twisted_plate_parts,
)
from .geometry import binormals_to_tangents

if TYPE_CHECKING:
    from .geometry import Kaleidocycle

__all__ = [
    "ClearanceError",
    "ClearanceIssue",
    "ClearanceReport",
    "ClearanceWarning",
    "check_hinge_fit",
    "check_kaleidocycle_clearance",
    "trace_kaleidocycle_motion",
]

# Default clearance (mm) required between links that do not share a hinge.
DEFAULT_MIN_CLEARANCE = 0.4
# Parts that are designed to be apart at a hinge must keep at least this gap.
_HINGE_TOLERANCE = 0.05
# A pin pressed into the bottom knuckles may touch them, but not cut into them.
_PRESS_FIT_VOLUME = 1e-3
# Surface sampling: any surface point lies within 0.6 sample spacings of a
# sample, so a sampled distance exceeds the true one by at most 1.2 spacings.
_SAMPLE_ERROR = 1.2
# Coarse screening keeps one fine sample per cubic voxel of this size (mm).
_VOXEL = 1.0
_VOXEL_ERROR = _VOXEL * math.sqrt(3.0)
_REFINE_SPACING = 0.1
# Closest approaches are measured (and reported) up to this far beyond the
# required clearance; farther pairs are reported as "no approach closer".
_REPORT_RANGE = 2.0
_MAX_CHECK_FRAMES = 360


class ClearanceWarning(UserWarning):
    """Parts of a kaleidocycle model touch or come too close."""


class ClearanceError(ValueError):
    """Raised instead of :class:`ClearanceWarning` when checks are strict."""


@dataclass(frozen=True)
class ClearanceIssue:
    """One contact found by a clearance check.

    ``clearance`` is the smallest distance in mm (``0`` when the parts touch
    or overlap, in which case ``overlap_volume`` gives the overlap in mm^3),
    and ``phase`` the position in the rotation cycle (``0 <= phase < 1``) or
    ``None`` for the hinge fit.
    """

    kind: str
    parts: tuple[str, str]
    clearance: float
    phase: float | None = None
    overlap_volume: float = 0.0

    def describe(self) -> str:
        """One-line human-readable description."""
        if self.overlap_volume > 0.0:
            amount = f"overlap {self.overlap_volume:.2f} mm^3"
        else:
            amount = f"clearance {self.clearance:.2f} mm"
        where = "" if self.phase is None else f" at {self.phase:.0%} of the cycle"
        return f"{self.kind}: {self.parts[0]} / {self.parts[1]}, {amount}{where}"


@dataclass
class ClearanceReport:
    """Result of :func:`check_kaleidocycle_clearance` or :func:`check_hinge_fit`.

    ``closest_adjacent`` and ``closest_non_adjacent`` are the smallest
    clearances found in mm: exact where a near miss was re-measured on the
    B-rep, otherwise a sampled estimate accurate to about the sample spacing;
    ``inf`` when no pair came within ``min_clearance + 2.6`` mm, and ``None``
    when the motion was not checked.
    """

    min_clearance: float
    hinge_type: str
    motion: str
    frames: int
    hinge_clearance: float
    closest_adjacent: float | None = None
    closest_non_adjacent: float | None = None
    issues: list[ClearanceIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether no contact was found."""
        return not self.issues

    def summary(self) -> str:
        """Multi-line human-readable summary."""

        def fmt(value: float | None) -> str:
            if value is None:
                return "not checked"
            if not math.isfinite(value):
                return f"more than {self.min_clearance + 2.6:.1f} mm"
            return f"{value:.2f} mm"

        lines = [
            f"Clearance check ({self.hinge_type} hinges, required "
            f"{self.min_clearance:.2f} mm between non-adjacent links): "
            + ("OK" if self.ok else f"{len(self.issues)} contact(s)"),
            f"  motion          : {self.motion} ({self.frames} poses checked)",
            f"  hinge fit       : {fmt(self.hinge_clearance)}",
            f"  adjacent links  : {fmt(self.closest_adjacent)} (away from the hinge)",
            f"  non-adjacent    : {fmt(self.closest_non_adjacent)}",
        ]
        lines += [f"  - {issue.describe()}" for issue in self.issues]
        return "\n".join(lines)


def _resolve_policy(check_clearance: bool | str) -> str:
    if check_clearance is True:
        return "warn"
    if check_clearance is False or check_clearance is None:
        return "off"
    policy = str(check_clearance).lower()
    if policy not in ("warn", "error", "off"):
        raise ValueError(
            "check_clearance must be True, False, 'warn', 'error' or 'off'"
        )
    return policy


def enforce_report(
    report: ClearanceReport, check_clearance: bool | str, *, stacklevel: int = 3
) -> None:
    """Warn about (or raise for) the contacts in ``report`` per the policy."""
    policy = _resolve_policy(check_clearance)
    if report.ok or policy == "off":
        return
    message = (
        report.summary()
        + "\nAdjust ribbon_width, edge_length, or the hinge clearances; pass "
        "check_clearance=False to skip this check."
    )
    if policy == "error":
        raise ClearanceError(message)
    warnings.warn(message, ClearanceWarning, stacklevel=stacklevel)


# --------------------------------------------------------------------------
# Hinge fit
# --------------------------------------------------------------------------

_HINGE_CACHE: dict[tuple[Any, ...], tuple[float, list[ClearanceIssue]]] = {}


def _as_shape(result: Any) -> Any:
    """Normalise a boolean result (``None``, a shape, or a list of shapes)."""
    if result is None or not isinstance(result, (list, tuple)):
        return result
    if not result:
        return None
    if len(result) == 1:
        return result[0]
    return _require_build123d().Compound(children=list(result))


def _overlap_volume(a: Any, b: Any) -> float:
    common = _as_shape(a.intersect(b))
    if common is None:
        return 0.0
    return float(sum(s.volume for s in common.solids()))


def _hinge_fit(cfg: TwistedPlateConfig) -> tuple[float, list[ClearanceIssue]]:
    key = _config_key(cfg)
    if key in _HINGE_CACHE:
        return _HINGE_CACHE[key]
    bd = _require_build123d()
    plate = build_twisted_plate_parts(cfg)["plate"]
    z0, z1 = cfg.z_bottom_hinge, cfg.z_top_hinge
    # The next-lower link, attached by its top hinge to this link's bottom
    # hinge and folded straight (its plate points away from this one).
    to_bottom = (
        bd.Location((0.0, 0.0, z0))
        * bd.Rotation(0.0, 0.0, -cfg.torsion_degrees)
        * bd.Location((0.0, 0.0, -z1))
    )
    neighbour = plate.moved(to_bottom)
    reach = cfg.hinge_radius + cfg.gap + 1.5
    zone = bd.Pos(0.0, 0.0, z0) * bd.Box(
        cfg.ribbon_width + 4.0, 2.0 * reach, 2.0 * reach
    )
    near_self = _as_shape(plate.intersect(zone))
    near_other = _as_shape(neighbour.intersect(zone))

    issues: list[ClearanceIssue] = []
    clearance = float(near_self.distance_to(near_other))
    if clearance < _HINGE_TOLERANCE:
        issues.append(
            ClearanceIssue(
                "hinge",
                ("bottom knuckles", "mating top knuckles"),
                clearance,
                overlap_volume=_overlap_volume(near_self, near_other),
            )
        )
    if cfg.uses_pins:
        pin = build_hinge_pin(cfg).moved(bd.Location((0.0, 0.0, z0)))
        pressed = _overlap_volume(pin, near_self)
        if pressed > _PRESS_FIT_VOLUME:
            issues.append(
                ClearanceIssue(
                    "hinge",
                    ("pin", "bottom knuckles"),
                    0.0,
                    overlap_volume=pressed,
                )
            )
        free = float(pin.distance_to(near_other))
        clearance = min(clearance, free)
        if free < _HINGE_TOLERANCE:
            issues.append(
                ClearanceIssue(
                    "hinge",
                    ("pin", "top knuckles"),
                    free,
                    overlap_volume=_overlap_volume(pin, near_other),
                )
            )
    _HINGE_CACHE[key] = (clearance, issues)
    return clearance, issues


def check_hinge_fit(
    config: TwistedPlateConfig | None = None,
    *,
    min_clearance: float = DEFAULT_MIN_CLEARANCE,
    **overrides: Any,
) -> ClearanceReport:
    """Check the knuckles, pins, cones and sockets of one hinge for contact.

    A link is joined at its bottom hinge to a copy of itself; parts that are
    meant to move must stay at least 0.05 mm apart, and a pin may touch, but
    not cut into, the bottom knuckles that hold it.
    """
    from .cad import _resolve_config

    cfg = _resolve_config(config, **overrides)
    clearance, issues = _hinge_fit(cfg)
    return ClearanceReport(
        min_clearance=float(min_clearance),
        hinge_type=cfg.hinge_type,
        motion="not checked (no kaleidocycle given)",
        frames=0,
        hinge_clearance=clearance,
        issues=list(issues),
    )


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------


def _fold_angles(hinges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed angle between the two links at every hinge."""
    tangents = binormals_to_tangents(hinges, normalize=True)
    prev = -np.roll(tangents, 1, axis=0)
    b = hinges[:-1]
    sin = np.einsum("ij,ij->i", np.cross(prev, tangents), b)
    cos = np.einsum("ij,ij->i", prev, tangents)
    return np.arctan2(sin, cos)


def _is_oriented(hinges: NDArray[np.float64]) -> bool:
    return bool(np.dot(hinges[-1], hinges[0]) > 0.0)


def trace_kaleidocycle_motion(
    kc_or_hinges: Kaleidocycle | NDArray[np.float64],
    *,
    step_size: float = 0.02,
    max_steps: int = 6000,
) -> tuple[list[NDArray[np.float64]], str]:
    """Trace one full rotation cycle of a kaleidocycle.

    Integrates the mKdV2, sine-Gordon (anti-oriented only) and mKdV flows in
    turn, keeping the first one that preserves the constraints (constant
    torsion and closure) and returns to the starting fold angles.  Returns
    the hinge frames of one cycle and a description; when no flow works,
    returns only the given pose.
    """
    hinges = _extract_hinges(kc_or_hinges)
    oriented = getattr(kc_or_hinges, "oriented", None)
    if oriented is None:
        oriented = _is_oriented(hinges)
    cos0 = np.einsum("ij,ij->i", hinges[:-1], hinges[1:])
    folds0 = _fold_angles(hinges)
    rules = ["mKdV2"] + ([] if oriented else ["sine-Gordon"]) + ["mKdV"]

    for rule in rules:
        frames = [hinges]
        current = hinges
        devs = [0.0]
        left_start = False
        for _ in range(max_steps):
            current = sine_gordon_step(
                current, step_size, rule=rule, oriented=bool(oriented)
            )
            cos = np.einsum("ij,ij->i", current[:-1], current[1:])
            closure = np.linalg.norm(
                binormals_to_tangents(current, normalize=True).sum(axis=0)
            )
            if np.max(np.abs(cos - cos0)) > 1e-4 or closure > 1e-4 * len(cos):
                break
            dev = float(
                np.max(np.abs(np.angle(np.exp(1j * (_fold_angles(current) - folds0)))))
            )
            left_start = left_start or dev > 0.3
            if left_start and devs[-1] < 0.03 and dev > devs[-1]:
                period = len(frames) - 1
                return frames[:period], f"{rule} flow, cycle of {period} steps"
            frames.append(current)
            devs.append(dev)
    return [hinges], "given pose only (the rotation could not be traced)"


@dataclass
class _Side:
    """Fine and voxel-coarse samples of a body, optionally away from an axis."""

    fine: NDArray[np.float64]
    coarse: NDArray[np.float64]
    fine_tree: cKDTree
    coarse_tree: cKDTree
    order: NDArray[np.int64]  # fine sample indices grouped by voxel
    starts: NDArray[np.int64]
    counts: NDArray[np.int64]

    def fine_near(self, voxels: NDArray[np.int64]) -> NDArray[np.int64]:
        """Indices of the fine samples in the given coarse voxels."""
        counts = self.counts[voxels]
        offsets = np.repeat(self.starts[voxels] - np.cumsum(counts) + counts, counts)
        return self.order[offsets + np.arange(counts.sum())]


@dataclass
class _Body:
    """Sampled surface of one link (plate, ring, pin) in its printing frame."""

    cfg: TwistedPlateConfig
    solids: list[Any]
    verts: NDArray[np.float64]
    tris: NDArray[np.int64]
    sides: dict[str, _Side]
    axes: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]
    reach: float
    center: NDArray[np.float64]
    radius: float
    trimmed: dict[str, Any] = field(default_factory=dict)
    tri_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    def away_from(self, side: str, pts: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Mask of ``pts`` outside the hinge zone of ``side`` (all for "all")."""
        if side == "all":
            return np.ones(len(pts), dtype=bool)
        point, direction = self.axes[side]
        return _axis_distance(pts, point, direction) >= self.reach


def _surface_samples(
    verts: NDArray[np.float64], tris: NDArray[np.int64], spacing: float
) -> NDArray[np.float64]:
    """Points on a triangle mesh with every surface point within
    ``0.6 * spacing`` of a sample."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    longest = np.max(
        [
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ],
        axis=0,
    )
    divisions = np.maximum(1, np.ceil(longest / spacing)).astype(int)
    pts = [verts[np.unique(tris)]]
    for k in np.unique(divisions):
        if k == 1:
            continue
        sel = divisions == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        u = (i[keep] / k)[None, :, None]
        v = (j[keep] / k)[None, :, None]
        a_s, b_s, c_s = a[sel][:, None, :], b[sel][:, None, :], c[sel][:, None, :]
        pts.append((a_s + u * (b_s - a_s) + v * (c_s - a_s)).reshape(-1, 3))
    return np.unique(np.round(np.vstack(pts), 5), axis=0)


def _axis_distance(
    pts: NDArray[np.float64], point: NDArray[np.float64], direction: NDArray[np.float64]
) -> NDArray[np.float64]:
    w = pts - point
    return np.linalg.norm(w - np.outer(w @ direction, direction), axis=1)


def _make_side(fine: NDArray[np.float64]) -> _Side:
    keys = np.floor(fine / _VOXEL).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    coarse = fine[first]
    return _Side(
        fine=fine,
        coarse=coarse,
        fine_tree=cKDTree(fine),
        coarse_tree=cKDTree(coarse),
        order=order,
        starts=starts,
        counts=counts,
    )


def _make_body(cfg: TwistedPlateConfig, spacing: float, with_pins: bool) -> _Body:
    bd = _require_build123d()
    parts = build_twisted_plate_parts(cfg)
    solids = list(parts.values())
    if with_pins and cfg.uses_pins:
        solids.append(
            build_hinge_pin(cfg).moved(bd.Location((0.0, 0.0, cfg.z_bottom_hinge)))
        )
    verts_all, tris_all, offset = [], [], 0
    for solid in solids:
        verts, tris = bd.Mesher._mesh_shape(solid, 0.02, 0.2)
        verts_all.append(np.array([[float(c) for c in v] for v in verts]))
        tris_all.append(np.asarray(tris, dtype=np.int64) + offset)
        offset += len(verts)
    verts = np.vstack(verts_all)
    tris = np.vstack(tris_all)
    fine = _surface_samples(verts, tris, spacing)

    alpha = math.radians(cfg.torsion_degrees)
    axes = {
        "bottom": (np.array([0.0, 0.0, cfg.z_bottom_hinge]), np.array([1.0, 0, 0])),
        "top": (
            np.array([0.0, 0.0, cfg.z_top_hinge]),
            np.array([math.cos(alpha), math.sin(alpha), 0.0]),
        ),
    }
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (lo + hi)
    body = _Body(
        cfg=cfg,
        solids=solids,
        verts=verts,
        tris=tris,
        sides={},
        axes=axes,
        # Points this close to a hinge axis belong to the hinge itself.
        reach=cfg.hinge_radius + cfg.gap + 1.0,
        center=center,
        radius=float(np.linalg.norm(verts - center, axis=1).max()),
    )
    for side in ("all", "bottom", "top"):
        body.sides[side] = _make_side(fine[body.away_from(side, fine)])
    return body


def _trimmed_solids(body: _Body, side: str) -> list[Any]:
    """The body's solids with the neighbourhood of one hinge axis removed."""
    if side == "all":
        return body.solids
    if side not in body.trimmed:
        bd = _require_build123d()
        cfg = body.cfg
        length = 4.0 * cfg.ribbon_width
        z = cfg.z_bottom_hinge if side == "bottom" else cfg.z_top_hinge
        cutter = (
            bd.Pos(0.0, 0.0, z)
            * bd.Rotation(0.0, 90.0, 0.0)
            * bd.Cylinder(body.reach, length)
        )
        if side == "top":
            cutter = bd.Rotation(0.0, 0.0, cfg.torsion_degrees) * cutter
        kept = []
        for solid in body.solids:
            rest = solid - cutter
            kept.extend(rest.solids() if rest is not None else [])
        body.trimmed[side] = kept
    return body.trimmed[side]


def _to_local(
    pts: NDArray[np.float64],
    src: tuple[NDArray[np.float64], NDArray[np.float64], float],
    dst: tuple[NDArray[np.float64], NDArray[np.float64], float],
) -> NDArray[np.float64]:
    """Map printing-frame points of one placed link into another's frame."""
    o_s, r_s, z_s = src
    o_d, r_d, z_d = dst
    world = o_s + (pts - np.array([0.0, 0.0, z_s])) @ r_s.T
    return (world - o_d) @ r_d + np.array([0.0, 0.0, z_d])


def _pair_distance(
    a: _Body,
    b: _Body,
    a_side: str,
    b_side: str,
    pose_a: tuple[NDArray[np.float64], NDArray[np.float64], float],
    pose_b: tuple[NDArray[np.float64], NDArray[np.float64], float],
    fine_bound: float,
    coarse_bound: float,
) -> tuple[float, float, NDArray[np.float64] | None]:
    """Sampled distances between two placed links.

    Returns ``(coarse, fine, near)``: the closest voxel-sample distance (``inf``
    beyond ``coarse_bound``); the closest fine-sample distance (``inf`` beyond
    ``fine_bound``), which exceeds the true distance by at most
    ``_SAMPLE_ERROR`` sample spacings; and the closest fine sample of ``a`` in
    its own frame.  Voxel samples locate the regions of ``a`` that may come
    within ``fine_bound`` of ``b``; only the fine samples there are measured.
    """
    sa, sb = a.sides[a_side], b.sides[b_side]
    local = _to_local(sa.coarse, pose_a, pose_b)
    dist, _ = sb.coarse_tree.query(local, k=1, distance_upper_bound=coarse_bound)
    coarse = float(dist.min())
    seeds = np.flatnonzero(dist <= fine_bound + 2.0 * _VOXEL_ERROR)
    if seeds.size == 0:
        return coarse, math.inf, None
    idx = sa.fine_near(seeds)
    local = _to_local(sa.fine[idx], pose_a, pose_b)
    dist, _ = sb.fine_tree.query(local, k=1, distance_upper_bound=fine_bound)
    k = int(np.argmin(dist))
    if not math.isfinite(dist[k]):
        return coarse, math.inf, None
    return coarse, float(dist[k]), sa.fine[idx[k]]


def _refine_distance(
    a: _Body,
    b: _Body,
    a_side: str,
    b_side: str,
    pose_a: tuple[NDArray[np.float64], NDArray[np.float64], float],
    pose_b: tuple[NDArray[np.float64], NDArray[np.float64], float],
    near_a: NDArray[np.float64],
    radius: float,
) -> float:
    """Re-measure a near miss with dense local samples (about 0.1 mm error)."""
    near_b = _to_local(near_a[None, :], pose_a, pose_b)[0]

    def local_samples(body: _Body, side: str, center: NDArray[np.float64]) -> Any:
        if body.tri_bounds is None:
            corners = body.verts[body.tris]
            body.tri_bounds = (corners.min(axis=1), corners.max(axis=1))
        lo, hi = body.tri_bounds
        sel = np.all((lo - radius <= center) & (center <= hi + radius), axis=1)
        if not sel.any():
            return np.empty((0, 3))
        pts = _surface_samples(body.verts, body.tris[sel], _REFINE_SPACING)
        pts = pts[np.linalg.norm(pts - center, axis=1) <= radius]
        return pts[body.away_from(side, pts)]

    pts_a = local_samples(a, a_side, near_a)
    pts_b = local_samples(b, b_side, near_b)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return math.inf
    dist, _ = cKDTree(pts_b).query(_to_local(pts_a, pose_a, pose_b), k=1)
    return float(dist.min())


def _overlap_between(
    a: _Body,
    b: _Body,
    a_side: str,
    b_side: str,
    pose_a: tuple[NDArray[np.float64], NDArray[np.float64], float],
    pose_b: tuple[NDArray[np.float64], NDArray[np.float64], float],
) -> float:
    """Overlap volume (mm^3) of two placed links, measured on the B-rep."""
    from .cad import _placement_location

    bd = _require_build123d()

    def place(solids: list[Any], pose: Any) -> list[Any]:
        origin, rotation, z = pose
        loc = _placement_location(bd, origin, rotation) * bd.Location((0.0, 0.0, -z))
        return [s.moved(loc) for s in solids]

    sa = place(_trimmed_solids(a, a_side), pose_a)
    sb = place(_trimmed_solids(b, b_side), pose_b)
    return sum(_overlap_volume(x, y) for x in sa for y in sb)


_MOTION_CACHE: dict[tuple[Any, ...], ClearanceReport] = {}


def check_kaleidocycle_clearance(
    kc_or_hinges: Kaleidocycle | NDArray[np.float64],
    *,
    ribbon_width: float = 43.0,
    edge_length: float = 29.0,
    scale: float | None = None,
    with_ring: bool = False,
    with_pins: bool = True,
    normalize_segments: bool = True,
    min_clearance: float = DEFAULT_MIN_CLEARANCE,
    motion: bool = True,
    sample_spacing: float = 0.5,
    max_frames: int = _MAX_CHECK_FRAMES,
    **kwargs: Any,
) -> ClearanceReport:
    """Check a kaleidocycle model for contacts through its whole rotation.

    Builds the links exactly as :func:`~kaleidocycle.cad.build_kaleidocycle_assembly`
    does (``ribbon_width``, ``edge_length``, ``with_ring``, hinge options in
    ``kwargs``), checks the hinge fit, then moves the ring through one cycle
    of its rotation (at most ``max_frames`` evenly spaced poses; only the
    given pose when ``motion=False``) and measures every pair of links.  The
    captive ring, when present, is checked at its printed position.
    """
    hinges = _extract_hinges(kc_or_hinges)
    cache_key = (
        np.round(hinges, 8).tobytes(),
        float(ribbon_width),
        float(edge_length),
        scale,
        bool(with_ring),
        bool(with_pins),
        bool(normalize_segments),
        float(min_clearance),
        bool(motion),
        float(sample_spacing),
        int(max_frames),
        tuple(sorted(kwargs.items())),
    )
    if cache_key in _MOTION_CACHE:
        return _MOTION_CACHE[cache_key]

    place_kw = {
        "edge_length": edge_length,
        "scale": scale,
        "normalize_segments": normalize_segments,
    }
    placements = _link_placements(hinges, **place_kw)
    n = len(placements)
    bodies: dict[tuple[Any, ...], _Body] = {}
    link_body: list[_Body] = []
    for torsion, length, _, _ in placements:
        cfg = TwistedPlateConfig(
            torsion_angle=torsion,
            ribbon_width=float(ribbon_width),
            edge_length=length,
            with_ring=bool(with_ring),
            **kwargs,
        )
        cfg.validate()
        key = _config_key(cfg)
        if key not in bodies:
            bodies[key] = _make_body(cfg, sample_spacing, with_pins)
        link_body.append(bodies[key])

    issues: list[ClearanceIssue] = []
    hinge_clearance = math.inf
    for body in bodies.values():
        clearance, found = _hinge_fit(body.cfg)
        hinge_clearance = min(hinge_clearance, clearance)
        issues.extend(found)

    if motion:
        frames, description = trace_kaleidocycle_motion(hinges)
    else:
        frames, description = [hinges], "given pose only"
    stride = max(1, math.ceil(len(frames) / max(1, int(max_frames))))
    checked = list(range(0, len(frames), stride))

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1:
                pairs.append((i, j, "adjacent", "top", "bottom"))
            elif i == 0 and j == n - 1:
                pairs.append((j, i, "adjacent", "top", "bottom"))
            else:
                pairs.append((i, j, "non-adjacent", "all", "all"))

    limit = float(min_clearance)
    margin = _SAMPLE_ERROR * sample_spacing
    # Every pose is screened for contacts closer than ``limit`` (allowing for
    # the sampling error); closest approaches up to ``_REPORT_RANGE`` beyond
    # are measured afterwards at each pair's closest pose only.
    critical = limit + margin
    report_bound = critical + _REPORT_RANGE
    coarse_bound = report_bound + 2.0 * _VOXEL_ERROR
    best: dict[tuple[int, int], tuple[float, int, Any, Any]] = {}
    nearest: dict[tuple[int, int], tuple[float, int, Any]] = {}
    for f in checked:
        poses = []
        for k, (_, _, origin, rotation) in enumerate(
            _link_placements(frames[f], **place_kw)
        ):
            poses.append((origin, rotation, link_body[k].cfg.z_bottom_hinge))
        centers = [
            p[0] + (link_body[k].center - np.array([0.0, 0.0, p[2]])) @ p[1].T
            for k, p in enumerate(poses)
        ]
        for a_i, b_i, _, a_side, b_side in pairs:
            a, b = link_body[a_i], link_body[b_i]
            gap = np.linalg.norm(centers[a_i] - centers[b_i]) - a.radius - b.radius
            if gap > coarse_bound:
                continue
            coarse, d, near = _pair_distance(
                a, b, a_side, b_side, poses[a_i], poses[b_i], critical, coarse_bound
            )
            if coarse < nearest.get((a_i, b_i), (math.inf,))[0]:
                nearest[(a_i, b_i)] = (coarse, f, poses)
            if d < best.get((a_i, b_i), (math.inf,))[0]:
                best[(a_i, b_i)] = (d, f, poses, near)

    # Pairs that stayed clear of ``critical``: estimate their closest approach.
    for a_i, b_i, _, a_side, b_side in pairs:
        if (a_i, b_i) in best or (a_i, b_i) not in nearest:
            continue
        _, f, poses = nearest[(a_i, b_i)]
        _, d, near = _pair_distance(
            link_body[a_i],
            link_body[b_i],
            a_side,
            b_side,
            poses[a_i],
            poses[b_i],
            report_bound,
            coarse_bound,
        )
        if math.isfinite(d):
            best[(a_i, b_i)] = (d, f, poses, near)

    # Re-measure near misses (and the closest pair of each kind) densely.
    closest_pair: dict[str, tuple[float, tuple[int, int]]] = {}
    for a_i, b_i, kind, _, _ in pairs:
        if (a_i, b_i) in best:
            d = best[(a_i, b_i)][0]
            if d < closest_pair.get(kind, (math.inf,))[0]:
                closest_pair[kind] = (d, (a_i, b_i))
    to_refine = {pair for _, pair in closest_pair.values()}
    to_refine |= {pair for pair, value in best.items() if value[0] < critical}

    closest = {"adjacent": math.inf, "non-adjacent": math.inf}
    for a_i, b_i, kind, a_side, b_side in pairs:
        if (a_i, b_i) not in best:
            continue
        d, f, poses, near = best[(a_i, b_i)]
        a, b = link_body[a_i], link_body[b_i]
        pose_a, pose_b = poses[a_i], poses[b_i]
        if (a_i, b_i) in to_refine:
            d = min(
                d,
                _refine_distance(
                    a, b, a_side, b_side, pose_a, pose_b, near, min(d, critical) + 0.5
                ),
            )
        closest[kind] = min(closest[kind], d)
        if d < limit:
            overlap = 0.0
            if d < 2.0 * _SAMPLE_ERROR * _REFINE_SPACING:
                overlap = _overlap_between(a, b, a_side, b_side, pose_a, pose_b)
            issues.append(
                ClearanceIssue(
                    kind,
                    (f"link {min(a_i, b_i)}", f"link {max(a_i, b_i)}"),
                    0.0 if overlap > 0.0 else d,
                    phase=f / len(frames),
                    overlap_volume=overlap,
                )
            )

    report = ClearanceReport(
        min_clearance=limit,
        hinge_type=link_body[0].cfg.hinge_type,
        motion=description,
        frames=len(checked),
        hinge_clearance=hinge_clearance,
        closest_adjacent=closest["adjacent"],
        closest_non_adjacent=closest["non-adjacent"],
        issues=issues,
    )
    _MOTION_CACHE[cache_key] = report
    return report
