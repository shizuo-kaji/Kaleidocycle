"""Parametric 3D-printable CAD model generation using ``build123d``.

This module generates parametric twisted-plate units (and full ring
assemblies) modeled after the print-in-place Möbius Kaleidocycle design
("Twisted Plate with Ring"), with configurable torsion angle, ribbon
width, edge length, hinge dimensions, and optional captive ring.

Adjacent units are joined by separately printed octagonal hinge pins, as
in NovemberKou's "Möbius Kaleidocycle (Material-saving Design)": each pin
is held by the bottom knuckles of one unit (a snug fit with a small radial
clearance at the octagon corners) and the top knuckles of the neighbouring
unit rotate on it with a wider clearance.  The pins
are laid out on the same print plate as the units.

``build123d`` is an optional dependency; importing this module does not
require ``build123d`` until one of the CAD build or export functions is
called.
"""

from __future__ import annotations

import importlib
import math
import zipfile
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape, quoteattr

import numpy as np
from numpy.typing import NDArray

from .geometry import binormals_to_tangents, tangents_to_curve

if TYPE_CHECKING:
    from .geometry import Kaleidocycle

_OCP_STATIC_PATCHES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "OCP.TopoDS",
        "TopoDS",
        (
            "Vertex",
            "Edge",
            "Wire",
            "Face",
            "Shell",
            "Solid",
            "CompSolid",
            "Compound",
        ),
    ),
    ("OCP.TopExp", "TopExp", ("Vertices", "MapShapesAndAncestors")),
    (
        "OCP.BRep",
        "BRep_Tool",
        (
            "Pnt",
            "Curve",
            "CurveOnPlane",
            "Surface",
            "IsClosed",
            "Degenerated",
            "Triangulation",
        ),
    ),
    (
        "OCP.BRepTools",
        "BRepTools",
        (
            "OuterWire",
            "UVBounds",
            "Clean",
            "Triangulation",
            "Write",
            "Read",
        ),
    ),
    (
        "OCP.BRepGProp",
        "BRepGProp",
        ("LinearProperties", "SurfaceProperties", "VolumeProperties"),
    ),
    ("OCP.BRepBndLib", "BRepBndLib", ("Add", "AddOptimal", "AddOBB")),
    ("OCP.BRepLib", "BRepLib", ("BuildCurves3d",)),
    ("OCP.BRepFill", "BRepFill", ("Face", "Shell")),
    ("OCP.BRepAlgo", "BRepAlgo", ("ConvertFace",)),
    ("OCP.GCPnts", "GCPnts_AbscissaPoint", ("Length",)),
    ("OCP.ShapeAnalysis", "ShapeAnalysis_Curve", ("IsPlanar",)),
    (
        "OCP.ShapeAnalysis",
        "ShapeAnalysis_FreeBounds",
        ("ConnectEdgesToWires",),
    ),
    ("OCP.ShapeCustom", "ShapeCustom", ("BSplineRestriction",)),
    ("OCP.GeomConvert", "GeomConvert", ("SplitBSplineCurve",)),
    ("OCP.Precision", "Precision", ("Confusion",)),
    ("OCP.Quantity", "Quantity_Color", ("StringName",)),
    ("OCP.Quantity", "Quantity_ColorRGBA", ("ColorFromName",)),
    ("OCP.Font", "Font_FontMgr", ("GetInstance",)),
    ("OCP.Message", "Message", ("DefaultMessenger",)),
    ("OCP.Interface", "Interface_Static", ("SetIVal",)),
    ("OCP.STEPControl", "STEPControl_Controller", ("Init",)),
    ("OCP.STEPCAFControl", "STEPCAFControl_Controller", ("Init",)),
    ("OCP.IGESControl", "IGESControl_Controller", ("Init",)),
    ("OCP.XCAFApp", "XCAFApp_Application", ("GetApplication",)),
    (
        "OCP.XCAFDoc",
        "XCAFDoc_DocumentTool",
        ("ShapeTool", "ColorTool", "SetLengthUnit", "IsXCAFDocument"),
    ),
    (
        "OCP.XCAFDoc",
        "XCAFDoc_ShapeTool",
        (
            "GetShape",
            "GetComponents",
            "GetLocation",
            "GetReferredShape",
            "IsAssembly",
            "IsReference",
            "SetAutoNaming",
        ),
    ),
    ("OCP.TDataStd", "TDataStd_Name", ("GetID", "Set")),
    ("OCP.BinTools", "BinTools", ("Write", "Read")),
    ("OCP.RWStl", "RWStl", ("ReadFile",)),
)


def _ensure_ocp_compat() -> None:
    """Patch OCP 7.8+ bindings when paired with ``build123d`` 0.7.0."""
    try:
        from OCP.TopoDS import TopoDS_Shape
    except ImportError:
        return

    if not hasattr(TopoDS_Shape, "HashCode"):
        TopoDS_Shape.HashCode = (
            lambda self, m=2147483647: (hash(self) & 0x7FFFFFFF) % m
            if m
            else (hash(self) & 0x7FFFFFFF)
        )

    for mod_name, cls_name, methods in _OCP_STATIC_PATCHES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        for meth_name in methods:
            alias_name = f"{meth_name}_s"
            if not hasattr(cls, alias_name) and hasattr(cls, meth_name):
                setattr(cls, alias_name, getattr(cls, meth_name))


def _require_build123d() -> Any:
    """Lazily import ``build123d`` with OCP compatibility shims."""
    _ensure_ocp_compat()
    try:
        import build123d as bd
    except ImportError as exc:
        raise ImportError(
            "build123d is required for 3D CAD model generation. "
            'Install it with `pip install "kaleidocycle[cad]"` '
            "(or `conda install -c conda-forge build123d`)."
        ) from exc
    return bd


# Minimum wall thickness (mm) kept around hinge bores and pin cores: two
# perimeters of a 0.4 mm nozzle.
_MIN_WALL = 0.8
# Height (mm) of the flat bed contact cut from the round bottom knuckles.
_KNUCKLE_FOOT_DEPTH = 0.3
# Hinge bores are printed with a horizontal axis, so their top is a 45-degree
# teardrop bridged by a flat this far (mm) above the bore circle.  The margin
# leaves room for the printed bridge to sag without touching the turning pin.
_BORE_FLAT_MARGIN = 0.3


HINGE_TYPES = ("pin", "cone")
# Default knuckle diameters per hinge type (mm), from the reference designs.
_DEFAULT_HINGE_DIAMETER = {"pin": 7.0, "cone": 6.0}
# Default axial gap between mating knuckles per hinge type (mm).  Pin hinges
# use 0.4 mm: 0.2 mm, as in the reference design, fitted but turned stiffly
# in a test print.  For cones the gap is also their axial play.
_DEFAULT_HINGE_GAP = {"pin": 0.4, "cone": 0.2}


@dataclass(frozen=True)
class TwistedPlateConfig:
    """Geometric parameters for a 3D-printable twisted plate link (units: mm, rad).

    Two hinge types reproduce NovemberKou's printed and tested designs:

    ``hinge_type="pin"`` ("Material-saving Design")
        A separately printed regular octagonal pin, 4 mm corner to corner,
        held in 4.18 mm bores of the bottom knuckles and rotating in 4.6 mm
        bores of the top knuckles of the neighbouring unit; 7 mm knuckles
        with 0.4 mm axial gaps.  The clearances are wider than the reference
        design's (4.0 mm and 4.4 mm bores, 0.2 mm gaps), with which a test
        print's pin did not go in and the knuckles turned stiffly.
    ``hinge_type="cone"`` ("Twisted Plate with Ring")
        No separate parts: the two outer bottom knuckles carry 60-degree cones
        (4 mm base) pointing inward, which snap into matching conical sockets
        in the outer ends of the neighbour's top knuckles; 6 mm knuckles.  The
        socketed knuckles keep a wide ``snap_gap`` on their inner side so they
        can flex past the cone tips during assembly.

    Attributes
    ----------
    torsion_angle:
        Twist angle between the bottom and top hinge axes in radians.
    ribbon_width:
        Length along each hinge axis (mm); also the hinge pin length.
    edge_length:
        Center-to-center distance between the bottom and top hinge axes (mm).
    body_thickness:
        Thickness of the twisted plate body (mm; default 2.4, six lines of a
        0.4 mm nozzle).  Thicker plates print more reliably but bring links
        that do not share a hinge closer together; the contact check reports
        the remaining margin.
    hinge_type:
        ``"pin"`` (separate octagonal pins) or ``"cone"`` (snap-in cone pivots).
    hinge_diameter:
        Outer diameter of the hinge knuckles (mm); ``None`` selects 7 mm for
        pins and 6 mm for cones.
    hinge_gap:
        Axial clearance between mating knuckles along a hinge axis (mm);
        ``None`` selects 0.4 mm for pins and 0.2 mm for cones.  For cone
        hinges this is also the gap between a cone's base face and the
        socketed knuckle, i.e. the axial play of the cone in its socket.
    gap:
        Radial clearance between a hinge barrel and the mating plate edge
        (mm; default 0.4).  Between its own bottom knuckles the plate's lower
        edge is printed as a short bridge over the mating knuckles, and the
        margin leaves room for it to sag.
    num_knuckles:
        Total odd number of hinge knuckles shared across the joint (>= 3).
    pin_diameter:
        Corner-to-corner (circumscribed) diameter of the octagonal hinge pin.
    pin_fit_clearance:
        Radial clearance between the pin corners and the bores of the bottom
        knuckles, which hold the pin (default 0.09 mm).  FDM printers make
        horizontal holes slightly undersized, so this is a snug push fit on
        most printers; in test prints 0 and 0.07 mm kept the pin out and
        0.1 mm let it slide out.  Raise it if the pin does not go in; lower it (``0`` is
        a press fit) if the pin slides out.
    pin_clearance:
        Radial clearance between the pin corners and the bores of the top
        knuckles, which rotate on the pin (default 0.3 mm).  Raise it if the
        hinge turns stiffly.
    pin_core_diameter:
        Corner-to-corner diameter of the hollow octagonal core of the pin;
        ``0`` for a solid pin.  The core saves material and lets the pin
        corners yield slightly when it is pressed in.
    pin_chamfer:
        45-degree chamfer on both pin ends (mm), easing insertion.
    cone_diameter:
        Base diameter of the snap-in cones and their sockets (cone hinges).
    cone_angle:
        Half-angle (degrees) between the cone axis and its surface.  60 degrees
        keeps the horizontal cones and sockets printable without support.
    snap_gap:
        Axial gap on the inner side of each socketed top knuckle (cone hinges),
        leaving room for the knuckle to flex past the cone tip.
    with_ring:
        Whether to generate the captive print-in-place ring (default False).
    cut_slits:
        Whether to cut window slits in the plate; defaults to ``with_ring``.
    ring_diameter:
        Outer diameter of the captive ring (mm); defaults to ``0.4 * ribbon_width``.
    ring_width:
        Radial width and axial height of the captive ring (mm).
    ring_gap:
        Clearance around the captive ring inside the window slits (mm).
    slit_rail_height:
        Height of the solid rails above/below the window slits (mm).
    loft_slices:
        Number of cross-section slices used to loft the twisted plate (>= 5).
    """

    torsion_angle: float = float(np.deg2rad(72.8))
    ribbon_width: float = 43.0
    edge_length: float = 29.0
    body_thickness: float = 2.4
    hinge_type: str = "pin"
    hinge_diameter: float | None = None
    hinge_gap: float | None = None
    gap: float = 0.4
    num_knuckles: int = 5
    pin_diameter: float = 4.0
    pin_fit_clearance: float = 0.09
    pin_clearance: float = 0.3
    pin_core_diameter: float = 1.8
    pin_chamfer: float = 0.5
    cone_diameter: float = 4.0
    cone_angle: float = 60.0
    snap_gap: float = 2.25
    with_ring: bool = False
    cut_slits: bool | None = None
    ring_diameter: float | None = None
    ring_width: float = 2.0
    ring_gap: float = 0.5
    slit_rail_height: float = 2.0
    loft_slices: int = 25

    @property
    def resolved_cut_slits(self) -> bool:
        """Return whether window slits are cut in the plate."""
        return bool(self.with_ring if self.cut_slits is None else self.cut_slits)

    @property
    def resolved_ring_diameter(self) -> float:
        """Return the effective captive ring outer diameter in mm."""
        if self.ring_diameter is None:
            return 0.4 * float(self.ribbon_width)
        return float(self.ring_diameter)

    @property
    def torsion_degrees(self) -> float:
        """Return ``torsion_angle`` in degrees."""
        return float(np.rad2deg(self.torsion_angle))

    @property
    def resolved_hinge_diameter(self) -> float:
        """Knuckle diameter, defaulting per ``hinge_type``."""
        if self.hinge_diameter is not None:
            return float(self.hinge_diameter)
        return _DEFAULT_HINGE_DIAMETER.get(self.hinge_type, 7.0)

    @property
    def resolved_hinge_gap(self) -> float:
        """Axial knuckle gap, defaulting per ``hinge_type``."""
        if self.hinge_gap is not None:
            return float(self.hinge_gap)
        return _DEFAULT_HINGE_GAP.get(self.hinge_type, 0.4)

    @property
    def hinge_radius(self) -> float:
        """Outer radius of each hinge knuckle."""
        return 0.5 * self.resolved_hinge_diameter

    @property
    def uses_pins(self) -> bool:
        """Whether units are joined by separate hinge pins."""
        return self.hinge_type == "pin"

    @property
    def pin_radius(self) -> float:
        """Circumradius of the octagonal hinge pin (centre to corner)."""
        return 0.5 * float(self.pin_diameter)

    @property
    def pin_inradius(self) -> float:
        """Inradius of the octagonal hinge pin (centre to flat face)."""
        return self.pin_radius * math.cos(math.pi / 8.0)

    @property
    def pin_core_inradius(self) -> float:
        """Inradius of the hollow pin core (``0`` for a solid pin)."""
        return 0.5 * float(self.pin_core_diameter) * math.cos(math.pi / 8.0)

    @property
    def fixed_bore_radius(self) -> float:
        """Bore radius of the bottom knuckles, which hold the pin."""
        return self.pin_radius + float(self.pin_fit_clearance)

    @property
    def free_bore_radius(self) -> float:
        """Bore radius of the top knuckles, which rotate on the pin."""
        return self.pin_radius + float(self.pin_clearance)

    @property
    def fixed_bore_flat_height(self) -> float:
        """Height of the flat teardrop top of the fixed bores above the axis."""
        return self.fixed_bore_radius + _BORE_FLAT_MARGIN

    @property
    def free_bore_flat_height(self) -> float:
        """Height of the flat teardrop top of the free bores above the axis."""
        return self.free_bore_radius + _BORE_FLAT_MARGIN

    @property
    def cone_radius(self) -> float:
        """Base radius of the snap-in cones and sockets."""
        return 0.5 * float(self.cone_diameter)

    @property
    def cone_height(self) -> float:
        """Axial height of the snap-in cones (depth of the sockets)."""
        return self.cone_radius / math.tan(math.radians(float(self.cone_angle)))

    @property
    def snap_deflection(self) -> float:
        """Axial deflection a socketed knuckle needs to pass a cone tip (mm)."""
        return max(self.cone_height - self.resolved_hinge_gap, 0.0)

    @property
    def z_bottom_hinge(self) -> float:
        """Height of the bottom hinge axis above the print bed (z=0).

        The round bottom knuckles are cut flat by ``_KNUCKLE_FOOT_DEPTH`` so
        that they have a bed contact strip.
        """
        return self.hinge_radius - _KNUCKLE_FOOT_DEPTH

    @property
    def z_top_hinge(self) -> float:
        """Height of the top hinge axis above the print bed (z=0)."""
        return self.z_bottom_hinge + float(self.edge_length)

    @property
    def z_plate_bottom(self) -> float:
        """Bottom z-coordinate of the twisted plate body."""
        return self.z_bottom_hinge + self.hinge_radius + float(self.gap)

    @property
    def z_plate_top(self) -> float:
        """Top z-coordinate of the twisted plate body."""
        return self.z_top_hinge - self.hinge_radius - float(self.gap)

    def knuckle_intervals(
        self,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Axial ``(start, end)`` extents of the bottom and top knuckles.

        Knuckle ``k`` occupies segment ``k`` of ``num_knuckles`` equal
        segments; bottom knuckles have even ``k``, top knuckles odd ``k``.
        Pin hinges split the axial gap (:attr:`resolved_hinge_gap`) between
        both neighbours of every interior boundary.  Cone hinges keep the
        bottom knuckles full length; each top knuckle gives way by the gap
        towards a cone and by
        ``snap_gap`` on the plain side of a socketed knuckle.
        """
        w = float(self.ribbon_width)
        n = int(self.num_knuckles)
        seg = w / n
        g = self.resolved_hinge_gap
        bottom: list[tuple[float, float]] = []
        top: list[tuple[float, float]] = []
        for k in range(n):
            x0 = -0.5 * w + k * seg
            x1 = x0 + seg
            if self.hinge_type == "cone":
                if k % 2 == 0:
                    bottom.append((x0, x1))
                    continue
                left_socket = k == 1
                right_socket = k == n - 2
                socketed = left_socket or right_socket
                lo = g if left_socket or not socketed else float(self.snap_gap)
                hi = g if right_socket or not socketed else float(self.snap_gap)
                top.append((x0 + lo, x1 - hi))
            else:
                lo = 0.5 * g if k > 0 else 0.0
                hi = 0.5 * g if k < n - 1 else 0.0
                (bottom if k % 2 == 0 else top).append((x0 + lo, x1 - hi))
        return bottom, top

    def validate(self) -> None:
        """Validate geometric and printability constraints."""
        float_fields = (
            ("torsion_angle", self.torsion_angle),
            ("ribbon_width", self.ribbon_width),
            ("edge_length", self.edge_length),
            ("body_thickness", self.body_thickness),
            ("hinge_diameter", self.resolved_hinge_diameter),
            ("hinge_gap", self.resolved_hinge_gap),
            ("gap", self.gap),
            ("pin_diameter", self.pin_diameter),
            ("pin_fit_clearance", self.pin_fit_clearance),
            ("pin_clearance", self.pin_clearance),
            ("pin_core_diameter", self.pin_core_diameter),
            ("pin_chamfer", self.pin_chamfer),
            ("cone_diameter", self.cone_diameter),
            ("cone_angle", self.cone_angle),
            ("snap_gap", self.snap_gap),
            ("ring_width", self.ring_width),
            ("ring_gap", self.ring_gap),
            ("slit_rail_height", self.slit_rail_height),
        )
        for name, val in float_fields:
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        if self.hinge_type not in HINGE_TYPES:
            raise ValueError(f"hinge_type must be one of {HINGE_TYPES}")
        if self.ribbon_width <= 0:
            raise ValueError("ribbon_width must be positive")
        if self.body_thickness <= 0:
            raise ValueError("body_thickness must be positive")
        if self.resolved_hinge_diameter <= 0:
            raise ValueError("hinge_diameter must be positive")
        if self.resolved_hinge_gap < 0:
            raise ValueError("hinge_gap must be non-negative")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")
        if self.edge_length <= self.resolved_hinge_diameter + 2.0 * self.gap:
            raise ValueError(
                "edge_length must exceed hinge_diameter + 2 * gap "
                "for positive plate height"
            )
        if self.num_knuckles < 3 or self.num_knuckles % 2 == 0:
            raise ValueError("num_knuckles must be an odd integer >= 3")
        if self.ribbon_width / self.num_knuckles <= self.resolved_hinge_gap:
            raise ValueError("ribbon_width / num_knuckles must exceed hinge_gap")
        if self.loft_slices < 5:
            raise ValueError("loft_slices must be at least 5")
        if self.uses_pins:
            self._validate_pin_hinge()
        else:
            self._validate_cone_hinge()

        if self.with_ring or self.resolved_cut_slits:
            if self.ring_width <= 0:
                raise ValueError("ring_width must be positive")
            if self.ring_gap <= 0:
                raise ValueError("ring_gap must be positive")
            if self.slit_rail_height < 0:
                raise ValueError("slit_rail_height must be non-negative")

            r_out = 0.5 * self.resolved_ring_diameter
            if not math.isfinite(r_out) or r_out <= 0:
                raise ValueError("ring_diameter must be finite and positive")
            r_in = r_out - self.ring_width
            if r_in <= 0:
                raise ValueError("ring_diameter must exceed 2 * ring_width")
            u_in = r_in - self.ring_gap
            u_out = r_out + self.ring_gap
            if u_in <= 0:
                raise ValueError(
                    "ring inner radius minus ring_gap must be positive"
                )
            if u_out >= 0.5 * self.ribbon_width:
                raise ValueError(
                    "ring outer radius plus ring_gap must be less than "
                    "half of ribbon_width"
                )
            if (
                self.with_ring
                and math.hypot(u_in, 0.5 * self.body_thickness) >= r_in
            ):
                raise ValueError(
                    "central pillar corners intersect ring bore; "
                    "increase ring_gap or decrease body_thickness"
                )

            z_slit_bot = self.z_plate_bottom + self.slit_rail_height
            z_slit_top = self.z_plate_top - self.slit_rail_height
            if z_slit_top <= z_slit_bot + self.ring_width + 2.0 * self.ring_gap:
                raise ValueError(
                    "edge_length is too small for slit height and captive ring"
                )

    def _validate_pin_hinge(self) -> None:
        if self.pin_diameter <= 0:
            raise ValueError("pin_diameter must be positive")
        if self.pin_clearance < 0:
            raise ValueError("pin_clearance must be non-negative")
        if self.fixed_bore_radius <= 0.5 * self.pin_radius:
            raise ValueError("pin_fit_clearance is too negative for pin_diameter")
        bore_top = max(self.fixed_bore_flat_height, self.free_bore_flat_height)
        if self.hinge_radius - bore_top < _MIN_WALL:
            raise ValueError(
                f"hinge_diameter must leave a {_MIN_WALL} mm wall around the pin "
                "bores; increase hinge_diameter or decrease pin_diameter"
            )
        if self.z_bottom_hinge - self.fixed_bore_radius < _MIN_WALL:
            raise ValueError("hinge_diameter is too small for the bottom knuckle foot")
        if self.pin_core_diameter < 0:
            raise ValueError("pin_core_diameter must be non-negative")
        if self.pin_chamfer < 0:
            raise ValueError("pin_chamfer must be non-negative")
        if self.pin_core_diameter > 0 and (
            self.pin_inradius - self.pin_core_inradius < _MIN_WALL
        ):
            raise ValueError(f"pin_core_diameter must leave a {_MIN_WALL} mm pin wall")
        if self.pin_inradius - self.pin_chamfer <= self.pin_core_inradius:
            raise ValueError("pin_chamfer is too large for the pin cross-section")
        if 2.0 * self.pin_chamfer >= self.ribbon_width:
            raise ValueError("pin_chamfer must be less than half of ribbon_width")

    def _validate_cone_hinge(self) -> None:
        if self.cone_diameter <= 0:
            raise ValueError("cone_diameter must be positive")
        if not 0.0 < self.cone_angle < 90.0:
            raise ValueError("cone_angle must be strictly between 0 and 90 degrees")
        if self.hinge_radius - self.cone_radius < _MIN_WALL:
            raise ValueError(
                f"hinge_diameter must leave a {_MIN_WALL} mm wall around the cone "
                "sockets; increase hinge_diameter or decrease cone_diameter"
            )
        if self.z_bottom_hinge < self.cone_radius:
            raise ValueError("cone_diameter is too large: the cones reach the bed")
        if self.snap_gap < self.snap_deflection:
            raise ValueError(
                "snap_gap must leave room for the socketed knuckles to flex "
                f"past the cone tips ({self.snap_deflection:.2f} mm)"
            )
        _, top = self.knuckle_intervals()
        n_sockets = 2 if self.num_knuckles == 3 else 1
        for x0, x1 in top:
            if x1 - x0 < n_sockets * self.cone_height + _MIN_WALL:
                raise ValueError(
                    "top knuckles are too short for the cone sockets; increase "
                    "ribbon_width or decrease num_knuckles, snap_gap, or cone size"
                )


def _resolve_config(
    config: TwistedPlateConfig | None = None,
    *,
    degrees: bool = False,
    **overrides: Any,
) -> TwistedPlateConfig:
    """Build and validate a ``TwistedPlateConfig`` with optional overrides."""
    base = config if config is not None else TwistedPlateConfig()
    cleaned = {k: v for k, v in overrides.items() if v is not None}
    if "torsion_angle" in cleaned and degrees:
        cleaned["torsion_angle"] = float(np.deg2rad(cleaned["torsion_angle"]))
    resolved = replace(base, **cleaned)
    resolved.validate()
    return resolved


def _loft_twisted_box(
    bd: Any,
    *,
    z_start: float,
    z_end: float,
    z_plate_bottom: float,
    z_plate_top: float,
    alpha_deg: float,
    width: float,
    thickness: float,
    n_slices: int,
    x_offset: float = 0.0,
) -> Any:
    """Loft a rectangular cross-section twisting linearly from bottom to top."""
    z_span = z_plate_top - z_plate_bottom
    sections = []
    for i in range(n_slices):
        frac = i / (n_slices - 1)
        z = z_start + frac * (z_end - z_start)
        ang = ((z - z_plate_bottom) / z_span) * alpha_deg
        plane = bd.Plane.XY.offset(z).rotated((0.0, 0.0, ang))
        rect = bd.Rectangle(width, thickness)
        if x_offset != 0.0:
            rect = bd.Pos(x_offset, 0.0) * rect
        sections.append(plane * rect)
    return bd.loft(sections)


def _build_bottom_knuckle_profile(bd: Any, cfg: TwistedPlateConfig) -> Any:
    """Create the YZ cross-section for bottom hinge knuckles.

    The knuckle is round about the hinge axis (cut flat on the bed for
    adhesion), so the neighbouring plate, which starts ``gap`` outside the
    knuckle radius, clears it at every fold angle.  A tangent wedge joins the
    knuckle to the plate above it.
    """
    r_h = cfg.hinge_radius
    y_neck = min(0.5 * cfg.body_thickness, r_h)
    z0 = cfg.z_bottom_hinge
    z_bot = cfg.z_plate_bottom
    zp = z_bot - z0
    dist_p = math.hypot(y_neck, zp)
    if dist_p > r_h and y_neck < r_h:
        theta_p = math.atan2(zp, y_neck)
        delta = math.acos(r_h / dist_p)
        theta_t = max(theta_p - delta, 0.0)
        y_tan = r_h * math.cos(theta_t)
        z_tan = z0 + r_h * math.sin(theta_t)
    else:
        y_tan = max(r_h, y_neck)
        z_tan = z0

    overlap = 0.02
    pts = [
        (-y_neck, z_bot + overlap),
        (-y_tan, z_tan),
        (0.0, z0),
        (y_tan, z_tan),
        (y_neck, z_bot + overlap),
    ]
    wedge = bd.Plane.YZ * bd.Polygon(*pts, align=None)
    disc = bd.Plane.YZ * bd.Pos(0.0, z0) * bd.Circle(r_h)
    above_bed = (
        bd.Plane.YZ
        * bd.Pos(0.0, 0.5 * (z0 + r_h))
        * bd.Rectangle(2.0 * r_h + 2.0, z0 + r_h)
    )
    return (disc & above_bed) + wedge


def _build_top_knuckle_profile(bd: Any, cfg: TwistedPlateConfig) -> Any:
    """Create the teardrop YZ cross-section for top hinge knuckles."""
    r_h = cfg.hinge_radius
    y_neck = min(0.5 * cfg.body_thickness, r_h)
    z1 = cfg.z_top_hinge
    z_top = cfg.z_plate_top
    zp = z_top - z1
    dist_p = math.hypot(y_neck, zp)
    if dist_p > r_h and y_neck < r_h:
        theta_p = math.atan2(zp, y_neck)
        delta = math.acos(r_h / dist_p)
        theta_t = min(theta_p + delta, 0.0)
        y_tan = r_h * math.cos(theta_t)
        z_tan = z1 + r_h * math.sin(theta_t)
    else:
        y_tan = max(r_h, y_neck)
        z_tan = z1

    overlap = 0.02
    pts = [
        (-y_neck, z_top - overlap),
        (y_neck, z_top - overlap),
        (y_tan, z_tan),
        (0.0, z1),
        (-y_tan, z_tan),
    ]
    wedge = bd.Plane.YZ * bd.Polygon(*pts, align=None)
    cyl = bd.Plane.YZ * bd.Pos(0.0, z1) * bd.Circle(r_h)
    return wedge + cyl


def _build_knuckles(
    bd: Any,
    profile: Any,
    intervals: list[tuple[float, float]],
    *,
    rotation_deg: float = 0.0,
) -> Any | None:
    """Extrude hinge knuckles over axial ``intervals`` and combine them."""
    rot = bd.Rotation(0.0, 0.0, rotation_deg) if rotation_deg != 0.0 else None
    combined = None
    for x_start, x_end in intervals:
        knuckle = bd.extrude(
            profile.moved(bd.Location((x_start, 0.0, 0.0))),
            x_end - x_start,
        )
        if rot is not None:
            knuckle = rot * knuckle
        combined = knuckle if combined is None else combined + knuckle
    return combined


def _build_cone(
    bd: Any,
    *,
    face_x: float,
    direction: float,
    radius: float,
    height: float,
    z_axis: float,
    back: float = 0.05,
    rotation_deg: float = 0.0,
) -> Any:
    """Revolve a cone about a horizontal X-parallel axis at height ``z_axis``.

    The base (``radius``) lies on the plane ``x = face_x`` and the tip points
    along ``direction`` (+1 or -1); a short cylindrical stub of length ``back``
    extends behind the base so the cone fuses with (or cuts cleanly through)
    the face it sits on.
    """
    d = 1.0 if direction > 0 else -1.0
    pts = [
        (face_x - d * back, z_axis),
        (face_x - d * back, z_axis + radius),
        (face_x, z_axis + radius),
        (face_x + d * height, z_axis),
    ]
    profile = bd.Plane.XZ * bd.Polygon(*pts, align=None)
    cone = bd.revolve(profile, bd.Axis((0.0, 0.0, z_axis), (1.0, 0.0, 0.0)))
    if rotation_deg != 0.0:
        cone = bd.Rotation(0.0, 0.0, rotation_deg) * cone
    return cone


def _build_bore(
    bd: Any,
    *,
    radius: float,
    flat_height: float,
    z_axis: float,
    length: float,
    rotation_deg: float = 0.0,
) -> Any:
    """Build a horizontal teardrop bore along X (optionally rotated about Z).

    The bore is a circle of ``radius`` extended upward by 45-degree tangents
    and cut flat ``flat_height`` above the axis, so it prints without support
    and the bridged top stays clear of the pin.
    """
    c = radius * math.sqrt(0.5)
    wedge = bd.Polygon(
        (0.0, 0.0), (c, c), (0.0, radius * math.sqrt(2.0)), (-c, c), align=None
    )
    span = 2.0 * radius + flat_height
    below_flat = bd.Pos(0.0, flat_height - 0.5 * span) * bd.Rectangle(
        4.0 * radius, span
    )
    profile = (bd.Circle(radius) + wedge) & below_flat
    # ``Plane.YZ`` maps local (x, y) to global (Y, Z); shift the start along X.
    face = (bd.Plane.YZ * bd.Pos(0.0, z_axis) * profile).moved(
        bd.Location((-0.5 * length, 0.0, 0.0))
    )
    bore = bd.extrude(face, length)
    if rotation_deg != 0.0:
        bore = bd.Rotation(0.0, 0.0, rotation_deg) * bore
    return bore


def _octagon_points(circumradius: float) -> list[tuple[float, float]]:
    """Regular octagon with a flat edge at the bottom (local -y)."""
    return [
        (
            circumradius * math.cos(math.radians(22.5 + 45.0 * k)),
            circumradius * math.sin(math.radians(22.5 + 45.0 * k)),
        )
        for k in range(8)
    ]


def build_hinge_pin(
    config: TwistedPlateConfig | None = None,
    **overrides: Any,
) -> Any:
    """Build the octagonal hinge pin that joins two adjacent units.

    The pin runs along the X axis, is centred on the origin, spans the full
    ``ribbon_width`` (so it passes through every knuckle of the joint), and
    rests on one of its flat faces so that it prints lying down without
    support.  Both ends carry a ``pin_chamfer`` lead-in; an optional hollow
    octagonal core of ``pin_core_diameter`` runs through it.
    """
    cfg = _resolve_config(config, **overrides)
    bd = _require_build123d()

    half = 0.5 * float(cfg.ribbon_width)
    r_pin = cfg.pin_radius
    chamfer = float(cfg.pin_chamfer)

    def section(x: float, circumradius: float) -> Any:
        return bd.Plane.YZ.offset(x) * bd.Polygon(
            *_octagon_points(circumradius), align=None
        )

    if chamfer > 0.0:
        r_end = r_pin - chamfer / math.cos(math.pi / 8.0)
        pin = bd.loft(
            [
                section(-half, r_end),
                section(-half + chamfer, r_pin),
                section(half - chamfer, r_pin),
                section(half, r_end),
            ],
            ruled=True,
        )
    else:
        pin = bd.extrude(section(-half, r_pin), 2.0 * half)

    if cfg.pin_core_diameter > 0.0:
        core = bd.extrude(
            section(-half - 1.0, 0.5 * float(cfg.pin_core_diameter)),
            2.0 * half + 2.0,
        )
        pin = pin - core

    pin_solid = pin.solid()
    pin_solid.label = "pin"
    return pin_solid


def _build_captive_ring(bd: Any, cfg: TwistedPlateConfig) -> Any:
    """Build a 45-degree chamfered captive ring encircling the central pillar."""
    r_out = 0.5 * cfg.resolved_ring_diameter
    r_in = r_out - cfg.ring_width
    z_rb = cfg.z_plate_bottom + cfg.slit_rail_height + cfg.ring_gap
    z_rt = z_rb + cfg.ring_width
    chamfer = 0.25 * cfg.ring_width
    pts = [
        (r_in + chamfer, z_rb),
        (r_out - chamfer, z_rb),
        (r_out, z_rb + chamfer),
        (r_out, z_rt - chamfer),
        (r_out - chamfer, z_rt),
        (r_in + chamfer, z_rt),
        (r_in, z_rt - chamfer),
        (r_in, z_rb + chamfer),
    ]
    ring_face = bd.Plane.XZ * bd.Polygon(*pts, align=None)
    ring = bd.revolve(ring_face, bd.Axis.Z)
    ring_solid = ring.solid()
    ring_solid.label = "ring"
    return ring_solid


def build_twisted_plate_parts(
    config: TwistedPlateConfig | None = None,
    *,
    torsion_angle: float | None = None,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    body_thickness: float | None = None,
    hinge_diameter: float | None = None,
    pin_diameter: float | None = None,
    pin_fit_clearance: float | None = None,
    pin_clearance: float | None = None,
    pin_core_diameter: float | None = None,
    pin_chamfer: float | None = None,
    hinge_type: str | None = None,
    cone_diameter: float | None = None,
    cone_angle: float | None = None,
    snap_gap: float | None = None,
    hinge_gap: float | None = None,
    gap: float | None = None,
    num_knuckles: int | None = None,
    with_ring: bool | None = None,
    cut_slits: bool | None = None,
    ring_diameter: float | None = None,
    ring_width: float | None = None,
    ring_gap: float | None = None,
    slit_rail_height: float | None = None,
    loft_slices: int | None = None,
    degrees: bool = False,
) -> dict[str, Any]:
    """Construct the twisted plate solid and optional captive ring separately.

    Returns a dictionary mapping ``"plate"`` (and ``"ring"`` when
    ``with_ring=True``) to ``build123d`` ``Solid`` objects.
    """
    cfg = _resolve_config(
        config,
        degrees=degrees,
        torsion_angle=torsion_angle,
        ribbon_width=ribbon_width,
        edge_length=edge_length,
        body_thickness=body_thickness,
        hinge_diameter=hinge_diameter,
        pin_diameter=pin_diameter,
        pin_fit_clearance=pin_fit_clearance,
        pin_clearance=pin_clearance,
        pin_core_diameter=pin_core_diameter,
        pin_chamfer=pin_chamfer,
        hinge_type=hinge_type,
        cone_diameter=cone_diameter,
        cone_angle=cone_angle,
        snap_gap=snap_gap,
        hinge_gap=hinge_gap,
        gap=gap,
        num_knuckles=num_knuckles,
        with_ring=with_ring,
        cut_slits=cut_slits,
        ring_diameter=ring_diameter,
        ring_width=ring_width,
        ring_gap=ring_gap,
        slit_rail_height=slit_rail_height,
        loft_slices=loft_slices,
    )
    bd = _require_build123d()

    z_bot = cfg.z_plate_bottom
    z_top = cfg.z_plate_top
    alpha_deg = cfg.torsion_degrees

    plate = _loft_twisted_box(
        bd,
        z_start=z_bot,
        z_end=z_top,
        z_plate_bottom=z_bot,
        z_plate_top=z_top,
        alpha_deg=alpha_deg,
        width=cfg.ribbon_width,
        thickness=cfg.body_thickness,
        n_slices=cfg.loft_slices,
    )

    if cfg.resolved_cut_slits:
        z_slit_bot = z_bot + cfg.slit_rail_height
        z_slit_top = z_top - cfg.slit_rail_height
        r_ring_out = 0.5 * cfg.resolved_ring_diameter
        r_ring_in = r_ring_out - cfg.ring_width
        u_in = r_ring_in - cfg.ring_gap
        u_out = r_ring_out + cfg.ring_gap
        slit_w = u_out - u_in
        slit_center = 0.5 * (u_in + u_out)
        slit_t = max(3.0 * cfg.body_thickness, 2.0 * cfg.body_thickness + 2.0)
        for sign in (-1.0, 1.0):
            slit = _loft_twisted_box(
                bd,
                z_start=z_slit_bot,
                z_end=z_slit_top,
                z_plate_bottom=z_bot,
                z_plate_top=z_top,
                alpha_deg=alpha_deg,
                width=slit_w,
                thickness=slit_t,
                n_slices=cfg.loft_slices,
                x_offset=sign * slit_center,
            )
            plate = plate - slit

    bottom_iv, top_iv = cfg.knuckle_intervals()
    bot_knuckles = _build_knuckles(
        bd, _build_bottom_knuckle_profile(bd, cfg), bottom_iv
    )
    if bot_knuckles is not None:
        plate = plate + bot_knuckles
    top_knuckles = _build_knuckles(
        bd, _build_top_knuckle_profile(bd, cfg), top_iv, rotation_deg=alpha_deg
    )
    if top_knuckles is not None:
        plate = plate + top_knuckles

    if cfg.uses_pins:
        # Bottom knuckles hold the pin (fixed bore); the top knuckles of this
        # unit rotate on the pin held by the next unit (free bore).
        bore_len = 2.0 * cfg.ribbon_width
        plate = plate - _build_bore(
            bd,
            radius=cfg.fixed_bore_radius,
            flat_height=cfg.fixed_bore_flat_height,
            z_axis=cfg.z_bottom_hinge,
            length=bore_len,
        )
        plate = plate - _build_bore(
            bd,
            radius=cfg.free_bore_radius,
            flat_height=cfg.free_bore_flat_height,
            z_axis=cfg.z_top_hinge,
            length=bore_len,
            rotation_deg=alpha_deg,
        )
    else:
        # The outer bottom knuckles carry inward-pointing cones; the matching
        # sockets sit in the outer ends of the outer top knuckles, so the
        # neighbour's socketed knuckles snap over this unit's cones.
        cone_kw = {"radius": cfg.cone_radius, "height": cfg.cone_height}
        plate = plate + _build_cone(
            bd,
            face_x=bottom_iv[0][1],
            direction=1.0,
            z_axis=cfg.z_bottom_hinge,
            **cone_kw,
        )
        plate = plate + _build_cone(
            bd,
            face_x=bottom_iv[-1][0],
            direction=-1.0,
            z_axis=cfg.z_bottom_hinge,
            **cone_kw,
        )
        socket_kw = {
            **cone_kw,
            "z_axis": cfg.z_top_hinge,
            "back": 1.0,
            "rotation_deg": alpha_deg,
        }
        plate = plate - _build_cone(bd, face_x=top_iv[0][0], direction=1.0, **socket_kw)
        plate = plate - _build_cone(
            bd, face_x=top_iv[-1][1], direction=-1.0, **socket_kw
        )

    plate_solid = plate.solid()
    plate_solid.label = "plate"
    parts: dict[str, Any] = {"plate": plate_solid}
    if cfg.with_ring:
        parts["ring"] = _build_captive_ring(bd, cfg)
    return parts


def build_twisted_plate(
    config: TwistedPlateConfig | None = None,
    *,
    torsion_angle: float | None = None,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    body_thickness: float | None = None,
    hinge_diameter: float | None = None,
    pin_diameter: float | None = None,
    pin_fit_clearance: float | None = None,
    pin_clearance: float | None = None,
    pin_core_diameter: float | None = None,
    pin_chamfer: float | None = None,
    hinge_type: str | None = None,
    cone_diameter: float | None = None,
    cone_angle: float | None = None,
    snap_gap: float | None = None,
    hinge_gap: float | None = None,
    gap: float | None = None,
    num_knuckles: int | None = None,
    with_ring: bool | None = None,
    cut_slits: bool | None = None,
    ring_diameter: float | None = None,
    ring_width: float | None = None,
    ring_gap: float | None = None,
    slit_rail_height: float | None = None,
    loft_slices: int | None = None,
    degrees: bool = False,
) -> Any:
    """Build a single twisted plate link as a ``Solid`` or ``Compound``.

    When ``with_ring=False`` (default), returns the twisted plate ``Solid``.
    When ``with_ring=True``, returns a ``Compound`` containing both the plate
    and the captive ring as separate solids.
    """
    parts = build_twisted_plate_parts(
        config,
        torsion_angle=torsion_angle,
        ribbon_width=ribbon_width,
        edge_length=edge_length,
        body_thickness=body_thickness,
        hinge_diameter=hinge_diameter,
        pin_diameter=pin_diameter,
        pin_fit_clearance=pin_fit_clearance,
        pin_clearance=pin_clearance,
        pin_core_diameter=pin_core_diameter,
        pin_chamfer=pin_chamfer,
        hinge_type=hinge_type,
        cone_diameter=cone_diameter,
        cone_angle=cone_angle,
        snap_gap=snap_gap,
        hinge_gap=hinge_gap,
        gap=gap,
        num_knuckles=num_knuckles,
        with_ring=with_ring,
        cut_slits=cut_slits,
        ring_diameter=ring_diameter,
        ring_width=ring_width,
        ring_gap=ring_gap,
        slit_rail_height=slit_rail_height,
        loft_slices=loft_slices,
        degrees=degrees,
    )
    if "ring" not in parts:
        return parts["plate"]
    bd = _require_build123d()
    return bd.Compound(children=[parts["plate"], parts["ring"]], label="twisted_plate")


def _extract_hinges(
    kc_or_hinges: Kaleidocycle | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract and validate a normalized ``(n + 1, 3)`` hinge array."""
    raw = kc_or_hinges.hinges if hasattr(kc_or_hinges, "hinges") else kc_or_hinges
    hinges = np.asarray(raw, dtype=float)
    if hinges.ndim != 2 or hinges.shape[1] != 3 or hinges.shape[0] < 4:
        raise ValueError(
            f"expected (n+1, 3) hinges array with n >= 3, got {hinges.shape}"
        )
    norms = np.linalg.norm(hinges, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("hinge vectors must be non-zero")
    return hinges / norms


def twisted_plate_from_kaleidocycle(
    kc_or_hinges: Kaleidocycle | NDArray[np.float64],
    *,
    segment: int = 0,
    ribbon_width: float = 43.0,
    edge_length: float | None = None,
    scale: float | None = None,
    with_ring: bool = False,
    **kwargs: Any,
) -> TwistedPlateConfig:
    """Create a ``TwistedPlateConfig`` from a kaleidocycle's hinge geometry.

    Parameters
    ----------
    kc_or_hinges:
        ``Kaleidocycle`` instance or ``(n + 1, 3)`` hinge vectors array.
    segment:
        Index of the link segment (``0 <= segment < n``) used to read torsion.
    ribbon_width:
        Ribbon width along the hinge axis in mm.
    edge_length:
        Explicit hinge-to-hinge edge length in mm. If omitted, derived from
        ``scale * ||B_i x B_{i+1}||`` when ``scale`` is given, or ``29.0`` mm.
    scale:
        Optional scale factor multiplying the unnormalized tangent segment
        length ``||B_i x B_{i+1}||`` when ``edge_length`` is ``None``.
    with_ring:
        Whether to enable the captive ring (default ``False``).
    **kwargs:
        Additional keyword arguments passed to :class:`TwistedPlateConfig`.
    """
    hinges = _extract_hinges(kc_or_hinges)
    n = hinges.shape[0] - 1
    if not (0 <= segment < n):
        raise ValueError(f"segment must satisfy 0 <= segment < {n}, got {segment}")

    b0 = hinges[segment]
    b1 = hinges[segment + 1]
    sin_norm = float(np.linalg.norm(np.cross(b0, b1)))
    cos_val = float(np.clip(np.dot(b0, b1), -1.0, 1.0))
    torsion = float(np.arctan2(sin_norm, cos_val))

    if edge_length is not None:
        resolved_edge = float(edge_length)
    elif scale is not None:
        tangents = binormals_to_tangents(hinges, normalize=False)
        resolved_edge = float(scale * np.linalg.norm(tangents[segment]))
    else:
        resolved_edge = 29.0

    cfg = TwistedPlateConfig(
        torsion_angle=torsion,
        ribbon_width=float(ribbon_width),
        edge_length=resolved_edge,
        with_ring=bool(with_ring),
        **kwargs,
    )
    cfg.validate()
    return cfg


def _link_placements(
    kc_or_hinges: Kaleidocycle | NDArray[np.float64],
    *,
    edge_length: float = 29.0,
    scale: float | None = None,
    normalize_segments: bool = True,
) -> list[tuple[float, float, NDArray[np.float64], NDArray[np.float64]]]:
    """Rigid placements of the ``n`` links of a kaleidocycle.

    Returns one ``(torsion, length, origin, rotation)`` per link: link ``i``
    has its bottom hinge axis on ``hinges[i]`` through ``origin`` and its top
    hinge axis on ``hinges[i + 1]``.  A point ``p`` in the link's printing
    frame (bottom hinge along X at height ``z_bottom_hinge``) maps to
    ``origin + rotation @ (p - (0, 0, z_bottom_hinge))``.
    """
    hinges = _extract_hinges(kc_or_hinges)
    n = hinges.shape[0] - 1
    raw_tangents = binormals_to_tangents(hinges, normalize=False)
    raw_norms = np.linalg.norm(raw_tangents, axis=1)
    if np.any(raw_norms <= 1e-10):
        raise ValueError("adjacent hinges are nearly collinear; cannot orient assembly")

    if scale is not None:
        curve = tangents_to_curve(raw_tangents, scale=float(scale), center=True)
    elif normalize_segments:
        unit_tangents = raw_tangents / raw_norms[:, np.newaxis]
        curve = tangents_to_curve(unit_tangents, scale=float(edge_length), center=True)
    else:
        mean_len = float(np.mean(raw_norms))
        curve = tangents_to_curve(
            raw_tangents, scale=float(edge_length) / mean_len, center=True
        )

    placements = []
    for i in range(n):
        seg_vec = curve[i + 1] - curve[i]
        d_i = float(np.linalg.norm(seg_vec))
        t_hat = seg_vec / d_i

        b_i = hinges[i]
        x_dir = b_i - float(np.dot(b_i, t_hat)) * t_hat
        x_norm = float(np.linalg.norm(x_dir))
        if x_norm <= 1e-10:
            raise ValueError(f"hinge {i} is parallel to tangent segment")
        x_dir = x_dir / x_norm
        y_dir = np.cross(t_hat, x_dir)

        b_next = hinges[i + 1]
        torsion_i = float(
            np.arctan2(
                float(np.dot(y_dir, b_next)),
                float(np.clip(np.dot(x_dir, b_next), -1.0, 1.0)),
            )
        )
        rotation = np.column_stack([x_dir, y_dir, t_hat])
        placements.append((torsion_i, d_i, np.asarray(curve[i], float), rotation))
    return placements


def build_kaleidocycle_assembly(
    kc_or_hinges: Kaleidocycle | NDArray[np.float64],
    *,
    ribbon_width: float = 43.0,
    edge_length: float = 29.0,
    scale: float | None = None,
    with_ring: bool = False,
    with_pins: bool = True,
    normalize_segments: bool = True,
    **kwargs: Any,
) -> Any:
    """Assemble all ``n`` twisted plate links along a closed kaleidocycle curve.

    Each link is positioned so its bottom hinge axis aligns with ``hinges[i]``
    at ``curve[i]`` and its top hinge axis aligns with ``hinges[i + 1]`` at
    ``curve[i + 1]``. Returned solids are placed in a single ``Compound``
    without boolean fusion so each moving unit remains distinct.  For pin
    hinges with ``with_pins=True`` (default) the hinge pin ``pin_i`` is
    inserted along hinge ``i``, i.e. through the bottom knuckles of link ``i``
    and the top knuckles of link ``i - 1``.
    """
    placements = _link_placements(
        kc_or_hinges,
        edge_length=edge_length,
        scale=scale,
        normalize_segments=normalize_segments,
    )
    bd = _require_build123d()
    cache: dict[tuple[float, float], tuple[TwistedPlateConfig, dict[str, Any]]] = {}
    pin_cache: dict[tuple[Any, ...], Any] = {}
    children: list[Any] = []

    for i, (torsion_i, d_i, origin, rotation) in enumerate(placements):
        key = (round(torsion_i, 6), round(d_i, 6))
        if key not in cache:
            cfg = TwistedPlateConfig(
                torsion_angle=torsion_i,
                ribbon_width=float(ribbon_width),
                edge_length=d_i,
                with_ring=bool(with_ring),
                **kwargs,
            )
            cfg.validate()
            cache[key] = (cfg, build_twisted_plate_parts(cfg))
        cfg, unit_parts = cache[key]

        frame_loc = _placement_location(bd, origin, rotation)
        transform = frame_loc * bd.Location((0.0, 0.0, -cfg.z_bottom_hinge))
        for part_name, solid in unit_parts.items():
            placed = solid.moved(transform)
            placed.label = f"{part_name}_{i}"
            children.append(placed)

        if with_pins and cfg.uses_pins:
            pin_key = _pin_key(cfg)
            if pin_key not in pin_cache:
                pin_cache[pin_key] = build_hinge_pin(cfg)
            # The pin is built centred on the X axis, which the placement maps
            # to the bottom hinge axis of this link.
            placed_pin = pin_cache[pin_key].moved(frame_loc)
            placed_pin.label = f"pin_{i}"
            children.append(placed_pin)

    return bd.Compound(children=children, label="kaleidocycle_assembly")


def _placement_location(
    bd: Any, origin: NDArray[np.float64], rotation: NDArray[np.float64]
) -> Any:
    """``build123d`` location for a link placement (see ``_link_placements``)."""
    return bd.Location(
        bd.Plane(
            origin=tuple(float(v) for v in origin),
            x_dir=tuple(float(v) for v in rotation[:, 0]),
            z_dir=tuple(float(v) for v in rotation[:, 2]),
        )
    )


def _config_key(cfg: TwistedPlateConfig) -> tuple[Any, ...]:
    """Hashable key identifying configs that produce the same unit geometry."""
    return tuple(round(v, 6) if isinstance(v, float) else v for v in astuple(cfg))


def _pin_key(cfg: TwistedPlateConfig) -> tuple[float, ...]:
    """Hashable key identifying configs that produce the same hinge pin."""
    return tuple(
        round(float(v), 6)
        for v in (
            cfg.ribbon_width,
            cfg.pin_diameter,
            cfg.pin_core_diameter,
            cfg.pin_chamfer,
        )
    )


def _iter_export_shapes(bd: Any, shape: Any) -> list[Any]:
    """Flatten Compound hierarchies while preserving child labels."""
    if isinstance(shape, bd.Compound) and getattr(shape, "children", None):
        items: list[Any] = []
        for child in shape.children:
            items.extend(_iter_export_shapes(bd, child))
        return items
    return [shape]


_3MF_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_3MF_MODEL_REL_TYPE = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
# Generator name written to the 3MF ``Application`` metadata.  It must not start
# with ``"BambuStudio-"``: Bambu Studio would then treat the file as one of its
# own projects and require a complete, version-matched project configuration.
_3MF_APPLICATION = "kaleidocycle"
# Centre of a 256 mm x 256 mm bed (Bambu Lab X1/P1/P2 series, etc.).
_DEFAULT_BED_CENTER: tuple[float, float] = (128.0, 128.0)


def _dedup_mesh(
    raw_verts: Any,
    raw_tris: Any,
    *,
    digits: int = 6,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Offset, round, and merge raw mesh vertices; drop degenerate triangles."""
    ox, oy, oz = (float(v) for v in offset)
    vert_map: dict[tuple[float, float, float], int] = {}
    unique_verts: list[tuple[float, float, float]] = []
    idx_remap: list[int] = []
    for x, y, z in raw_verts:
        key = (
            round(float(x) + ox, digits),
            round(float(y) + oy, digits),
            round(float(z) + oz, digits),
        )
        idx = vert_map.get(key)
        if idx is None:
            idx = len(unique_verts)
            vert_map[key] = idx
            unique_verts.append(key)
        idx_remap.append(idx)

    tris: list[tuple[int, int, int]] = []
    for a, b, c in raw_tris:
        ma, mb, mc = idx_remap[a], idx_remap[b], idx_remap[c]
        if ma != mb and mb != mc and ma != mc:
            tris.append((ma, mb, mc))
    return unique_verts, tris


def _compute_plate_positions(
    count: int,
    unit_width: float,
    unit_depth: float,
    *,
    spacing: float = 6.0,
    center: tuple[float, float] = (128.0, 128.0),
) -> list[tuple[float, float]]:
    """Compute a compact 2D grid of non-overlapping positions for ``count`` units."""
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if spacing < 0:
        raise ValueError(f"spacing must be non-negative, got {spacing}")

    dx = float(unit_width) + float(spacing)
    dy = float(unit_depth) + float(spacing)
    n_cols = max(1, math.ceil(math.sqrt(count * dy / max(dx, 1e-6))))
    n_rows = math.ceil(count / n_cols)

    positions: list[tuple[float, float]] = []
    for idx in range(count):
        col = idx % n_cols
        row = idx // n_cols
        x = center[0] + (col - 0.5 * (n_cols - 1)) * dx
        y = center[1] + (row - 0.5 * (n_rows - 1)) * dy
        positions.append((float(x), float(y)))
    return positions


def _plate_layout(
    unit_count: int,
    unit_width: float,
    unit_depth: float,
    *,
    pin_count: int = 0,
    pin_length: float = 0.0,
    pin_width: float = 0.0,
    spacing: float = 6.0,
    center: tuple[float, float] = (128.0, 128.0),
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Lay out units and hinge pins on one plate without overlaps.

    Units occupy the first cells of the grid of :func:`_compute_plate_positions`;
    the pins, lying along X, are stacked ``pin_width + spacing`` apart in the
    remaining cells.  Returns ``(unit_positions, pin_positions)``.
    """
    if pin_count <= 0:
        units = _compute_plate_positions(
            unit_count, unit_width, unit_depth, spacing=spacing, center=center
        )
        return units, []

    cell_w = max(float(unit_width), float(pin_length))
    cell_d = max(float(unit_depth), float(pin_width))
    pitch = float(pin_width) + float(spacing)
    per_cell = int(math.floor((cell_d - pin_width) / pitch + 1e-9)) + 1
    pin_cells = math.ceil(pin_count / per_cell)
    cells = _compute_plate_positions(
        unit_count + pin_cells, cell_w, cell_d, spacing=spacing, center=center
    )

    pins: list[tuple[float, float]] = []
    remaining = pin_count
    for cx, cy in cells[unit_count:]:
        k = min(per_cell, remaining)
        pins.extend((cx, cy + (j - 0.5 * (k - 1)) * pitch) for j in range(k))
        remaining -= k
    return cells[:unit_count], pins


def _unit_bounds(parts: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Return ``(min_x, max_x, min_y, max_y, min_z)`` across all solids in a unit."""
    solids = list(parts.values())
    bboxes = [solid.bounding_box() for solid in solids]
    return (
        min(float(bb.min.X) for bb in bboxes),
        max(float(bb.max.X) for bb in bboxes),
        min(float(bb.min.Y) for bb in bboxes),
        max(float(bb.max.Y) for bb in bboxes),
        min(float(bb.min.Z) for bb in bboxes),
    )


def _mesh_object_xml(
    object_id: int,
    name: str,
    verts: list[tuple[float, float, float]],
    tris: list[tuple[int, int, int]],
) -> str:
    """Serialize one triangle mesh as a 3MF core ``<object type="model">``."""
    v_lines = "\n".join(
        f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in verts
    )
    t_lines = "\n".join(
        f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tris
    )
    return (
        f'  <object id="{object_id}" name={quoteattr(name)} type="model">\n'
        "   <mesh>\n"
        "    <vertices>\n"
        f"{v_lines}\n"
        "    </vertices>\n"
        "    <triangles>\n"
        f"{t_lines}\n"
        "    </triangles>\n"
        "   </mesh>\n"
        "  </object>"
    )


def _write_3mf_archive(
    bd: Any,
    filepath: Path,
    unit_entries: list[tuple[str, dict[str, Any]]],
    positions: list[tuple[float, float]],
    *,
    tolerance: float = 1e-3,
    angular_tolerance: float = 0.1,
    unit_name: str = "unit",
    object_names: dict[str, str] | None = None,
) -> Path:
    """Write a vendor-neutral ``.3mf`` archive (3MF core specification only).

    Every distinct unit geometry (identified by its key in ``unit_entries``) is
    tessellated once.  Its meshes are translated so that the unit is centred on
    the XY origin and rests on ``z = 0``; each entry of ``unit_entries`` then
    becomes one ``<build>`` item translated to the matching XY position.  A
    unit made of several parts (e.g. plate + captive ring) is written as a
    component object, so slicers load it as one object with several parts.
    Objects are named ``object_names[key]`` when given, otherwise
    ``unit_name`` (numbered when there are several distinct keys).

    No slicer-specific project data is embedded.  Bambu Studio, OrcaSlicer,
    PrusaSlicer, etc. therefore open the file as plain geometry and keep the
    printer, filament, and process presets currently selected by the user.
    """
    if len(unit_entries) != len(positions):
        raise ValueError("unit_entries and positions must have the same length")
    if not unit_entries:
        raise ValueError("at least one unit is required for 3MF export")

    distinct_keys = list(dict.fromkeys(key for key, _ in unit_entries))
    resources: list[str] = []
    unit_object_ids: dict[str, int] = {}
    next_id = 1

    for key, parts in unit_entries:
        if key in unit_object_ids:
            continue
        if not parts:
            raise ValueError("each unit must contain at least one part")
        if object_names and key in object_names:
            name = object_names[key]
        elif len(distinct_keys) == 1:
            name = unit_name
        else:
            name = f"{unit_name}_{distinct_keys.index(key) + 1}"
        # Centre and ground the unit on its tessellated vertices: B-rep
        # bounding boxes of rotated or lofted solids can be loose.
        raw_meshes = [
            (part_name, *bd.Mesher._mesh_shape(solid, tolerance, angular_tolerance))
            for part_name, solid in parts.items()
        ]
        pts = np.array(
            [[float(c) for c in v] for _, verts, _ in raw_meshes for v in verts]
        )
        if pts.size == 0:
            raise ValueError("unit produced an empty mesh")
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        offset = (-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), -lo[2])

        mesh_ids: list[int] = []
        for part_name, raw_verts, raw_tris in raw_meshes:
            verts, tris = _dedup_mesh(raw_verts, raw_tris, offset=offset)
            if not tris:
                raise ValueError(f"part {part_name!r} produced an empty mesh")
            mesh_name = name if len(parts) == 1 else f"{name}_{part_name}"
            resources.append(_mesh_object_xml(next_id, mesh_name, verts, tris))
            mesh_ids.append(next_id)
            next_id += 1

        if len(mesh_ids) == 1:
            unit_object_ids[key] = mesh_ids[0]
        else:
            components = "\n".join(
                f'    <component objectid="{mesh_id}"/>' for mesh_id in mesh_ids
            )
            resources.append(
                f'  <object id="{next_id}" name={quoteattr(name)} type="model">\n'
                "   <components>\n"
                f"{components}\n"
                "   </components>\n"
                "  </object>"
            )
            unit_object_ids[key] = next_id
            next_id += 1

    items = "\n".join(
        f'  <item objectid="{unit_object_ids[key]}" '
        f'transform="1 0 0 0 1 0 0 0 1 {tx:.6f} {ty:.6f} 0"/>'
        for (key, _), (tx, ty) in zip(unit_entries, positions, strict=True)
    )
    model_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<model unit="millimeter" xml:lang="en-US" xmlns="{_3MF_CORE_NS}">',
            f' <metadata name="Application">{_3MF_APPLICATION}</metadata>',
            f' <metadata name="Title">{escape(filepath.stem)}</metadata>',
            " <resources>",
            *resources,
            " </resources>",
            " <build>",
            items,
            " </build>",
            "</model>",
            "",
        ]
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        "</Types>\n"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Relationships "
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel0" '
        f'Type="{_3MF_MODEL_REL_TYPE}"/>\n'
        "</Relationships>\n"
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("3D/3dmodel.model", model_xml)
    return filepath


def _resolve_plate_units(
    kc_or_config: Kaleidocycle | NDArray[np.float64] | TwistedPlateConfig | None = None,
    *,
    count: int | None = None,
    segment: int = 0,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    scale: float | None = None,
    with_ring: bool | None = None,
    normalize_segments: bool = True,
    degrees: bool = False,
    **kwargs: Any,
) -> tuple[
    list[tuple[str, dict[str, Any]]], tuple[float, float], list[TwistedPlateConfig]
]:
    """Build unit part dicts (cached by geometry) and compute unit XY footprint.

    Returns ``(unit_entries, (max_width, max_depth), configs)`` with one config
    per unit.
    """
    if kc_or_config is None or isinstance(kc_or_config, TwistedPlateConfig):
        cfg = _resolve_config(
            kc_or_config,
            degrees=degrees,
            ribbon_width=ribbon_width,
            edge_length=edge_length,
            with_ring=with_ring,
            **kwargs,
        )
        resolved_count = 1 if count is None else int(count)
        if resolved_count < 1:
            raise ValueError(f"count must be at least 1, got {resolved_count}")
        configs = [cfg] * resolved_count
    else:
        hinges = _extract_hinges(kc_or_config)
        n = hinges.shape[0] - 1
        if not (0 <= segment < n):
            raise ValueError(f"segment must satisfy 0 <= segment < {n}, got {segment}")
        resolved_count = n if count is None else int(count)
        if resolved_count < 1:
            raise ValueError(f"count must be at least 1, got {resolved_count}")

        raw_tangents = binormals_to_tangents(hinges, normalize=False)
        raw_norms = np.linalg.norm(raw_tangents, axis=1)
        mean_len = float(np.mean(raw_norms))
        resolved_w = 43.0 if ribbon_width is None else float(ribbon_width)
        base_d = 29.0 if edge_length is None else float(edge_length)
        resolved_ring = False if with_ring is None else bool(with_ring)

        configs = []
        for idx in range(resolved_count):
            seg = (segment + idx) % n
            b0 = hinges[seg]
            b1 = hinges[seg + 1]
            torsion_i = float(
                np.arctan2(
                    float(np.linalg.norm(np.cross(b0, b1))),
                    float(np.clip(np.dot(b0, b1), -1.0, 1.0)),
                )
            )
            if scale is not None:
                d_i = float(scale * raw_norms[seg])
            elif normalize_segments:
                d_i = base_d
            else:
                d_i = float(base_d * raw_norms[seg] / max(mean_len, 1e-12))

            cfg = TwistedPlateConfig(
                torsion_angle=torsion_i,
                ribbon_width=resolved_w,
                edge_length=d_i,
                with_ring=resolved_ring,
                **kwargs,
            )
            cfg.validate()
            configs.append(cfg)

    parts_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    bounds_cache: dict[str, tuple[float, float]] = {}
    unit_entries: list[tuple[str, dict[str, Any]]] = []
    max_w = 0.0
    max_d = 0.0

    for cfg in configs:
        cache_key = _config_key(cfg)
        key_str = repr(cache_key)
        if cache_key not in parts_cache:
            parts = build_twisted_plate_parts(cfg)
            parts_cache[cache_key] = parts
            min_x, max_x, min_y, max_y, _ = _unit_bounds(parts)
            bounds_cache[key_str] = (max_x - min_x, max_y - min_y)
        unit_w, unit_d = bounds_cache[key_str]
        max_w = max(max_w, unit_w)
        max_d = max(max_d, unit_d)
        unit_entries.append((key_str, parts_cache[cache_key]))

    return unit_entries, (max_w, max_d), configs


def _plate_items(
    kc_or_config: Kaleidocycle | NDArray[np.float64] | TwistedPlateConfig | None,
    *,
    spacing: float,
    center: tuple[float, float],
    with_pins: bool,
    **unit_kwargs: Any,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[float, float]], dict[str, str]]:
    """Collect the units (and one hinge pin per unit) of a print plate.

    Returns ``(entries, xy_positions, object_names)``; identical units and
    identical pins share a key so that they are meshed only once.
    """
    unit_entries, (unit_w, unit_d), configs = _resolve_plate_units(
        kc_or_config, **unit_kwargs
    )
    unit_keys = list(dict.fromkeys(key for key, _ in unit_entries))
    names = {
        key: "twisted_plate" if len(unit_keys) == 1 else f"twisted_plate_{i + 1}"
        for i, key in enumerate(unit_keys)
    }

    pin_entries: list[tuple[str, dict[str, Any]]] = []
    pin_len = pin_w = 0.0
    if with_pins:
        pins: dict[str, dict[str, Any]] = {}
        for cfg in (c for c in configs if c.uses_pins):
            key = f"pin{_pin_key(cfg)!r}"
            if key not in pins:
                pins[key] = {"pin": build_hinge_pin(cfg)}
                min_x, max_x, min_y, max_y, _ = _unit_bounds(pins[key])
                pin_len = max(pin_len, max_x - min_x)
                pin_w = max(pin_w, max_y - min_y)
            pin_entries.append((key, pins[key]))
        names.update(
            {
                key: "hinge_pin" if len(pins) == 1 else f"hinge_pin_{i + 1}"
                for i, key in enumerate(pins)
            }
        )

    unit_pos, pin_pos = _plate_layout(
        len(unit_entries),
        unit_w,
        unit_d,
        pin_count=len(pin_entries),
        pin_length=pin_len,
        pin_width=pin_w,
        spacing=spacing,
        center=center,
    )
    return unit_entries + pin_entries, unit_pos + pin_pos, names


def build_print_plate(
    kc_or_config: Kaleidocycle | NDArray[np.float64] | TwistedPlateConfig | None = None,
    *,
    count: int | None = None,
    spacing: float = 6.0,
    center: tuple[float, float] = (0.0, 0.0),
    segment: int = 0,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    scale: float | None = None,
    with_ring: bool | None = None,
    with_pins: bool = True,
    normalize_segments: bool = True,
    degrees: bool = False,
    **kwargs: Any,
) -> Any:
    """Arrange ``count`` twisted-plate units flat on ``z = 0`` for batch printing.

    When ``kc_or_config`` is a :class:`Kaleidocycle` or hinge array and
    ``count`` is omitted, defaults to ``n`` (all links in the cycle).  With
    ``with_pins=True`` (default) one hinge pin per unit is laid next to the
    units (a closed ring of ``n`` units has ``n`` hinges).
    """
    bd = _require_build123d()
    entries, positions, _ = _plate_items(
        kc_or_config,
        spacing=spacing,
        center=center,
        with_pins=with_pins,
        count=count,
        segment=segment,
        ribbon_width=ribbon_width,
        edge_length=edge_length,
        scale=scale,
        with_ring=with_ring,
        normalize_segments=normalize_segments,
        degrees=degrees,
        **kwargs,
    )

    children: list[Any] = []
    label_counts: dict[str, int] = {}
    for (_, parts), (tx, ty) in zip(entries, positions, strict=True):
        min_x, max_x, min_y, max_y, min_z = _unit_bounds(parts)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        loc = bd.Location((tx - cx, ty - cy, -min_z))
        for part_name, solid in parts.items():
            idx = label_counts.get(part_name, 0)
            label_counts[part_name] = idx + 1
            placed = solid.moved(loc)
            placed.label = f"{part_name}_{idx}"
            children.append(placed)

    return bd.Compound(children=children, label="print_plate")


def _run_clearance_check(
    kc_or_config: Kaleidocycle | NDArray[np.float64] | TwistedPlateConfig | None,
    check_clearance: bool | str,
    *,
    min_clearance: float = 0.4,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    scale: float | None = None,
    with_ring: bool | None = None,
    with_pins: bool = True,
    normalize_segments: bool = True,
    degrees: bool = False,
    **kwargs: Any,
) -> None:
    """Run the automatic contact checks used by the export functions."""
    from .cad_check import (
        _resolve_policy,
        check_hinge_fit,
        check_kaleidocycle_clearance,
        enforce_report,
    )

    if _resolve_policy(check_clearance) == "off":
        return
    if kc_or_config is None or isinstance(kc_or_config, TwistedPlateConfig):
        cfg = _resolve_config(
            kc_or_config,
            degrees=degrees,
            ribbon_width=ribbon_width,
            edge_length=edge_length,
            with_ring=with_ring,
            **kwargs,
        )
        report = check_hinge_fit(cfg, min_clearance=min_clearance)
    else:
        report = check_kaleidocycle_clearance(
            kc_or_config,
            ribbon_width=43.0 if ribbon_width is None else float(ribbon_width),
            edge_length=29.0 if edge_length is None else float(edge_length),
            scale=scale,
            with_ring=bool(with_ring),
            with_pins=with_pins,
            normalize_segments=normalize_segments,
            min_clearance=min_clearance,
            **kwargs,
        )
    enforce_report(report, check_clearance, stacklevel=4)


def export_print_plate(
    filepath: str | Path,
    kc_or_config: Kaleidocycle | NDArray[np.float64] | TwistedPlateConfig | None = None,
    *,
    count: int | None = None,
    spacing: float = 6.0,
    segment: int = 0,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    scale: float | None = None,
    with_ring: bool | None = None,
    with_pins: bool = True,
    normalize_segments: bool = True,
    degrees: bool = False,
    tolerance: float = 1e-3,
    angular_tolerance: float = 0.1,
    bed_center: tuple[float, float] = _DEFAULT_BED_CENTER,
    check_clearance: bool | str = "warn",
    min_clearance: float = 0.4,
    **kwargs: Any,
) -> Path:
    """Export an ``N``-unit print plate to ``.3mf``, ``.stl``, or ``.step``.

    For ``.3mf`` files, writes a standard 3MF archive whose ``count`` units
    are laid out in a grid around ``bed_center`` (default: centre of a
    256 mm bed), each resting on ``z = 0``.  With ``with_pins=True``
    (default) one octagonal hinge pin per unit (object ``hinge_pin``) lies
    flat next to the units.  Identical units and pins share one mesh, and a
    unit with a captive ring is a single object with ``plate`` and ``ring``
    parts.  No slicer project settings are embedded, so the slicer keeps its
    currently selected printer, filament, and process presets.

    Before writing, the parts are checked for contacts
    (:mod:`kaleidocycle.cad_check`): the hinge fit always, and for a
    kaleidocycle also every pair of links through a full rotation, which must
    stay ``min_clearance`` apart.  Contacts raise a ``ClearanceWarning``
    (``check_clearance="warn"``, default) or ``ClearanceError`` (``"error"``);
    ``check_clearance=False`` skips the check.
    """
    _run_clearance_check(
        kc_or_config,
        check_clearance,
        min_clearance=min_clearance,
        ribbon_width=ribbon_width,
        edge_length=edge_length,
        scale=scale,
        with_ring=with_ring,
        with_pins=with_pins,
        normalize_segments=normalize_segments,
        degrees=degrees,
        **kwargs,
    )
    out_path = Path(filepath)
    suffix = out_path.suffix.lower()

    if suffix == ".3mf":
        bd = _require_build123d()
        entries, positions, names = _plate_items(
            kc_or_config,
            spacing=spacing,
            center=bed_center,
            with_pins=with_pins,
            count=count,
            segment=segment,
            ribbon_width=ribbon_width,
            edge_length=edge_length,
            scale=scale,
            with_ring=with_ring,
            normalize_segments=normalize_segments,
            degrees=degrees,
            **kwargs,
        )
        return _write_3mf_archive(
            bd,
            out_path,
            entries,
            positions,
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            unit_name="twisted_plate",
            object_names=names,
        )

    plate_compound = build_print_plate(
        kc_or_config,
        count=count,
        spacing=spacing,
        segment=segment,
        ribbon_width=ribbon_width,
        edge_length=edge_length,
        scale=scale,
        with_ring=with_ring,
        with_pins=with_pins,
        normalize_segments=normalize_segments,
        degrees=degrees,
        **kwargs,
    )
    return export_shape(
        plate_compound,
        out_path,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def export_shape(
    shape: Any,
    filepath: str | Path,
    *,
    tolerance: float = 1e-3,
    angular_tolerance: float = 0.1,
    bed_center: tuple[float, float] = _DEFAULT_BED_CENTER,
) -> Path:
    """Export a ``build123d`` ``Shape``/``Compound`` to STL, STEP, or 3MF.

    For ``.3mf``, the shape becomes one object (one part per solid) centred
    on ``bed_center`` and resting on ``z = 0``.
    """
    bd = _require_build123d()
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".stl":
        bd.export_stl(
            shape,
            str(out_path),
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
        )
    elif suffix in (".step", ".stp"):
        bd.export_step(shape, str(out_path))
    elif suffix == ".3mf":
        parts: dict[str, Any] = {}
        for idx, item in enumerate(_iter_export_shapes(bd, shape)):
            label = getattr(item, "label", "") or f"part_{idx}"
            if label in parts:
                label = f"{label}_{idx}"
            parts[label] = item
        return _write_3mf_archive(
            bd,
            out_path,
            [("shape", parts)],
            [bed_center],
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            unit_name=getattr(shape, "label", "") or out_path.stem,
        )
    else:
        raise ValueError(
            f"Unsupported CAD export format {suffix!r}; "
            "expected .stl, .step, .stp, or .3mf"
        )
    return out_path


def export_twisted_plate(
    filepath: str | Path,
    config: TwistedPlateConfig | None = None,
    *,
    count: int = 1,
    spacing: float = 6.0,
    tolerance: float = 1e-3,
    angular_tolerance: float = 0.1,
    torsion_angle: float | None = None,
    ribbon_width: float | None = None,
    edge_length: float | None = None,
    body_thickness: float | None = None,
    hinge_diameter: float | None = None,
    pin_diameter: float | None = None,
    pin_fit_clearance: float | None = None,
    pin_clearance: float | None = None,
    pin_core_diameter: float | None = None,
    pin_chamfer: float | None = None,
    hinge_type: str | None = None,
    cone_diameter: float | None = None,
    cone_angle: float | None = None,
    snap_gap: float | None = None,
    hinge_gap: float | None = None,
    gap: float | None = None,
    num_knuckles: int | None = None,
    with_ring: bool | None = None,
    cut_slits: bool | None = None,
    ring_diameter: float | None = None,
    ring_width: float | None = None,
    ring_gap: float | None = None,
    slit_rail_height: float | None = None,
    loft_slices: int | None = None,
    with_pins: bool = True,
    degrees: bool = False,
    check_clearance: bool | str = "warn",
) -> Path:
    """Build one or ``count`` twisted plate links and export to STL, STEP, or 3MF.

    ``.3mf`` files and ``count > 1`` produce a print plate (see
    :func:`export_print_plate`), with one hinge pin per unit unless
    ``with_pins=False``.  A single-unit ``.stl``/``.step`` holds the unit only.
    The hinge fit is checked for contacts first (see ``check_clearance`` in
    :func:`export_print_plate`).
    """
    cfg = _resolve_config(
        config,
        degrees=degrees,
        torsion_angle=torsion_angle,
        ribbon_width=ribbon_width,
        edge_length=edge_length,
        body_thickness=body_thickness,
        hinge_diameter=hinge_diameter,
        pin_diameter=pin_diameter,
        pin_fit_clearance=pin_fit_clearance,
        pin_clearance=pin_clearance,
        pin_core_diameter=pin_core_diameter,
        pin_chamfer=pin_chamfer,
        hinge_type=hinge_type,
        cone_diameter=cone_diameter,
        cone_angle=cone_angle,
        snap_gap=snap_gap,
        hinge_gap=hinge_gap,
        gap=gap,
        num_knuckles=num_knuckles,
        with_ring=with_ring,
        cut_slits=cut_slits,
        ring_diameter=ring_diameter,
        ring_width=ring_width,
        ring_gap=ring_gap,
        slit_rail_height=slit_rail_height,
        loft_slices=loft_slices,
    )
    _run_clearance_check(cfg, check_clearance)
    out_path = Path(filepath)
    if out_path.suffix.lower() == ".3mf" or count > 1:
        return export_print_plate(
            out_path,
            cfg,
            count=count,
            spacing=spacing,
            with_pins=with_pins,
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            check_clearance=False,
        )

    shape = build_twisted_plate(cfg)
    return export_shape(
        shape,
        out_path,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )
