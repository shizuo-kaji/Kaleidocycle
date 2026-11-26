/*
 * 3D-printable twisted-plate links for Kaleidocycle Studio.
 *
 * A port of the geometry of src/kaleidocycle/cad.py (build123d) to manifold-3d,
 * which runs in the browser.  Dimensions, defaults and validation follow
 * TwistedPlateConfig; the B-rep lofts are replaced by meshed equivalents, so
 * volumes agree with the Python models to a fraction of a percent
 * (test_print.js).  STEP output needs the Python package.
 *
 * Coordinates are millimetres in a link's printing frame: the bottom hinge
 * runs along X at height z_bottom_hinge above the bed (z = 0) and the top
 * hinge, edge_length higher, is turned by the torsion angle about Z.
 *
 * Loadable as a classic script (window.KaleidocyclePrint) and from Node.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.KaleidocyclePrint = api;
})(typeof window === "undefined" ? globalThis : window, function factory() {
  "use strict";

  // TwistedPlateConfig defaults (tests/test_cad.py checks that they agree).
  const DEFAULTS = {
    "torsion_angle": 1.270599695451872,
    "ribbon_width": 43.0,
    "edge_length": 29.0,
    "body_thickness": 2.4,
    "hinge_type": "pin",
    "hinge_diameter": null,
    "hinge_gap": null,
    "gap": 0.4,
    "num_knuckles": 5,
    "pin_diameter": 4.0,
    "pin_fit_clearance": 0.09,
    "pin_clearance": 0.3,
    "pin_core_diameter": 1.8,
    "pin_chamfer": 0.5,
    "cone_diameter": 4.0,
    "cone_angle": 60.0,
    "snap_gap": 2.25,
    "with_ring": false,
    "cut_slits": null,
    "ring_diameter": null,
    "ring_width": 2.0,
    "ring_gap": 0.5,
    "slit_rail_height": 2.0,
    "loft_slices": 25
  };
  // Module constants of cad.py.
  const CONSTANTS = {
    "min_wall": 0.8,
    "knuckle_foot_depth": 0.3,
    "bore_flat_margin": 0.3,
    "default_hinge_diameter": { "pin": 7.0, "cone": 6.0 },
    "default_hinge_gap": { "pin": 0.4, "cone": 0.2 }
  };
  const HINGE_TYPES = ["pin", "cone"];
  const MIN_CLEARANCE = 0.4;
  const BED_CENTRE = [128, 128];

  // Mesh resolution: grid step of the twisted plate (mm) and circle segments.
  const QUALITY = { grid: 0.5, knuckle: 96, bore: 96, cone: 64, ring: 128 };

  /** A resolved configuration with the derived quantities of cad.py. */
  function config(overrides = {}) {
    const raw = { ...DEFAULTS };
    for (const [key, value] of Object.entries(overrides)) {
      if (!(key in DEFAULTS)) throw new Error(`unknown print setting ${key}`);
      if (value !== undefined) raw[key] = value;
    }
    const c = raw;
    const hingeDiameter = c.hinge_diameter ?? CONSTANTS.default_hinge_diameter[c.hinge_type] ?? 7.0;
    const hingeGap = c.hinge_gap ?? CONSTANTS.default_hinge_gap[c.hinge_type] ?? 0.4;
    const hingeRadius = 0.5 * hingeDiameter;
    const pinRadius = 0.5 * c.pin_diameter;
    const fixedBore = pinRadius + c.pin_fit_clearance;
    const freeBore = pinRadius + c.pin_clearance;
    const coneRadius = 0.5 * c.cone_diameter;
    const coneHeight = coneRadius / Math.tan((c.cone_angle * Math.PI) / 180);
    const zBottomHinge = hingeRadius - CONSTANTS.knuckle_foot_depth;
    const zTopHinge = zBottomHinge + c.edge_length;
    const derived = {
      hingeDiameter,
      hingeGap,
      hingeRadius,
      usesPins: c.hinge_type === "pin",
      pinRadius,
      pinInradius: pinRadius * Math.cos(Math.PI / 8),
      pinCoreInradius: 0.5 * c.pin_core_diameter * Math.cos(Math.PI / 8),
      fixedBoreRadius: fixedBore,
      freeBoreRadius: freeBore,
      fixedBoreFlatHeight: fixedBore + CONSTANTS.bore_flat_margin,
      freeBoreFlatHeight: freeBore + CONSTANTS.bore_flat_margin,
      coneRadius,
      coneHeight,
      snapDeflection: Math.max(coneHeight - hingeGap, 0),
      zBottomHinge,
      zTopHinge,
      zPlateBottom: zBottomHinge + hingeRadius + c.gap,
      zPlateTop: zTopHinge - hingeRadius - c.gap,
      torsionDegrees: (c.torsion_angle * 180) / Math.PI,
      ringDiameter: c.ring_diameter ?? 0.4 * c.ribbon_width,
      cutSlits: c.cut_slits ?? c.with_ring,
    };
    return Object.freeze({ ...c, ...derived });
  }

  /** Axial (start, end) extents of the bottom and top knuckles. */
  function knuckleIntervals(cfg) {
    const w = cfg.ribbon_width;
    const n = cfg.num_knuckles;
    const seg = w / n;
    const g = cfg.hingeGap;
    const bottom = [];
    const top = [];
    for (let k = 0; k < n; k += 1) {
      const x0 = -0.5 * w + k * seg;
      const x1 = x0 + seg;
      if (cfg.hinge_type === "cone") {
        if (k % 2 === 0) {
          bottom.push([x0, x1]);
          continue;
        }
        const leftSocket = k === 1;
        const rightSocket = k === n - 2;
        const socketed = leftSocket || rightSocket;
        const lo = leftSocket || !socketed ? g : cfg.snap_gap;
        const hi = rightSocket || !socketed ? g : cfg.snap_gap;
        top.push([x0 + lo, x1 - hi]);
      } else {
        const lo = k > 0 ? 0.5 * g : 0;
        const hi = k < n - 1 ? 0.5 * g : 0;
        (k % 2 === 0 ? bottom : top).push([x0 + lo, x1 - hi]);
      }
    }
    return { bottom, top };
  }

  /** The validation of TwistedPlateConfig.validate(); throws on failure. */
  function validate(cfg) {
    const finite = [
      "torsion_angle", "ribbon_width", "edge_length", "body_thickness", "gap",
      "pin_diameter", "pin_fit_clearance", "pin_clearance", "pin_core_diameter",
      "pin_chamfer", "cone_diameter", "cone_angle", "snap_gap", "ring_width",
      "ring_gap", "slit_rail_height",
    ];
    for (const name of finite) {
      if (!Number.isFinite(cfg[name])) throw new Error(`${name} must be finite`);
    }
    const wall = CONSTANTS.min_wall;
    if (!HINGE_TYPES.includes(cfg.hinge_type)) {
      throw new Error(`hinge type must be one of ${HINGE_TYPES.join(", ")}`);
    }
    if (cfg.ribbon_width <= 0) throw new Error("ribbon width must be positive");
    if (cfg.body_thickness <= 0) throw new Error("plate thickness must be positive");
    if (cfg.hingeDiameter <= 0) throw new Error("hinge diameter must be positive");
    if (cfg.hingeGap < 0) throw new Error("knuckle gap must be non-negative");
    if (cfg.gap < 0) throw new Error("plate-edge gap must be non-negative");
    if (cfg.edge_length <= cfg.hingeDiameter + 2 * cfg.gap) {
      throw new Error("edge length must exceed the hinge diameter + 2 × plate-edge gap");
    }
    if (cfg.num_knuckles < 3 || cfg.num_knuckles % 2 === 0) {
      throw new Error("the number of knuckles must be an odd integer ≥ 3");
    }
    if (cfg.ribbon_width / cfg.num_knuckles <= cfg.hingeGap) {
      throw new Error("ribbon width / knuckles must exceed the knuckle gap");
    }
    if (cfg.loft_slices < 5) throw new Error("loft slices must be at least 5");
    if (cfg.usesPins) {
      if (cfg.pin_diameter <= 0) throw new Error("pin diameter must be positive");
      if (cfg.pin_clearance < 0) throw new Error("pin clearance must be non-negative");
      if (cfg.fixedBoreRadius <= 0.5 * cfg.pinRadius) {
        throw new Error("pin fit clearance is too negative for the pin");
      }
      const boreTop = Math.max(cfg.fixedBoreFlatHeight, cfg.freeBoreFlatHeight);
      if (cfg.hingeRadius - boreTop < wall) {
        throw new Error(
          `the knuckles must leave a ${wall} mm wall around the pin bores; ` +
            "reduce the pin clearances",
        );
      }
      if (cfg.zBottomHinge - cfg.fixedBoreRadius < wall) {
        throw new Error("the hinge diameter is too small for the bottom knuckle foot");
      }
      if (cfg.pin_core_diameter < 0) throw new Error("pin core must be non-negative");
      if (cfg.pin_chamfer < 0) throw new Error("pin chamfer must be non-negative");
      if (cfg.pin_core_diameter > 0 && cfg.pinInradius - cfg.pinCoreInradius < wall) {
        throw new Error(`the pin core must leave a ${wall} mm pin wall`);
      }
      if (cfg.pinInradius - cfg.pin_chamfer <= cfg.pinCoreInradius) {
        throw new Error("the pin chamfer is too large for the pin cross-section");
      }
      if (2 * cfg.pin_chamfer >= cfg.ribbon_width) {
        throw new Error("the pin chamfer must be less than half of the ribbon width");
      }
    } else {
      if (cfg.cone_diameter <= 0) throw new Error("cone diameter must be positive");
      if (!(cfg.cone_angle > 0 && cfg.cone_angle < 90)) {
        throw new Error("cone angle must lie strictly between 0 and 90 degrees");
      }
      if (cfg.hingeRadius - cfg.coneRadius < wall) {
        throw new Error(`the knuckles must leave a ${wall} mm wall around the cone sockets`);
      }
      if (cfg.zBottomHinge < cfg.coneRadius) {
        throw new Error("the cone diameter is too large: the cones reach the bed");
      }
      if (cfg.snap_gap < cfg.snapDeflection) {
        throw new Error(
          `the snap gap must leave room for the cone tips (${cfg.snapDeflection.toFixed(2)} mm)`,
        );
      }
      const sockets = cfg.num_knuckles === 3 ? 2 : 1;
      for (const [x0, x1] of knuckleIntervals(cfg).top) {
        if (x1 - x0 < sockets * cfg.coneHeight + wall) {
          throw new Error("the top knuckles are too short for the cone sockets");
        }
      }
    }
    if (cfg.with_ring || cfg.cutSlits) {
      if (cfg.ring_width <= 0) throw new Error("ring width must be positive");
      if (cfg.ring_gap <= 0) throw new Error("ring gap must be positive");
      if (cfg.slit_rail_height < 0) throw new Error("slit rail height must be non-negative");
      const rOut = 0.5 * cfg.ringDiameter;
      if (!(rOut > 0)) throw new Error("ring diameter must be positive");
      const rIn = rOut - cfg.ring_width;
      if (rIn <= 0) throw new Error("ring diameter must exceed 2 × ring width");
      const uIn = rIn - cfg.ring_gap;
      const uOut = rOut + cfg.ring_gap;
      if (uIn <= 0) throw new Error("ring inner radius minus ring gap must be positive");
      if (uOut >= 0.5 * cfg.ribbon_width) {
        throw new Error("the ring plus its gap must fit within the ribbon width");
      }
      if (cfg.with_ring && Math.hypot(uIn, 0.5 * cfg.body_thickness) >= rIn) {
        throw new Error("the plate is too thick for the captive ring");
      }
      const slitBottom = cfg.zPlateBottom + cfg.slit_rail_height;
      const slitTop = cfg.zPlateTop - cfg.slit_rail_height;
      if (slitTop <= slitBottom + cfg.ring_width + 2 * cfg.ring_gap) {
        throw new Error("the edge length is too small for the slits and captive ring");
      }
    }
    return cfg;
  }

  // ---------------------------------------------------------------------------
  // Geometry (needs the manifold-3d module returned by KaleidocycleManifold.load)

  function signedArea(points) {
    let area = 0;
    for (let i = 0; i < points.length; i += 1) {
      const [x0, y0] = points[i];
      const [x1, y1] = points[(i + 1) % points.length];
      area += x0 * y1 - x1 * y0;
    }
    return 0.5 * area;
  }

  /** Column-major 4 × 4 matrix of the affine map p ↦ R p + o. */
  function affine(columns, origin) {
    const [a, b, c] = columns;
    return [
      a[0], a[1], a[2], 0,
      b[0], b[1], b[2], 0,
      c[0], c[1], c[2], 0,
      origin[0], origin[1], origin[2], 1,
    ];
  }

  function createBuilder(wasm) {
    const { Manifold, CrossSection, Mesh } = wasm;

    /** Combine and free: ``fn`` gets the inputs, which are deleted afterwards. */
    function consume(inputs, fn) {
      try {
        return fn(...inputs);
      } finally {
        inputs.forEach((item) => item.delete());
      }
    }

    function polygon(points) {
      const ordered = signedArea(points) < 0 ? points.slice().reverse() : points;
      return new CrossSection([ordered]);
    }

    function rectangle(cx, cy, width, height) {
      const hw = 0.5 * width;
      const hh = 0.5 * height;
      return polygon([
        [cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh],
      ]);
    }

    function circle(cx, cy, radius, segments) {
      return consume([CrossSection.circle(radius, segments)], (disc) =>
        disc.translate([cx, cy]));
    }

    /** Extrude a (y, z) profile along +X from ``x0`` over ``length``. */
    function extrudeAlongX(section, x0, length) {
      return consume([Manifold.extrude(section, length)], (solid) =>
        solid.transform(affine([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [x0, 0, 0])));
    }

    /**
     * Rectangular cross-section twisting linearly from z_plate_bottom to
     * z_plate_top (cad._loft_twisted_box), meshed on a fine grid so that the
     * ruled faces stay true to the loft.
     */
    function twistedBox({ zStart, zEnd, zPlateBottom, zPlateTop, alpha, width, thickness, xOffset = 0 }) {
      const nx = Math.max(2, Math.ceil(width / QUALITY.grid));
      const twistLevels = Math.ceil((Math.abs(alpha) * (zEnd - zStart)) /
        ((zPlateTop - zPlateBottom) * ((1.5 * Math.PI) / 180)));
      const nz = Math.max(2, Math.ceil((zEnd - zStart) / QUALITY.grid), twistLevels);
      const ring = [];
      for (let i = 0; i <= nx; i += 1) ring.push([xOffset - 0.5 * width + (width * i) / nx, -0.5 * thickness]);
      for (let i = nx; i >= 0; i -= 1) ring.push([xOffset - 0.5 * width + (width * i) / nx, 0.5 * thickness]);
      const perimeter = ring.length;
      const vertices = new Float32Array(3 * perimeter * (nz + 1));
      for (let k = 0; k <= nz; k += 1) {
        const z = zStart + ((zEnd - zStart) * k) / nz;
        const angle = ((z - zPlateBottom) / (zPlateTop - zPlateBottom)) * alpha;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        ring.forEach(([x, y], index) => {
          const offset = 3 * (k * perimeter + index);
          vertices[offset] = x * cos - y * sin;
          vertices[offset + 1] = x * sin + y * cos;
          vertices[offset + 2] = z;
        });
      }
      const triangles = [];
      const at = (k, index) => k * perimeter + (index % perimeter);
      for (let k = 0; k < nz; k += 1) {
        for (let index = 0; index < perimeter; index += 1) {
          const a0 = at(k, index);
          const b0 = at(k, index + 1);
          const a1 = at(k + 1, index);
          const b1 = at(k + 1, index + 1);
          triangles.push(a0, b0, b1, a0, b1, a1);
        }
      }
      // Caps: front row i ↔ back row (perimeter − 1 − i).
      for (let i = 0; i < nx; i += 1) {
        const f0 = i;
        const f1 = i + 1;
        const b0 = perimeter - 1 - i;
        const b1 = perimeter - 2 - i;
        triangles.push(at(0, f0), at(0, b0), at(0, b1), at(0, f0), at(0, b1), at(0, f1));
        triangles.push(at(nz, f0), at(nz, b1), at(nz, b0), at(nz, f0), at(nz, f1), at(nz, b1));
      }
      const mesh = new Mesh({
        numProp: 3,
        vertProperties: vertices,
        triVerts: new Uint32Array(triangles),
      });
      return new Manifold(mesh);
    }

    function bottomKnuckleProfile(cfg) {
      const r = cfg.hingeRadius;
      const yNeck = Math.min(0.5 * cfg.body_thickness, r);
      const z0 = cfg.zBottomHinge;
      const zBot = cfg.zPlateBottom;
      const zp = zBot - z0;
      const distance = Math.hypot(yNeck, zp);
      let yTan;
      let zTan;
      if (distance > r && yNeck < r) {
        const thetaP = Math.atan2(zp, yNeck);
        const delta = Math.acos(r / distance);
        const thetaT = Math.max(thetaP - delta, 0);
        yTan = r * Math.cos(thetaT);
        zTan = z0 + r * Math.sin(thetaT);
      } else {
        yTan = Math.max(r, yNeck);
        zTan = z0;
      }
      const overlap = 0.02;
      const wedge = polygon([
        [-yNeck, zBot + overlap], [-yTan, zTan], [0, z0], [yTan, zTan], [yNeck, zBot + overlap],
      ]);
      const disc = consume(
        [circle(0, z0, r, QUALITY.knuckle), rectangle(0, 0.5 * (z0 + r), 2 * r + 2, z0 + r)],
        (round, aboveBed) => round.intersect(aboveBed),
      );
      return consume([disc, wedge], (a, b) => a.add(b));
    }

    function topKnuckleProfile(cfg) {
      const r = cfg.hingeRadius;
      const yNeck = Math.min(0.5 * cfg.body_thickness, r);
      const z1 = cfg.zTopHinge;
      const zTop = cfg.zPlateTop;
      const zp = zTop - z1;
      const distance = Math.hypot(yNeck, zp);
      let yTan;
      let zTan;
      if (distance > r && yNeck < r) {
        const thetaP = Math.atan2(zp, yNeck);
        const delta = Math.acos(r / distance);
        const thetaT = Math.min(thetaP + delta, 0);
        yTan = r * Math.cos(thetaT);
        zTan = z1 + r * Math.sin(thetaT);
      } else {
        yTan = Math.max(r, yNeck);
        zTan = z1;
      }
      const overlap = 0.02;
      const wedge = polygon([
        [-yNeck, zTop - overlap], [yNeck, zTop - overlap], [yTan, zTan], [0, z1], [-yTan, zTan],
      ]);
      return consume([wedge, circle(0, z1, r, QUALITY.knuckle)], (a, b) => a.add(b));
    }

    function knuckles(profile, intervals, rotationDeg = 0) {
      const pieces = intervals.map(([x0, x1]) => extrudeAlongX(profile, x0, x1 - x0));
      return consume(pieces, (...all) => {
        const union = Manifold.union(all);
        if (!rotationDeg) return union;
        return consume([union], (solid) => solid.rotate([0, 0, rotationDeg]));
      });
    }

    /** Horizontal teardrop bore along X, flat-topped (cad._build_bore). */
    function bore({ radius, flatHeight, zAxis, length, rotationDeg = 0 }) {
      const c = radius * Math.SQRT1_2;
      const span = 2 * radius + flatHeight;
      const profile = consume(
        [
          circle(0, 0, radius, QUALITY.bore),
          polygon([[0, 0], [c, c], [0, radius * Math.SQRT2], [-c, c]]),
          rectangle(0, flatHeight - 0.5 * span, 4 * radius, span),
        ],
        (round, wedge, belowFlat) =>
          consume([round.add(wedge)], (drop) => drop.intersect(belowFlat)),
      );
      const placed = consume([profile], (section) => section.translate([0, zAxis]));
      const solid = consume([placed], (section) => extrudeAlongX(section, -0.5 * length, length));
      if (!rotationDeg) return solid;
      return consume([solid], (body) => body.rotate([0, 0, rotationDeg]));
    }

    /** Cone about an X-parallel axis, base on x = faceX (cad._build_cone). */
    function cone({ faceX, direction, radius, height, zAxis, back = 0.05, rotationDeg = 0 }) {
      const d = direction > 0 ? 1 : -1;
      const along = consume(
        [
          Manifold.cylinder(back, radius, radius, QUALITY.cone),
          Manifold.cylinder(height, radius, 0, QUALITY.cone),
        ],
        (stub, tip) => consume([tip.translate([0, 0, back])], (moved) => stub.add(moved)),
      );
      // +Z ↦ d·X, then move the stub start to faceX − d·back.
      const oriented = consume([along], (solid) => solid.rotate([0, d * 90, 0]));
      const placed = consume([oriented], (solid) =>
        solid.translate([faceX - d * back, 0, zAxis]));
      if (!rotationDeg) return placed;
      return consume([placed], (solid) => solid.rotate([0, 0, rotationDeg]));
    }

    function captiveRing(cfg) {
      const rOut = 0.5 * cfg.ringDiameter;
      const rIn = rOut - cfg.ring_width;
      const zb = cfg.zPlateBottom + cfg.slit_rail_height + cfg.ring_gap;
      const zt = zb + cfg.ring_width;
      const ch = 0.25 * cfg.ring_width;
      const section = polygon([
        [rIn + ch, zb], [rOut - ch, zb], [rOut, zb + ch], [rOut, zt - ch],
        [rOut - ch, zt], [rIn + ch, zt], [rIn, zt - ch], [rIn, zb + ch],
      ]);
      return consume([section], (s) => Manifold.revolve(s, QUALITY.ring));
    }

    function octagon(x, circumradius) {
      const points = [];
      for (let k = 0; k < 8; k += 1) {
        const angle = ((22.5 + 45 * k) * Math.PI) / 180;
        points.push([x, circumradius * Math.cos(angle), circumradius * Math.sin(angle)]);
      }
      return points;
    }

    /** The octagonal hinge pin along X, centred on the origin, flat face down. */
    function pin(cfg) {
      const half = 0.5 * cfg.ribbon_width;
      const r = cfg.pinRadius;
      const chamfer = cfg.pin_chamfer;
      const sections = chamfer > 0
        ? (() => {
            const rEnd = r - chamfer / Math.cos(Math.PI / 8);
            return [
              ...octagon(-half, rEnd), ...octagon(-half + chamfer, r),
              ...octagon(half - chamfer, r), ...octagon(half, rEnd),
            ];
          })()
        : [...octagon(-half, r), ...octagon(half, r)];
      const outer = Manifold.hull(sections);
      if (!(cfg.pin_core_diameter > 0)) return outer;
      const core = Manifold.hull([
        ...octagon(-half - 1, 0.5 * cfg.pin_core_diameter),
        ...octagon(half + 1, 0.5 * cfg.pin_core_diameter),
      ]);
      return consume([outer, core], (a, b) => a.subtract(b));
    }

    /** Plate (and captive ring) of one link (cad.build_twisted_plate_parts). */
    function unit(cfg) {
      validate(cfg);
      const zb = cfg.zPlateBottom;
      const zt = cfg.zPlateTop;
      const alpha = cfg.torsion_angle;
      const alphaDeg = cfg.torsionDegrees;
      let plate = twistedBox({
        zStart: zb, zEnd: zt, zPlateBottom: zb, zPlateTop: zt, alpha,
        width: cfg.ribbon_width, thickness: cfg.body_thickness,
      });
      if (cfg.cutSlits) {
        const rOut = 0.5 * cfg.ringDiameter;
        const uIn = rOut - cfg.ring_width - cfg.ring_gap;
        const uOut = rOut + cfg.ring_gap;
        const slitThickness = Math.max(3 * cfg.body_thickness, 2 * cfg.body_thickness + 2);
        const slits = [-1, 1].map((sign) => twistedBox({
          zStart: zb + cfg.slit_rail_height,
          zEnd: zt - cfg.slit_rail_height,
          zPlateBottom: zb,
          zPlateTop: zt,
          alpha,
          width: uOut - uIn,
          thickness: slitThickness,
          xOffset: sign * 0.5 * (uIn + uOut),
        }));
        plate = consume([plate, ...slits], (body, ...cuts) => Manifold.difference([body, ...cuts]));
      }
      const { bottom, top } = knuckleIntervals(cfg);
      const bottomProfile = bottomKnuckleProfile(cfg);
      const topProfile = topKnuckleProfile(cfg);
      const additions = [
        knuckles(bottomProfile, bottom),
        knuckles(topProfile, top, alphaDeg),
      ];
      bottomProfile.delete();
      topProfile.delete();
      const cuts = [];
      if (cfg.usesPins) {
        const length = 2 * cfg.ribbon_width;
        cuts.push(
          bore({ radius: cfg.fixedBoreRadius, flatHeight: cfg.fixedBoreFlatHeight, zAxis: cfg.zBottomHinge, length }),
          bore({ radius: cfg.freeBoreRadius, flatHeight: cfg.freeBoreFlatHeight, zAxis: cfg.zTopHinge, length, rotationDeg: alphaDeg }),
        );
      } else {
        const size = { radius: cfg.coneRadius, height: cfg.coneHeight };
        additions.push(
          cone({ ...size, faceX: bottom[0][1], direction: 1, zAxis: cfg.zBottomHinge }),
          cone({ ...size, faceX: bottom.at(-1)[0], direction: -1, zAxis: cfg.zBottomHinge }),
        );
        const socket = { ...size, zAxis: cfg.zTopHinge, back: 1, rotationDeg: alphaDeg };
        cuts.push(
          cone({ ...socket, faceX: top[0][0], direction: 1 }),
          cone({ ...socket, faceX: top.at(-1)[1], direction: -1 }),
        );
      }
      plate = consume([plate, ...additions], (...all) => Manifold.union(all));
      if (cuts.length) {
        plate = consume([plate, ...cuts], (body, ...holes) => Manifold.difference([body, ...holes]));
      }
      const parts = { plate };
      if (cfg.with_ring) parts.ring = captiveRing(cfg);
      return parts;
    }

    /**
     * ``body`` with the neighbourhood of one hinge axis removed, so that two
     * links sharing that hinge can be compared away from it (cad_check).
     */
    function awayFromHinge(body, cfg, side) {
      const reach = cfg.hingeRadius + cfg.gap + 1.0;
      const length = 4 * cfg.ribbon_width;
      const z = side === "bottom" ? cfg.zBottomHinge : cfg.zTopHinge;
      let cutter = consume([Manifold.cylinder(length, reach, reach, QUALITY.knuckle, true)], (c) =>
        c.rotate([0, 90, 0]));
      cutter = consume([cutter], (c) => c.translate([0, 0, z]));
      if (side === "top") cutter = consume([cutter], (c) => c.rotate([0, 0, cfg.torsionDegrees]));
      return consume([cutter], (c) => body.subtract(c));
    }

    /** One link with its pin, whole and trimmed at either hinge. */
    function linkBodies(parts, pinSolid, cfg) {
      const all = pinSolid
        ? consume([pinSolid.translate([0, 0, cfg.zBottomHinge])], (moved) => parts.plate.add(moved))
        : parts.plate.translate([0, 0, 0]);
      return {
        all,
        top: awayFromHinge(all, cfg, "top"),
        bottom: awayFromHinge(all, cfg, "bottom"),
        delete() {
          [this.all, this.top, this.bottom].forEach((item) => item.delete());
        },
      };
    }

    return { unit, pin, linkBodies, twistedBox, Manifold };
  }

  // ---------------------------------------------------------------------------
  // Placement of the links along a ring (cad._link_placements)

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const scaled = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const normalised = (a) => scaled(a, 1 / Math.hypot(a[0], a[1], a[2]));

  /**
   * Rigid placement of each link: its bottom hinge axis along binormal i
   * through points[i], its top hinge axis along binormal i + 1.  Returns
   * { alpha, length, origin, x, y, t } per link (alpha is the signed torsion).
   */
  function linkPlacements(points, binormals) {
    const n = points.length - 1;
    const placements = [];
    for (let i = 0; i < n; i += 1) {
      const segment = sub(points[i + 1], points[i]);
      const length = Math.hypot(...segment);
      const t = scaled(segment, 1 / length);
      const b = normalised(binormals[i]);
      const x = normalised(sub(b, scaled(t, dot(b, t))));
      const y = cross(t, x);
      const next = normalised(binormals[i + 1]);
      placements.push({
        alpha: Math.atan2(dot(y, next), Math.max(-1, Math.min(1, dot(x, next)))),
        length,
        origin: points[i],
        x,
        y,
        t,
      });
    }
    return placements;
  }

  /** Column-major matrix taking link (mm) coordinates to ring coordinates. */
  function placementMatrix(placement, scale, zBottomHinge, lift = true) {
    const { x, y, t, origin } = placement;
    const shift = lift ? zBottomHinge * scale : 0;
    return affine(
      [scaled(x, scale), scaled(y, scale), scaled(t, scale)],
      sub(origin, scaled(t, shift)),
    );
  }

  // ---------------------------------------------------------------------------
  // Meshes and file formats

  /** Positions (Float32Array) and indices (Uint32Array) of a manifold. */
  function meshArrays(manifold) {
    const mesh = manifold.getMesh();
    const stride = mesh.numProp;
    const count = mesh.vertProperties.length / stride;
    const positions = new Float32Array(3 * count);
    for (let i = 0; i < count; i += 1) {
      positions[3 * i] = mesh.vertProperties[stride * i];
      positions[3 * i + 1] = mesh.vertProperties[stride * i + 1];
      positions[3 * i + 2] = mesh.vertProperties[stride * i + 2];
    }
    return { positions, indices: new Uint32Array(mesh.triVerts) };
  }

  function transformPoint(m, p) {
    return [
      m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
      m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
      m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ];
  }

  function bounds(meshes) {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const { positions } of meshes) {
      for (let i = 0; i < positions.length; i += 3) {
        for (let axis = 0; axis < 3; axis += 1) {
          lo[axis] = Math.min(lo[axis], positions[i + axis]);
          hi[axis] = Math.max(hi[axis], positions[i + axis]);
        }
      }
    }
    return { lo, hi };
  }

  /** Binary STL of meshes, each optionally moved by a column-major matrix. */
  function stl(meshes, title = "kaleidocycle") {
    const triangles = meshes.reduce((sum, item) => sum + item.indices.length / 3, 0);
    const buffer = new ArrayBuffer(84 + 50 * triangles);
    const view = new DataView(buffer);
    const header = `${title} - Kaleidocycle Studio`.slice(0, 79);
    for (let i = 0; i < header.length; i += 1) view.setUint8(i, header.charCodeAt(i) & 0x7f);
    view.setUint32(80, triangles, true);
    let offset = 84;
    for (const { positions, indices, matrix } of meshes) {
      const vertex = (index) => {
        const p = [positions[3 * index], positions[3 * index + 1], positions[3 * index + 2]];
        return matrix ? transformPoint(matrix, p) : p;
      };
      for (let f = 0; f < indices.length; f += 3) {
        const a = vertex(indices[f]);
        const b = vertex(indices[f + 1]);
        const c = vertex(indices[f + 2]);
        const normal = cross(sub(b, a), sub(c, a));
        const size = Math.hypot(...normal) || 1;
        [...scaled(normal, 1 / size), ...a, ...b, ...c].forEach((value) => {
          view.setFloat32(offset, value, true);
          offset += 4;
        });
        view.setUint16(offset, 0, true);
        offset += 2;
      }
    }
    return buffer;
  }

  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n += 1) {
      let c = n;
      for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
    return table;
  })();

  function crc32(bytes) {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i += 1) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  }

  async function deflateRaw(bytes) {
    if (typeof CompressionStream === "undefined") return null;
    const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  /** A ZIP archive (deflated where the browser can, stored otherwise). */
  async function zip(entries) {
    const encoder = new TextEncoder();
    const local = [];
    const central = [];
    let offset = 0;
    for (const entry of entries) {
      const name = encoder.encode(entry.name);
      const data = typeof entry.data === "string" ? encoder.encode(entry.data) : entry.data;
      const compressed = await deflateRaw(data);
      const method = compressed ? 8 : 0;
      const body = compressed || data;
      const crc = crc32(data);
      const header = new DataView(new ArrayBuffer(30));
      header.setUint32(0, 0x04034b50, true);
      header.setUint16(4, 20, true);
      header.setUint16(6, 0x0800, true);
      header.setUint16(8, method, true);
      header.setUint16(10, 0, true);
      header.setUint16(12, 0x21, true);
      header.setUint32(14, crc, true);
      header.setUint32(18, body.length, true);
      header.setUint32(22, data.length, true);
      header.setUint16(26, name.length, true);
      header.setUint16(28, 0, true);
      local.push(new Uint8Array(header.buffer), name, body);
      const record = new DataView(new ArrayBuffer(46));
      record.setUint32(0, 0x02014b50, true);
      record.setUint16(4, 20, true);
      record.setUint16(6, 20, true);
      record.setUint16(8, 0x0800, true);
      record.setUint16(10, method, true);
      record.setUint16(12, 0, true);
      record.setUint16(14, 0x21, true);
      record.setUint32(16, crc, true);
      record.setUint32(20, body.length, true);
      record.setUint32(24, data.length, true);
      record.setUint16(28, name.length, true);
      record.setUint32(42, offset, true);
      central.push(new Uint8Array(record.buffer), name);
      offset += 30 + name.length + body.length;
    }
    const centralSize = central.reduce((sum, part) => sum + part.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, entries.length, true);
    end.setUint16(10, entries.length, true);
    end.setUint32(12, centralSize, true);
    end.setUint32(16, offset, true);
    const parts = [...local, ...central, new Uint8Array(end.buffer)];
    const total = parts.reduce((sum, part) => sum + part.length, 0);
    const out = new Uint8Array(total);
    let position = 0;
    for (const part of parts) {
      out.set(part, position);
      position += part.length;
    }
    return out;
  }

  const xmlEscape = (text) => String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  /**
   * Vendor-neutral 3MF (core specification) as written by cad._write_3mf_archive:
   * each object is meshed once, centred on the XY origin and resting on z = 0,
   * and placed by build items.  objects: [{ key, name, parts: [{ name, mesh }] }],
   * items: [{ key, x, y }].
   */
  async function threeMF({ title, objects, items }) {
    const resources = [];
    const ids = new Map();
    let nextId = 1;
    const fixed = (value) => (Math.abs(value) < 5e-7 ? 0 : value).toFixed(6);
    for (const object of objects) {
      const { lo, hi } = bounds(object.parts.map((part) => part.mesh));
      const shift = [-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), -lo[2]];
      const meshIds = [];
      for (const part of object.parts) {
        const { positions, indices } = part.mesh;
        const vertices = [];
        for (let i = 0; i < positions.length; i += 3) {
          vertices.push(
            `     <vertex x="${fixed(positions[i] + shift[0])}" y="${fixed(positions[i + 1] + shift[1])}" z="${fixed(positions[i + 2] + shift[2])}"/>`,
          );
        }
        const triangles = [];
        for (let f = 0; f < indices.length; f += 3) {
          triangles.push(`     <triangle v1="${indices[f]}" v2="${indices[f + 1]}" v3="${indices[f + 2]}"/>`);
        }
        const meshName = object.parts.length === 1 ? object.name : `${object.name}_${part.name}`;
        resources.push(
          `  <object id="${nextId}" name="${xmlEscape(meshName)}" type="model">\n` +
            "   <mesh>\n    <vertices>\n" + vertices.join("\n") + "\n    </vertices>\n" +
            "    <triangles>\n" + triangles.join("\n") + "\n    </triangles>\n   </mesh>\n  </object>",
        );
        meshIds.push(nextId);
        nextId += 1;
      }
      if (meshIds.length === 1) {
        ids.set(object.key, meshIds[0]);
      } else {
        resources.push(
          `  <object id="${nextId}" name="${xmlEscape(object.name)}" type="model">\n   <components>\n` +
            meshIds.map((id) => `    <component objectid="${id}"/>`).join("\n") +
            "\n   </components>\n  </object>",
        );
        ids.set(object.key, nextId);
        nextId += 1;
      }
    }
    const build = items.map(({ key, x, y }) =>
      `  <item objectid="${ids.get(key)}" transform="1 0 0 0 1 0 0 0 1 ${x.toFixed(6)} ${y.toFixed(6)} 0"/>`);
    const model = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
      ' <metadata name="Application">kaleidocycle</metadata>',
      ` <metadata name="Title">${xmlEscape(title)}</metadata>`,
      " <resources>",
      ...resources,
      " </resources>",
      " <build>",
      ...build,
      " </build>",
      "</model>",
      "",
    ].join("\n");
    const contentTypes =
      '<?xml version="1.0" encoding="UTF-8"?>\n' +
      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n' +
      ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n' +
      ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n' +
      "</Types>\n";
    const rels =
      '<?xml version="1.0" encoding="UTF-8"?>\n' +
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n' +
      ' <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n' +
      "</Relationships>\n";
    return zip([
      { name: "[Content_Types].xml", data: contentTypes },
      { name: "_rels/.rels", data: rels },
      { name: "3D/3dmodel.model", data: model },
    ]);
  }

  /** Grid of unit and pin positions on the bed (cad._plate_layout). */
  function plateLayout(unitCount, unitWidth, unitDepth, { pinCount = 0, pinLength = 0, pinWidth = 0, spacing = 6, centre = BED_CENTRE } = {}) {
    const grid = (count, width, depth) => {
      const dx = width + spacing;
      const dy = depth + spacing;
      const columns = Math.max(1, Math.ceil(Math.sqrt((count * dy) / Math.max(dx, 1e-6))));
      const rows = Math.ceil(count / columns);
      return Array.from({ length: count }, (_, index) => [
        centre[0] + ((index % columns) - 0.5 * (columns - 1)) * dx,
        centre[1] + (Math.floor(index / columns) - 0.5 * (rows - 1)) * dy,
      ]);
    };
    if (pinCount <= 0) return { units: grid(unitCount, unitWidth, unitDepth), pins: [] };
    const cellWidth = Math.max(unitWidth, pinLength);
    const cellDepth = Math.max(unitDepth, pinWidth);
    const pitch = pinWidth + spacing;
    const perCell = Math.floor((cellDepth - pinWidth) / pitch + 1e-9) + 1;
    const pinCells = Math.ceil(pinCount / perCell);
    const cells = grid(unitCount + pinCells, cellWidth, cellDepth);
    const pins = [];
    let remaining = pinCount;
    for (const [cx, cy] of cells.slice(unitCount)) {
      const k = Math.min(perCell, remaining);
      for (let j = 0; j < k; j += 1) pins.push([cx, cy + (j - 0.5 * (k - 1)) * pitch]);
      remaining -= k;
    }
    return { units: cells.slice(0, unitCount), pins };
  }

  /**
   * One cycle of the ring's motion under an integrable flow, as in
   * cad_check.trace_kaleidocycle_motion: steps until the curvature angles
   * return to their starting values.  ``model`` is a studio Model (it is
   * moved; pass a copy).  Returns { states: [{ points, binormals }], periodic }.
   */
  function traceCycle(model, kind, order, { maxSteps = 6000 } = {}) {
    const angles = () => model.curvatures.map((k) => 2 * Math.atan(k / 2));
    const start = angles();
    const snapshot = () => ({
      points: model.displayVertices(),
      binormals: model.hinges(),
      state: typeof model.snapshot === "function" ? model.snapshot() : null,
    });
    const states = [snapshot()];
    let leftStart = false;
    let previous = 0;
    for (let step = 1; step <= maxSteps; step += 1) {
      model.advanceFlow(kind, order, 1);
      if (step % 12 === 0) model.project(2);
      const now = angles();
      let deviation = 0;
      for (let i = 0; i < now.length; i += 1) {
        const difference = Math.atan2(Math.sin(now[i] - start[i]), Math.cos(now[i] - start[i]));
        deviation = Math.max(deviation, Math.abs(difference));
      }
      leftStart = leftStart || deviation > 0.3;
      if (leftStart && previous < 0.03 && deviation > previous) {
        return { states, periodic: true };
      }
      states.push(snapshot());
      previous = deviation;
    }
    return { states, periodic: false };
  }

  /**
   * Fold angle at each hinge of a pose: the angle, about the shared hinge axis,
   * from link i's axis t_i to the next link's t_(i+1), measured in link i's
   * frame (whose top hinge points along (cos α, sin α, 0)).  Two adjacent links
   * of a ring are placed relative to each other by this angle alone.
   */
  function foldAngles(placements, alpha) {
    const n = placements.length;
    const c = Math.cos(alpha);
    const s = Math.sin(alpha);
    return placements.map((link, i) => {
      const next = placements[(i + 1) % n].t;
      const tx = dot(link.x, next);
      const ty = dot(link.y, next);
      const tz = dot(link.t, next);
      return Math.atan2(tx * s - ty * c, tz);
    });
  }

  /** Placement of the next link, in link coordinates, at fold angle theta. */
  function nextLinkMatrix(cfg, theta) {
    const a = cfg.torsion_angle;
    const h = [Math.cos(a), Math.sin(a), 0];
    const side = cross(h, [0, 0, 1]);
    const t = [side[0] * Math.sin(theta), side[1] * Math.sin(theta), Math.cos(theta)];
    const y = cross(t, h);
    const zb = cfg.zBottomHinge;
    return affine([h, y, t], [-zb * t[0], -zb * t[1], cfg.edge_length + zb - zb * t[2]]);
  }

  /**
   * Smallest gaps between the links of a ring over its motion, as in
   * cad_check.check_kaleidocycle_clearance.
   *
   * Links that do not share a hinge are compared whole in each of ``poses``
   * (lists of placement matrices in millimetres).  Two adjacent links sit
   * relative to each other by the fold angle at their hinge only, so they are
   * compared once per fold angle in ``angles`` (all hinges, all poses), away
   * from the shared hinge: first on a 2-degree grid, then every 0.25 degrees
   * where the grid comes within the angular sampling error of ``limit``.
   */
  async function linkGaps(bodies, cfg, { poses, angles, limit = MIN_CLEARANCE, searchLength = 3, onProgress } = {}) {
    const degree = Math.PI / 180;
    const coarse = 2 * degree;
    const fine = 0.25 * degree;
    // Every point of the moving link lies within this distance of the hinge.
    const reach = Math.hypot(0.5 * cfg.ribbon_width, cfg.edge_length + 2 * cfg.hingeRadius);
    const cells = new Map();
    for (const theta of angles) {
      const key = Math.round(theta / coarse);
      const cell = cells.get(key) || { lo: theta, hi: theta };
      cell.lo = Math.min(cell.lo, theta);
      cell.hi = Math.max(cell.hi, theta);
      cells.set(key, cell);
    }
    const adjacentAt = (theta) => {
      const moved = bodies.bottom.transform(nextLinkMatrix(cfg, theta));
      try {
        return bodies.top.minGap(moved, searchLength);
      } finally {
        moved.delete();
      }
    };
    const n = poses[0].length;
    const total = cells.size + poses.length;
    let done = 0;
    const tick = async () => {
      done += 1;
      if (onProgress && done % 4 === 0) await onProgress(done, total);
    };

    const adjacent = [];
    for (const [key, cell] of cells) {
      const theta = Math.min(cell.hi, Math.max(cell.lo, key * coarse));
      adjacent.push({ theta, gap: adjacentAt(theta), cell });
      await tick();
    }
    const margin = 0.5 * coarse * reach;
    const refined = [];
    for (const sample of adjacent) {
      if (sample.gap >= limit + margin) continue;
      for (let theta = sample.cell.lo; theta <= sample.cell.hi + 1e-12; theta += fine) {
        refined.push({ theta, gap: adjacentAt(theta) });
      }
      refined.push({ theta: sample.cell.hi, gap: adjacentAt(sample.cell.hi) });
    }
    const adjacentSamples = [...adjacent, ...refined];
    const adjacentClosest = adjacentSamples.reduce((best, item) => (!best || item.gap < best.gap ? item : best), null);
    const contactAngles = adjacentSamples.filter((item) => item.gap < limit).map((item) => item.theta);

    const best = new Map();
    for (let p = 0; p < poses.length; p += 1) {
      const all = poses[p].map((matrix) => bodies.all.transform(matrix));
      try {
        for (let i = 0; i < n; i += 1) {
          for (let j = i + 2; j < n; j += 1) {
            if (i === 0 && j === n - 1) continue;
            const gap = all[i].minGap(all[j], searchLength);
            const key = `${i}/${j}`;
            const previous = best.get(key);
            if (!previous || gap < previous.gap) best.set(key, { i, j, gap, pose: p });
          }
        }
      } finally {
        all.forEach((item) => item.delete());
      }
      await tick();
    }
    const pairs = [...best.values()].sort((a, b) => a.gap - b.gap);
    return {
      adjacent: adjacentClosest,
      adjacentContactAngles: contactAngles,
      adjacentSamples: adjacentSamples.length,
      nonAdjacent: pairs[0] || null,
      nonAdjacentContacts: pairs.filter((pair) => pair.gap < limit),
      fineStep: fine,
    };
  }

  return Object.freeze({
    BED_CENTRE,
    CONSTANTS,
    DEFAULTS,
    MIN_CLEARANCE,
    affine,
    bounds,
    config,
    createBuilder,
    crc32,
    knuckleIntervals,
    linkPlacements,
    foldAngles,
    linkGaps,
    meshArrays,
    nextLinkMatrix,
    placementMatrix,
    plateLayout,
    stl,
    threeMF,
    traceCycle,
    transformPoint,
    validate,
    zip,
  });
});
