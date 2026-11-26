#!/usr/bin/env node
/*
 * Node checks of printmodel.js (the 3D-print links of Kaleidocycle Studio)
 * against the build123d models of src/kaleidocycle/cad.py:
 *
 *   node web/test_print.js
 *
 * Reference volumes and bounding boxes were computed with build123d; the
 * clearance references are those of cad_check.check_kaleidocycle_clearance.
 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const zlib = require("zlib");

const here = __dirname;
vm.runInThisContext(fs.readFileSync(path.join(here, "vendor", "manifold-classic.js"), "utf8"), {
  filename: "manifold-classic.js",
  importModuleDynamically: vm.constants.USE_MAIN_CONTEXT_DEFAULT_LOADER,
});
require(path.join(here, "sim.js"));
const Print = require(path.join(here, "printmodel.js"));
const { Model } = globalThis.Kaleidocycle;
const sample = (name) => JSON.parse(
  fs.readFileSync(path.join(here, "..", "data", "kaleidocycles", `${name}.json`), "utf8"),
);

// build123d volumes (mm³) and bounding boxes (mm) with the current defaults.
const K7 = 1.2709341212741452;
const REFERENCE = {
  pin_k7: {
    config: { torsion_angle: K7 },
    plate: { volume: 3197.76, min: [-21.53, -20.9, 0], max: [21.53, 20.9, 35.7] },
    pin: { volume: 385.19, min: [-21.5, -1.85, -1.85], max: [21.5, 1.85, 1.85] },
  },
  pin_60deg_w36: {
    config: { torsion_angle: Math.PI / 3, ribbon_width: 36, edge_length: 24, num_knuckles: 3 },
    plate: { volume: 2264.02, min: [-18.04, -16.19, 0], max: [18.04, 16.19, 30.7] },
    pin: { volume: 322.03, min: [-18, -1.85, -1.85], max: [18, 1.85, 1.85] },
  },
  cone_k7: {
    config: { torsion_angle: K7, hinge_type: "cone" },
    plate: { volume: 3421.65, min: [-21.53, -20.9, 0], max: [21.53, 20.9, 34.7] },
  },
  ring_k7: {
    config: { torsion_angle: K7, with_ring: true },
    plate: { volume: 2950.08, min: [-21.53, -20.9, 0], max: [21.53, 20.9, 35.7] },
    ring: { volume: 167.13, min: [-8.6, -8.6, 9.6], max: [8.6, 8.6, 11.6] },
  },
  pin_mirror: {
    config: { torsion_angle: Math.PI - K7 },
    plate: { volume: 3197.77, min: [-21.53, -21.53, 0], max: [21.53, 21.53, 35.7] },
  },
};

const close = (actual, expected, tolerance, label) => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual} differs from ${expected} by more than ${tolerance}`,
  );
};

function checkConfig() {
  const cfg = Print.config();
  close(cfg.fixedBoreRadius, 2.09, 1e-12, "fixed bore");
  close(cfg.freeBoreRadius, 2.3, 1e-12, "free bore");
  close(cfg.freeBoreFlatHeight, 2.6, 1e-12, "free bore flat");
  assert.strictEqual(cfg.hingeGap, 0.4);
  assert.strictEqual(Print.config({ hinge_type: "cone" }).hingeGap, 0.2);
  assert.strictEqual(Print.config({ hinge_gap: 0.3 }).hingeGap, 0.3);
  Print.validate(cfg);
  assert.throws(() => Print.validate(Print.config({ pin_clearance: 0.6 })), /wall/);
  assert.throws(() => Print.validate(Print.config({ edge_length: 6 })), /edge length/);
  assert.throws(() => Print.config({ nonsense: 1 }), /unknown/);
  const { bottom, top } = Print.knuckleIntervals(cfg);
  assert.strictEqual(bottom.length, 3);
  assert.strictEqual(top.length, 2);
}

function checkParts(builder) {
  for (const [name, reference] of Object.entries(REFERENCE)) {
    const cfg = Print.config(reference.config);
    const parts = builder.unit(cfg);
    if (cfg.usesPins) parts.pin = builder.pin(cfg);
    for (const [part, expected] of Object.entries(reference)) {
      if (part === "config") continue;
      const solid = parts[part];
      assert.ok(solid, `${name}: missing ${part}`);
      assert.strictEqual(solid.status(), "NoError", `${name}.${part} is not a valid solid`);
      // Circles are polygons with 64-128 sides: volumes agree to 0.2 %.
      close(solid.volume() / expected.volume, 1, 2e-3, `${name}.${part} volume`);
      const box = solid.boundingBox();
      for (let axis = 0; axis < 3; axis += 1) {
        close(box.min[axis], expected.min[axis], 0.02, `${name}.${part} min[${axis}]`);
        close(box.max[axis], expected.max[axis], 0.02, `${name}.${part} max[${axis}]`);
      }
    }
    Object.values(parts).forEach((solid) => solid.delete());
  }
}

function placementsOf(state, scale) {
  return Print.linkPlacements(state.points.map((p) => p.map((v) => v * scale)), state.binormals);
}

function checkPlacements(trace, cfg) {
  const placements = placementsOf(trace.states[123], cfg.edge_length);
  const n = placements.length;
  placements.forEach((placement) => {
    close(placement.length, cfg.edge_length, 1e-9, "edge length");
    close(placement.alpha, cfg.torsion_angle, 1e-9, "torsion");
  });
  // nextLinkMatrix reproduces the relative placement of neighbouring links;
  // the closing hinge of an anti-oriented ring differs by the link's 180°
  // symmetry about its axis.
  const angles = Print.foldAngles(placements, cfg.torsion_angle);
  const matrix = (p) => Print.placementMatrix(p, 1, cfg.zBottomHinge);
  const apply = (m, p) => Print.transformPoint(m, p);
  const probe = [[7.5, 1.2, 3.9], [-12, 0.4, 20], [3, -2, 30]];
  placements.forEach((placement, i) => {
    const next = placements[(i + 1) % n];
    const relative = Print.nextLinkMatrix(cfg, angles[i]);
    probe.forEach((point) => {
      const viaRelative = apply(matrix(placement), apply(relative, point));
      const flipped = [-point[0], -point[1], point[2]];
      const direct = [apply(matrix(next), point), apply(matrix(next), flipped)];
      const error = Math.min(...direct.map((q) => Math.hypot(
        q[0] - viaRelative[0], q[1] - viaRelative[1], q[2] - viaRelative[2],
      )));
      close(error, 0, 1e-8, `relative placement at hinge ${i}`);
    });
  });
}

async function clearance(builder, data, overrides) {
  const trace = Print.traceCycle(new Model(data), "hierarchy", 2);
  assert.ok(trace.periodic, "the flow returns to its start");
  const alpha = placementsOf(trace.states[0], 1)[0].alpha;
  const cfg = Print.config({ torsion_angle: alpha, ...overrides });
  const parts = builder.unit(cfg);
  const pin = builder.pin(cfg);
  const bodies = builder.linkBodies(parts, pin, cfg);
  const stride = Math.ceil(trace.states.length / 48);
  const poses = trace.states.filter((_, i) => i % stride === 0).map((state) =>
    placementsOf(state, cfg.edge_length).map((p) => Print.placementMatrix(p, 1, cfg.zBottomHinge)));
  const angles = trace.states.flatMap((state) => Print.foldAngles(placementsOf(state, 1), alpha));
  try {
    return { trace, cfg, ...(await Print.linkGaps(bodies, cfg, { poses, angles })) };
  } finally {
    bodies.delete();
    parts.plate.delete();
    pin.delete();
  }
}

async function checkClearance(builder) {
  // Python: principal K7, 43 mm: non-adjacent 0.66 mm, adjacent > 3 mm.
  const principal = await clearance(builder, sample("principal_k7_minimum"), {});
  checkPlacements(principal.trace, principal.cfg);
  close(principal.nonAdjacent.gap, 0.66, 0.02, "principal K7 non-adjacent gap");
  assert.ok(principal.adjacent.gap >= 3, "principal K7 adjacent links stay apart");
  assert.strictEqual(principal.nonAdjacentContacts.length, 0);
  assert.strictEqual(principal.adjacentContactAngles.length, 0);
  // Python: mirror principal K7 (Möbius sample), 36 mm: the seven adjacent
  // pairs touch away from their hinge; non-adjacent links stay > 3 mm apart.
  const mirror = await clearance(builder, sample("mobius_k7_oriented"), { ribbon_width: 36 });
  assert.ok(mirror.adjacentContactAngles.length > 0, "mirror K7 adjacent contact");
  assert.ok(mirror.adjacent.gap < 0.4);
  assert.ok(mirror.nonAdjacent.gap >= 3);
}

function inflate(archive) {
  const entries = {};
  let offset = 0;
  while (archive.readUInt32LE(offset) === 0x04034b50) {
    const method = archive.readUInt16LE(offset + 8);
    const crc = archive.readUInt32LE(offset + 14);
    const size = archive.readUInt32LE(offset + 18);
    const nameLength = archive.readUInt16LE(offset + 26);
    const name = archive.subarray(offset + 30, offset + 30 + nameLength).toString();
    const body = archive.subarray(offset + 30 + nameLength, offset + 30 + nameLength + size);
    const data = method === 8 ? zlib.inflateRawSync(body) : body;
    assert.strictEqual(Print.crc32(data), crc, `${name} CRC`);
    entries[name] = data.toString();
    offset += 30 + nameLength + size;
  }
  return entries;
}

async function checkFiles(builder) {
  const cfg = Print.config({ torsion_angle: K7 });
  const plate = builder.unit(cfg).plate;
  const pin = builder.pin(cfg);
  const plateMesh = Print.meshArrays(plate);
  const pinMesh = Print.meshArrays(pin);

  // Binary STL: header, triangle count and record size.
  const stl = Buffer.from(Print.stl([plateMesh, { ...pinMesh, matrix: Print.affine([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 5]) }]));
  const triangles = (plateMesh.indices.length + pinMesh.indices.length) / 3;
  assert.strictEqual(stl.readUInt32LE(80), triangles);
  assert.strictEqual(stl.length, 84 + 50 * triangles);

  // Plate layout as cad._plate_layout (7 units 43.06 × 41.81 mm, 7 pins).
  const layout = Print.plateLayout(7, 43.06, 41.81, { pinCount: 7, pinLength: 43, pinWidth: 3.7 });
  const units = [[78.94, 80.19], [128, 80.19], [177.06, 80.19], [78.94, 128], [128, 128], [177.06, 128], [78.94, 175.81]];
  const pins = [[128, 161.26], [128, 170.96], [128, 180.66], [128, 190.36], [177.06, 166.11], [177.06, 175.81], [177.06, 185.51]];
  layout.units.forEach((p, i) => p.forEach((v, k) => close(v, units[i][k], 1e-9, "unit position")));
  layout.pins.forEach((p, i) => p.forEach((v, k) => close(v, pins[i][k], 1e-9, "pin position")));

  // 3MF: one object per distinct part, one build item per unit and pin, and
  // every mesh closed (each edge shared by two triangles in opposite order).
  const archive = Buffer.from(await Print.threeMF({
    title: "test",
    objects: [
      { key: "unit", name: "twisted_plate", parts: [{ name: "plate", mesh: plateMesh }] },
      { key: "pin", name: "hinge_pin", parts: [{ name: "pin", mesh: pinMesh }] },
    ],
    items: [
      ...layout.units.map(([x, y]) => ({ key: "unit", x, y })),
      ...layout.pins.map(([x, y]) => ({ key: "pin", x, y })),
    ],
  }));
  const entries = inflate(archive);
  assert.deepStrictEqual(Object.keys(entries), ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]);
  const model = entries["3D/3dmodel.model"];
  assert.ok(model.includes('<metadata name="Application">kaleidocycle</metadata>'));
  assert.strictEqual((model.match(/<object /g) || []).length, 2);
  assert.strictEqual((model.match(/<item /g) || []).length, 14);
  for (const object of model.split("<object ").slice(1)) {
    const zs = [...object.matchAll(/z="([-\d.]+)"/g)].map((m) => Number(m[1]));
    close(Math.min(...zs), 0, 1e-6, "object rests on the bed");
    const edges = new Map();
    for (const [, a, b, c] of object.matchAll(/v1="(\d+)" v2="(\d+)" v3="(\d+)"/g)) {
      [[a, b], [b, c], [c, a]].forEach(([u, v]) => {
        const key = `${u}>${v}`;
        assert.ok(!edges.has(key), "consistently oriented triangles");
        edges.set(key, true);
      });
    }
    for (const key of edges.keys()) {
      const [u, v] = key.split(">");
      assert.ok(edges.has(`${v}>${u}`), "closed mesh");
    }
  }
  plate.delete();
  pin.delete();
}

(async () => {
  const wasm = await globalThis.KaleidocycleManifold.load();
  const builder = Print.createBuilder(wasm);
  checkConfig();
  checkParts(builder);
  await checkFiles(builder);
  await checkClearance(builder);
  console.log("web/printmodel.js: all print-model checks passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
