"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

require("./sim.js");

const K = global.Kaleidocycle;
const fixture = (name) =>
  JSON.parse(
    fs.readFileSync(
      path.join(__dirname, "..", "data", "kaleidocycles", name),
      "utf8",
    ),
  );
const close = (actual, expected, tolerance = 2e-11) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) < tolerance,
      `${value} != ${expected[index]} at ${index}`,
    );
  });
};

assert.deepEqual(K.twistedShift([1, 2, 3, 4], 1, -1), [2, 3, 4, -1]);
assert.deepEqual(K.twistedShift([1, 2, 3, 4], -1, -1), [-4, 1, 2, 3]);

const curvature = [0.3, -0.8, 1.2, 0.1, -0.4, 0.7];
close(K.hierarchyField(curvature, 1, -1), [
  -0.051125, 0.522, 0.612, -0.802, 0.312, 0.056125,
]);
close(K.hierarchyField(curvature, 3, -1), [
  -0.13461755703125017,
  -0.0737350375,
  1.801762,
  -1.249866875,
  1.391550875,
  -1.50937488671875,
]);
close(K.hierarchyField(curvature, 6, -1), [
  4.7471517791455025,
  5.211505936536689,
  -7.149141846059457,
  7.973489671670534,
  -9.288617134987161,
  6.641183653821565,
]);
close(K.sineGordonField(curvature, 0.9, -1), [
  -0.26179078142996653,
  -0.4472046829537062,
  -0.36055696628005324,
  0.12783813768758465,
  0.01704436532360291,
  0.12882146645625406,
]);

const model = new K.Model(fixture("generic_k15_noncritical.json"));
const initial = model.diagnostics();
assert.ok(initial.closure < 1e-12);
assert.ok(initial.monodromy < 1e-12);

for (let index = 0; index < 80; index += 1) {
  model.advanceFlow("hierarchy", 3, 0.001);
  if (index % 12 === 0) model.project(2);
}
const evolved = model.diagnostics();
assert.ok(evolved.closure < 1e-7);
assert.ok(evolved.monodromy < 1e-7);
assert.ok(Math.abs(evolved.hamiltonian1 - initial.hamiltonian1) < 1e-9);
assert.ok(Math.abs(evolved.hamiltonian2 - initial.hamiltonian2) < 1e-9);

model.restore();
const originalTorsion = model.torsionAngle;
assert.equal(model.setTorsion(originalTorsion + 0.1), true);
assert.ok(Math.abs(model.torsionAngle - originalTorsion - 0.1) < 1e-12);

const vertex = model.displayVertices()[7];
const target = [vertex[0] + 0.08, vertex[1] - 0.05, vertex[2] + 0.03];
assert.equal(model.dragVertex(7, target), true);
const edited = model.diagnostics();
assert.ok(edited.closure < 1e-6);
assert.ok(edited.monodromy < 1e-6);

// Check standard normalisation and the K12 jump against the certified example.
const crossing = new K.Model(fixture("counterexample_k12_self_crossing.json"));
const twist = 12 * 1.2 / (2 * Math.PI);
close([crossing.diagnostics().writhe, crossing.diagnostics().linking], [4.5 - twist, 4.5]);
for (let index = 0; index < 40; index += 1) {
  crossing.advanceFlow("hierarchy", 1, 0.0001);
}
close([crossing.diagnostics().writhe, crossing.diagnostics().linking], [2.5 - twist, 2.5]);
crossing.restore();
close([crossing.diagnostics().linking], [4.5]);

const certificate = JSON.parse(fs.readFileSync(
  path.join(__dirname, "..", "data", "counterexamples", "mkdv_self_crossing_k12.json"),
  "utf8",
));
crossing.curvatures = certificate.center.slice(0, certificate.n).map(
  (angle) => 2 * Math.tan(Number(angle) / 2),
);
crossing.update();
assert.ok(Number.isNaN(crossing.diagnostics().writhe));
assert.ok(Number.isNaN(crossing.diagnostics().linking));

close([K.gaussWrithe([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])], [0]);
assert.ok(Number.isNaN(K.gaussWrithe(
  [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
)));

// ---------------------------------------------------------------------------
// Exact families (families.js), checked against src/kaleidocycle/elliptic.py.
// ---------------------------------------------------------------------------
require("./families.js");
const F = global.KaleidocycleFamilies;

close([F.ellipticK(0.7), F.ellipticK(0.999)], [1.8456939983747234, 4.49559639584215], 1e-13);
const jacobi = F.jacobiFunctions(1.3, 0.8);
close(
  [jacobi.sn, jacobi.cn, jacobi.dn],
  [0.9055026584496213, 0.4243405890798904, 0.6893776604634267],
  1e-14,
);
const far = F.jacobiFunctions(-7.1, 0.95);
close(
  [far.sn, far.cn, far.dn],
  [0.9751280297859277, -0.2216423369435876, 0.3766106826512598],
  1e-13,
);

// Closing moduli k_{N,m,j} and the torsion branch c_m(k).
close(F.closingModuli(9, 3), [0.7271056052008956], 1e-12);
close(F.closingModuli(13, 6), [0.46793299962768586, 0.8275542333103058], 1e-12);
close(F.closingModuli(11, 5), [0.5457339915717958, 0.9157749012505968], 1e-12);
close([F.ellipticFamily(10, 4, 0.7).torsionCosine], [0.3662931248373162], 1e-13);

// Minimum torsion mu_N / pi from the paper's table.
const paperMinimum = { 7: 0.40455, 8: 0.34427, 9: 0.30101, 12: 0.22043, 21: 0.12369 };
Object.entries(paperMinimum).forEach(([n, value]) => {
  const circle = F.criticalCircle(Number(n), 3, 1);
  assert.ok(Math.abs(circle.torsionAngle / Math.PI - value) < 6e-6, `mu_${n}`);
});

const exact = (hinges) => new K.Model({ n: hinges.length - 1, hinges });
for (const [n, m, level, selfLinking] of [
  [9, 3, 1, 1.5],
  [11, 4, 1, 2],
  [11, 5, 2, 7.5],
  [17, 7, 3, 17.5],
]) {
  const circle = F.criticalCircle(n, m, level);
  for (const phase of [0.1, 1.3]) {
    const ring = exact(circle.hingesAt(phase));
    const d = ring.diagnostics();
    assert.ok(d.closure < 1e-12 && d.monodromy < 1e-12, `closure ${n},${m},${level}`);
    assert.ok(d.criticality < 1e-11, `criticality ${n},${m},${level}`);
    assert.equal(ring.sign, m % 2 === 0 ? 1 : -1);
    close([ring.torsionAngle, d.linking], [circle.torsionAngle, selfLinking], 1e-10);
  }
  assert.deepEqual(circle.inertia, [n - m - 3, m - 3, 1]);

  // Mirror duality: mu* = pi - mu, sigma* = (-1)^N sigma, SL* = N/2 - SL.
  const mirror = F.criticalCircle(n, m, level, { mirror: true });
  const dual = exact(mirror.hingesAt(0.1));
  close(
    [dual.torsionAngle, dual.diagnostics().linking],
    [Math.PI - circle.torsionAngle, n / 2 - selfLinking],
    1e-10,
  );
  assert.equal(dual.sign, (n % 2 === 0 ? 1 : -1) * circle.family.sign);
  assert.deepEqual(mirror.inertia, [m - 3, n - m - 3, 1]);
}

// Right-angle circles: the classical K6 and the energy minima of the paper.
for (const [n, level, energy] of [[6, 1, 3 * Math.log(4)], [8, 1, 4 * Math.log(2)]]) {
  const circle = F.criticalCircle(n, n / 2, level);
  const ring = exact(circle.hingesAt(0.4));
  const d = ring.diagnostics();
  assert.ok(d.closure < 1e-12 && d.criticality < 1e-11);
  close([ring.torsionAngle, d.hamiltonian1], [Math.PI / 2, energy], 1e-12);
  assert.equal(ring.sign, (n / 2) % 2 === 0 ? 1 : -1);
}
assert.equal(F.closingLevelCount(10, 5), 2);
assert.equal(F.closingLevelCount(9, 3), 1);

// Explicit torus-knot ring: SL = cells * turns, and its mirror.
const knot = exact(F.torusKnotHinges(5, 2));
assert.equal(knot.sign, 1);
close([knot.diagnostics().linking], [10], 1e-10);
assert.ok(knot.curvatures.every((value) => value > 0));
const knotMirror = exact(F.torusKnotHinges(5, 2, { mirror: true }));
assert.equal(knotMirror.sign, -1);
close([knotMirror.diagnostics().linking], [7.5 - 10], 1e-10);
assert.throws(() => F.torusKnotHinges(6, 2));

// Phase changes keep N, sigma and the frame gauge, so setHinges applies.
const principal = F.criticalCircle(9, 3, 1);
const moving = exact(principal.hingesAt(0));
moving.setHinges(principal.hingesAt(2.0));
close(moving.curvatures, principal.family.curvatures(2.0), 1e-10);

// The generic sample is far from the critical locus.
assert.ok(model.diagnostics().criticality > 0.1);

// Horn alignment recovers a known rotation.
const cloud = [[1, 0.2, -0.3], [-0.4, 1.1, 0.5], [0.3, -0.9, 0.8], [-0.9, -0.4, -1]];
const angle = 2.1;
const [c, s] = [Math.cos(angle), Math.sin(angle)];
const rotation = [c, -s, 0, s, c, 0, 0, 0, 1];
const turned = cloud.map(([x, y, z]) => [c * x - s * y, s * x + c * y, z]);
close(K.bestRotation(cloud, turned), rotation, 1e-12);

// Every catalogue sample loads; the exact critical ones are critical.
const catalogue = fixture("catalog.json");
catalogue.samples.forEach((entry) => {
  const sample = new K.Model(fixture(entry.file));
  const d = sample.diagnostics();
  assert.ok(d.closure < 1e-6, entry.name);
  if (entry.kind === "critical") assert.ok(d.criticality < 1e-10, entry.name);
});

console.log("web/sim.js: all mathematical checks passed");
