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

console.log("web/sim.js: all mathematical checks passed");
