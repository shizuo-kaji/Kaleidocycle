# Kaleidocycle Studio

**[Open the hosted studio](https://shizuo-kaji.github.io/Kaleidocycle/)**

Kaleidocycle Studio is a local, interactive laboratory for closed,
constant-torsion kaleidocycles. It reconstructs the ring from Cayley curvature
coordinates, evolves it with the integrable hierarchy and keeps direct edits on
the closed-configuration locus.

The interface uses British English throughout.

## Run it without a server

Open [`index.html`](index.html) directly in a modern browser. A `file://` URL is
fully supported; no HTTP server, package installation or network connection is
required.

This is possible because:

- three.js is vendored in `vendor/` rather than imported from a CDN;
- the application uses ordinary local scripts rather than ES modules;
- the generated `sample-fallback.js` mirrors the canonical JSON catalogue; and
- arbitrary JSON files are read with `FileReader`, by file picker or drop.

Serving the repository root over HTTP is recommended while developing: in that
mode the studio reads each built-in directly from `data/kaleidocycles/`.

## Integrable motion

The state variable is the Cayley curvature

```text
kappa[n] = 2 tan(phi[n] / 2),
```

with the twisted boundary condition `x[n + N] = sign x[n]`. The application
implements the same positive semi-discrete mKdV hierarchy as
`src/kaleidocycle/integrable.py`:

- `X(1)` and `X(2)` use the explicit local formulae;
- every higher `X(j)` is generated from the exact gradient of the logarithmic
  twisted-Floquet Hamiltonian and the common Poisson operator; and
- the order control runs from 1 through the finite Cayley–Hamilton bound `N`.

The semi-discrete sine–Gordon flow is also available for anti-oriented cycles.
Time integration uses RK4. The time step is automatically reduced when a high
hierarchy field is large, and a small periodic constraint correction controls
floating-point drift. The displayed `t` is the actual integrated flow time.

The Diagnostics panel updates **Wr** (standard Gauss writhe) and the
normalised self-linking **SL = Wr + Nμ/(2π)** with each flow step or edit;
`2·SL` is the number of half-twists of the hinge ribbon. Anti-oriented cycles
have half-integer SL. Both values display `—` at a detected self-contact or
when the closure error exceeds `1e-6`; the normalisation matches
`SelfCrossing.ipynb` and the paper. (Earlier versions labelled the same
quantity “Lk”.)

**Criticality** is the relative residual of the stationary recurrence

```text
D[n] (kappa[n-1] + kappa[n+1]) = lambda kappa[n],   lambda = 2 cos(mu) + E2 / N,
```

with `D[n] = 1 + kappa[n]^2 / 4`. It vanishes exactly at torsion-critical
configurations and is of order one for generic rings.

## Model lab: exact families

The Model lab builds exact configurations in the browser (`families.js`), with
no optimisation:

- **Critical circle.** The Jacobi elliptic family
  `kappa[n] = A cn(omega + n delta; k)` with `delta = 2 K(k) m / N`. For
  `3 ≤ m < N/2` the studio computes the closing angle `Theta(k)` of the frame
  monodromy, which decreases from 0 to `−m pi`, and solves `Theta(k) = −2 pi j`
  for each closing index `j = 1, …, ⌊(m − 1)/2⌋`. For `m = N/2` it uses the
  right-angle circles `k = sin(2 pi j / N)`, `1 ≤ j < N/4`. The information line
  shows `k`, `mu`, `sigma = (−1)^m` and the torsion Hessian inertia
  `(N − m − 3, m − 3, 1)`: `m = 3` is the minimum (**Principal family**),
  `m = N − 3` the maximum, and everything in between is a saddle. **Mirror dual**
  gives `mu* = pi − mu` and winding `N − m`.
- **Torus knot.** The explicit oriented ring with `N = 3M` hinges and
  `M`-fold symmetry. Its centre line is the `(turns, M)` torus knot and its
  self-linking `M · turns` grows quadratically with `N`.

The **Phase on the critical circle** slider moves `omega` over one period `4K`.
This traverses the critical circle exactly, which the first mKdV flow `X(1)`
also does, up to time reparametrisation. The JavaScript elliptic functions use
the AGM and the descending Landen transformation. Its closing moduli agree with
`src/kaleidocycle/elliptic.py` to about `1e-15`.

The sample menu, the closing index and the mirror setting are recorded in the
URL hash (`#sample=…`, `#critical=N,m,j&phase=…`, `#knot=M,turns`, optionally
`&mirror=1`), so **Copy link** (⧉) reproduces the current model.

## Analysis plots

- **Curvature**: the current `kappa[n]` as bars, with the initial state as
  ticks. Hierarchy flows move these values as travelling waves.
- **Phase plane**: the pairs `(kappa[n], kappa[n+1])` together with the QRT
  biquadric `(1 + x²/4)(1 + y²/4) − (lambda/4) x y = J`. At a critical
  configuration every pair lies on the curve; generic rings scatter off it.
- **Closing Θ(k)**: the closing-angle curve for the Model lab's `N` and `m`,
  with its crossings of `−2 pi j` and the current model highlighted.

## Mirror duality and display tools

**Mirror** (key `M`) applies `B[n] ↦ (−1)^n B[n]`: `mu ↦ pi − mu`,
`sigma ↦ (−1)^N sigma` and `SL ↦ N/2 − SL`. The torsion Hessian inertia swaps its
first two entries.

The Appearance card can colour links along the ring, by curvature (a diverging
map of `kappa[n]` at each hinge), alternately, or in one colour. **Remove rigid
drift** shows each frame by the rotation that best matches the previous one
(Horn's quaternion method), so a motion is seen without its overall
spin. This only affects the display; exported JSON is unchanged.

The view tools reset the camera (`V`), toggle turntable rotation (`A`), save a
PNG, record a WebM/MP4 clip of the canvas, copy a link, and list the keyboard
shortcuts (`?`). `Space` runs or pauses the flow, `.` takes a single step, `B`
reverses it, `[`/`]` change the hierarchy order, `R` resets the shape, and
`P` shows the printable links.

## 3D-printable links

The **3D print** card builds the twisted-plate links of
`src/kaleidocycle/cad.py` in the browser and shows the assembled ring in place
of the tetrahedra (**Show printable links**, `P`). The links follow the flow,
the phase slider and vertex editing; a change of torsion (the slider, the Model
lab or another sample) rebuilds them. All N links share the current torsion
angle `mu` and the print settings:

- ribbon width, edge length, plate thickness (2.4 mm by default), pin or
  snap-in cone hinges, and the captive print-in-place ring;
- the **hinge fit**: pin hold (0.09 mm), pin turn (0.3 mm), knuckle gap (empty
  for 0.4 mm with pins, 0.2 mm with cones) and plate-edge gap (0.4 mm). These
  are the `pin_fit_clearance`, `pin_clearance`, `hinge_gap` and `gap` of
  `TwistedPlateConfig`, with the same defaults.

**Print plate** saves a vendor-neutral 3MF with all N links and their pins
laid out around the centre of a 256 mm bed, **Link STL** one link in its
printing orientation, and **Assembly** the assembled ring in its current pose
as STL. STEP files need the Python package.

**Check contacts** traces one cycle of the selected flow from the current pose
and measures, as `check_kaleidocycle_clearance` does, the smallest gap between
links that do not share a hinge (in up to 72 poses) and between adjacent links
away from their shared hinge. Two adjacent links are placed relative to each
other by the fold angle at their hinge alone, so that gap is evaluated for
every fold angle that occurs in the cycle: on a 2° grid, refined to 0.25°
wherever it comes within the sampling error of the 0.4 mm limit. Links in
contact turn red, and **Closest pose** moves the ring to the closest approach.
The fit of the hinge itself is checked by the Python package.

The geometry engine, manifold-3d (Apache-2.0), loads on first use from
`vendor/manifold-classic.js`. That file is the npm package's `manifold.js` and
`manifold.wasm`, repackaged by `python scripts/vendor_manifold.py` as a classic
script with the WebAssembly embedded, so the card also works from `file://`.
The links agree with the build123d models to 0.2 % in volume and 0.02 mm in
bounding box, and the contact check reproduces the Python results
(`node test_print.js`).

## Interactive initial geometry

The geometry editor operates on curvature coordinates rather than moving mesh
vertices independently.

### Pulling a vertex

Select a white centre-line handle and drag it in the screen-facing plane. At
each update the editor:

1. differentiates closure and monodromy with respect to all curvatures;
2. constructs the tangent-space projector of those six constraints;
3. solves for the closest motion of the selected vertex within that tangent
   space; and
4. projects the result back onto exact closure and monodromy.

The selected point follows the pointer only as far as the local configuration
space permits. The rest of the ring moves coherently and constant torsion is
preserved by reconstruction.

### Changing torsion

The torsion-angle slider changes `mu` by continuation. Each small change in
`mu` is followed by a least-norm curvature correction. If the requested value
leaves the current local branch, the previous valid configuration is restored
and the interface reports that the limit was reached.

After editing, **Use as initial** makes the current shape the reset state.
**Export JSON** writes the constrained configuration in the package's usual
`hinges`, `tangents` and `curve` format.

## Built-in data

The repository-level `data/kaleidocycles/` directory is the single source of
truth. It contains exact torsion-critical and knotted models (below), the
K7–K9 Möbius configurations, a counterexample, an explicit theta solution and
three generic samples, including the non-critical, anti-oriented K15 used in
`IntegrableDeformations.ipynb`. The generic configurations were selected from
many feasible seeds using separation, spatial extent and three-dimensionality,
so they read as open rings instead of collapsed tangles. K15 is the default
because it exposes a five-dimensional span of independent hierarchy motions.

The menu groups the built-ins by kind. The featured exact models, regenerated by
`python scripts/build_featured_samples.py`, are:

| Sample | What it shows |
| --- | --- |
| Classical K6 · right-angle circle | The hexagonal toy: `N = 6`, `mu = pi/2`, the right-angle circle `m = 3`, `j = 1`; half-width `1/√2` gives regular tetrahedra |
| Principal K7 · minimum torsion | The global torsion minimum `mu_7 = 0.40455 pi`, inertia `(1, 0, 1)` |
| Principal K48 · continuum limit | `k ≈ k_inf`, `N mu ≈ C_inf` |
| Elliptic K11 · oriented saddle (m = 4) | Inertia `(4, 1, 1)` |
| Elliptic K11 · (2,5) torus knot | `m = 5`, `j = 2`, `SL = 15/2` |
| Elliptic K17 · (3,7) torus knot | `m = 7`, `j = 3`, `SL = 35/2` |
| Right-angle K8 · oriented energy minimum | `E1 = 4 log 2`, two-dimensional fixed-torsion fibre |
| Right-angle K10 · energy-minimising circle | Energy inertia `(4, 0, 1)`, four-dimensional motion fibre |
| Torus-knot K15 · SL = 10 | Explicit oriented `(2,5)` torus knot; its mirror has `SL = −5/2` |

The knot types of the `j ≥ 2` elliptic branches are numerical identifications
(Alexander polynomials of a generic projection), not theorems. The optimised
Möbius K7–K9 samples are the mirrors of the principal family. Their torsion
`pi − mu_N` is the maximum for their monodromy, and their descriptions now say
so.

Select **Knotted K15 · explicit theta solution** for the knotted,
anti-oriented K15 from `ExplicitConstructionWithTheta.ipynb` (`N = 15`,
`m = 5`, `t = 0`, `cos(mu) ≈ 0.649663`). It is the `j = 2` closing branch of
the `m = 5` cn family, and numerically a `(2,5)` torus knot. Its canonical sample is
`explicit_k15_knotted.json`; the generating theta parameters are recorded in
`metadata.theta_parameters`. Reconstruct it with
`generate_theta_binormals(v, z, r, y, N, t=t, gamma=gamma)` and
`Kaleidocycle(hinges=binormals, oriented=False)`; using the binormals directly
preserves the frame at the zero-curvature vertices of this initial state.

With a server running from the repository root, the browser loads
`data/kaleidocycles/catalog.json` and its JSON entries directly. Direct-file
mode cannot fetch neighbouring JSON under browser security rules, so it uses
`sample-fallback.js`, a generated mirror of exactly the same files.

Notebook code can load or promote a shared sample with:

```python
from kaleidocycle import export_sample, load_sample

cycle = load_sample("generic_k15_noncritical")
export_sample(cycle, "my_named_sample", metadata={"kind": "generic"})
```

After adding or changing a canonical file, rebuild the catalogue and direct-file
fallback:

```bash
python scripts/build_web_samples.py
```

## Files

- `index.html` — British-English interface and responsive styling.
- `viewer.js` — three.js rendering, camera gestures, picking and UI wiring.
- `sim.js` — server-independent curvature model, constraints, flows,
  criticality diagnostics and display alignment; it can also be loaded by Node
  for mathematical tests.
- `families.js` — Jacobi elliptic functions, closing moduli, critical circles,
  right-angle circles, torus-knot rings and mirror duality.
- `printmodel.js` — printable twisted-plate links, pins and captive rings (the
  geometry of `cad.py` on manifold-3d), link placement along the ring,
  contact check, and STL and 3MF writers.
- `test_sim.js` — Node checks of both scripts against the Python package
  (`node test_sim.js`).
- `test_print.js` — Node checks of `printmodel.js` against build123d and the
  Python contact check (`node test_print.js`).
- `../scripts/build_featured_samples.py` — exact featured samples.
- `sample-fallback.js` — generated direct-file fallback; do not edit by hand.
- `../scripts/build_web_samples.py` — catalogue/fallback generator.
- `../data/kaleidocycles/` — canonical named JSON configurations and catalogue.
- `vendor/` — local three.js and manifold-3d builds and their licences.
- `../scripts/vendor_manifold.py` — repackages manifold-3d for `file://` use.

The web editor deliberately replaces the old Rapier rigid-body simulation.
That simulation approximated hinge constraints dynamically; the new editor is
instead designed around the closed constant-torsion locus and the integrable
flows studied in the paper.

## Self-crossing counterexample

Select **Self-crossing K12 · first mKdV flow** in the studio's built-in menu.
This loads `counterexample_k12_self_crossing`, an embedded pre-contact
configuration at `t = -0.002` on the certified K12 first-flow orbit. Evolve it
with `X(1)` to explore the crossing.

`notebooks/SelfCrossing.ipynb` executes the interval certificate and provides
static diagnostics and an interactive view. Rebuild the shared sample and
studio's offline fallback with `python scripts/build_self_crossing.py` from
`data/counterexamples/mkdv_self_crossing_k12.json`.
