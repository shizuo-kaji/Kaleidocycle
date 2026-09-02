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
truth. It contains the K7–K9 reference configurations and three generic samples,
including the non-critical, anti-oriented K15 used in
`IntegrableDeformations.ipynb`. The generic configurations were selected from
many feasible seeds using separation, spatial extent and three-dimensionality,
so they read as open rings instead of collapsed tangles. K15 is the default
because it exposes a five-dimensional span of independent hierarchy motions.

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
- `sim.js` — server-independent curvature model, constraints and flows; it can
  also be loaded by Node for mathematical tests.
- `sample-fallback.js` — generated direct-file fallback; do not edit by hand.
- `../scripts/build_web_samples.py` — catalogue/fallback generator.
- `../data/kaleidocycles/` — canonical named JSON configurations and catalogue.
- `vendor/` — local three.js build and licence.

The web editor deliberately replaces the old Rapier rigid-body simulation.
That simulation approximated hinge constraints dynamically; the new editor is
instead designed around the closed constant-torsion locus and the integrable
flows studied in the paper.
