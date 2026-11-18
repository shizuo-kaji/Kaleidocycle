# Kaleidocycle Python Toolkit

This repository ports the Maple, Mathematica, and MATLAB kaleidocycle code into
a Python package. The package works with an N-kaleidocycle as `N + 1`
unit hinge vectors, also called binormals, in `R^3`. Adjacent hinges have
constant torsion, meaning their pairwise inner products are constant. Oriented
cycles end with the same hinge direction they started with; non-oriented cycles
end with the opposite direction.

The current Python package includes geometry, integrable curvature flows,
energies, constraints, solvers, theta-function constructions, animation tools,
reports, I/O, and plotting.

## Install

```bash
python -m pip install -e ".[dev]"
```

JAX is included in the development dependencies. For a smaller runtime install
with automatic differentiation support, use `python -m pip install -e ".[jax]"`.

Run the test suite with:

```bash
pytest
```

## Notebooks

Interactive examples live in `notebooks/`. Use `%matplotlib widget` for 3D
interaction and make sure the notebook kernel points at the project environment.

- `FindKaleidocycles.ipynb` explores optimization-based construction.
- `ExplicitConstructionWithTheta.ipynb` demonstrates analytic theta-function
  solutions.
- `Animation.ipynb` explores animation utilities.
- `IntegrableDeformations.ipynb` demonstrates the positive mKdV hierarchy,
  its spectral Hamiltonians and lifts, the anti-periodic sine--Gordon flow,
  and numerical conservation.
- `IntegrableTheorems.ipynb` is an interactive tour of Theorems A–E of
  *Integrable deformations of Kaleidocycles*: dimension counts and rank loss,
  commuting flows, the QRT curve, the torsion index of the elliptic families,
  the global minimum and nearby fibres, and the continuum limit.
- `SelfCrossing.ipynb` certifies a K12 self-crossing under the first mKdV flow,
  with static diagnostics and an interactive 3D time slider.
- `KaleidocycleProperties.ipynb` inspects structural properties and residuals.
- `AnimationScalarProperties.ipynb` computes and plots animation diagnostics.
- `BackendComparison.ipynb` compares NumPy and optional JAX solver backends.
- `LocalDoF.ipynb` analyzes infinitesimal and finite degrees of freedom of the
  constraint-preserving motion at a given configuration.
- `CADModelGeneration.ipynb` demonstrates parametric 3D-printable twisted-plate
  CAD export (`.stl`, `.step`, `.3mf`) using optional `build123d`.

## Interactive web studio

Select **Self-crossing K12 · first mKdV flow** in the studio's built-in menu
for the pre-contact counterexample (`counterexample_k12_self_crossing`). Its
certified decimal inputs are in `data/counterexamples/`, with the certificate
and interactive diagnostics in `notebooks/SelfCrossing.ipynb`. Rebuild the
shared sample and offline fallback with `python scripts/build_self_crossing.py`.

`web/` contains a three.js studio for the integrable hierarchy and constrained
editing of closed constant-torsion kaleidocycles. When it is served over HTTP,
the built-in menu reads the same canonical JSON files in
`data/kaleidocycles/` that notebooks load through `load_sample()`.

**[Open Kaleidocycle Studio](https://shizuo-kaji.github.io/Kaleidocycle/)**

```bash
python -m http.server 8000   # from this directory
```

then open <http://localhost:8000/web/>. Directly opening `web/index.html` also
works, using a generated static fallback because browsers do not permit a
`file://` page to fetch neighbouring JSON files. See `web/README.md` for details.

### Shared notebook/web samples

```python
from kaleidocycle import export_sample, load_sample

cycle = load_sample("generic_k15_noncritical")

# Promote a notebook result to the canonical built-in catalogue.
export_sample(
    cycle,
    "my_named_sample",
    metadata={"kind": "generic", "non_critical": True},
)
```

`export_json()` now writes a top-level `name` too; an explicit `name=...` wins,
then `Kaleidocycle.name`, and finally the output filename is used. After adding
or changing canonical samples, run `python scripts/build_web_samples.py` to
refresh `catalog.json` and the generated fallback used only in direct-file mode.

## Package Layout

- `src/kaleidocycle/geometry.py`: hinge frames, curves, curvature, torsion,
  writhe, twist, linking number, transformations between curves, tangents, and
  binormals, plus the `Kaleidocycle` class.
- `src/kaleidocycle/animation.py`: sine-Gordon flow, step/random animation
  generation, frame cleaning, alignment, sorting, curve conversion, and
  `KaleidocycleAnimation`.
- `src/kaleidocycle/integrable.py`: twisted curvature coordinates, discrete
  Frenet reconstruction, the positive mKdV hierarchy and compatible lifts,
  twisted-Floquet Hamiltonians, the anti-periodic sine--Gordon field, Poisson
  operator, variational recurrence, QRT invariant, and time integration.
- `src/kaleidocycle/elliptic.py`: Jacobi elliptic torsion-critical families
  `kappa_n = A cn(omega + n delta; k)`, the torsion branch and closing angle,
  closing moduli, the principal family and minimum torsion `mu_N`, the Jacobi
  operator, the torsion Hessian (index theorem), and continuum constants.
- `src/kaleidocycle/theta.py`: Jacobi theta functions, closure equations,
  analytic curves, theta binormals, and theta animations.
- `src/kaleidocycle/constraints.py`: unit norm, closure, alignment, constant
  torsion, anchor, and curvature-recursion residuals.
- `src/kaleidocycle/energies.py`: bending, torsion, and dipole energies.
- `src/kaleidocycle/solvers.py`: optimization dispatch, Newton iteration,
  pseudoinverse utilities, multi-seed search, and curvature/torsion solvers.
- `src/kaleidocycle/optimality.py`: gradients, constraint Jacobians, tangent
  projection, and stationarity checks.
- `src/kaleidocycle/rigidity.py`: rigidity-matrix diagnostics, mechanism and
  self-stress bases, Calladine/Pellegrino counts, Connelly second-order stress
  tests, and finite-motion tracing wrappers.
- `src/kaleidocycle/visualization.py`: centreline, hinge, band, tetrahedron,
  paper-model, energy, and vertex-value plotting.
- `src/kaleidocycle/cad.py`: optional `build123d`-based parametric 3D CAD
  generator for print-in-place twisted-plate links and assemblies.
- `src/kaleidocycle/io.py`: JSON and CSV import/export.
- `src/kaleidocycle/samples.py`: shared named-sample catalogue, promotion, and
  presentable generic-sample selection.
- `src/kaleidocycle/report.py`: formatted diagnostic reports.

## Core Properties

`Kaleidocycle` and `KaleidocycleAnimation` expose the same high-level checks:

| Property | Meaning |
| --- | --- |
| `n` | Number of tetrahedra, equal to `len(hinges) - 1`. |
| `oriented` | Whether the first and last hinge directions agree. |
| `is_closed` | Whether the tangent closure residual is below tolerance. |
| `is_aligned` | Whether endpoint hinges satisfy the alignment rule. |
| `is_unit_norm` | Whether all hinge vectors have unit norm. |
| `constant_torsion` | Common torsion value, or `None` if not constant. |

The corresponding lower-level residuals are available from
`kaleidocycle.constraints`: `closure_residual`, `alignment_residuals`,
`unit_norm_residuals`, `constant_torsion_residuals`, `constraint_residuals`, and
`constraint_penalty`.

## Animation Diagnostics

`KaleidocycleAnimation.compute_scalar_property()` supports built-in scalar
properties:

| Name | Description |
| --- | --- |
| `bending_energy` or `bending` | Bobenko-Suris bending energy of the tangent vectors. |
| `mean_torsion` | Mean torsion angle across hinges. |
| `mean_curvature` | Mean discrete curvature across hinges. |
| `penalty` | Sum of squared constraint residuals. |
| `linking_number` | Topological linking number. |

Custom scalar properties can be added with a function:

```python
def max_curvature(hinges):
    from kaleidocycle.geometry import binormals_to_tangents, pairwise_curvature

    tangents = binormals_to_tangents(hinges, normalize=True)
    curvature = pairwise_curvature(hinges, tangents)
    return float(abs(curvature).max())

anim.compute_scalar_property("max_curvature", func=max_curvature)
```

## Analytic Theta Construction

`generate_theta_curve()` and `generate_theta_binormals()` implement the explicit
theta-function construction from `Kaleidocycle.m`. The binormal path follows the
Mathematica `Bx`, `By`, and `Bz` formulas. For parameters where the reference
formula has a removable near-singularity around `G = 0`, the implementation
automatically re-evaluates the same formula with higher precision instead of
falling back to curve-derived signs.

```python
from kaleidocycle import generate_theta_binormals

v = 0.07227972073349694
r = 0.30353311936556515
y = 0.9155431292909612
binormals = generate_theta_binormals(v, 0.0, r, y, N=38, t=0.0)
```

## Reports And Stationarity

Use `kc.report()` for a compact summary of geometric, topological, energetic,
and constraint diagnostics. Use `kc.is_stationary(energy, config=...)` to
project an energy gradient onto the constraint tangent space and estimate
whether a configuration is a constrained critical point.

```python
from kaleidocycle import Kaleidocycle
from kaleidocycle.constraints import ConstraintConfig

kc = Kaleidocycle(8, oriented=True, seed=42)
config = ConstraintConfig(oriented=True, constant_torsion=True)

print(kc.report())
print(kc.is_stationary("bending", tolerance=1e-4, config=config))
```

## Local Degrees Of Freedom

Several complementary routines analyze how many independent motions preserve
the constraints at a given configuration.

For the highest Jacobian accuracy, use the optional JAX backend in this
workflow whenever it is available:

```python
BACKEND = "jax"
```

`kc.local_dof()` performs a *linear* (infinitesimal) analysis: it returns the
dimension of `ker(J)`, where `J` is the constraint Jacobian. Global rigid
rotations `δh_i = ω × h_i` always lie in this nullspace, so they are
quotiented out by default (`subtract_rigid=True`). The returned dictionary
contains `dof`, `raw_dof`, `rigid_dof`, `rank`, singular values, and
optionally an orthonormal `basis` of shape `(n+1, 3, dof)`.

`rigidity_svd(hinges, config)` exposes the same Jacobian as a rigidity matrix
and returns explicit bases for infinitesimal mechanisms (`ker J`) and
self-stresses (`ker J.T`). `calladine_summary()` reports the
Pellegrino/Calladine identity
`mechanisms - self_stresses = variables - constraints`, with optional rigid
rotation subtraction. `connelly_second_order_test()` evaluates the
stress-weighted Hessian `d²(w·F)` on infinitesimal mechanisms for a
second-order obstruction check.

`kc.finite_motion_dof()` performs a *nonlinear* (finite-motion) analysis via
predictor-corrector continuation. It samples tangent directions from the
infinitesimal nullspace, takes finite predictor steps, projects back onto the
constraint manifold with Newton iteration, and reports the numerical rank of
the resulting displacement matrix. At a singular point of the constraint
variety, `finite_dof < infinitesimal_dof` because some tangents fail to
integrate to finite paths.

`kc.find_nearby_stationary(energy="mean_cos", config=...)` finds the nearest
critical point of an energy under the constraints by solving
`Π_{ker J} ∇E = 0` directly with `scipy.optimize.root`. Unlike gradient
descent it converges at saddle points — essential for analytically
constructed (e.g. theta-function) kaleidocycles that are saddles rather than
local minima.

`kc.follow_motion(config=..., direction_index=0)` runs a single bidirectional
predictor-corrector continuation along a chosen tangent direction and
returns the array of hinge frames.

`plot_vertex_trajectories(frames)` and `trajectory_dimensionality(curves)`
(in `kaleidocycle.visualization`) visualize and quantify the motion. The
helper `align_first_three(curve)` rigidly fixes the first three vertices to
kill the rigid-motion ambiguity.

For Kutzbach-Grübler-style DoF counting (`dof = N − 6` generically) use
`ConstraintConfig(full_alignment=True, reference_torsion=h_0·h_1)` — the
default scalar alignment has rank-0 contribution at the manifold.

```python
from kaleidocycle import (
    Kaleidocycle,
    ConstraintConfig,
    calladine_summary,
    connelly_second_order_test,
    rigidity_svd,
    trace_finite_motion,
)

kc = Kaleidocycle(8, oriented=True, seed=42)
BACKEND = "jax"

# Kutzbach-Grübler "physical" constraint set
cfg = ConstraintConfig(
    oriented=True, constant_torsion=True, full_alignment=True,
    reference_torsion=float(kc.hinges[0] @ kc.hinges[1]),
)

rig = rigidity_svd(kc.hinges, cfg, tol=1e-8, backend=BACKEND)
summary = calladine_summary(kc.hinges, cfg, tol=1e-8, backend=BACKEND)

lin = kc.local_dof(config=cfg, tol=1e-6, backend=BACKEND)
print(f"infinitesimal DoF: {lin['dof']}")
print(f"self stresses:     {summary.self_stresses}")

if rig.self_stress_count:
    second = connelly_second_order_test(
        kc.hinges, cfg, stress_index=0, backend=BACKEND
    )
    print(second.eigenvalues)

fin = kc.finite_motion_dof(
    config=cfg, seed=0, step_size=1e-4, n_steps=10, backend=BACKEND
)
print(f"finite DoF:        {fin['finite_dof']}")

frames = trace_finite_motion(
    kc.hinges, cfg, direction_index=0, step_size=5e-4,
    n_steps=80, backend=BACKEND,
)
```

See `notebooks/LocalDoF.ipynb` for a complete worked example covering the
K-G count, rigidity matrix rank, Calladine/Pellegrino self-stress diagnostics,
Connelly second-order checks, Möbius / theta-construction stationary detection,
and the 1-DoF vertex-trajectory visualization. The notebook uses the JAX backend
for Jacobian-based calculations wherever the current APIs support it.


## Integrable deformation

```python
import numpy as np

from kaleidocycle import (
    framed_polygon_from_binormals,
    import_json,
    integrate_curvature_flow,
)

kc = import_json("notebooks/output/kaleidocycle_k10_nonoriented_bending.json")
initial = framed_polygon_from_binormals(kc.hinges)
evolution = integrate_curvature_flow(
    initial.curvatures,
    initial.torsion_angle,
    np.linspace(0.0, 1.0, 101),
    flow="sine-gordon",
    sign=initial.sign,
    initial_frame=initial.frames[0],
)

configurations = evolution.configurations()
print(np.ptp(evolution.first_hamiltonian))
print(np.linalg.norm(configurations[-1].closure_residual))
```

The integrable API uses the Cayley curvature
`kappa = 2*tan(phi/2)`. `sign=1` means periodic/oriented binormals and
`sign=-1` means anti-periodic/anti-oriented binormals. Every positive mKdV flow
is available through `flow="mkdvN"` (or `flow=N`) for both signs; the
sine--Gordon flow is defined only for `sign=-1`.

## Elliptic critical families

The torsion-critical configurations are Jacobi elliptic families with an
integer winding `m`. They close at explicit moduli, and the principal family
`m = 3` realises the global minimum `mu_N` of the torsion angle of
anti-oriented rings:

```python
from kaleidocycle import (
    closing_moduli,
    continuum_constants,
    elliptic_family,
    principal_family,
    torsion_hessian,
)

family = principal_family(9)          # m = 3, closing modulus k_9
print(family.torsion_angle)           # mu_9 = 0.945665469...
hinges = family.binormals(phase=0.3)  # closed anti-oriented ring, shape (10, 3)

# Index theorem: Hess(mu) modulo rotations has inertia (N-m-3, m-3, 1).
for k in closing_moduli(13, 5):
    saddle = elliptic_family(13, 5, k)
    print(torsion_hessian(saddle.binormals(), sign=saddle.sign).inertia)

print(continuum_constants())          # (k_inf, C_inf) = (0.63092..., 8.09349...)
```

`Kaleidocycle.local_dof()` and the other tangent-space methods use
`Kaleidocycle.analysis_config`, which constrains the terminal hinge by the full
3-vector alignment. At a regular closed configuration the default count is the
design-space dimension `N - 5`; with `reference_torsion` set it is `N - 6`, or
`N - 5` at torsion-critical configurations.

## 3D Printable CAD Models (Optional)

Parametric 3D CAD export (`.stl`, `.step`, `.3mf`) for twisted-plate links
uses `build123d` as an optional extra:

```bash
python -m pip install -e ".[cad]"
```

If `build123d` is already installed in your conda environment, no extra install
is needed.

```python
from kaleidocycle import (
    Kaleidocycle,
    check_kaleidocycle_clearance,
    export_print_plate,
    export_twisted_plate,
)

# Export a single link with custom torsion angle, ribbon width, and edge length
export_twisted_plate(
    "unit.stl",
    torsion_angle=72.8,
    ribbon_width=43.0,
    edge_length=29.0,
    degrees=True,
    with_ring=False,  # set True to include the captive print-in-place ring
)

# Or derive the torsion angle directly from a Kaleidocycle instance
kc = Kaleidocycle(7, oriented=False, seed=42)
# Exporting .3mf from a Kaleidocycle arranges all N units on one print plate
kc.export_twisted_plate("k7_plate.3mf", ribbon_width=43.0, edge_length=29.0, with_ring=True)
kc.export_twisted_plate("k7_plate_cone.3mf", hinge_type="cone")  # snap-in hinges
export_print_plate("k7_plate_no_ring.3mf", kc, ribbon_width=43.0, edge_length=29.0)
kc.export_twisted_plate("k7_assembly.step", assembly=True)

# Contact check through a full rotation (also run by every export above)
print(check_kaleidocycle_clearance(kc, ribbon_width=43.0, edge_length=29.0).summary())
```

`.3mf` output is a plain 3MF (core specification) without slicer project
settings, so Bambu Studio, OrcaSlicer, and PrusaSlicer open it as geometry and
keep the printer, filament, and process presets currently selected. Each unit
is one object (with `plate` and `ring` parts when `with_ring=True`), resting on
the bed and laid out around `bed_center=(128, 128)` (the centre of a 256 mm
bed; pass another value for smaller printers). Bambu Studio 02.08.02 shows a
"3mf file has invalid config, load geometry data only" notice for any 3MF
without Bambu project settings; this was fixed in 02.08.04, and the geometry
loads either way.

Two hinge types (`hinge_type`) reproduce NovemberKou's printed designs:

- `"pin"` (default, "Material-saving Design"): adjacent links are joined by
  octagonal hinge pins (`build_hinge_pin`). Print plates include one pin per
  unit (object `hinge_pin`, lying flat; pass `with_pins=False` to omit them),
  and assemblies show the pins in place. The pin is 4 mm corner to corner
  (`pin_diameter`) with a hollow 1.8 mm core and 0.5 mm end chamfers. The
  bottom knuckles hold it in 4.0 mm bores, zero clearance at the corners
  (`pin_fit_clearance=0.0`), so the pin presses in and does not wobble. The
  top knuckles of the neighbouring link turn on it in 4.4 mm bores
  (`pin_clearance=0.2`). Knuckles are 7 mm in diameter with 0.2 mm axial gaps
  (`hinge_gap`), and the bores have a flat teardrop top so that they print
  without support. If your printer makes holes tight, raise
  `pin_fit_clearance` slightly (for example to 0.05); if the hinge binds,
  raise `pin_clearance`.
- `"cone"` ("Twisted Plate with Ring"): no separate parts. The two outer
  bottom knuckles carry 60-degree cones with a 4 mm base (`cone_diameter`,
  `cone_angle`) that snap into matching sockets in the outer ends of the
  neighbour's top knuckles, with 0.2 mm axial play (0.17 mm normal to the
  cone). Knuckles are 6 mm in diameter; each socketed knuckle keeps a 2.25 mm
  gap (`snap_gap`) on its inner side so that it can flex past the cone tip
  (about 0.95 mm) during assembly.

The default ribbon is 43 mm wide: for the `Kaleidocycle(7, oriented=False)`
above with 29 mm edges, links two apart graze each other near the centre of
the ring at 44-45 mm, while 43 mm keeps them at least 0.9 mm apart through
the whole rotation.

Every export checks the parts for contacts first (`kaleidocycle.cad_check`).
`check_hinge_fit` joins a link to a copy of itself and requires the knuckles,
pins, cones and sockets to stay apart, except the intended press fit of a pin.
For a kaleidocycle, `check_kaleidocycle_clearance` also traces one full
rotation with the integrable mKdV / sine-Gordon flow and measures every pair
of links: links that do not share a hinge must stay `min_clearance` (0.4 mm)
apart, and adjacent links must not touch away from their hinge. Contacts raise
a `ClearanceWarning` (or `ClearanceError` with `check_clearance="error"`);
`check_clearance=False` skips the check. The captive ring is checked at its
printed position.
