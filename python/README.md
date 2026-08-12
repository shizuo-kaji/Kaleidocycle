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

Use the project environment while porting or running notebooks:

```bash
python -m pip install -e ".[dev]"
```

JAX is included in the development dependencies. For a smaller runtime install
with automatic differentiation support, use `python -m pip install -e ".[jax]"`.

Run the test suite with:

```bash
pytest
```

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
`sign=-1` means anti-periodic/anti-oriented binormals. The two mKdV flows are
available for both signs; the sine--Gordon flow is defined only for `sign=-1`.

## Notebooks

Interactive examples live in `notebooks/`. Use `%matplotlib widget` for 3D
interaction and make sure the notebook kernel points at the project environment.

- `FindKaleidocycles.ipynb` explores optimization-based construction.
- `ExplicitConstructionWithTheta.ipynb` demonstrates analytic theta-function
  solutions.
- `Animation.ipynb` explores animation utilities.
- `IntegrableDeformations.ipynb` demonstrates curvature reconstruction, both
  mKdV flows, the anti-periodic sine--Gordon flow, and numerical conservation.
- `KaleidocycleProperties.ipynb` inspects structural properties and residuals.
- `AnimationScalarProperties.ipynb` computes and plots animation diagnostics.
- `BackendComparison.ipynb` compares NumPy and optional JAX solver backends.
- `LocalDoF.ipynb` analyzes infinitesimal and finite degrees of freedom of the
  constraint-preserving motion at a given configuration.

## Package Layout

- `src/kaleidocycle/geometry.py`: hinge frames, curves, curvature, torsion,
  writhe, twist, linking number, transformations between curves, tangents, and
  binormals, plus the `Kaleidocycle` class.
- `src/kaleidocycle/animation.py`: sine-Gordon flow, step/random animation
  generation, frame cleaning, alignment, sorting, curve conversion, and
  `KaleidocycleAnimation`.
- `src/kaleidocycle/integrable.py`: twisted curvature coordinates, discrete
  Frenet reconstruction, compatible lifts, two commuting mKdV fields, the
  anti-periodic sine--Gordon field, Hamiltonians, Poisson operator, variational
  recurrence, QRT invariant, and high-accuracy time integration.
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
- `src/kaleidocycle/io.py`: JSON and CSV import/export.
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
    config=cfg, seed=0, step_size=1e-3, n_steps=15, backend=BACKEND
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
