# Python API

## Integrable deformation API

`kaleidocycle.integrable` is the canonical implementation of the manuscript
in `Integrable deformations of kaleidocycles/`. It uses one period of Cayley
curvatures `kappa[n] = 2 tan(phi[n]/2)` and an explicit monodromy sign:

- `sign=1`: `x[n+N] = x[n]` (oriented);
- `sign=-1`: `x[n+N] = -x[n]` (anti-oriented).

Do not pass curvature angles directly to a function whose argument is named
`curvatures`; convert them with `cayley_curvatures()` first.

### Coordinates and reconstruction

| API | Purpose |
| --- | --- |
| `curvature_angles`, `cayley_curvatures` | Convert between `kappa` and the signed angle `phi`. |
| `twisted_shift` | Evaluate finite sequences with periodic or anti-periodic boundary conditions. |
| `curvature_weights` | Compute `D[n] = 1 + kappa[n]^2/4`. |
| `reconstruct_framed_polygon` | Reconstruct `F`, vertices, tangents, normals, and binormals by the discrete Frenet equation. |
| `framed_polygon_from_binormals` | Validate a constant-torsion binormal cycle and extract `kappa`, `mu`, the sign, and its frame. |
| `FramedPolygon` | Immutable reconstruction result with closure and monodromy residuals. |

`binormals` must include the terminal signed copy, so its shape is `(N+1, 3)`.
Curvature arrays contain one period and therefore have shape `(N,)`.

### Flows and compatible lifts

| API | Equation or result |
| --- | --- |
| `mkdv1_field` | `kappa_dot[n] = D[n](kappa[n+1]-kappa[n-1])/2`. |
| `mkdv2_field` | The second commuting polynomial mKdV field. |
| `mkdv_hierarchy_field` | Any positive hierarchy field `X^(order)`; orders 1 and 2 share the explicit implementations above. |
| `hierarchy_orbit_jacobian` | Columns `X^(1), ..., X^(count)` of the multi-time orbit-map Jacobian; defaults to the `N`-field Cayley--Hamilton bound. |
| `sine_gordon_potential` | Unique anti-periodic `u` satisfying `phi[n]=u[n-1]-u[n]`. |
| `sine_gordon_field` | Semi-discrete sine--Gordon field; requires `sign=-1`. |
| `lift_coefficients` | Moving-frame vertex and angular velocities `(v[n], w[n])` for `mkdvN` at any positive order and sine--Gordon. |
| `lifted_velocities` | World-coordinate vertex and binormal velocities, including the terminal value. |
| `integrate_curvature_flow` | DOP853 integration sampled at user-provided times. |
| `IntegrableEvolution` | Curvature history, arbitrary spectral/Hamiltonian histories, and lazy frame reconstruction. |

### Hamiltonian and variational structure

| API | Purpose |
| --- | --- |
| `first_hamiltonian` | `E1 = sum(log(D[n]))`. |
| `second_hamiltonian` | `E2 = sum(kappa[n] kappa[n+1])/2`. |
| `spectral_integrals`, `spectral_integral_gradients` | Coefficients `I_j` of the logarithmic twisted-Floquet expansion and their gradients. |
| `hierarchy_hamiltonians`, `hierarchy_hamiltonian_gradients` | Hamiltonians `E_j` obtained from the Chebyshev/binomial combination in the paper. |
| `twisted_trace_polynomial` | Coefficients of `P(lambda)` in `Delta_sigma(zeta) = lambda^(-N/2) P(lambda)`, `lambda=zeta^2`. |
| `spectral_curve_diagnostics` | Branch polynomial, closed-curve singular-factor check, normalised genus, reciprocal quotient genus, and Prym dimension. |
| `SpectralCurveDiagnostics` | Immutable result returned by `spectral_curve_diagnostics`. |
| `poisson_operator`, `poisson_bracket` | `P = D(S-S^-1)D` and its bracket. |
| `variational_u` | `U[n] = D[n] kappa[n-1] kappa[n+1] - kappa[n]^2`. |
| `recurrence_multiplier` | Least-squares `lambda` for the three-term recurrence. |
| `critical_torsion_cosine` | Recover `c` from a nonzero critical curvature sequence. |
| `critical_multiplier` | Evaluate the critical-locus identity `lambda = 2*c + E2/N`. |
| `variational_recurrence_residual` | Residual of `D[n](kappa[n-1]+kappa[n+1]) = lambda kappa[n]`. |
| `qrt_invariant` | Biquadratic invariant of the recurrence. |

`bending_energy()` now treats a one-dimensional input as Cayley curvature and
returns `E1`. Its two-dimensional tangent-vector form evaluates the equivalent
angle expression.

## Legacy API inventory

This document captures every callable routine in the legacy Mathematica, Maple, and MATLAB sources so we can plan the Python rewrite without missing features. Each entry lists the current responsibility, important inputs/outputs, and the Python module that will eventually absorb the logic.

## Mathematica (`Kaleidocycle.m`)

| Routine | Purpose & I/O | Planned Python Target |
| --- | --- | --- |
| `Setup[n, ori, sl, freeangle]` | Allocates hinge vectors, global constraint lists, and energy expressions; inputs are counts/flags, outputs are global symbols side effects. | `kaleidocycle.geometry.setup` |
| `RndH[]` | Builds deterministic random initial points for solver variables (returns list of substitution rules). | `kaleidocycle.geometry.initialization` |
| `MKn[target, localop, opts]` | Wrapper around `FindMinimum`/`NMinimize` to solve constrained problems and print solutions; returns solver result. | `kaleidocycle.solvers.optimize_cycle` |
| `M2B[M]` | Converts centre-line segment list into normalized binormal frame. | `kaleidocycle.geometry.frames` |
| `Tw[B]` | Computes total twist from binormals; accepts hinge/binormal arrays, returns scalar twist. | `kaleidocycle.analysis.twist` |
| `TwM[M]` | Applies `M2B` then `Tw` to measure torsion of a centre-line. | `kaleidocycle.analysis.twist` |
| `Dipole[C, P]` | Evaluates dipole energy between hinge directions and positions. | `kaleidocycle.energies.dipole` |
| `PrintSol[sol]` | Formats solver output listing twist, writhe, energies. | `kaleidocycle.reporting.console` |
| `Disp[sol, width, matform]` | Produces `Graphics3D` panels of the kaleidocycle; returns graphic. | `kaleidocycle.visualization.solids` |
| `Lwrithe[u, eps]` | Local writhe measure for four points (returns signed angle). | `kaleidocycle.analysis.writhe` |
| `Writhe[M]` | Global writhe accumulation across the curve. | `kaleidocycle.analysis.writhe` |

## Maple (`Kaleidocycle.mpl`)

### Indexing & Setup

| Routine | Purpose & I/O | Planned Python Target |
| --- | --- | --- |
| `h(i, j)` / `p(i)` / `q(i)` | Symbol builders for hinge, primal, and dual Maple variables. | `kaleidocycle.symbolics.naming` |
| `setupK(EndPoint, variableTorsion)` | Core Maple setup: declares globals (`N`, constraints) and seeds hinge/centre-line arrays. | `kaleidocycle.geometry.setup` |
| `fix_global(variableTorsion)` | Applies endpoint and torsion fixing rules to remove rigid motion. | `kaleidocycle.constraints.symmetry` |
| `infoX(X1, B1)` | Reports curvature, torsion, writhe for a given configuration and returns `(X, B, T, K)`. | `kaleidocycle.reporting.console` |
| `clean_eqs(eqlist)` | Deduplicates/algebraically simplifies constraint sets. | `kaleidocycle.constraints.algebra` |

### Sampling, Search & Feasibility

| Routine | Purpose & I/O | Target |
| --- | --- | --- |
| `modN(a)` / `signed_modN(L, i)` | Periodic index helpers (with orientation awareness). | `kaleidocycle.geometry.indexing` |
| `rnd2sp()` / `rndips()` / `rndip(eqn)` | Random initializers for hinge directions and solver seeds. | `kaleidocycle.geometry.initialization` |
| `find(cn, target, vars)` | High-level solver orchestrator that iterates over constraints. | `kaleidocycle.solvers.dispatch` |
| `MPinv(A, eps)` | Moore–Penrose pseudoinverse wrapper with tolerance. | `kaleidocycle.linear_algebra.linalg` |
| `newton(eqlist, vlist, J, ip0, max_iter)` | Newton iteration for constraint solving with manual Jacobians. | `kaleidocycle.solvers.newton` |
| `subdiv(B)` / `mirror(B)` / `involution(B)` / `negativeTwist(B)` / `reverse(B)` | Geometry transforms applied to binormal sequences (refine, mirror, invert). | `kaleidocycle.geometry.transforms` |
| `feasible(B, verbose, total)` | Validates current binormal set against constraints, optionally printing stats. | `kaleidocycle.constraints.validation` |
| `closestsol(U, u, istart, iend)` / `dist(diff)` | Helpers for selecting closest solution candidates using norms. | `kaleidocycle.solvers.search` |

### Energies & Invariants

| Routine | Purpose & I/O | Target |
| --- | --- | --- |
| `cos_invariant(K, mKdV)` | Computes cosine-based invariants from curvature list. | `kaleidocycle.analysis.invariants` |
| `elastic_energy2(T, d)` / `elastic_energy(T, d)` | Elastic energies on tangent fields (squared angle vs. log energy). | `kaleidocycle.energies.elastic` |
| `torsion_energy(B)` / `torsion_energy2(B)` | Two torsion metrics mirroring Mathematica energies. | `kaleidocycle.energies.torsion` |
| `dipole_energy(B, X0)` / `coulomb_energy(X, a)` | Long-range interaction energies; inputs are frames or centres. | `kaleidocycle.energies.long_range` |

### Curvature, Writhe & Axes

| Routine | Purpose & I/O | Target |
| --- | --- | --- |
| `curvature_recursion(K)` / `curvature_recursion_T(T)` | Recurrence relations linking curvature samples. | `kaleidocycle.geometry.curvature` |
| `lwrithe(u)` / `lwrithe2(u)` | Local writhe integrals for four-point windows. | `kaleidocycle.analysis.writhe` |
| `sangle(u, v, w)` / `sarea(U)` | Spherical angle/area helpers. | `kaleidocycle.analysis.spherical` |
| `writhe(C)` / `Tw(B)` / `TwX(X)` / `torsion(B)` / `axis(B, K)` | Global writhe/twist/torsion metrics and axis reconstruction. | `kaleidocycle.analysis` |

### Frame Transformations

| Routine | Purpose & I/O | Target |
| --- | --- | --- |
| `B2T(B, normalise)` / `B2K(B, T0, signed)` / `B2K2(B)` | Convert binormals into tangents and curvature angles. | `kaleidocycle.geometry.frames` |
| `B2X(B)` / `X2T(X)` / `T2B(T)` / `X2B(X)` | Map between binormal, tangent, and spatial coordinates. | `kaleidocycle.geometry.frames` |
| `match_sgn(...)` / `B2N(B)` / `K2B(K)` | Sign/orientation alignment and curvature-to-binormal conversion. | `kaleidocycle.geometry.transforms` |

### Theta-Based Parametrizations & Visualization

| Routine | Purpose & I/O | Target |
| --- | --- | --- |
| `periodic_theta(...)` | Periodic theta-function helper for analytic solutions. | `kaleidocycle.analysis.theta` |
| `coords(...)` / `coordsTrefoil(...)` | Construct spatial coordinates from analytic expressions. | `kaleidocycle.geometry.parametric` |
| `vis(...)` / `visTet(...)` / `visBand(...)` | Maple 3D visualization wrappers. | `kaleidocycle.visualization.solids` |
| `sweepTorus(...)` / `dispCurve(...)` / `dispBd(...)` / `apex(...)` / `tetsheet(...)` / `drawtrig(...)` / `paper_model(...)` | Specialized plotting routines and printable nets. | `kaleidocycle.visualization` |
| `signed_list(...)` | Applies orientation-aware signs to sequences (used by visualization). | `kaleidocycle.geometry.indexing` |

### Angle Transforms & Animation

| Routine | Purpose & I/O | Target |
| --- | --- | --- |
| `K2omega(K)` / `omega2K(omega)` / `K2theta(K)` / `theta2K(theta)` | Convert between curvature and auxiliary angle parameterizations. | `kaleidocycle.analysis.transforms` |
| `diffX(...)` / `sGmoveX(...)` / `diffB(...)` / `sGmoveB(...)` | Discrete sine-Gordon moves on X/B frames. | `kaleidocycle.simulation.sine_gordon` |
| `project_sol(...)` | Projects iterates back into the constraint manifold. | `kaleidocycle.solvers.projection` |
| `anim_step(...)` / `anim_rnd(...)` / `anim_dd(...)` / `anim_ddX(...)` / `anim_load(filename)` | Animation builders (stepwise, random, data-driven, loader). | `kaleidocycle.animation.builder` |
| `clean_anim(S, eps, twistFix, Xs)` / `sort_anim()` / `fix_rot(Bs1, Ts1)` | Post-processing for animation sequences (cleaning, ordering, alignment). | `kaleidocycle.animation.processing` |
| `disp_anim(Vs, func)` / `XYZ(v, z, r, y, N, t)` | Display helpers returning point clouds or animated plots. | `kaleidocycle.visualization.anim` |

## MATLAB (`matlab/`)

| Routine | Purpose & I/O | Planned Python Target |
| --- | --- | --- |
| `Kaleidocycle.m` (script) | Full optimization workflow: symbol setup, constraint export, objective wiring, frame stepping. | `examples/optimization_notebook.ipynb` & CLI wrappers |
| `advance(x, eps)` | Applies sine-Gordon evolution to hinge array (`x` vector) to seed next frame. | `kaleidocycle.simulation.sine_gordon` |
| `curvature(x)` | Computes signed curvature angles from hinge vectorization. | `kaleidocycle.geometry.curvature` |
| `dispK(sol)` | MATLAB visualization of strips from hinge data. | `kaleidocycle.visualization.solids` |
| `energy_bend(sol)` | Bending energy of mid-axis (log form). | `kaleidocycle.energies.elastic` |
| `energy_dipole(sol)` | Dipole energy computed via pairwise sums over centre points. | `kaleidocycle.energies.long_range` |
| `gen_constr(x)` + `ft_1`..`ft_4` | Symbolic constraint and gradient evaluators for fmincon (and helper chunks). | `kaleidocycle.constraints.symbolic` |
| `hessfinal(X, lambda)` | Aggregates Lagrange multipliers with symbolic Hessians. | `kaleidocycle.solvers.multipliers` |
| `localWrithe(u1,u2,u3,u4)` / `writhe(centre)` | Local/global writhe utilities identical to Mathematica versions. | `kaleidocycle.analysis.writhe` |
| `prevstep_constr(X)` | Adds step-size proximity constraint relative to previous frame. | `kaleidocycle.constraints.temporal` |
| `rndX(n)` | Random hinge initializer for MATLAB optimizers. | `kaleidocycle.geometry.initialization` |
| `twist(x)` | Half-twist plus writhe calculation; uses `writhe`. | `kaleidocycle.analysis.twist` |
| `pathdef()` | MATLAB search-path generator (reference only; no Python port required). | _Not planned_ |

All routines above now have an explicit Python destination so contributors can track migration progress and avoid duplicating work.
