# Kaleidocycle Python Port (WIP)

This repository is transitioning Maple, Mathematica, and MATLAB tooling for kaleidocycles into a tested Python package plus notebooks.
See `AGENTS.md` and `api.md` for the migration plan.

N-Kaleidocycles are represented by a sequence of N+1 unit-length hinge vectors (binormals) in R^3 with **constant torsion** (constant inner product between adjacent hinges).
For an oriented kaleidocycle, the first and last hinge are identical, while for a non-oriented one they are negatives of each other.
(this is why we include N+1 hinges for an N-kaleidocycle; when considering geometric properties such as writhe and twist, this point is important).

## Quick Start

### Interactive Notebooks

Explore the capabilities of the package through interactive Jupyter notebooks in the `notebooks/` directory:

## Project Structure

- `src/kaleidocycle/` — geometry primitives, energy functionals, constraint helpers, and optimization wrappers
  - `geometry.py` — core geometric functions (writhe, twist, curvature, frames) and **Kaleidocycle class**
  - `animation.py` — sine-Gordon flow and animation utilities
  - `theta.py` — exact analytic solutions via Jacobi theta functions
  - `constraints.py` — constraint checking and validation
  - `energies.py` — energy functionals
  - `solvers.py` — optimization routines
  - `report.py` — property report generation
  - `visualization.py` — plotting utilities
  - `io.py` — import/export functions
- `tests/` — test suites
- `notebooks/` — interactive Jupyter demonstrations
