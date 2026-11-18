# Repository Guidelines

- Target layout: `src/kaleidocycle/` (geometry, energies, solvers, visualization), `tests/`, `notebooks/`, `data/`, and `figures/`. Keep notebooks thin wrappers around functions exposed by `src/`.

## Creating interactive jupyter notebooks
- %matplotlib widget for 3D interactivity.

## Build, Test, and Development Commands
- use conda env: /opt/homebrew/Caskroom/miniforge/base/envs/py313/bin/python
- `pip install -e .[dev]` installs the package plus lint/test extras.
- `pytest` runs the suite; `pytest -k geometry` narrows focus while porting.

## Coding Style & Naming Conventions
- Follow PEP 8 with 88-character lines; `ruff format` is the source of truth.
- Modules use noun names (`geometry`, `energies`); exported APIs use verbs (`build_cycle`, `optimise_cycle`).
- Require type hints, preferring `numpy.typing.NDArray` for arrays, and describe math objects clearly in docstrings.

**Test Coverage:**
- Geometry: 29 tests in `test_geometry.py` ✓
- Animation: 33 tests in `test_animation.py` ✓ (includes 5 tests for `sort_animation_frames`)
- Curve generation: 10 tests in `test_curve_generation.py` ✓
- Newton: 13 tests in `test_newton.py` ✓
- Theta functions: 17 tests in `test_theta.py` ✓
- Constraints: 4 tests in `test_constraints.py` ✓ (includes validation of constant torsion)
- IO: 7 tests in `test_io.py` ✓
- Visualization: 11 tests in `test_visualization.py` ✓
- **Total: 124 tests, all passing** ✓

**Important Note on Test Fixtures:**
All tests now use properly optimized kaleidocycles with **constant torsion** (constant inner product between adjacent hinges), which is the fundamental defining property of kaleidocycles. Test fixtures verify:
- Constant torsion: std < 1e-4
- Unit norm: ||residual|| < 1e-4
- Closure: ||residual|| < 2e-3


## Function Port Status

This table tracks the migration of functions from legacy codebases (Maple/Mathematica/MATLAB) to Python.
The remaining notable gaps are:

- `coulomb_energy`
- `anim_ddX`
