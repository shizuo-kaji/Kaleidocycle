# New Properties Added to Kaleidocycle and KaleidocycleAnimation

## Overview

Six new properties have been added to both the `Kaleidocycle` and `KaleidocycleAnimation` classes to provide convenient access to structural and geometric information.

## Properties

### 1. `n: int`
Number of tetrahedra in the kaleidocycle.

**Kaleidocycle:**
```python
kc = Kaleidocycle(8, seed=42)
print(kc.n)  # 8
```

**KaleidocycleAnimation:**
```python
anim = KaleidocycleAnimation(frames=frames)
print(anim.n)  # Number of tetrahedra (n_vertices - 1)
```

### 2. `oriented: bool`
Whether the kaleidocycle is oriented (first and last hinge point in same direction).

**Kaleidocycle:**
```python
kc_oriented = Kaleidocycle(8, oriented=True)
print(kc_oriented.oriented)  # True

kc_non = Kaleidocycle(7, oriented=False)
print(kc_non.oriented)  # False
```

**KaleidocycleAnimation:**
```python
# Automatically determined from first frame
anim = KaleidocycleAnimation(frames=frames)
print(anim.oriented)  # True or False
```

### 3. `is_closed: bool`
Whether the closure constraint is satisfied (tangent vectors sum to zero).

This checks if `closure_residual` is small, meaning the structure forms a closed spatial polygon.

**Kaleidocycle:**
```python
kc = Kaleidocycle(6, seed=42)
print(kc.is_closed)  # True if closure constraint satisfied
```

**KaleidocycleAnimation:**
- `True` if ALL frames satisfy the closure constraint
- Optimized kaleidocycles satisfy this constraint

```python
anim = KaleidocycleAnimation(frames=frames)
print(anim.is_closed)  # True if all frames are closed
```

### 4. `is_aligned: bool`
Whether the alignment constraint is satisfied (first and last hinge match properly).

This checks if `alignment_residuals` is small:
- For oriented: first and last hinges should be equal (h[0] ≈ h[-1])
- For non-oriented: first and last hinges should be opposite (h[0] ≈ -h[-1])

**Kaleidocycle:**
```python
kc = Kaleidocycle(6, seed=42)
print(kc.is_aligned)  # True if alignment constraint satisfied
```

**KaleidocycleAnimation:**
- `True` if ALL frames satisfy the alignment constraint
- Checks based on each frame's orientation

```python
anim = KaleidocycleAnimation(frames=frames)
print(anim.is_aligned)  # True if all frames are aligned
```

### 5. `is_unit_norm: bool`
Whether all hinge vectors have unit norm (within tolerance).

```python
kc = Kaleidocycle(6, seed=42)
print(kc.is_unit_norm)  # True for optimized kaleidocycles

anim = KaleidocycleAnimation(frames=frames)
print(anim.is_unit_norm)  # True if all frames have unit norm
```

### 6. `constant_torsion: float | None`
Constant torsion value if torsion is constant, `None` otherwise.

```python
kc = Kaleidocycle(6, seed=42)
torsion = kc.constant_torsion
if torsion is not None:
    print(f"Constant torsion: {torsion:.4f} radians")
else:
    print("Torsion is not constant")
```

## Implementation Details

### Tolerances

- **Orientation check**: cosine > 0.9999 between first and last hinge
- **Closure check**: `norm(closure_residual) < 1e-3`
- **Alignment check**: `alignment_residuals < 1e-3`
- **Unit norm check**: `np.allclose(norms, 1.0, rtol=1e-6, atol=1e-6)`
- **Constant torsion check**: std(torsion) < 1e-4

### Constraint Residuals

The `is_closed` and `is_aligned` properties correspond directly to constraint residuals:

- **`closure_residual(hinges)`**: Returns 3D vector (sum of tangent vectors). Should be zero for closed structures.
- **`alignment_residuals(hinges, oriented)`**: Returns scalar norm of difference between first and last hinges.
  - For oriented: `norm(h[0] - h[-1])`
  - For non-oriented: `norm(h[0] + h[-1])`

### Performance

All properties are computed on-the-fly (not cached). For animations with many frames, consider caching the results if used repeatedly.

## Examples

See `examples/properties_demo.py` for comprehensive demonstrations of all properties.

## Tests

Comprehensive tests have been added:
- `tests/test_kaleidocycle_animation.py`: 6 new tests for animation properties
- `tests/test_kaleidocycle_from_n.py`: 6 new tests for kaleidocycle properties

All 62 tests pass.
