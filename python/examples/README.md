# Kaleidocycle Examples

This directory contains example scripts demonstrating how to use the Kaleidocycle package.

## Available Examples

### properties_demo.py

Demonstrates the new properties available on both `Kaleidocycle` and `KaleidocycleAnimation` classes.

**Properties demonstrated:**
- `n` - Number of tetrahedra
- `oriented` - Whether the kaleidocycle is oriented
- `is_closed` - Whether closure constraint is satisfied (sum of tangents = 0)
- `is_aligned` - Whether alignment constraint is satisfied (first/last hinge match)
- `is_unit_norm` - Whether all vectors have unit norm
- `constant_torsion` - Constant torsion value (or None if not constant)

For animations, `is_closed` and `is_aligned` check if ALL frames satisfy the constraints.

**Usage:**
```bash
python examples/properties_demo.py
```

### plot_scalar_properties.py

Comprehensive example demonstrating how to compute and plot scalar properties of kaleidocycle animations.

**Features:**
- Generate animations using different evolution rules (sine-Gordon, step-based)
- Compute built-in scalar properties (bending energy, mean torsion, mean curvature)
- Create custom scalar properties
- Visualize property evolution over time
- Compare properties across different kaleidocycles

**Usage:**
```bash
python examples/plot_scalar_properties.py
```

**Output:**
The script generates four plots saved as PNG files:
- `sine_gordon_scalar_properties.png` - Properties evolution for sine-Gordon flow
- `step_scalar_properties.png` - Properties evolution for step-based method
- `compare_bending_energy.png` - Comparison of bending energy across different n
- `custom_property.png` - Example of custom property (max curvature)

## Quick Example

Here's a minimal example to get started:

```python
from kaleidocycle import KaleidocycleAnimation, generate_animation, random_hinges
import matplotlib.pyplot as plt

# Create initial kaleidocycle
n = 8
hinges = random_hinges(n, seed=42, oriented=True).as_array()

# Generate animation
frames = generate_animation(hinges, num_frames=30, rule="sine-Gordon")

# Create animation object
anim = KaleidocycleAnimation(frames=frames, evolution_rule="sine-Gordon")

# Compute scalar properties
anim.compute_scalar_property("bending_energy")
anim.compute_scalar_property("mean_torsion")
anim.compute_scalar_property("mean_curvature")

# Plot bending energy evolution
plt.figure(figsize=(10, 4))
plt.plot(anim.scalar_properties["bending_energy"], marker='o')
plt.xlabel("Frame")
plt.ylabel("Bending Energy")
plt.title("Bending Energy Evolution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bending_energy.png", dpi=150)
plt.show()
```

## Scalar Properties

The `KaleidocycleAnimation.compute_scalar_property()` method supports the following built-in properties:

| Property Name | Description |
|--------------|-------------|
| `bending_energy` or `bending` | Bobenko-Suris bending energy of the tangent vectors |
| `mean_torsion` | Mean torsion angle across all hinges |
| `mean_curvature` | Mean curvature across all hinges |
| `penalty` | Constraint penalty (sum of squared residuals) |
| `linking_number` | Topological linking number |

You can also compute custom properties by providing a function:

```python
def my_property(hinges):
    # Your custom computation
    return scalar_value

anim.compute_scalar_property("my_property", func=my_property)
```

## Requirements

- kaleidocycle package
- numpy
- scipy
- matplotlib

Install the package in development mode:
```bash
pip install -e .
```
