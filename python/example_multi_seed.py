"""Example usage of optimize_multi_seed function."""

from kaleidocycle import ConstraintConfig, optimize_multi_seed

# Example 1: Run optimization with 10 random trials
print("Example 1: Run 10 random trials")
print("-" * 50)

config = ConstraintConfig(oriented=True, constant_torsion=True)
results = optimize_multi_seed(
    n=9,
    config=config,
    n_trials=10,
    objective="bending",
)

print(f"Generated {len(results)} kaleidocycles")
print(f"\nFirst result properties:")
print(f"  Seed: {results[0].metadata['seed']}")
print(f"  Mean cosine: {results[0].mean_cosine:.6f}")
print(f"  Bending energy: {results[0].metadata['optimization']['energy']:.6f}")
print(f"  Success: {results[0].metadata['optimization']['success']}")

# Example 2: With explicit seeds
print("\n\nExample 2: Using explicit seeds")
print("-" * 50)

results = optimize_multi_seed(
    n=8,
    config=ConstraintConfig(oriented=False, constant_torsion=True),
    seeds=[42, 123, 456, 789],
    objective="mean_cos",
)

print(f"Generated {len(results)} kaleidocycles with seeds: [42, 123, 456, 789]")
for kc in results:
    print(f"  Seed {kc.metadata['seed']}: mean_cos = {kc.mean_cosine:.6f}")

# Example 3: With dataframe output (requires pandas)
print("\n\nExample 3: With dataframe output")
print("-" * 50)

try:
    results, df = optimize_multi_seed(
        n=9,
        config=config,
        n_trials=5,
        objective="bending",
        return_dataframe=True,
    )

    print(f"Generated dataframe with {len(df)} rows and {len(df.columns)} columns")
    print(f"\nDataframe columns: {list(df.columns)}")
    print(f"\nDataframe summary:")
    print(df[['seed', 'mean_cos', 'bending_energy', 'linking_number', 'penalty']].to_string())

    # Find best result by bending energy
    best_idx = df['bending_energy'].idxmin()
    print(f"\nBest result (lowest bending energy):")
    print(f"  Seed: {df.loc[best_idx, 'seed']}")
    print(f"  Bending energy: {df.loc[best_idx, 'bending_energy']:.6f}")
    print(f"  Mean cosine: {df.loc[best_idx, 'mean_cos']:.6f}")

except ImportError:
    print("Pandas not installed - skipping dataframe example")
    print("Install with: pip install pandas")

# Example 4: With linking constraint
print("\n\nExample 4: With linking constraint")
print("-" * 50)

config_linking = ConstraintConfig(oriented=True, constant_torsion=True, closure=True)
results = optimize_multi_seed(
    n=9,
    config=config_linking,
    seeds=[10, 20, 30],
    optimizer="optimize_with_linking_constraint",
    target_linking=2.0,
    objective="bending",
)

print(f"Generated {len(results)} kaleidocycles with target linking = 2.0π")
for kc in results:
    opt_info = kc.metadata['optimization']
    print(f"  Seed {kc.metadata['seed']}: "
          f"success={opt_info['success']}, "
          f"energy={opt_info['energy']:.4f}")
