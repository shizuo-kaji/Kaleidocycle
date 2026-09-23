# A transverse self-crossing of the first mKdV flow

`mkdv_self_crossing_k12.json` specifies an anti-oriented K12 with torsion angle
`mu = 6/5`. Edge indices are zero-based. Edges 0 and 6 meet at time zero.
All decimal strings in `center` and `inverse_jacobian` are exact certificate
inputs; they must not be rounded or converted through binary floats before
interval evaluation.

The 14 centre coordinates are `(phi_0, ..., phi_11, u, v)`. The five angles
at indices `[3, 5, 6, 7, 9]` are held fixed. The other seven angles and the
interior edge fractions `u, v` vary in an infinity-norm box of radius `1e-20`.
Reconstruct with `F_0 = I`, `gamma_0 = 0` and
`F_(n+1) = F_n R1(mu) R3(phi_(n+1))`, using `phi_12 = -phi_0`.

The nine equations are position closure, the three skew components of
`F_12 diag(1,-1,-1)`, and
`gamma_0 + u T_0 - gamma_6 - v T_6 = 0`.
The checker also verifies that the frame trace is greater than 2.9, excluding
the half-turn solutions of the skew equations.

The stored matrix B approximates the inverse Jacobian. Outward-rounded
70-digit interval arithmetic, with automatic derivatives over the full box,
verifies `||B f(center)||_inf < 1e-58` and
`sup ||I - B Df||_inf < 1e-15`. The Newton map is a contraction from the box
into itself, giving a unique exact contact. No root-finder output is trusted
without these checks.

At the contact, all joints are regular and all 53 other non-adjacent edge
pairs have distance greater than 0.0129. For the first-flow vertex velocities
`V_n = F_n (cos(mu), -z_n cos(mu), -z_n sin(mu))`, the relative velocity
`W = (1-u)V_0 + uV_1 - (1-v)V_6 - vV_7` satisfies
`-3.283 < dot(cross(T_0,T_6), W) < -3.281`.
The contact equation has an invertible Jacobian in `(time,u,v)`. It follows
that the crossing is isolated and the nearby trajectories on either side
are embedded. Choosing a nearby negative time as the initial time disproves
unconditional embeddedness preservation.

Run from `python/`, using the project Python environment:

```bash
python -c 'from kaleidocycle.collision_certificate import certify_crossing; print(certify_crossing())'
python scripts/build_self_crossing.py
pytest tests/test_collisions.py
```

`notebooks/SelfCrossing.ipynb` executes the certificate and displays static and
interactive views. In the studio's built-in menu, select **Self-crossing K12 ·
first mKdV flow** (`counterexample_k12_self_crossing`) for the numerical
pre-contact configuration. The exact input is this certificate,
not the rounded geometry in the visualisation assets.

Trajectory samples use DOP853 without constraint projection. Their plotting
interval is `[-0.002, 0.002]`; the interval proof concerns the exact local
crossing, not every time-integration sample. Standard Gauss writhe jumps by
-2, with `Tw = 12*1.2/(2*pi)`. The half-integral self-linking number
`Lk = Tw + Wr` changes from 4.5 to 2.5; the integer twisting number is
`T = 2 Lk`. Wr and Lk are not evaluated at the contact. This normalisation
is distinct from the legacy `geometry.writhe` API.
