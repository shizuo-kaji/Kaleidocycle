/* Exact kaleidocycle families: Jacobi elliptic (cn) critical circles,
 * right-angle circles and the explicit torus-knot rings. No modules or
 * network APIs; load after sim.js. The formulae follow
 * src/kaleidocycle/elliptic.py and "Integrable deformations of Kaleidocycles".
 */
(function initialiseFamilies(global) {
  "use strict";

  const Mathematics = global.Kaleidocycle;
  if (!Mathematics) throw new Error("Load sim.js before families.js.");

  const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const column = (frame, index) => [frame[index], frame[3 + index], frame[6 + index]];
  const applyMatrix = (matrix, vector) => [
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
    matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
    matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
  ];

  function gcd(a, b) {
    let x = Math.abs(a);
    let y = Math.abs(b);
    while (y) [x, y] = [y, x % y];
    return x;
  }

  function checkModulus(modulus) {
    const k = Number(modulus);
    if (!Number.isFinite(k) || k <= 0 || k >= 1) {
      throw new Error("The elliptic modulus must lie strictly between 0 and 1.");
    }
    return k;
  }

  /** Complete elliptic integral K(k) of the first kind, by the AGM. */
  function ellipticK(modulus) {
    const k = checkModulus(modulus);
    let a = 1;
    let b = Math.sqrt((1 - k) * (1 + k));
    for (let index = 0; index < 64 && Math.abs(a - b) > 1e-16 * a; index += 1) {
      const mean = 0.5 * (a + b);
      b = Math.sqrt(a * b);
      a = mean;
    }
    return Math.PI / (2 * a);
  }

  /** Jacobi sn, cn and dn of modulus k by the descending Landen transform. */
  function jacobiFunctions(u, modulus) {
    const k = Number(modulus);
    if (k * k < 1e-30) return { sn: Math.sin(u), cn: Math.cos(u), dn: 1 };
    const a = [1];
    const c = [k];
    let b = Math.sqrt((1 - k) * (1 + k));
    let level = 0;
    while (Math.abs(c[level]) > 1e-16 * a[level] && level < 64) {
      a.push(0.5 * (a[level] + b));
      c.push(0.5 * (a[level] - b));
      b = Math.sqrt(a[level] * b);
      level += 1;
    }
    let amplitude = 2 ** level * a[level] * u;
    for (let index = level; index >= 1; index -= 1) {
      amplitude = 0.5 *
        (amplitude + Math.asin((c[index] / a[index]) * Math.sin(amplitude)));
    }
    const sn = Math.sin(amplitude);
    return {
      sn,
      cn: Math.cos(amplitude),
      dn: Math.sqrt(Math.max(0, 1 - k * k * sn * sn)),
    };
  }

  function rotationAngleAbout(rotation, axis) {
    const length = norm(axis);
    const unit = axis.map((value) => value / length);
    const smallest = unit
      .map((value, index) => [Math.abs(value), index])
      .sort((left, right) => left[0] - right[0])[0][1];
    const helper = [0, 0, 0];
    helper[smallest] = 1;
    const raw = cross(unit, helper);
    const rawLength = norm(raw);
    const first = raw.map((value) => value / rawLength);
    const image = applyMatrix(rotation, first);
    return Math.atan2(dot(unit, cross(first, image)), dot(first, image));
  }

  /**
   * The cn family kappa[n] = A cn(omega + n delta; k) with delta = 2 K m / N.
   * Its torsion cosine is recovered from the multiplier identity
   * lambda = 2 c + E2 / N, which holds for every modulus.
   */
  function ellipticFamily(n, winding, modulus) {
    const size = Number(n);
    const m = Number(winding);
    if (!Number.isInteger(size) || size < 3) throw new Error("N must be an integer ≥ 3.");
    if (!Number.isInteger(m) || m <= 0 || m >= size) {
      throw new Error("The winding m must satisfy 0 < m < N.");
    }
    const k = checkModulus(modulus);
    const quarterPeriod = ellipticK(k);
    const step = (2 * quarterPeriod * m) / size;
    const { sn, cn, dn } = jacobiFunctions(step, k);
    const amplitude = (2 * k * sn) / dn;
    const multiplier = (2 * cn) / (dn * dn);
    const sign = m % 2 === 0 ? 1 : -1;
    const curvatures = (phase = 0) =>
      Array.from({ length: size }, (_, index) =>
        amplitude * jacobiFunctions(phase + index * step, k).cn);
    const sample = curvatures(0);
    const shifted = Mathematics.twistedShift(sample, 1, sign);
    const secondHamiltonian = 0.5 * sample.reduce(
      (sum, value, index) => sum + value * shifted[index],
      0,
    );
    const torsionCosine = 0.5 * (multiplier - secondHamiltonian / size);
    return {
      n: size,
      winding: m,
      modulus: k,
      sign,
      quarterPeriod,
      phasePeriod: 4 * quarterPeriod,
      step,
      amplitude,
      multiplier,
      torsionCosine,
      curvatures,
    };
  }

  function familyTorsionAngle(family) {
    if (!(family.torsionCosine > -1 && family.torsionCosine < 1)) {
      throw new Error("The torsion cosine has left (−1, 1).");
    }
    return Math.acos(family.torsionCosine);
  }

  function multiplierVector(family, phase = 0) {
    const kappa = family.curvatures(phase);
    const c = family.torsionCosine;
    const s = Math.sqrt(1 - c * c);
    const rho = -1 / (2 * family.n);
    return [
      (rho * (family.multiplier - 2 * c - (kappa[0] * kappa[1]) / 2)) / s,
      (rho * (c * kappa[0] - kappa[1])) / s,
      rho * kappa[0],
    ];
  }

  /** Principal value of the closing angle of the frame monodromy. */
  function closingAngle(family, phase = 0) {
    const configuration = Mathematics.reconstruct(
      family.curvatures(phase),
      familyTorsionAngle(family),
      family.sign,
      IDENTITY,
    );
    const last = configuration.frames[configuration.frames.length - 1];
    const sigma = family.sign;
    const rotation = [
      last[0], sigma * last[1], sigma * last[2],
      last[3], sigma * last[4], sigma * last[5],
      last[6], sigma * last[7], sigma * last[8],
    ];
    return rotationAngleAbout(rotation, multiplierVector(family, phase));
  }

  function bisect(fn, lower, upper, iterations = 200) {
    let left = lower;
    let right = upper;
    let leftValue = fn(left);
    for (let index = 0; index < iterations; index += 1) {
      const middle = 0.5 * (left + right);
      if (middle === left || middle === right) break;
      const value = fn(middle);
      if ((value <= 0) === (leftValue <= 0)) {
        left = middle;
        leftValue = value;
      } else {
        right = middle;
      }
    }
    return 0.5 * (left + right);
  }

  /** Modulus where the increasing torsion branch c_m(k) reaches `cosine`. */
  function torsionLimitModulus(n, winding, cosine = 1 - 1e-8) {
    if (!(0 < 2 * winding && 2 * winding < n)) {
      throw new Error("The torsion limit requires 0 < m < N/2.");
    }
    return bisect(
      (k) => ellipticFamily(n, winding, k).torsionCosine - cosine,
      1e-12,
      1 - 1e-15,
    );
  }

  /** Moduli and the continuous lift Theta(k), normalised by Theta(0+) = 0. */
  function closingAngleCurve(n, winding, samples = 240) {
    const upper = torsionLimitModulus(n, winding);
    let grid = Array.from(
      { length: samples },
      (_, index) => 1e-4 + ((upper - 1e-4) * index) / (samples - 1),
    );
    const principal = (values) =>
      values.map((k) => closingAngle(ellipticFamily(n, winding, k)));
    const wrap = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));
    let angles = principal(grid);
    for (let pass = 0; pass < 6; pass += 1) {
      const coarse = [];
      for (let index = 0; index + 1 < grid.length; index += 1) {
        if (Math.abs(wrap(angles[index + 1] - angles[index])) > Math.PI / 2) {
          coarse.push(0.5 * (grid[index] + grid[index + 1]));
        }
      }
      if (!coarse.length) break;
      grid = [...grid, ...coarse].sort((left, right) => left - right);
      angles = principal(grid);
    }
    const theta = [angles[0]];
    for (let index = 1; index < grid.length; index += 1) {
      theta.push(theta[index - 1] + wrap(angles[index] - angles[index - 1]));
    }
    const offset = 2 * Math.PI * Math.round(theta[0] / (2 * Math.PI));
    return { moduli: grid, theta: theta.map((value) => value - offset) };
  }

  const modulusCache = new Map();

  /** Closing moduli k_{N,m,j}, j = 1, …, floor((m − 1)/2). */
  function closingModuli(n, winding) {
    const key = `${n}:${winding}`;
    if (modulusCache.has(key)) return modulusCache.get(key);
    const m = Number(winding);
    let roots = [];
    if (2 * m === n) {
      for (let level = 1; level < n / 4; level += 1) {
        roots.push(Math.sin((2 * Math.PI * level) / n));
      }
    } else {
      if (!(0 < 2 * m && 2 * m < n)) {
        throw new Error("Closing moduli need 0 < m ≤ N/2.");
      }
      const curve = closingAngleCurve(n, m);
      roots = [];
      for (let level = 1; level <= Math.floor((m - 1) / 2); level += 1) {
        const target = -2 * Math.PI * level;
        const index = curve.theta.findIndex((value, position) =>
          position + 1 < curve.theta.length &&
          (value - target) * (curve.theta[position + 1] - target) <= 0);
        if (index < 0) {
          throw new Error(`Closing level ${level} is not bracketed for N=${n}, m=${m}.`);
        }
        roots.push(bisect(
          (k) => closingAngle(ellipticFamily(n, m, k)),
          curve.moduli[index],
          curve.moduli[index + 1],
        ));
      }
    }
    modulusCache.set(key, roots);
    return roots;
  }

  /** Number of closing branches for N hinges and winding 3 ≤ m ≤ N/2. */
  function closingLevelCount(n, winding) {
    if (2 * winding === n) return Math.ceil(n / 4) - 1;
    return Math.floor((winding - 1) / 2);
  }

  function hingesFromCurvatures(curvatures, torsionAngle, sign) {
    return Mathematics.reconstruct(curvatures, torsionAngle, sign, IDENTITY)
      .frames.map((frame) => column(frame, 2));
  }

  const mirrorHinges = (hinges) =>
    hinges.map((hinge, index) =>
      index % 2 === 0 ? hinge.slice() : hinge.map((value) => -value));

  /**
   * A closed critical circle in the cn family. `mirror` applies the mirror
   * duality B[n] -> (−1)^n B[n], so μ* = π − μ and m* = N − m.
   */
  function criticalCircle(n, winding, level = 1, { mirror = false } = {}) {
    const roots = closingModuli(n, winding);
    if (!(level >= 1 && level <= roots.length)) {
      throw new Error(`Choose a closing index 1…${roots.length} for N=${n}, m=${winding}.`);
    }
    const family = ellipticFamily(n, winding, roots[level - 1]);
    const torsionAngle = familyTorsionAngle(family);
    const effectiveWinding = mirror ? n - winding : winding;
    const inertia = n >= 7
      ? [n - effectiveWinding - 3, effectiveWinding - 3, 1]
      : null;
    const sign = mirror ? family.sign * (n % 2 === 0 ? 1 : -1) : family.sign;
    const rightAngle = 2 * winding === n;
    return {
      family,
      level,
      mirror,
      n,
      winding: effectiveWinding,
      sign,
      torsionAngle: mirror ? Math.PI - torsionAngle : torsionAngle,
      inertia,
      rightAngle,
      hingesAt(phase) {
        const hinges = hingesFromCurvatures(family.curvatures(phase), torsionAngle, family.sign);
        return mirror ? mirrorHinges(hinges) : hinges;
      },
    };
  }

  /** Explicit torus-knot ring of the quadratic-linking proposition. */
  function torusKnotHinges(cells, turns, { mirror = false } = {}) {
    const m = Number(cells);
    const k = Number(turns);
    if (!(Number.isInteger(m) && m >= 3 && k >= 1 && 2 * k < m && gcd(m, k) === 1)) {
      throw new Error("Need cells ≥ 3, 1 ≤ turns < cells/2 and gcd(cells, turns) = 1.");
    }
    const alpha = (2 * Math.PI * k) / m;
    const beta = (Math.PI - alpha) / 3;
    const h = Math.cos(beta);
    const d = (1 + h) * (1 + 3 * h);
    const r = Math.sqrt((1 + 4 * h) / d);
    const z = (-Math.sqrt(3) * h) / Math.sqrt(d);
    const t = (Math.sqrt(3) * (1 + 2 * h)) / (2 * Math.sqrt(d));
    const triple = [
      [r * h, r * Math.sin(beta), z],
      [r / 2, 0, t],
      [r * h, -r * Math.sin(beta), z],
    ];
    const hinges = [];
    for (let cell = 0; cell < m; cell += 1) {
      const c = Math.cos(cell * alpha);
      const s = Math.sin(cell * alpha);
      triple.forEach(([x, y, w]) => hinges.push([c * x - s * y, s * x + c * y, w]));
    }
    hinges.push(triple[0].slice());
    return mirror ? mirrorHinges(hinges) : hinges;
  }

  global.KaleidocycleFamilies = Object.freeze({
    closingAngle,
    closingAngleCurve,
    closingLevelCount,
    closingModuli,
    criticalCircle,
    ellipticFamily,
    ellipticK,
    gcd,
    jacobiFunctions,
    mirrorHinges,
    torsionLimitModulus,
    torusKnotHinges,
  });
})(typeof window === "undefined" ? globalThis : window);
