/* Curvature-coordinate kaleidocycle model. No modules or network APIs. */
(function initialiseKaleidocycle(global) {
  "use strict";

  const EPSILON = 1e-12;
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scale = (a, value) => [a[0] * value, a[1] * value, a[2] * value];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm = (a) => Math.hypot(...a);

  function normalise(a) {
    const length = norm(a);
    if (!Number.isFinite(length) || length < EPSILON) {
      throw new Error("Expected a non-zero finite vector.");
    }
    return scale(a, 1 / length);
  }

  function matrixMultiply(a, b) {
    const result = new Array(9).fill(0);
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        for (let index = 0; index < 3; index += 1) {
          result[3 * row + column] +=
            a[3 * row + index] * b[3 * index + column];
        }
      }
    }
    return result;
  }

  const transpose = (a) => [
    a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8],
  ];
  const matrixColumn = (a, column) => [a[column], a[3 + column], a[6 + column]];

  function rotation1(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [1, 0, 0, 0, c, -s, 0, s, c];
  }

  function rotation3(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [c, -s, 0, s, c, 0, 0, 0, 1];
  }

  function twistedShift(values, offset, sign = 1) {
    const length = values.length;
    return values.map((_, index) => {
      const extendedIndex = index + offset;
      const quotient = Math.floor(extendedIndex / length);
      const remainder = ((extendedIndex % length) + length) % length;
      const factor = sign === -1 && Math.abs(quotient % 2) === 1 ? -1 : 1;
      return factor * values[remainder];
    });
  }

  const curvatureAngles = (curvatures) =>
    curvatures.map((value) => 2 * Math.atan(value / 2));
  const curvatureWeights = (curvatures) =>
    curvatures.map((value) => 1 + (value * value) / 4);
  const vectorNorm = (values) => Math.hypot(...values);

  function clampVector(values, maximumNorm) {
    const length = vectorNorm(values);
    if (length <= maximumNorm || length < EPSILON) return values;
    return values.map((value) => (value * maximumNorm) / length);
  }

  function solveLinear(matrix, rightHandSide) {
    const size = rightHandSide.length;
    const augmented = matrix.map((row, index) => [...row, rightHandSide[index]]);
    for (let column = 0; column < size; column += 1) {
      let pivot = column;
      for (let row = column + 1; row < size; row += 1) {
        if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) {
          pivot = row;
        }
      }
      if (Math.abs(augmented[pivot][column]) < 1e-14) return null;
      [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
      const divisor = augmented[column][column];
      for (let index = column; index <= size; index += 1) {
        augmented[column][index] /= divisor;
      }
      for (let row = 0; row < size; row += 1) {
        if (row === column) continue;
        const factor = augmented[row][column];
        for (let index = column; index <= size; index += 1) {
          augmented[row][index] -= factor * augmented[column][index];
        }
      }
    }
    return augmented.map((row) => row[size]);
  }

  const transposeRectangular = (matrix) =>
    matrix[0].map((_, column) => matrix.map((row) => row[column]));

  function multiplyRectangular(a, b) {
    const result = Array.from({ length: a.length }, () =>
      new Array(b[0].length).fill(0),
    );
    for (let row = 0; row < a.length; row += 1) {
      for (let index = 0; index < b.length; index += 1) {
        for (let column = 0; column < b[0].length; column += 1) {
          result[row][column] += a[row][index] * b[index][column];
        }
      }
    }
    return result;
  }

  const matrixVector = (matrix, vector) =>
    matrix.map((row) =>
      row.reduce((sum, value, index) => sum + value * vector[index], 0),
    );
  const addDiagonal = (matrix, value) =>
    matrix.map((row, rowIndex) =>
      row.map((entry, columnIndex) =>
        rowIndex === columnIndex ? entry + value : entry,
      ),
    );

  function leastNormCorrection(jacobian, residual, damping = 1e-10) {
    const transposeJacobian = transposeRectangular(jacobian);
    const gram = addDiagonal(
      multiplyRectangular(jacobian, transposeJacobian),
      damping,
    );
    const multiplier = solveLinear(gram, residual);
    if (multiplier === null) return new Array(jacobian[0].length).fill(0);
    return matrixVector(transposeJacobian, multiplier).map((value) => -value);
  }

  const matrix2Multiply = (a, b) => [
    a[0] * b[0] + a[1] * b[2],
    a[0] * b[1] + a[1] * b[3],
    a[2] * b[0] + a[3] * b[2],
    a[2] * b[1] + a[3] * b[3],
  ];
  const matrix2Add = (a, b) => a.map((value, index) => value + b[index]);
  const matrix2Scale = (a, value) => a.map((entry) => entry * value);

  function binomial(n, k) {
    if (k < 0 || k > n) return 0;
    const reflected = Math.min(k, n - k);
    let value = 1;
    for (let index = 1; index <= reflected; index += 1) {
      value = (value * (n - reflected + index)) / index;
    }
    return value;
  }

  function spectralIntegralGradients(curvatures, count, sign) {
    const n = curvatures.length;
    let coefficients = Array.from({ length: count }, (_, degree) =>
      degree === 0 ? [1, 0, 0, 1] : [0, 0, 0, 0],
    );
    let gradients = Array.from({ length: count }, () =>
      Array.from({ length: n }, () => [0, 0, 0, 0]),
    );

    curvatures.forEach((curvature, site) => {
      const z = curvature / 2;
      const weight = 1 + z * z;
      const factorScale = weight ** -0.5;
      const scaleGradient = -curvature / (4 * weight ** 1.5);
      const factor0Base = [1, 0, -z, 0];
      const factor1Base = [0, z, 0, 1];
      const factor0 = matrix2Scale(factor0Base, factorScale);
      const factor1 = matrix2Scale(factor1Base, factorScale);
      const factor0Gradient = matrix2Add(
        matrix2Scale(factor0Base, scaleGradient),
        matrix2Scale([0, 0, -0.5, 0], factorScale),
      );
      const factor1Gradient = matrix2Add(
        matrix2Scale(factor1Base, scaleGradient),
        matrix2Scale([0, 0.5, 0, 0], factorScale),
      );
      const updated = Array.from({ length: count }, () => [0, 0, 0, 0]);
      const updatedGradients = Array.from({ length: count }, () =>
        Array.from({ length: n }, () => [0, 0, 0, 0]),
      );

      for (let degree = 0; degree < count; degree += 1) {
        updated[degree] = matrix2Multiply(factor0, coefficients[degree]);
        for (let variable = 0; variable < n; variable += 1) {
          updatedGradients[degree][variable] = matrix2Multiply(
            factor0,
            gradients[degree][variable],
          );
        }
        updatedGradients[degree][site] = matrix2Add(
          updatedGradients[degree][site],
          matrix2Multiply(factor0Gradient, coefficients[degree]),
        );
        if (degree > 0) {
          updated[degree] = matrix2Add(
            updated[degree],
            matrix2Multiply(factor1, coefficients[degree - 1]),
          );
          for (let variable = 0; variable < n; variable += 1) {
            updatedGradients[degree][variable] = matrix2Add(
              updatedGradients[degree][variable],
              matrix2Multiply(factor1, gradients[degree - 1][variable]),
            );
          }
          updatedGradients[degree][site] = matrix2Add(
            updatedGradients[degree][site],
            matrix2Multiply(factor1Gradient, coefficients[degree - 1]),
          );
        }
      }
      coefficients = updated;
      gradients = updatedGradients;
    });

    const trace = coefficients.map((entry) => entry[0] + sign * entry[3]);
    const traceGradient = gradients.map((degree) =>
      degree.map((entry) => entry[0] + sign * entry[3]),
    );
    const multiplier = new Array(count).fill(0);
    const multiplierGradient = Array.from({ length: count }, () =>
      new Array(n).fill(0),
    );
    multiplier[0] = trace[0];
    multiplierGradient[0] = [...traceGradient[0]];
    for (let degree = 1; degree < count; degree += 1) {
      let remainder = 0;
      const remainderGradient = new Array(n).fill(0);
      for (let index = 1; index < degree; index += 1) {
        const left = multiplier[index] - trace[index];
        const right = multiplier[degree - index];
        remainder += left * right;
        for (let variable = 0; variable < n; variable += 1) {
          const leftGradient =
            multiplierGradient[index][variable] - traceGradient[index][variable];
          remainderGradient[variable] +=
            leftGradient * right +
            left * multiplierGradient[degree - index][variable];
        }
      }
      if (degree === n) remainder += sign;
      multiplier[degree] = trace[degree] - remainder / multiplier[0];
      for (let variable = 0; variable < n; variable += 1) {
        multiplierGradient[degree][variable] =
          traceGradient[degree][variable] -
          remainderGradient[variable] / multiplier[0] +
          (remainder * multiplierGradient[0][variable]) / multiplier[0] ** 2;
      }
    }

    const logarithm = new Array(count).fill(0);
    const logarithmGradient = Array.from({ length: count }, () =>
      new Array(n).fill(0),
    );
    logarithm[0] = Math.log(multiplier[0]);
    logarithmGradient[0] = multiplierGradient[0].map(
      (value) => value / multiplier[0],
    );
    for (let degree = 1; degree < count; degree += 1) {
      let numerator = degree * multiplier[degree];
      const numeratorGradient = multiplierGradient[degree].map(
        (value) => degree * value,
      );
      for (let index = 1; index < degree; index += 1) {
        numerator -= index * logarithm[index] * multiplier[degree - index];
        for (let variable = 0; variable < n; variable += 1) {
          numeratorGradient[variable] -=
            index *
            (logarithmGradient[index][variable] * multiplier[degree - index] +
              logarithm[index] * multiplierGradient[degree - index][variable]);
        }
      }
      const denominator = degree * multiplier[0];
      logarithm[degree] = numerator / denominator;
      for (let variable = 0; variable < n; variable += 1) {
        logarithmGradient[degree][variable] =
          (numeratorGradient[variable] * denominator -
            numerator * degree * multiplierGradient[0][variable]) /
          denominator ** 2;
      }
    }
    return logarithmGradient.map((gradient) =>
      gradient.map((value) => -2 * value),
    );
  }

  function hierarchyWeights(order) {
    const power = order - 1;
    const weights = new Array(order).fill(0);
    const stop = power === 0 ? 0 : Math.floor((power - 1) / 2) + 1;
    for (let index = 0; index < stop; index += 1) {
      weights[power - 2 * index] += binomial(power, index);
    }
    if (power % 2 === 0) weights[0] += binomial(power, power / 2);
    return weights;
  }

  function poissonOperator(curvatures, covector, sign) {
    const weights = curvatureWeights(curvatures);
    const weighted = weights.map((value, index) => value * covector[index]);
    const forward = twistedShift(weighted, 1, sign);
    const backward = twistedShift(weighted, -1, sign);
    return weights.map(
      (value, index) => value * (forward[index] - backward[index]),
    );
  }

  function hierarchyField(curvatures, order, sign) {
    const weights = curvatureWeights(curvatures);
    if (order === 1) {
      const forward = twistedShift(curvatures, 1, sign);
      const backward = twistedShift(curvatures, -1, sign);
      return weights.map(
        (value, index) => 0.5 * value * (forward[index] - backward[index]),
      );
    }
    if (order === 2) {
      const weightsForward = twistedShift(weights, 1, 1);
      const weightsBackward = twistedShift(weights, -1, 1);
      const plusTwo = twistedShift(curvatures, 2, sign);
      const minusTwo = twistedShift(curvatures, -2, sign);
      return weights.map(
        (value, index) =>
          0.5 * value *
          (weightsForward[index] * (curvatures[index] + plusTwo[index]) -
            weightsBackward[index] * (minusTwo[index] + curvatures[index])),
      );
    }
    const integralGradients = spectralIntegralGradients(curvatures, order, sign);
    const coefficients = hierarchyWeights(order);
    const gradient = new Array(curvatures.length).fill(0);
    for (let index = 0; index < order; index += 1) {
      for (let site = 0; site < curvatures.length; site += 1) {
        gradient[site] += coefficients[index] * integralGradients[index][site];
      }
    }
    return poissonOperator(curvatures, gradient, sign);
  }

  function sineGordonField(curvatures, torsionAngle, sign) {
    if (sign !== -1) {
      throw new Error("The sine–Gordon flow requires an anti-oriented cycle.");
    }
    const angles = curvatureAngles(curvatures);
    const potential = new Array(curvatures.length);
    potential[0] = 0.5 *
      (angles.slice(1).reduce((sum, value) => sum + value, 0) - angles[0]);
    for (let index = 1; index < potential.length; index += 1) {
      potential[index] = potential[index - 1] - angles[index];
    }
    const previous = twistedShift(potential, -1, -1);
    const factor = -(1 - Math.cos(torsionAngle));
    const weights = curvatureWeights(curvatures);
    return potential.map(
      (value, index) =>
        weights[index] * factor * (Math.sin(value) + Math.sin(previous[index])),
    );
  }

  const firstHamiltonian = (curvatures) =>
    curvatureWeights(curvatures).reduce(
      (sum, value) => sum + Math.log(value),
      0,
    );
  function secondHamiltonian(curvatures, sign) {
    const forward = twistedShift(curvatures, 1, sign);
    return 0.5 * curvatures.reduce(
      (sum, value, index) => sum + value * forward[index],
      0,
    );
  }

  const frameFromColumns = (tangent, normal, binormal) => [
    tangent[0], normal[0], binormal[0],
    tangent[1], normal[1], binormal[1],
    tangent[2], normal[2], binormal[2],
  ];

  function reconstruct(curvatures, torsionAngle, sign, initialFrame) {
    const frames = [initialFrame.slice()];
    const vertices = [[0, 0, 0]];
    const shiftedAngles = twistedShift(curvatureAngles(curvatures), 1, sign);
    const torsionRotation = rotation1(torsionAngle);
    for (let index = 0; index < curvatures.length; index += 1) {
      vertices.push(add(vertices[index], matrixColumn(frames[index], 0)));
      frames.push(matrixMultiply(
        frames[index],
        matrixMultiply(torsionRotation, rotation3(shiftedAngles[index])),
      ));
    }
    return { frames, vertices };
  }

  function centredVertices(configuration) {
    const period = configuration.vertices.length - 1;
    const centre = configuration.vertices.slice(0, period).reduce(add, [0, 0, 0]);
    const mean = scale(centre, 1 / period);
    return configuration.vertices.map((vertex) => subtract(vertex, mean));
  }

  function inferCycle(data) {
    if (!data || !Array.isArray(data.hinges) || data.hinges.length < 4) {
      throw new Error("The JSON file must contain a ‘hinges’ array.");
    }
    const statedN = Number(data.n ?? data.metadata?.n);
    const n = Number.isInteger(statedN) ? statedN : data.hinges.length - 1;
    if (data.hinges.length < n + 1) {
      throw new Error("The hinges array must include its terminal value.");
    }
    const binormals = data.hinges.slice(0, n + 1).map(normalise);
    const same = norm(subtract(binormals[n], binormals[0]));
    const opposite = norm(add(binormals[n], binormals[0]));
    const sign = same <= opposite ? 1 : -1;
    if (Math.min(same, opposite) > 2e-4) {
      throw new Error("The terminal hinge does not close with either orientation.");
    }
    const cosines = [];
    const tangents = [];
    for (let index = 0; index < n; index += 1) {
      cosines.push(dot(binormals[index], binormals[index + 1]));
      tangents.push(normalise(cross(binormals[index], binormals[index + 1])));
    }
    const cosineMean = cosines.reduce((sum, value) => sum + value, 0) / n;
    const cosineSpread = Math.sqrt(
      cosines.reduce((sum, value) => sum + (value - cosineMean) ** 2, 0) / n,
    );
    if (cosineSpread > 2e-4) {
      throw new Error(
        `Adjacent hinges do not have constant torsion (σ = ${cosineSpread.toExponential(2)}).`,
      );
    }
    if (Math.abs(cosineMean) > 1 - 1e-10) {
      throw new Error("The torsion angle must lie strictly between 0 and π.");
    }
    const torsionAngle = Math.acos(Math.max(-1, Math.min(1, cosineMean)));
    const normals = binormals.slice(0, n).map((binormal, index) =>
      cross(binormal, tangents[index]),
    );
    const curvatures = tangents.map((tangent, index) => {
      const previous = tangents[(index - 1 + n) % n];
      const sine = dot(tangent, cross(binormals[index], previous));
      const cosine = dot(previous, tangent);
      return 2 * Math.tan(Math.atan2(sine, cosine) / 2);
    });
    return {
      name: data.name,
      n,
      sign,
      binormals,
      torsionAngle,
      curvatures,
      initialFrame: frameFromColumns(tangents[0], normals[0], binormals[0]),
      metadata: { ...(data.metadata || {}) },
    };
  }

  class Model {
    constructor(data) {
      const parsed = inferCycle(data);
      Object.assign(this, parsed);
      this.name = String(parsed.name || parsed.metadata.name || `k${parsed.n}`);
      this.configuration = reconstruct(
        this.curvatures,
        this.torsionAngle,
        this.sign,
        this.initialFrame,
      );
      this.initialState = this.snapshot();
      this.flowTime = 0;
    }

    snapshot() {
      return {
        curvatures: this.curvatures.slice(),
        torsionAngle: this.torsionAngle,
        initialFrame: this.initialFrame.slice(),
      };
    }

    restore(state = this.initialState) {
      this.curvatures = state.curvatures.slice();
      this.torsionAngle = state.torsionAngle;
      this.initialFrame = state.initialFrame.slice();
      this.flowTime = 0;
      this.update();
    }

    update() {
      this.configuration = reconstruct(
        this.curvatures,
        this.torsionAngle,
        this.sign,
        this.initialFrame,
      );
    }

    displayVertices(curvatures = this.curvatures, torsion = this.torsionAngle) {
      return centredVertices(
        reconstruct(curvatures, torsion, this.sign, this.initialFrame),
      );
    }

    residual(curvatures = this.curvatures, torsion = this.torsionAngle) {
      const configuration = reconstruct(
        curvatures,
        torsion,
        this.sign,
        this.initialFrame,
      );
      const lastVertex = configuration.vertices[configuration.vertices.length - 1];
      const diagonal = [1, 0, 0, 0, this.sign, 0, 0, 0, this.sign];
      const target = matrixMultiply(this.initialFrame, diagonal);
      const lastFrame = configuration.frames[configuration.frames.length - 1];
      const relative = matrixMultiply(transpose(target), lastFrame);
      return [
        ...lastVertex,
        0.5 * (relative[7] - relative[5]),
        0.5 * (relative[2] - relative[6]),
        0.5 * (relative[3] - relative[1]),
      ];
    }

    constraintJacobian(curvatures = this.curvatures, torsion = this.torsionAngle) {
      const rows = Array.from({ length: 6 }, () =>
        new Array(curvatures.length).fill(0),
      );
      for (let column = 0; column < curvatures.length; column += 1) {
        const epsilon = 2e-5 * Math.max(1, Math.abs(curvatures[column]));
        const plus = curvatures.slice();
        const minus = curvatures.slice();
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const residualPlus = this.residual(plus, torsion);
        const residualMinus = this.residual(minus, torsion);
        for (let row = 0; row < 6; row += 1) {
          rows[row][column] =
            (residualPlus[row] - residualMinus[row]) / (2 * epsilon);
        }
      }
      return rows;
    }

    project(maxIterations = 8, tolerance = 2e-8) {
      for (let iteration = 0; iteration < maxIterations; iteration += 1) {
        const residual = this.residual();
        if (vectorNorm(residual) < tolerance) {
          this.update();
          return true;
        }
        const correction = clampVector(
          leastNormCorrection(this.constraintJacobian(), residual, 1e-10),
          0.35,
        );
        this.curvatures = this.curvatures.map(
          (value, index) => value + correction[index],
        );
      }
      this.update();
      return vectorNorm(this.residual()) < 2e-5;
    }

    setTorsion(target) {
      const next = Math.max(0.08, Math.min(Math.PI - 0.08, Number(target)));
      const previous = this.snapshot();
      const steps = Math.max(
        1,
        Math.ceil(Math.abs(next - this.torsionAngle) / 0.025),
      );
      for (let step = 1; step <= steps; step += 1) {
        this.torsionAngle = previous.torsionAngle +
          ((next - previous.torsionAngle) * step) / steps;
        if (!this.project(12, 1e-8)) {
          this.curvatures = previous.curvatures;
          this.torsionAngle = previous.torsionAngle;
          this.update();
          return false;
        }
      }
      return true;
    }

    dragVertex(vertexIndex, target) {
      const index = Math.max(0, Math.min(this.n - 1, vertexIndex));
      for (let iteration = 0; iteration < 7; iteration += 1) {
        const residual = this.residual();
        const constraintJacobian = this.constraintJacobian();
        const constraintTranspose = transposeRectangular(constraintJacobian);
        const constraintGram = addDiagonal(
          multiplyRectangular(constraintJacobian, constraintTranspose),
          1e-9,
        );
        const identity = Array.from({ length: this.n }, (_, row) =>
          Array.from({ length: this.n }, (_, column) => (row === column ? 1 : 0)),
        );
        const inverseColumns = [];
        for (let column = 0; column < 6; column += 1) {
          const basis = new Array(6).fill(0);
          basis[column] = 1;
          inverseColumns.push(solveLinear(constraintGram, basis));
        }
        if (inverseColumns.some((column) => column === null)) return false;
        const inverseGram = transposeRectangular(inverseColumns);
        const projectorCorrection = multiplyRectangular(
          multiplyRectangular(constraintTranspose, inverseGram),
          constraintJacobian,
        );
        const projector = identity.map((row, rowIndex) =>
          row.map((value, columnIndex) =>
            value - projectorCorrection[rowIndex][columnIndex],
          ),
        );
        const constraintStep = leastNormCorrection(
          constraintJacobian,
          residual,
          1e-9,
        );
        const positionJacobian = Array.from({ length: 3 }, () =>
          new Array(this.n).fill(0),
        );
        for (let column = 0; column < this.n; column += 1) {
          const epsilon = 2e-5 * Math.max(1, Math.abs(this.curvatures[column]));
          const plus = this.curvatures.slice();
          const minus = this.curvatures.slice();
          plus[column] += epsilon;
          minus[column] -= epsilon;
          const plusPosition = this.displayVertices(plus)[index];
          const minusPosition = this.displayVertices(minus)[index];
          for (let row = 0; row < 3; row += 1) {
            positionJacobian[row][column] =
              (plusPosition[row] - minusPosition[row]) / (2 * epsilon);
          }
        }
        const position = this.displayVertices()[index];
        const wanted = subtract(target, position);
        const alreadyMoved = matrixVector(positionJacobian, constraintStep);
        const tangentJacobian = multiplyRectangular(positionJacobian, projector);
        const dragGram = addDiagonal(
          multiplyRectangular(
            tangentJacobian,
            transposeRectangular(positionJacobian),
          ),
          3e-4,
        );
        const dragMultiplier = solveLinear(
          dragGram,
          subtract(wanted, alreadyMoved),
        );
        if (dragMultiplier === null) return false;
        const tangentStep = matrixVector(
          multiplyRectangular(projector, transposeRectangular(positionJacobian)),
          dragMultiplier,
        );
        const totalStep = clampVector(
          constraintStep.map(
            (value, site) => value + 0.72 * tangentStep[site],
          ),
          0.28,
        );
        this.curvatures = this.curvatures.map(
          (value, site) => value + totalStep[site],
        );
      }
      this.project(5, 2e-8);
      return true;
    }

    vectorField(kind, order = 1, values = this.curvatures) {
      return kind === "sine-gordon"
        ? sineGordonField(values, this.torsionAngle, this.sign)
        : hierarchyField(values, order, this.sign);
    }

    advanceFlow(kind, order, requestedStep) {
      const first = this.vectorField(kind, order);
      const maximum = Math.max(...first.map(Math.abs), EPSILON);
      const safeStep = Math.sign(requestedStep) * Math.min(
        Math.abs(requestedStep),
        0.018 / maximum,
      );
      const substeps = Math.max(
        1,
        Math.min(8, Math.ceil((Math.abs(safeStep) * maximum) / 0.009)),
      );
      const step = safeStep / substeps;
      for (let substep = 0; substep < substeps; substep += 1) {
        const start = this.curvatures;
        const k1 = this.vectorField(kind, order, start);
        const atK2 = start.map(
          (value, index) => value + 0.5 * step * k1[index],
        );
        const k2 = this.vectorField(kind, order, atK2);
        const atK3 = start.map(
          (value, index) => value + 0.5 * step * k2[index],
        );
        const k3 = this.vectorField(kind, order, atK3);
        const atK4 = start.map((value, index) => value + step * k3[index]);
        const k4 = this.vectorField(kind, order, atK4);
        this.curvatures = start.map(
          (value, index) => value +
            (step / 6) *
              (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index]),
        );
      }
      this.flowTime += safeStep;
      this.update();
      return safeStep;
    }

    diagnostics() {
      const closure = norm(
        this.configuration.vertices[this.configuration.vertices.length - 1],
      );
      const diagonal = [1, 0, 0, 0, this.sign, 0, 0, 0, this.sign];
      const target = matrixMultiply(this.initialFrame, diagonal);
      const finalFrame = this.configuration.frames[this.configuration.frames.length - 1];
      const monodromy = Math.hypot(
        ...finalFrame.map((value, index) => value - target[index]),
      );
      return {
        closure,
        monodromy,
        hamiltonian1: firstHamiltonian(this.curvatures),
        hamiltonian2: secondHamiltonian(this.curvatures, this.sign),
      };
    }

    toJSON() {
      const vertices = centredVertices(this.configuration);
      const hinges = this.configuration.frames.map((frame) => matrixColumn(frame, 2));
      const tangents = this.configuration.frames
        .slice(0, this.n)
        .map((frame) => matrixColumn(frame, 0));
      return {
        name: this.name,
        metadata: {
          ...this.metadata,
          name: this.name,
          n: this.n,
          oriented: this.sign === 1,
          edited_in: "Kaleidocycle Studio",
        },
        n: this.n,
        curve: vertices,
        hinges,
        tangents,
        cos_mean: Math.cos(this.torsionAngle),
        cos_std: 0,
      };
    }
  }

  global.Kaleidocycle = Object.freeze({
    Model,
    curvatureAngles,
    hierarchyField,
    inferCycle,
    reconstruct,
    sineGordonField,
    twistedShift,
  });
})(typeof window === "undefined" ? globalThis : window);
