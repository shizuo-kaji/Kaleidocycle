"""Interval certificate for the isolated K12 contact, not for a time integrator.

All decimal inputs and interval operations use outward-rounded mpmath intervals.
The Jacobian enclosure is obtained by forward automatic differentiation over
an entire box. The fixed point of y -> y - B f(y) proves an exact closed contact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mpmath.ctx_iv import MPIntervalContext

from .collisions import closest_segment_points, load_crossing_data


def certify_crossing(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify existence, regularity, separation and transverse first-flow speed.

    Raises ArithmeticError if any certificate inequality fails. This verifies
    the stored centre and inverse matrix without trusting a root finder.
    The returned bounds are deliberately rounded outwards for readable output.
    """
    data = load_crossing_data() if data is None else data
    iv = MPIntervalContext()
    iv.dps = 70
    n = data["n"]
    sign = data["sign"]
    active = data["active"]
    size = len(active)
    centre = [iv.mpf(x) for x in data["center"]]
    radius = iv.mpf(data["root_radius"])
    mu = iv.mpf(data["torsion_angle"])
    cosine, sine = iv.cos(mu), iv.sin(mu)
    edge_i, edge_j = data["edges"]

    class Dual:
        def __init__(self, value: Any, derivative: list[Any] | None = None) -> None:
            self.value = iv.mpf(value)
            self.derivative = (
                derivative if derivative is not None else [iv.mpf(0)] * size
            )

        def __add__(self, other: Any) -> Dual:
            other = other if isinstance(other, Dual) else Dual(other)
            return Dual(
                self.value + other.value,
                [a + b for a, b in zip(self.derivative, other.derivative, strict=True)],
            )

        __radd__ = __add__

        def __neg__(self) -> Dual:
            return Dual(-self.value, [-d for d in self.derivative])

        def __sub__(self, other: Any) -> Dual:
            return self + -(other if isinstance(other, Dual) else Dual(other))

        def __mul__(self, other: Any) -> Dual:
            other = other if isinstance(other, Dual) else Dual(other)
            return Dual(
                self.value * other.value,
                [
                    a * other.value + self.value * b
                    for a, b in zip(self.derivative, other.derivative, strict=True)
                ],
            )

        __rmul__ = __mul__

        def __truediv__(self, other: Any) -> Dual:
            return self * (iv.mpf(1) / other)

    def sin(value: Dual) -> Dual:
        return Dual(
            iv.sin(value.value), [iv.cos(value.value) * d for d in value.derivative]
        )

    def cos(value: Dual) -> Dual:
        return Dual(
            iv.cos(value.value), [-iv.sin(value.value) * d for d in value.derivative]
        )

    def multiply(a: list[list[Any]], b: list[list[Any]]) -> list[list[Any]]:
        return [
            [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]

    def evaluate(values: list[Any]) -> tuple[Any, Any, Any, Any, Any]:
        xs = [
            Dual(value, [iv.mpf(int(i == k)) for k in active])
            for i, value in enumerate(values)
        ]
        torsion = [[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]]
        frame = [[Dual(int(i == j)) for j in range(3)] for i in range(3)]
        point = [Dual(0) for _ in range(3)]
        frames, points = [], []
        for i in range(n):
            frames.append(frame)
            points.append(point)
            point = [point[k] + frame[k][0] for k in range(3)]
            angle = xs[i + 1] if i < n - 1 else sign * xs[0]
            c, s = cos(angle), sin(angle)
            frame = multiply(
                multiply(frame, torsion), [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            )
        points.append(point)
        monodromy = multiply(frame, [[1, 0, 0], [0, sign, 0], [0, 0, sign]])
        contact = [
            points[edge_i][k]
            + xs[n] * frames[edge_i][k][0]
            - points[edge_j][k]
            - xs[n + 1] * frames[edge_j][k][0]
            for k in range(3)
        ]
        equations = (
            points[-1]
            + [
                (monodromy[2][1] - monodromy[1][2]) / 2,
                (monodromy[0][2] - monodromy[2][0]) / 2,
                (monodromy[1][0] - monodromy[0][1]) / 2,
            ]
            + contact
        )
        return (
            iv.matrix([e.value for e in equations]),
            iv.matrix([e.derivative for e in equations]),
            [iv.matrix([[v.value for v in row] for row in f]) for f in frames],
            [iv.matrix([v.value for v in p]) for p in points],
            sum(monodromy[i][i].value for i in range(3)),
        )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ArithmeticError(message)

    def norm_upper(matrix: Any) -> Any:
        rows = [
            sum(abs(matrix[i, j]) for j in range(matrix.cols)).b
            for i in range(matrix.rows)
        ]
        return max(rows)

    inverse = iv.matrix(data["inverse_jacobian"])
    f0, jacobian0, _, centre_points, _ = evaluate(centre)
    box = [
        v + iv.mpf([-radius.b, radius.b]) if i in active else v
        for i, v in enumerate(centre)
    ]
    _, jacobian_box, frames, points, trace = evaluate(box)
    eta = norm_upper(inverse * f0)
    contraction = norm_upper(iv.eye(size) - inverse * jacobian_box)
    require(
        norm_upper(iv.eye(size) - inverse * jacobian0) < 1,
        "B is not certified invertible",
    )
    require(eta < iv.mpf("1e-58"), "Newton correction exceeds bound")
    require(contraction < iv.mpf("1e-15"), "Newton map is not a certified contraction")
    require(eta + contraction * radius < radius, "Newton map leaves its box")
    require(trace.a > iv.mpf("2.9"), "Skew frame equations may select a half-turn")
    require(all(abs(v).b < iv.pi.a for v in box[:n]), "A joint may fold back")
    require(all(v.a > 0 and v.b < 1 for v in box[n:]), "Contact is not interior")

    velocities = []
    for angle, frame in zip(box[:n], frames, strict=True):
        z = iv.tan(angle / 2)
        velocities.append(frame * iv.matrix([cosine, -cosine * z, -sine * z]))
    velocities.append(velocities[0])
    u, v = box[-2:]
    relative = (
        (1 - u) * velocities[edge_i]
        + u * velocities[edge_i + 1]
        - (1 - v) * velocities[edge_j]
        - v * velocities[edge_j + 1]
    )
    a, b = frames[edge_i][:, 0], frames[edge_j][:, 0]
    normal = iv.matrix(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
    speed = sum(normal[k] * relative[k] for k in range(3))
    require(
        speed.a > iv.mpf("-3.283") and speed.b < iv.mpf("-3.281"),
        "Transverse speed does not satisfy its nonzero bound",
    )

    # Floating-point nearest points only choose separating directions. The
    # actual endpoint separation and direction lengths are checked by intervals.
    approximate = np.asarray(
        [[float(p[k].mid) for k in range(3)] for p in centre_points]
    )
    checked_pairs = 0
    for i in range(n):
        for j in range(i + 2, n):
            if (i, j) in [(0, n - 1), (edge_i, edge_j)]:
                continue
            _, u0, v0 = closest_segment_points(
                *approximate[i : i + 2], *approximate[j : j + 2]
            )
            direction = (
                approximate[i]
                + u0 * (approximate[i + 1] - approximate[i])
                - approximate[j]
                - v0 * (approximate[j + 1] - approximate[j])
            )
            direction /= np.linalg.norm(direction)
            direction_iv = [iv.mpf(str(component)) for component in direction]
            require(
                sum(d * d for d in direction_iv).b < iv.mpf("1.0001") ** 2,
                "Separating direction has excessive norm",
            )

            def project(index: int, direction: list[Any] = direction_iv) -> Any:
                return sum(direction[k] * points[index][k] for k in range(3))

            gap = min(project(i).a, project(i + 1).a) - max(
                project(j).b, project(j + 1).b
            )
            require(
                gap.a > iv.mpf("0.0129") * iv.mpf("1.0001"),
                f"Edges {i}, {j} are not certified separated",
            )
            checked_pairs += 1
    return {
        "verified": True,
        "root_radius": data["root_radius"],
        "newton_correction_upper": "1e-58",
        "contraction_upper": "1e-15",
        "transverse_speed_bounds": [-3.283, -3.281],
        "other_edge_distance_lower": 0.0129,
        "separated_edge_pairs": checked_pairs,
        "interval_digits": iv.dps,
        "scope": (
            "Exact local contact and crossing; sampled trajectories are numerical."
        ),
    }
