"""Jacobi theta functions and related utilities for exact kaleidocycle solutions.

This module provides Jacobi theta functions and derived quantities needed
for computing exact analytic solutions to kaleidocycle curves using
integrable systems theory.

References:
    Corresponds to theta function definitions in Maple code (lines 520-545)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _jtheta1(z: complex | NDArray, q: complex) -> complex | NDArray:
    """Jacobi theta function 1 computed via series expansion.

    theta1(z, q) = 2 * sum_{n=0}^inf (-1)^n * q^{(n+1/2)^2} * sin((2n+1)z)

    Args:
        z: Argument
        q: Nome parameter

    Returns:
        Theta function value
    """
    if abs(q) >= 1:
        raise ValueError(f"Nome parameter |q| must be < 1, got |q| = {abs(q)}")

    result = 0.0 + 0j
    max_terms = 100
    tol = 1e-15

    for n in range(max_terms):
        term_power = (n + 0.5) ** 2
        term = (-1) ** n * (q ** term_power) * np.sin((2 * n + 1) * z)

        result += term

        # Check convergence
        if abs(term) < tol * abs(result) and abs(result) > tol:
            break

    return 2 * result


def _jtheta2(z: complex | NDArray, q: complex) -> complex | NDArray:
    """Jacobi theta function 2 computed via series expansion.

    theta2(z, q) = 2 * sum_{n=0}^inf q^{(n+1/2)^2} * cos((2n+1)z)

    Args:
        z: Argument
        q: Nome parameter

    Returns:
        Theta function value
    """
    if abs(q) >= 1:
        raise ValueError(f"Nome parameter |q| must be < 1, got |q| = {abs(q)}")

    result = 0.0 + 0j
    max_terms = 100
    tol = 1e-15

    for n in range(max_terms):
        term_power = (n + 0.5) ** 2
        term = (q ** term_power) * np.cos((2 * n + 1) * z)

        result += term

        # Check convergence
        if abs(term) < tol * abs(result) and abs(result) > tol:
            break

    return 2 * result


def _jtheta3(z: complex | NDArray, q: complex) -> complex | NDArray:
    """Jacobi theta function 3 computed via series expansion.

    theta3(z, q) = 1 + 2 * sum_{n=1}^inf q^{n^2} * cos(2nz)

    Args:
        z: Argument
        q: Nome parameter

    Returns:
        Theta function value
    """
    if abs(q) >= 1:
        raise ValueError(f"Nome parameter |q| must be < 1, got |q| = {abs(q)}")

    result = 1.0 + 0j
    max_terms = 100
    tol = 1e-15

    for n in range(1, max_terms):
        term = (q ** (n * n)) * np.cos(2 * n * z)

        result += 2 * term

        # Check convergence
        if abs(term) < tol * abs(result):
            break

    return result


def _jtheta4(z: complex | NDArray, q: complex) -> complex | NDArray:
    """Jacobi theta function 4 computed via series expansion.

    theta4(z, q) = 1 + 2 * sum_{n=1}^inf (-1)^n * q^{n^2} * cos(2nz)

    Args:
        z: Argument
        q: Nome parameter

    Returns:
        Theta function value
    """
    if abs(q) >= 1:
        raise ValueError(f"Nome parameter |q| must be < 1, got |q| = {abs(q)}")

    result = 1.0 + 0j
    max_terms = 100
    tol = 1e-15

    for n in range(1, max_terms):
        term = (-1) ** n * (q ** (n * n)) * np.cos(2 * n * z)

        result += 2 * term

        # Check convergence
        if abs(term) < tol * abs(result):
            break

    return result


def theta1(x: complex | NDArray, y: float) -> complex | NDArray:
    """Jacobi theta function 1.

    Args:
        x: Argument (can be complex or array)
        y: Nome parameter (real, positive)

    Returns:
        Theta function value

    References:
        Maple: th1(x, y) = JacobiTheta1(Pi*x, exp(-Pi*y))
    """
    z = np.pi * x
    q = np.exp(-np.pi * y)
    return _jtheta1(z, q)


def theta2(x: complex | NDArray, y: float) -> complex | NDArray:
    """Jacobi theta function 2.

    Args:
        x: Argument (can be complex or array)
        y: Nome parameter (real, positive)

    Returns:
        Theta function value

    References:
        Maple: th2(x, y) = JacobiTheta2(Pi*x, exp(-Pi*y))
    """
    z = np.pi * x
    q = np.exp(-np.pi * y)
    return _jtheta2(z, q)


def theta3(x: complex | NDArray, y: float) -> complex | NDArray:
    """Jacobi theta function 3.

    Args:
        x: Argument (can be complex or array)
        y: Nome parameter (real, positive)

    Returns:
        Theta function value

    References:
        Maple: th3(x, y) = JacobiTheta3(Pi*x, exp(-Pi*y))
    """
    z = np.pi * x
    q = np.exp(-np.pi * y)
    return _jtheta3(z, q)


def theta4(x: complex | NDArray, y: float) -> complex | NDArray:
    """Jacobi theta function 4.

    Args:
        x: Argument (can be complex or array)
        y: Nome parameter (real, positive)

    Returns:
        Theta function value

    References:
        Maple: th4(x, y) = JacobiTheta4(Pi*x, exp(-Pi*y))
    """
    z = np.pi * x
    q = np.exp(-np.pi * y)
    return _jtheta4(z, q)


def theta1_derivative(x: complex, y: float, eps: float = 1e-8) -> complex:
    """Derivative of theta1 with respect to x.

    Args:
        x: Argument
        y: Nome parameter
        eps: Step size for numerical differentiation

    Returns:
        Derivative value
    """
    return (theta1(x + eps, y) - theta1(x - eps, y)) / (2 * eps)


def theta2_derivative(x: complex, y: float, eps: float = 1e-8) -> complex:
    """Derivative of theta2 with respect to x."""
    return (theta2(x + eps, y) - theta2(x - eps, y)) / (2 * eps)


def theta3_derivative(x: complex, y: float, eps: float = 1e-8) -> complex:
    """Derivative of theta3 with respect to x."""
    return (theta3(x + eps, y) - theta3(x - eps, y)) / (2 * eps)


def theta4_derivative(x: complex, y: float, eps: float = 1e-8) -> complex:
    """Derivative of theta4 with respect to x."""
    return (theta4(x + eps, y) - theta4(x - eps, y)) / (2 * eps)


def logarithmic_derivative_1(r: complex, y: float) -> complex:
    """Logarithmic derivative d1 = dth1/th1.

    Args:
        r: Argument
        y: Nome parameter

    Returns:
        Logarithmic derivative

    References:
        Maple: d1 := (r, y) -> dth1(r, y)/th1(r, y)
    """
    return theta1_derivative(r, y) / theta1(r, y)


def logarithmic_derivative_3(r: complex, y: float) -> complex:
    """Logarithmic derivative d3 = dth3/th3.

    Args:
        r: Argument
        y: Nome parameter

    Returns:
        Logarithmic derivative

    References:
        Maple: d3 := (r, y) -> dth3(r, y)/th3(r, y)
    """
    return theta3_derivative(r, y) / theta3(r, y)


def R1(v: float, r: float, y: float) -> complex:
    """Ratio function R1.

    Args:
        v: Parameter v (real)
        r: Parameter r (real)
        y: Nome parameter (real, positive)

    Returns:
        Ratio value

    References:
        Maple: R1 := (v, r, y) -> th1(-1/2*I*v + r, y)/th1(1/2*I*v + r, y)
    """
    arg1 = -0.5j * v + r
    arg2 = 0.5j * v + r
    return theta1(arg1, y) / theta1(arg2, y)


def R3(v: float, r: float, y: float) -> complex:
    """Ratio function R3.

    Args:
        v: Parameter v (real)
        r: Parameter r (real)
        y: Nome parameter (real, positive)

    Returns:
        Ratio value

    References:
        Maple: R3 := (v, r, y) -> th3(-1/2*I*v + r, y)/th3(1/2*I*v + r, y)
    """
    arg1 = -0.5j * v + r
    arg2 = 0.5j * v + r
    return theta3(arg1, y) / theta3(arg2, y)


def delta1(v: float, r: float, y: float) -> complex:
    """Delta function delta1.

    Args:
        v: Parameter v (real)
        r: Parameter r (real)
        y: Nome parameter (real, positive)

    Returns:
        Delta value

    References:
        Maple: delta1 := (v, r, y) -> d1(-1/2*I*v + r, y) - d1(1/2*I*v + r, y)
    """
    arg1 = -0.5j * v + r
    arg2 = 0.5j * v + r
    return logarithmic_derivative_1(arg1, y) - logarithmic_derivative_1(arg2, y)


def delta3(v: float, r: float, y: float) -> complex:
    """Delta function delta3.

    Args:
        v: Parameter v (real)
        r: Parameter r (real)
        y: Nome parameter (real, positive)

    Returns:
        Delta value

    References:
        Maple: delta3 := (v, r, y) -> d3(-1/2*I*v + r, y) - d3(1/2*I*v + r, y)
    """
    arg1 = -0.5j * v + r
    arg2 = 0.5j * v + r
    return logarithmic_derivative_3(arg1, y) - logarithmic_derivative_3(arg2, y)


def alpha2(r: float, y: float) -> complex:
    """Alpha2 parameter.

    Args:
        r: Parameter r (real)
        y: Nome parameter (real, positive)

    Returns:
        Alpha2 value

    References:
        Maple: alpha2 := (r, y) -> sqrt(th3(2*r, y)*th3(0, y))
    """
    return np.sqrt(theta3(2 * r, y) * theta3(0, y))


def eC(v: float, r: float, y: float) -> complex:
    """Evolution coefficient C.

    Args:
        v: Parameter v (real)
        r: Parameter r (real)
        y: Nome parameter (real, positive)

    Returns:
        C coefficient

    References:
        Maple: eC := (v, r, y) -> 4*I*Pi/(y*(delta3(v, r, y) - delta1(v, r, y)))
    """
    return 4j * np.pi / (y * (delta3(v, r, y) - delta1(v, r, y)))


def Gamma(r: float, y: float, b: float) -> complex:
    """Gamma phase parameter.

    Args:
        r: Parameter r (real)
        y: Nome parameter (real, positive)
        b: Beta parameter

    Returns:
        Gamma value

    References:
        Mathematica: Gam[r,y,b]:=-Pi/y*(4*r+1-2*b)
    """
    return -np.pi / y * (4 * r + 1 - 2 * b)


def f_wave(v: float, n: int, z: float, r: float, y: float, t: float) -> complex:
    """Wave function f for theta-based binormal computation.

    Args:
        v: Parameter v
        n: Index (0 to N)
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        t: Time parameter

    Returns:
        Complex wave function value

    References:
        Mathematica: f[v,n,z,r,y,t] (line 191)
    """
    d3 = delta3(v, r, y)
    d1 = delta1(v, r, y)
    c = eC(v, r, y)
    gam = Gamma(r, y, 1)

    exp_term = np.exp(
        (n + 0.5) * d3 / (d3 - d1) * z + c * t * z / 2 - gam * 1j * t / 2
    )
    th_arg = 1j * v * n + z / (d3 - d1) + r + 1j * t

    return (
        theta3(-0.5j * v + r, y)
        * R3(v, r, y) ** n
        * exp_term
        * theta2(th_arg, y)
    )


def g_wave(v: float, n: int, z: float, r: float, y: float, t: float) -> complex:
    """Wave function g for theta-based binormal computation.

    Args:
        v: Parameter v
        n: Index (0 to N)
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        t: Time parameter

    Returns:
        Complex wave function value

    References:
        Mathematica: g[v,n,z,r,y,t] (line 193)
    """
    d3 = delta3(v, r, y)
    d1 = delta1(v, r, y)
    c = eC(v, r, y)
    gam = Gamma(r, y, 1)

    exp_term = np.exp(
        (n + 0.5) * d3 / (d3 - d1) * z + c * t * z / 2 + gam * 1j * t / 2
    )
    th_arg = 1j * v * n + z / (d3 - d1) - r + 1j * t

    return (
        theta1(0.5j * v + r, y)
        * R1(v, r, y) ** (-n)
        * exp_term
        * theta4(th_arg, y)
    )


def G_func(v: float, n: int, z: float, r: float, y: float, t: float) -> complex:
    """Function G for theta-based binormal computation.

    Args:
        v: Parameter v
        n: Index (0 to N)
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        t: Time parameter

    Returns:
        G value

    References:
        Mathematica: G[v,n,z,r,y,t] (line 196)
    """
    d3 = delta3(v, r, y)
    d1 = delta1(v, r, y)
    c = eC(v, r, y)
    a2 = alpha2(r, y)

    exp_term = np.exp(n * d3 / (d3 - d1) * z + c * t * z / 2)
    th_arg = 1j * t + 1j * v * (n - 0.5) + z / (d3 - d1)

    return (
        a2
        * (theta1(1j * v, y) / theta3(0, y))
        * exp_term
        * R1(v, r, y) ** (-n)
        * R3(v, r, y) ** n
        * theta4(th_arg, y)
    )


def eF(v: float, z: float, r: float, y: float, j: int, t: float) -> complex:
    """Function F for curve generation.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Index (0 to N)
        t: Time parameter

    Returns:
        F value

    References:
        Maple: eF (line 538)
    """
    d3 = delta3(v, r, y)
    d1 = delta1(v, r, y)
    c = eC(v, r, y)
    a2 = alpha2(r, y)

    arg = 1j * t + 1j * v * (j - 0.5) + z / (d3 - d1)
    exp_term = np.exp(j * d3 * z / (d3 - d1) + 0.5 * c * t * z)

    return a2 * exp_term * theta2(arg, y)


def eH(
    v: float, z: float, r: float, y: float, j: int, t: float, gamma: float = 0.0
) -> complex:
    """Function H for curve generation.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Index (0 to N)
        t: Time parameter
        gamma: Phase parameter (default 0.0)

    Returns:
        H value

    References:
        Maple: eH (line 539)
    """
    d3 = delta3(v, r, y)
    d1 = delta1(v, r, y)
    c = eC(v, r, y)
    a2 = alpha2(r, y)
    r1 = R1(v, r, y)
    r3 = R3(v, r, y)

    # Theta function products
    th_prod = (
        theta1(-0.5j * v + r, y)
        * theta3(-0.5j * v + r, y)
        * theta1(0.5j * v + r, y)
        * theta3(0.5j * v + r, y)
    )

    # Exponential term
    exp_term = np.exp(
        j * d3 * z / (d3 - d1) + 0.5 * c * t * z + 1j * gamma * t
    )

    # Argument for theta4
    arg = 1j * t + 1j * v * (j - 0.5) + z / (d3 - d1) - 2 * r

    # Denominator
    denom = a2 * theta3(2 * r, y) * theta1(1j * v, y)

    return th_prod * exp_term * r1 ** (-j) * r3 ** (-j) * theta4(arg, y) / denom


def eX(
    v: float, z: float, r: float, y: float, j: int, t: float, gamma: float = 0.0
) -> float:
    """X coordinate function.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Index (0 to N)
        t: Time parameter
        gamma: Phase parameter (default 0.0)

    Returns:
        X coordinate (real)

    References:
        Maple: eX := (v, z, r, y, j, t) -> 2*Re(eH(...))/eF(...)
    """
    h = eH(v, z, r, y, j, t, gamma)
    f = eF(v, z, r, y, j, t)
    return 2 * np.real(h) / np.real(f)


def eY(
    v: float, z: float, r: float, y: float, j: int, t: float, gamma: float = 0.0
) -> float:
    """Y coordinate function.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Index (0 to N)
        t: Time parameter
        gamma: Phase parameter (default 0.0)

    Returns:
        Y coordinate (real)

    References:
        Maple: eY := (v, z, r, y, j, t) -> 2*Im(eH(...))/eF(...)
    """
    h = eH(v, z, r, y, j, t, gamma)
    f = eF(v, z, r, y, j, t)
    return 2 * np.imag(h) / np.real(f)


def eZ(v: float, z: float, r: float, y: float, j: int, t: float) -> float:
    """Z coordinate function.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Index (0 to N)
        t: Time parameter

    Returns:
        Z coordinate (real)

    References:
        Maple: eZ := (v, z, r, y, j, t) -> j - 2*diff(log(eF(...)), z)
    """
    # Numerical derivative of log(eF) with respect to z
    eps = 1e-8
    f_plus = eF(v, z + eps, r, y, j, t)
    f_minus = eF(v, z - eps, r, y, j, t)

    # d/dz log(f) = (1/f) * df/dz
    deriv = (np.log(f_plus) - np.log(f_minus)) / (2 * eps)

    return float(j - 2 * np.real(deriv))


def eBx(
    v: float, z: float, r: float, y: float, j: int, t: float, gamma: float = 0.0
) -> float:
    """X component of binormal (hinge) vector from theta functions.

    Direct implementation from Mathematica using wave functions f and g.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Spatial index (0 to N)
        t: Time parameter
        gamma: Phase parameter (default 0.0, included for API compatibility)

    Returns:
        Bx component (real)

    References:
        Mathematica: Bx[v,n,z,r,y,t] (line 227)
        Bx = Sign[K] * 1/(2*I*F*|G|) * (conj(g_n)*f_{n-1} - conj(g_{n-1})*f_n
                                          - conj(f_{n-1})*g_n + conj(f_n)*g_{n-1})
    """
    # Compute curvature sign
    k = eK(v, z, r, y, j, t)
    sign_k = np.sign(k) if abs(k) > 1e-10 else 1.0

    # Evaluate wave functions at j and j-1
    f_n = f_wave(v, j, z, r, y, t)
    f_nm1 = f_wave(v, j - 1, z, r, y, t)
    g_n = g_wave(v, j, z, r, y, t)
    g_nm1 = g_wave(v, j - 1, z, r, y, t)

    # Evaluate F and G
    F_n = eF(v, z, r, y, j, t)
    G_n = G_func(v, j, z, r, y, t)

    # Compute the numerator (Mathematica formula)
    numerator = (
        np.conj(g_n) * f_nm1
        - np.conj(g_nm1) * f_n
        - np.conj(f_nm1) * g_n
        + np.conj(f_n) * g_nm1
    )

    # Denominator: 2*I*F*|G|
    denominator = 2j * F_n * abs(G_n)

    # Bx = Sign[K] * numerator / denominator
    result = sign_k * numerator / denominator

    return float(np.real(result))


def eBy(
    v: float, z: float, r: float, y: float, j: int, t: float, gamma: float = 0.0
) -> float:
    """Y component of binormal (hinge) vector from theta functions.

    Direct implementation from Mathematica using wave functions f and g.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Spatial index (0 to N)
        t: Time parameter
        gamma: Phase parameter (default 0.0, included for API compatibility)

    Returns:
        By component (real)

    References:
        Mathematica: By[v,n,z,r,y,t] (line 229)
        By = Sign[K] * 1/(2*F*|G|) * (conj(g_n)*f_{n-1} - conj(g_{n-1})*f_n
                                       + conj(f_{n-1})*g_n - conj(f_n)*g_{n-1})
    """
    # Compute curvature sign
    k = eK(v, z, r, y, j, t)
    sign_k = np.sign(k) if abs(k) > 1e-10 else 1.0

    # Evaluate wave functions at j and j-1
    f_n = f_wave(v, j, z, r, y, t)
    f_nm1 = f_wave(v, j - 1, z, r, y, t)
    g_n = g_wave(v, j, z, r, y, t)
    g_nm1 = g_wave(v, j - 1, z, r, y, t)

    # Evaluate F and G
    F_n = eF(v, z, r, y, j, t)
    G_n = G_func(v, j, z, r, y, t)

    # Compute the numerator (note: different signs than Bx)
    numerator = (
        np.conj(g_n) * f_nm1
        - np.conj(g_nm1) * f_n
        + np.conj(f_nm1) * g_n
        - np.conj(f_n) * g_nm1
    )

    # Denominator: 2*F*|G| (note: no I factor, unlike Bx)
    denominator = 2 * F_n * abs(G_n)

    # By = Sign[K] * numerator / denominator
    result = sign_k * numerator / denominator

    return float(np.real(result))


def eBz(
    v: float, z: float, r: float, y: float, j: int, t: float, gamma: float = 0.0
) -> float:
    """Z component of binormal (hinge) vector from theta functions.

    Direct implementation from Mathematica using wave functions f and g.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Spatial index (0 to N)
        t: Time parameter
        gamma: Phase parameter (default 0.0, included for API compatibility)

    Returns:
        Bz component (real)

    References:
        Mathematica: Bz[v,n,z,r,y,t] (line 231)
        Bz = Sign[K] * 1/(2*I*F*|G|) * (conj(f_n)*f_{n-1} - conj(f_{n-1})*f_n
                                         - conj(g_n)*g_{n-1} + conj(g_{n-1})*g_n)
    """
    # Compute curvature sign
    k = eK(v, z, r, y, j, t)
    sign_k = np.sign(k) if abs(k) > 1e-10 else 1.0

    # Evaluate wave functions at j and j-1
    f_n = f_wave(v, j, z, r, y, t)
    f_nm1 = f_wave(v, j - 1, z, r, y, t)
    g_n = g_wave(v, j, z, r, y, t)
    g_nm1 = g_wave(v, j - 1, z, r, y, t)

    # Evaluate F and G
    F_n = eF(v, z, r, y, j, t)
    G_n = G_func(v, j, z, r, y, t)

    # Compute the numerator (f and g products)
    numerator = (
        np.conj(f_n) * f_nm1
        - np.conj(f_nm1) * f_n
        - np.conj(g_n) * g_nm1
        + np.conj(g_nm1) * g_n
    )

    # Denominator: 2*I*F*|G|
    denominator = 2j * F_n * abs(G_n)

    # Bz = Sign[K] * numerator / denominator
    result = sign_k * numerator / denominator

    return float(np.real(result))


def generate_theta_binormals(
    v: float,
    z: float,
    r: float,
    y: float,
    N: int,
    t: float = 0.0,
    gamma: float = 0.0,
) -> NDArray[np.float64]:
    """Generate binormal (hinge) vectors directly from theta functions.

    Uses Mathematica's direct formulas based on wave functions f and g to compute
    binormals without relying on curve generation. The binormals are computed from
    differences of complex wave functions at adjacent spatial indices.

    Args:
        v: Parameter v (controls shape)
        z: Parameter z (usually 0)
        r: Parameter r (controls shape)
        y: Nome parameter (controls shape)
        N: Number of tetrahedra
        t: Time parameter for evolution (default 0.0)
        gamma: Phase parameter (default 0.0)

    Returns:
        Array of binormal (hinge) vectors, shape (N+1, 3)

    References:
        Mathematica: Bx, By, Bz functions (lines 227-231)
        Uses wave functions f, g, and normalization functions F, G

    Example:
        >>> v, r, y = 0.0723, 0.3035, 0.9155
        >>> N = 38
        >>> binormals = generate_theta_binormals(v, 0, r, y, N, t=0.0)
        >>> binormals.shape
        (39, 3)
    """
    binormals = np.zeros((N + 1, 3), dtype=float)

    # Compute raw binormals from Mathematica formulas
    for j in range(N + 1):
        binormals[j, 0] = eBx(v, z, r, y, j, t, gamma)
        binormals[j, 1] = eBy(v, z, r, y, j, t, gamma)
        binormals[j, 2] = eBz(v, z, r, y, j, t, gamma)

    # normalize binormals
    for j in range(N + 1):
        norm = np.linalg.norm(binormals[j])
        if norm > 1e-10:
            binormals[j] /= norm
        else:
            binormals[j] = np.array([0.0, 0.0, 1.0])  # default direction if zero

    # Apply sign-fixing for frame continuity (matches curve_to_binormals convention)
    # Reference binormal for first frame
    Bp0 = np.array([0.0, 0.0, 1.0])

    # Fix sign of first binormal to match reference
    if np.dot(binormals[0], Bp0) < 0:
        binormals[0] = -binormals[0]

    # Compute tangent vectors for sign-fixing
    tangents = np.zeros((N, 3), dtype=float)
    for k in range(N):
        k_next = (k + 1) if k < N else 0
        tangents[k, 0] = eX(v, z, r, y, k_next, t, gamma) - eX(v, z, r, y, k, t, gamma)
        tangents[k, 1] = eY(v, z, r, y, k_next, t, gamma) - eY(v, z, r, y, k, t, gamma)
        tangents[k, 2] = eZ(v, z, r, y, k_next, t) - eZ(v, z, r, y, k, t)

    # Fix signs of subsequent binormals for continuity
    for i in range(1, N + 1):
        cross = np.cross(binormals[i - 1], binormals[i])
        sign_val = np.sign(np.dot(cross, tangents[(i - 1) % N]))
        if sign_val == 0:
            sign_val = 1.0
        if sign_val < 0:
            binormals[i] = -binormals[i]

    return binormals


def generate_theta_curve(
    v: float,
    z: float,
    r: float,
    y: float,
    N: int,
    t: float = 0.0,
    gamma: float = 0.0,
) -> NDArray[np.float64]:
    """Generate exact analytic kaleidocycle curve using Jacobi theta functions.

    This is the Python equivalent of Maple's XYZ function, computing exact
    solutions using integrable systems theory based on theta functions.

    Args:
        v: Parameter v (controls shape)
        z: Parameter z (usually 0)
        r: Parameter r (controls shape)
        y: Nome parameter (controls shape)
        N: Number of tetrahedra
        t: Time parameter for evolution (default 0.0)
        gamma: Phase parameter (default 0.0)

    Returns:
        Array of 3D curve points, shape (N+1, 3)

    References:
        Maple: XYZ := proc(v, z, r, y, N, t) ...

    Example:
        >>> # Parameters for a specific kaleidocycle
        >>> v, r, y = 0.0723, 0.3035, 0.9155
        >>> N = 38
        >>> curve = generate_theta_curve(v, 0, r, y, N, t=0.0)
        >>> curve.shape
        (39, 3)

    Note:
        This function computes exact analytic solutions based on
        Jacobi theta functions and integrable systems theory.
        Parameters (v, r, y, N) must satisfy closure conditions
        for the curve to form a closed kaleidocycle.
    """
    points = np.zeros((N + 1, 3), dtype=float)

    for j in range(N + 1):
        points[j, 0] = eX(v, z, r, y, j, t, gamma)
        points[j, 1] = eY(v, z, r, y, j, t, gamma)
        points[j, 2] = eZ(v, z, r, y, j, t)

    return points


def generate_animation_theta(
    v: float,
    z: float,
    r: float,
    y: float,
    N: int,
    num_frames: int,
    t_step: float = 0.05,
    gamma: float = 0.0,
    output: str = "binormals", # or "curve"
) -> list[NDArray[np.float64]]:
    """Generate exact analytic animation using Jacobi theta functions.

    Args:
        v: Parameter v
        z: Parameter z (usually 0)
        r: Parameter r
        y: Nome parameter
        N: Number of tetrahedra
        num_frames: Number of animation frames
        t_step: Time step between frames
        gamma: Phase parameter

    Returns:
        List of curve arrays, each with shape (N+1, 3)

    References:
        Maple: Xs := [Threads:-Seq(eval(XYZ(..., step*t), vals), t = 1 .. frames)]

    Example:
        >>> v, r, y = 0.0723, 0.3035, 0.9155
        >>> N = 38
        >>> frames = generate_animation_theta(v, 0, r, y, N, num_frames=50, t_step=0.05)
        >>> len(frames)
        50
    """
    frames = []

    for frame_idx in range(num_frames):
        t = frame_idx * t_step
        if output == "binormals":
            binormals = generate_theta_binormals(v, z, r, y, N, t, gamma)
            frames.append(binormals)
        else:  # output == "curve"
            curve = generate_theta_curve(v, z, r, y, N, t, gamma)
            frames.append(curve)

    return frames


def close1(v: float, r: float, N: int, m: int) -> complex:
    """First closing condition for kaleidocycle.

    For a kaleidocycle to close properly, this must equal zero.

    Args:
        v: Parameter v
        r: Parameter r
        N: Number of tetrahedra
        m: Winding number

    Returns:
        Complex value (should be 0 for closure)

    References:
        Mathematica: close1[v,r,y]:=I*y*(delta3[v,r,y]+delta1[v,r,y])+4*Pi*v
        With constraint: y = k*v/m
    """
    y = N * v / m
    d3 = delta3(v, r, y)
    d1 = delta1(v, r, y)
    return 1j * y * (d3 + d1) + 4 * np.pi * v


def close2(v: float, r: float, N: int, m: int) -> complex:
    """Second closing condition for kaleidocycle.

    For a kaleidocycle to close properly, this must equal zero.

    Args:
        v: Parameter v
        r: Parameter r
        N: Number of tetrahedra
        m: Winding number

    Returns:
        Complex value (should be 0 for closure)

    References:
        Mathematica: close2[v,r,y,k,m]:=1-Exp[m*Pi*I*(4*r+1)]*R1[v,r,y]^(-k)*R3[v,r,y]^(-k)
        With constraint: y = k*v/m
    """
    y = N * v / m
    r1 = R1(v, r, y)
    r3 = R3(v, r, y)
    exp_term = np.exp(m * np.pi * 1j * (4 * r + 1))
    return 1 - exp_term * (r1 ** (-N)) * (r3 ** (-N))


def solve_closure_conditions(
    N: int,
    m: int = 3,
    initial_guess: tuple[float, float] | None = None,
) -> tuple[float, float, float] | None:
    """Solve closing conditions to find parameters (v, r, y) for a given N.

    Given the number of tetrahedra N, solves the system:
        close1(v, r, N, m) = 0
        close2(v, r, N, m) = 0

    with the constraint y = N*v/m, reducing to 2 unknowns (v, r).

    Args:
        N: Number of tetrahedra
        m: Winding number (if None, set to N for default topology)
        initial_guess: Initial guess for (v, r). If None, uses reasonable defaults.

    Returns:
        Tuple (v, r, y) if solution found, None otherwise
        where y is computed from y = N*v/m

    References:
        Mathematica: PlotClose1, PlotClose2 contour plots for finding intersections
        Constraint: y = k*v/m where k=N

    Example:
        >>> # Find parameters for 8-tetrahedra kaleidocycle
        >>> result = solve_closure_conditions(8)
        >>> if result:
        ...     v, r, y = result
        ...     print(f"v={v:.6f}, r={r:.6f}, y={y:.6f}")
    """
    from scipy.optimize import least_squares

    # Default initial guess based on typical kaleidocycle parameters
    if initial_guess is None:
        # Start with reasonable values from known solutions
        v_init = 0.5
        r_init = 0.3
    else:
        v_init, r_init = initial_guess

    def residuals(x: NDArray) -> NDArray:
        """Compute residuals of closing conditions (split into real/imag)."""
        v, r = x

        # Ensure parameters are in valid ranges
        if r <= 0.0 or r > 1.0:
            return np.array([1e10, 1e10, 1e10, 1e10])
        if v <= 0.0 or v > 2.0:
            return np.array([1e10, 1e10, 1e10, 1e10])

        try:
            # Evaluate closing conditions with y = N*v/m constraint
            c1 = close1(v, r, N, m)
            c2 = close2(v, r, N, m)

            # Split complex equations into real and imaginary parts
            # We have 2 complex equations = 4 real equations, 2 unknowns
            # This is an overdetermined system, use least squares
            return np.array([np.real(c1), np.imag(c1), np.real(c2), np.imag(c2)])
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.array([1e10, 1e10, 1e10, 1e10])

    # Solve the system using least squares
    x0 = np.array([v_init, r_init])
    bounds = ([0.001, 0.001], [2.0, 1.0])

    sol = least_squares(residuals, x0, jac='3-point',bounds=bounds, max_nfev=10000, ftol=1e-18, xtol=1e-18)

    if sol.success:
        v, r = sol.x
        y = N * v / m  # Compute y from constraint
        # Verify solution
        res = residuals(sol.x)
        residual_norm = np.linalg.norm(res)
        if residual_norm < 1e-4:
            return (float(v), float(r), float(y))

    return None


def eTorsion(v: float, z: float, r: float, y: float, t: float) -> float:
    """Compute torsion angle from theta function ratios.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        t: Time parameter

    Returns:
        Torsion angle (real)

    References:
        Maple: eTor := (v, z, r, y, t) -> Re(-I*log(R1(v, r, y)/R3(v, r, y)))
    """
    r1 = R1(v, r, y)
    r3 = R3(v, r, y)
    return float(np.real(-1j * np.log(r1 / r3)))


def eK(v: float, z: float, r: float, y: float, j: int, t: float) -> float:
    """Compute curvature angle at index j.

    Args:
        v: Parameter v
        z: Parameter z
        r: Parameter r
        y: Nome parameter
        j: Index
        t: Time parameter

    Returns:
        Curvature angle (real)

    References:
        Maple: eK (line 545)
    """
    arg = 1j * t + 1j * v * (j - 0.5) + z / (delta3(v, r, y) - delta1(v, r, y))

    numerator = -1j * theta1(1j * v, y) * theta4(arg, y)
    denominator = theta3(1j * v, y) * theta2(arg, y)

    return float(np.real(2 * np.arctan(numerator / denominator)))
