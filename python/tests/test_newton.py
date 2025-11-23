"""Tests for Newton solver utilities."""

from __future__ import annotations

import numpy as np
import pytest

from kaleidocycle.solvers import moore_penrose_inverse, newton_solve


def test_moore_penrose_inverse_full_rank() -> None:
    """Test pseudoinverse for full-rank square matrix."""
    # Create a simple full-rank matrix
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    A_pinv = moore_penrose_inverse(A)

    # For full-rank square matrix, pseudoinverse equals matrix inverse
    A_inv = np.linalg.inv(A)

    np.testing.assert_allclose(A_pinv, A_inv, atol=1e-10)

    # Verify A * A^+ = I
    np.testing.assert_allclose(A @ A_pinv, np.eye(2), atol=1e-10)


def test_moore_penrose_inverse_singular() -> None:
    """Test pseudoinverse for singular matrix."""
    # Create a rank-deficient matrix
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1

    A_pinv = moore_penrose_inverse(A)

    # Check A * A^+ * A = A (defining property of pseudoinverse)
    np.testing.assert_allclose(A @ A_pinv @ A, A, atol=1e-10)

    # Check A^+ * A * A^+ = A^+ (another defining property)
    np.testing.assert_allclose(A_pinv @ A @ A_pinv, A_pinv, atol=1e-10)


def test_moore_penrose_inverse_rectangular() -> None:
    """Test pseudoinverse for rectangular matrix."""
    # Overdetermined system (more rows than columns)
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    A_pinv = moore_penrose_inverse(A)

    # Check shape
    assert A_pinv.shape == (2, 3)

    # Check defining properties
    np.testing.assert_allclose(A @ A_pinv @ A, A, atol=1e-10)
    np.testing.assert_allclose(A_pinv @ A @ A_pinv, A_pinv, atol=1e-10)


def test_moore_penrose_inverse_compare_numpy() -> None:
    """Test that our pseudoinverse matches numpy's for typical matrices."""
    A = np.random.randn(5, 3)

    our_pinv = moore_penrose_inverse(A)
    numpy_pinv = np.linalg.pinv(A)

    np.testing.assert_allclose(our_pinv, numpy_pinv, atol=1e-10)


def test_moore_penrose_inverse_threshold() -> None:
    """Test that eps parameter controls singular value threshold."""
    # Matrix with small singular value
    A = np.array([[1.0, 0.0], [0.0, 1e-10]])

    # With default threshold, small singular value should be zeroed
    A_pinv_strict = moore_penrose_inverse(A, eps=1e-8)

    # With loose threshold, small singular value should be kept
    A_pinv_loose = moore_penrose_inverse(A, eps=1e-12)

    # The two should be different
    assert not np.allclose(A_pinv_strict, A_pinv_loose)


def test_newton_solve_linear_system() -> None:
    """Test Newton solver on a linear system."""
    # Solve Ax = b where A = [[2, 1], [1, 3]], b = [5, 7]
    # Solution: 2x + y = 5, x + 3y = 7 => x = 1.6, y = 1.8

    def residual(x: np.ndarray) -> np.ndarray:
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        return A @ x - b

    def jacobian(x: np.ndarray) -> np.ndarray:
        # Jacobian of Ax - b is just A
        return np.array([[2.0, 1.0], [1.0, 3.0]])

    x0 = np.array([0.0, 0.0])

    x_sol, converged, num_iter = newton_solve(residual, jacobian, x0)

    assert converged
    assert num_iter < 100  # Should converge in few iterations for linear problem

    # Check solution matches numpy
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 7.0])
    expected = np.linalg.solve(A, b)
    np.testing.assert_allclose(x_sol, expected, atol=1e-6)

    # Check residual is small
    assert np.max(np.abs(residual(x_sol))) < 1e-8


def test_newton_solve_nonlinear_system() -> None:
    """Test Newton solver on a nonlinear system."""
    # Solve: x^2 + y^2 = 1, x - y = 0
    # Solution: x = y = sqrt(2)/2

    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[0] - x[1]])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0], 2 * x[1]], [1.0, -1.0]])

    # Start near solution
    x0 = np.array([0.5, 0.5])

    x_sol, converged, num_iter = newton_solve(residual, jacobian, x0)

    assert converged

    # Check solution
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(x_sol, expected, atol=1e-6)

    # Check residual is small
    assert np.max(np.abs(residual(x_sol))) < 1e-8


def test_newton_solve_overdetermined() -> None:
    """Test Newton solver on overdetermined system (least squares)."""
    # Overdetermined system: Ax = b with more equations than unknowns
    # Find least squares solution

    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([1.0, 2.0, 3.0])

    def residual(x: np.ndarray) -> np.ndarray:
        return A @ x - b

    def jacobian(x: np.ndarray) -> np.ndarray:
        return A

    x0 = np.zeros(2)

    x_sol, converged, num_iter = newton_solve(residual, jacobian, x0)

    assert converged

    # Compare with numpy's least squares
    x_expected, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(x_sol, x_expected, atol=1e-6)


def test_newton_solve_max_iter() -> None:
    """Test that Newton solver respects max_iter limit."""
    # Create a difficult problem that won't converge quickly
    def residual(x: np.ndarray) -> np.ndarray:
        # Stiff problem
        return np.array([np.exp(x[0]) - 1000.0])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[np.exp(x[0])]])

    x0 = np.array([0.0])

    # Run with very few iterations
    x_sol, converged, num_iter = newton_solve(
        residual, jacobian, x0, max_iter=5, verbose=False
    )

    assert not converged
    assert num_iter == 5


def test_newton_solve_tolerance() -> None:
    """Test that Newton solver respects tolerance."""
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 - 4.0])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0]]])

    x0 = np.array([1.0])

    # Solve with loose tolerance
    x_loose, converged_loose, iter_loose = newton_solve(
        residual, jacobian, x0, tol=1e-4
    )

    # Solve with tight tolerance
    x_tight, converged_tight, iter_tight = newton_solve(
        residual, jacobian, x0, tol=1e-10
    )

    assert converged_loose
    assert converged_tight

    # Tight tolerance should require more iterations
    assert iter_tight >= iter_loose

    # Both should be close to x = 2
    assert abs(x_loose[0] - 2.0) < 1e-3
    assert abs(x_tight[0] - 2.0) < 1e-9


def test_newton_solve_adaptive_step() -> None:
    """Test that adaptive step size prevents divergence."""
    # Problem where full Newton step might overshoot
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 3 - 8.0])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[3 * x[0] ** 2]])

    # Start far from solution
    x0 = np.array([10.0])

    x_sol, converged, num_iter = newton_solve(
        residual, jacobian, x0, max_step_factor=0.1
    )

    # Should still converge due to adaptive stepping
    assert converged
    assert abs(x_sol[0] - 2.0) < 1e-6


def test_newton_solve_verbose_output(capsys) -> None:
    """Test that verbose mode produces output."""
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 - 4.0])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0]]])

    x0 = np.array([1.0])

    newton_solve(residual, jacobian, x0, verbose=True, max_iter=500)

    captured = capsys.readouterr()
    assert "converged" in captured.out.lower()


def test_newton_solve_zero_jacobian() -> None:
    """Test Newton solver handles near-zero Jacobian gracefully."""
    def residual(x: np.ndarray) -> np.ndarray:
        # Residual where Jacobian becomes very small
        return np.array([x[0] ** 2 - 1e-20])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0]]])

    x0 = np.array([1e-12])  # Start very close to zero

    # Should not crash, even if it doesn't converge perfectly
    x_sol, converged, num_iter = newton_solve(
        residual, jacobian, x0, max_iter=100
    )

    # Just check it doesn't crash
    assert isinstance(converged, bool)
    assert isinstance(num_iter, int)
