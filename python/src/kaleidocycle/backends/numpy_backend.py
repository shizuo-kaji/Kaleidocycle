"""NumPy backend using finite differences for differentiation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .base import Backend


class NumPyBackend(Backend):
    """NumPy backend with finite difference differentiation."""

    @property
    def name(self) -> str:
        """Name of the backend."""
        return 'numpy'

    @property
    def array_module(self) -> Any:
        """NumPy array module."""
        return np

    def grad(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute gradient using central finite differences.

        Args:
            func: Function to differentiate, must return scalar
            argnums: Argument index to differentiate with respect to

        Returns:
            Gradient function that returns array of partial derivatives
        """
        def grad_func(*args, eps: float = 1e-8, **kwargs):
            """Gradient function using finite differences."""
            x = np.asarray(args[argnums], dtype=float)
            original_shape = x.shape
            x_flat = x.flatten()
            grad = np.zeros_like(x_flat)

            # Central differences for each component
            for i in range(len(x_flat)):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps

                # Reconstruct args with perturbed x
                x_plus_reshaped = x_plus.reshape(original_shape)
                x_minus_reshaped = x_minus.reshape(original_shape)

                args_plus = list(args)
                args_minus = list(args)
                args_plus[argnums] = x_plus_reshaped
                args_minus[argnums] = x_minus_reshaped

                f_plus = func(*args_plus, **kwargs)
                f_minus = func(*args_minus, **kwargs)

                grad[i] = (f_plus - f_minus) / (2 * eps)

            return grad.reshape(original_shape)

        return grad_func

    def jacobian(self, func: Callable) -> Callable:
        """Compute Jacobian using finite differences.

        Args:
            func: Function to compute Jacobian of, must return vector

        Returns:
            Jacobian function that returns matrix of partial derivatives
        """
        def jac_func(x, eps: float = 1e-8):
            """Jacobian function using finite differences."""
            x = np.asarray(x, dtype=float)
            original_shape = x.shape
            x_flat = x.flatten()
            n_vars = len(x_flat)

            # Evaluate function once to get output shape
            f_x = func(x)
            f_x_flat = np.atleast_1d(f_x).flatten()
            n_outputs = len(f_x_flat)

            # Initialize Jacobian matrix
            jac = np.zeros((n_outputs, n_vars), dtype=float)

            # Central differences for each input variable
            for i in range(n_vars):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps

                f_plus = func(x_plus.reshape(original_shape))
                f_minus = func(x_minus.reshape(original_shape))

                f_plus_flat = np.atleast_1d(f_plus).flatten()
                f_minus_flat = np.atleast_1d(f_minus).flatten()

                jac[:, i] = (f_plus_flat - f_minus_flat) / (2 * eps)

            return jac

        return jac_func

    def to_numpy(self, array: Any) -> NDArray[np.float64]:
        """Convert to NumPy array (already NumPy, just ensure type).

        Args:
            array: Input array

        Returns:
            NumPy array
        """
        return np.asarray(array, dtype=float)

    def asarray(self, array: Any, dtype: Any = None) -> NDArray[np.float64]:
        """Convert to NumPy array.

        Args:
            array: Input array
            dtype: Desired dtype (default: float)

        Returns:
            NumPy array
        """
        if dtype is None:
            dtype = float
        return np.asarray(array, dtype=dtype)
