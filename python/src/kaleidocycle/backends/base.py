"""Abstract base class for computational backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class Backend(ABC):
    """Abstract interface for computational backends (NumPy, JAX, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend (e.g., 'numpy', 'jax')."""

    @property
    @abstractmethod
    def array_module(self) -> Any:
        """NumPy-like array module (numpy or jax.numpy)."""

    @abstractmethod
    def grad(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute gradient of function.

        For NumPy backend: uses finite differences
        For JAX backend: uses automatic differentiation (jax.grad)

        Args:
            func: Function to differentiate
            argnums: Argument index to differentiate with respect to

        Returns:
            Gradient function
        """

    @abstractmethod
    def jacobian(self, func: Callable) -> Callable:
        """Compute Jacobian matrix of function.

        For NumPy backend: uses finite differences
        For JAX backend: uses automatic differentiation (jax.jacfwd or jax.jacrev)

        Args:
            func: Function to compute Jacobian of

        Returns:
            Jacobian function
        """

    @abstractmethod
    def to_numpy(self, array: Any) -> NDArray[np.float64]:
        """Convert backend array to NumPy array.

        For NumPy backend: returns input (already NumPy)
        For JAX backend: converts jax.Array to np.ndarray

        Args:
            array: Backend-specific array

        Returns:
            NumPy array
        """

    @abstractmethod
    def asarray(self, array: Any, dtype: Any = None) -> Any:
        """Convert to backend array.

        Args:
            array: Input array (any format)
            dtype: Desired dtype

        Returns:
            Backend-specific array
        """

    def __repr__(self) -> str:
        """String representation of backend."""
        return f"{self.__class__.__name__}()"
