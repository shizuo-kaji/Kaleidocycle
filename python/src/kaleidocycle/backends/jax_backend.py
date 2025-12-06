"""JAX backend using automatic differentiation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .base import Backend


class JAXBackend(Backend):
    """JAX backend with automatic differentiation."""

    def __init__(self):
        """Initialize JAX backend (imports JAX on demand)."""
        try:
            import jax
            import jax.numpy as jnp

            # Enable 64-bit precision for numerical accuracy
            jax.config.update("jax_enable_x64", True)

            self._jax = jax
            self._jnp = jnp
        except ImportError as e:
            msg = (
                "JAX backend requires jax and jaxlib. "
                "Install with: pip install kaleidocycle[jax]"
            )
            raise ImportError(msg) from e

    @property
    def name(self) -> str:
        """Name of the backend."""
        return 'jax'

    @property
    def array_module(self) -> Any:
        """JAX numpy module."""
        return self._jnp

    @property
    def jax(self) -> Any:
        """Access to JAX module for advanced usage."""
        return self._jax

    def grad(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute gradient using JAX automatic differentiation.

        Args:
            func: Function to differentiate, must return scalar
            argnums: Argument index to differentiate with respect to

        Returns:
            Gradient function computed via autodiff
        """
        return self._jax.grad(func, argnums=argnums)

    def jacobian(self, func: Callable) -> Callable:
        """Compute Jacobian using JAX automatic differentiation.

        Args:
            func: Function to compute Jacobian of

        Returns:
            Jacobian function computed via autodiff (using forward-mode)
        """
        # Use forward-mode AD (jacfwd) which is efficient for tall Jacobians
        # (many outputs, few inputs). For wide Jacobians, jacrev would be better.
        return self._jax.jacfwd(func)

    def to_numpy(self, array: Any) -> NDArray[np.float64]:
        """Convert JAX array to NumPy array.

        Args:
            array: JAX array

        Returns:
            NumPy array
        """
        # JAX arrays can be converted to NumPy via np.asarray
        return np.asarray(array, dtype=float)

    def asarray(self, array: Any, dtype: Any = None) -> Any:
        """Convert to JAX array.

        Args:
            array: Input array (NumPy, JAX, or list)
            dtype: Desired dtype (default: float32 for JAX)

        Returns:
            JAX array
        """
        if dtype is None:
            dtype = self._jnp.float64  # Use float64 for numerical precision
        return self._jnp.asarray(array, dtype=dtype)
