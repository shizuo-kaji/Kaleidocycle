"""Computational backend abstraction for Kaleidocycle.

This module provides a backend abstraction layer that allows switching between
NumPy (with finite differences) and JAX (with automatic differentiation) for
gradient and Jacobian computations.

Examples
--------
Default NumPy backend:
    >>> from kaleidocycle import Kaleidocycle
    >>> kc = Kaleidocycle(n=9, oriented=True)
    >>> result = kc.is_stationary('bending')  # Uses NumPy

Switch to JAX backend globally:
    >>> from kaleidocycle.backends import set_backend, get_available_backends
    >>> if 'jax' in get_available_backends():
    ...     set_backend('jax')
    ...     kc = Kaleidocycle(n=9, oriented=True)
    ...     result = kc.is_stationary('bending')  # Uses JAX autodiff

Per-function backend selection:
    >>> from kaleidocycle.optimality import check_stationarity
    >>> result = check_stationarity(hinges, 'bending', config, backend='jax')
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import Backend
from .numpy_backend import NumPyBackend

__all__ = [
    'Backend',
    'get_backend',
    'set_backend',
    'get_available_backends',
    'list_backends',
]

# Backend registry
_BACKENDS: Dict[str, Optional[Backend]] = {
    'numpy': None,  # Lazy initialization
    'jax': None,    # Lazy initialization
}

# Track which backends are available
_AVAILABLE_BACKENDS: Dict[str, bool] = {
    'numpy': True,  # Always available
}

# Check JAX availability
try:
    import jax  # noqa: F401
    import jaxopt  # noqa: F401
    _AVAILABLE_BACKENDS['jax'] = True
except ImportError:
    _AVAILABLE_BACKENDS['jax'] = False

# Current backend
_CURRENT_BACKEND: str = 'numpy'


def get_available_backends() -> list[str]:
    """Get list of available backends on this system.

    Returns
    -------
    list[str]
        List of backend names that can be used

    Examples
    --------
    >>> from kaleidocycle.backends import get_available_backends
    >>> backends = get_available_backends()
    >>> print(backends)
    ['numpy', 'jax']  # If JAX is installed
    """
    return [name for name, available in _AVAILABLE_BACKENDS.items() if available]


def list_backends() -> Dict[str, bool]:
    """Get dictionary of all backends and their availability status.

    Returns
    -------
    dict[str, bool]
        Dictionary mapping backend names to availability status

    Examples
    --------
    >>> from kaleidocycle.backends import list_backends
    >>> backends = list_backends()
    >>> print(backends)
    {'numpy': True, 'jax': False}  # If JAX is not installed
    """
    return _AVAILABLE_BACKENDS.copy()


def set_backend(name: str) -> None:
    """Set the global computational backend.

    Parameters
    ----------
    name : str
        Backend name ('numpy' or 'jax')

    Raises
    ------
    ValueError
        If backend name is unknown
    ImportError
        If backend is not available (e.g., JAX not installed)

    Examples
    --------
    >>> from kaleidocycle.backends import set_backend
    >>> set_backend('numpy')  # Use NumPy (default)

    >>> set_backend('jax')  # Use JAX (if installed)
    Traceback (most recent call last):
        ...
    ImportError: Backend 'jax' not available. Install with: pip install kaleidocycle[jax]
    """
    global _CURRENT_BACKEND

    if name not in _BACKENDS:
        available = ', '.join(_BACKENDS.keys())
        msg = f"Unknown backend: {name}. Available backends: {available}"
        raise ValueError(msg)

    if not _AVAILABLE_BACKENDS.get(name, False):
        msg = (
            f"Backend '{name}' not available. "
            f"Install with: pip install kaleidocycle[{name}]"
        )
        raise ImportError(msg)

    _CURRENT_BACKEND = name


def get_backend(name: Optional[str] = None) -> Backend:
    """Get a backend instance.

    Parameters
    ----------
    name : str, optional
        Backend name ('numpy' or 'jax'). If None, returns current global backend.

    Returns
    -------
    Backend
        Backend instance

    Raises
    ------
    ValueError
        If backend name is unknown
    ImportError
        If backend is not available

    Examples
    --------
    >>> from kaleidocycle.backends import get_backend
    >>> backend = get_backend()  # Get current backend
    >>> print(backend.name)
    'numpy'

    >>> backend = get_backend('jax')  # Get specific backend
    >>> print(backend.name)
    'jax'
    """
    # Use current backend if name not specified
    if name is None:
        name = _CURRENT_BACKEND

    # Validate backend name
    if name not in _BACKENDS:
        available = ', '.join(_BACKENDS.keys())
        msg = f"Unknown backend: {name}. Available backends: {available}"
        raise ValueError(msg)

    if not _AVAILABLE_BACKENDS.get(name, False):
        msg = (
            f"Backend '{name}' not available. "
            f"Install with: pip install kaleidocycle[{name}]"
        )
        raise ImportError(msg)

    # Lazy initialization of backend
    if _BACKENDS[name] is None:
        if name == 'numpy':
            _BACKENDS[name] = NumPyBackend()
        elif name == 'jax':
            from .jax_backend import JAXBackend
            _BACKENDS[name] = JAXBackend()

    return _BACKENDS[name]  # type: ignore


# Re-export Backend base class
__all__ += ['Backend']
