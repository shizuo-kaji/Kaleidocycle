"""Tests for backend abstraction and switching."""

from __future__ import annotations

import pytest

from kaleidocycle.backends import (
    get_backend,
    get_available_backends,
    list_backends,
    set_backend,
)


class TestBackendRegistry:
    """Test backend registry and availability."""

    def test_numpy_always_available(self):
        """NumPy backend should always be available."""
        backends = get_available_backends()
        assert 'numpy' in backends

    def test_list_backends_returns_dict(self):
        """list_backends should return availability dictionary."""
        backends = list_backends()
        assert isinstance(backends, dict)
        assert 'numpy' in backends
        assert 'jax' in backends
        assert backends['numpy'] is True  # Always available

    def test_get_available_backends_returns_list(self):
        """get_available_backends should return list of available backends."""
        backends = get_available_backends()
        assert isinstance(backends, list)
        assert 'numpy' in backends


class TestBackendSwitching:
    """Test backend switching functionality."""

    def test_default_backend_is_numpy(self):
        """Default backend should be NumPy."""
        backend = get_backend()
        assert backend.name == 'numpy'

    def test_set_backend_numpy(self):
        """Should be able to set NumPy backend."""
        set_backend('numpy')
        backend = get_backend()
        assert backend.name == 'numpy'

    @pytest.mark.jax
    def test_set_backend_jax(self):
        """Should be able to set JAX backend if available."""
        pytest.importorskip("jax")

        set_backend('jax')
        backend = get_backend()
        assert backend.name == 'jax'

        # Reset to numpy
        set_backend('numpy')

    def test_invalid_backend_raises(self):
        """Setting invalid backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend('invalid_backend')

    def test_unavailable_backend_raises(self):
        """Setting unavailable backend should raise ImportError."""
        backends = list_backends()

        # Find an unavailable backend (if any)
        if not backends.get('jax', True):
            with pytest.raises(ImportError, match="Backend 'jax' not available"):
                set_backend('jax')

    def test_get_specific_backend(self):
        """Should be able to get specific backend by name."""
        backend = get_backend('numpy')
        assert backend.name == 'numpy'

    @pytest.mark.jax
    def test_get_specific_jax_backend(self):
        """Should be able to get JAX backend by name."""
        pytest.importorskip("jax")

        backend = get_backend('jax')
        assert backend.name == 'jax'


class TestNumPyBackend:
    """Test NumPy backend functionality."""

    def test_numpy_backend_array_module(self):
        """NumPy backend should provide numpy module."""
        import numpy as np

        backend = get_backend('numpy')
        assert backend.array_module is np

    def test_numpy_backend_grad(self):
        """NumPy backend should provide gradient function."""
        backend = get_backend('numpy')

        def f(x):
            return float((x ** 2).sum())

        grad_f = backend.grad(f)
        x = backend.asarray([1.0, 2.0, 3.0])

        # Compute gradient
        grad = grad_f(x, eps=1e-7)

        # Should be approximately [2, 4, 6]
        expected = backend.asarray([2.0, 4.0, 6.0])
        assert grad.shape == expected.shape
        # Finite differences are approximate
        import numpy as np
        assert np.allclose(grad, expected, rtol=1e-5, atol=1e-6)

    def test_numpy_backend_to_numpy(self):
        """NumPy backend to_numpy should return numpy array."""
        import numpy as np

        backend = get_backend('numpy')
        arr = np.array([1.0, 2.0, 3.0])
        result = backend.to_numpy(arr)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)


@pytest.mark.jax
class TestJAXBackend:
    """Test JAX backend functionality (requires JAX)."""

    def test_jax_backend_available(self):
        """JAX backend should be importable."""
        pytest.importorskip("jax")

        backend = get_backend('jax')
        assert backend.name == 'jax'

    def test_jax_backend_array_module(self):
        """JAX backend should provide jax.numpy module."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        backend = get_backend('jax')
        assert backend.array_module is jnp

    def test_jax_backend_grad(self):
        """JAX backend should provide gradient function via autodiff."""
        pytest.importorskip("jax")
        import numpy as np

        backend = get_backend('jax')

        def f(x):
            return (x ** 2).sum()

        grad_f = backend.grad(f)
        x = backend.asarray([1.0, 2.0, 3.0])

        # Compute gradient using JAX autodiff
        grad = grad_f(x)

        # Should be exactly [2, 4, 6] (no approximation error)
        expected = np.array([2.0, 4.0, 6.0])
        assert np.allclose(grad, expected, rtol=1e-10)

    def test_jax_backend_jacobian(self):
        """JAX backend should provide Jacobian function via autodiff."""
        pytest.importorskip("jax")
        import numpy as np

        backend = get_backend('jax')

        def f(x):
            """f: R^3 -> R^2"""
            jnp = backend.array_module
            return jnp.array([x[0] ** 2 + x[1], x[1] * x[2]])

        jac_f = backend.jacobian(f)
        x = backend.asarray([1.0, 2.0, 3.0])

        # Compute Jacobian using JAX autodiff
        jac = jac_f(x)

        # Expected Jacobian:
        # ∂f₁/∂x = [2x₀, 1, 0] = [2, 1, 0]
        # ∂f₂/∂x = [0, x₂, x₁] = [0, 3, 2]
        expected = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
        assert np.allclose(jac, expected, rtol=1e-10)

    def test_jax_backend_to_numpy(self):
        """JAX backend to_numpy should convert JAX arrays to NumPy."""
        pytest.importorskip("jax")
        import numpy as np

        backend = get_backend('jax')
        jax_arr = backend.asarray([1.0, 2.0, 3.0])
        numpy_arr = backend.to_numpy(jax_arr)

        assert isinstance(numpy_arr, np.ndarray)
        assert np.array_equal(numpy_arr, [1.0, 2.0, 3.0])


class TestBackendContextManagement:
    """Test backend context and state management."""

    def test_backend_instances_are_cached(self):
        """Backend instances should be reused (lazy initialization)."""
        backend1 = get_backend('numpy')
        backend2 = get_backend('numpy')

        # Should be the same instance
        assert backend1 is backend2

    @pytest.mark.jax
    def test_jax_backend_instances_are_cached(self):
        """JAX backend instances should be reused."""
        pytest.importorskip("jax")

        backend1 = get_backend('jax')
        backend2 = get_backend('jax')

        # Should be the same instance
        assert backend1 is backend2

    def test_set_backend_persists(self):
        """Setting backend should persist across get_backend calls."""
        set_backend('numpy')
        backend1 = get_backend()
        backend2 = get_backend()

        assert backend1.name == 'numpy'
        assert backend2.name == 'numpy'
