"""
Test sigmoid activation and SiLU patterns with JAX backend.

These tests verify that sigmoid and the SiLU (Swish) activation pattern work correctly
with JAX tracers, including gradient computation and numerical stability.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import grad
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_sigmoid_activation():
    """
    Test sigmoid activation with dynamic shapes.

    This test verifies:
    - SiLU pattern (x * sigmoid(x))
    - Gradient flow through activation
    - Numerical stability with JAX
    """
    # Arrange - dynamic batch size
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))

    # Act - compute sigmoid and SiLU
    sigmoid_out = pt.sigmoid(x)
    silu_out = x * sigmoid_out  # SiLU activation

    # Compute gradients
    loss = silu_out.sum()
    grad_x = grad(loss, x)

    # Test data with various ranges
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 64, 32, 32)).astype("float32")

    # Assert
    fn, (sig_val, silu_val, grad_val) = compare_jax_and_py(
        [x], [sigmoid_out, silu_out, grad_x], [x_val]
    )

    # Verify sigmoid properties
    assert np.all(sig_val >= 0) and np.all(sig_val <= 1), "Sigmoid output not in [0, 1]"

    # Verify SiLU gradient exists
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is all zeros"


def test_sigmoid_numerical_stability():
    """
    Test sigmoid numerical stability with extreme values.

    This test verifies:
    - No NaN/Inf for large positive inputs
    - No NaN/Inf for large negative inputs
    - Correct asymptotic behavior
    """
    # Arrange - extreme values
    x = pt.vector("x", dtype="float32")

    # Act
    y = pt.sigmoid(x)
    grad_y = grad(y.sum(), x)

    # Test data with extreme values
    x_val = np.array([-100, -50, -10, 0, 10, 50, 100], dtype="float32")

    # Assert
    fn, (y_val, grad_val) = compare_jax_and_py([x], [y, grad_y], [x_val])

    # Verify numerical stability
    assert np.all(np.isfinite(y_val)), f"Sigmoid produced NaN/Inf: {y_val}"
    assert np.all(np.isfinite(grad_val)), f"Gradient has NaN/Inf: {grad_val}"

    # Verify asymptotic behavior
    assert y_val[0] < 1e-6, f"sigmoid(-100) = {y_val[0]}, expected ~0"
    assert y_val[-1] > 1 - 1e-6, f"sigmoid(100) = {y_val[-1]}, expected ~1"


def test_silu_pattern_complete():
    """
    Test complete SiLU (Swish) activation pattern.

    This test verifies:
    - Multiplication of tensor with its sigmoid
    - Gradient computation through SiLU
    - Pattern matches expected SiLU behavior
    """
    # Arrange - typical ConvBNSiLU dimensions
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # Act - SiLU pattern
    silu = x * pt.sigmoid(x)

    # Gradient
    loss = silu.sum()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal(size=(2, 256, 32, 32)).astype("float32") * 0.5

    # Assert
    fn, (silu_val, grad_val) = compare_jax_and_py([x], [silu, grad_x], [x_val])

    # Manually compute expected SiLU for verification
    def silu_numpy(x):
        return x / (1 + np.exp(-x))

    expected_silu = silu_numpy(x_val)
    np.testing.assert_allclose(
        silu_val, expected_silu, rtol=1e-5, err_msg="SiLU output doesn't match expected"
    )

    # Verify gradient shape
    assert grad_val.shape == x_val.shape, "Gradient shape mismatch"


@pytest.fixture
def activation_test_data():
    """Test data for activation functions."""
    rng = np.random.default_rng(42)
    return {
        "small": rng.standard_normal(10).astype("float32"),
        "medium": rng.standard_normal(size=(2, 64, 32, 32)).astype("float32"),
        "extreme": np.array([-100, -10, 0, 10, 100], dtype="float32"),
    }


def test_sigmoid_with_various_shapes(activation_test_data):
    """Test sigmoid works with various tensor shapes."""
    for name, data in activation_test_data.items():
        x = pt.tensor("x", dtype="float32", shape=(None,) * data.ndim)
        y = pt.sigmoid(x)

        fn, y_val = compare_jax_and_py([x], y, [data])

        # Verify output is in valid range
        assert np.all(y_val >= 0) and np.all(y_val <= 1), (
            f"Sigmoid output for {name} data not in [0, 1]"
        )
        assert np.all(np.isfinite(y_val)), f"Sigmoid output for {name} data has NaN/Inf"
