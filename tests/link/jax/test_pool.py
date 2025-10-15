"""Tests for JAX dispatch of pooling operations."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.tensor.pool import pool_2d
from tests.link.jax.test_basic import compare_jax_and_py


# Skip if JAX not available
jax = pytest.importorskip("jax")

# Set tolerances based on precision
floatX = config.floatX
RTOL = ATOL = 1e-6 if floatX.endswith("64") else 1e-3


# ============================================================================
# Test Category 1: Basic Pooling Tests
# ============================================================================


def test_maxpool_2x2_no_padding():
    """
    Test MaxPool with 2x2 window and no padding.

    This is the most common pooling configuration - reduces spatial
    dimensions by half (stride equals window size by default).
    """
    # Arrange: Define symbolic variables
    x = pt.tensor4("x", dtype="float32")

    # Act: Create max pooling operation
    out = pool_2d(x, ws=(2, 2), mode="max")

    # Arrange: Generate test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")  # (N, C, H, W)

    # Assert: JAX output matches Python backend
    compare_jax_and_py(
        [x],
        [out],
        [x_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
    )


def test_maxpool_3x3_no_padding():
    """
    Test MaxPool with 3x3 window.

    Larger pooling windows capture features over bigger regions.
    Used in YOLO SPPF blocks.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(3, 3), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 9, 9)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


@pytest.mark.parametrize("padding", [(1, 1), (2, 2), (1, 2)])
def test_maxpool_with_padding(padding):
    """
    Test MaxPool with explicit padding.

    Padding allows controlling output size more precisely.
    Padded regions use -inf so they never affect max.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), padding=padding, mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


# ============================================================================
# Test Category 2: Stride Variations
# ============================================================================


@pytest.mark.parametrize("window_size", [2, 3, 4])
def test_maxpool_stride_equals_window(window_size):
    """
    Test MaxPool where stride equals window size (non-overlapping).

    This is the default and most common: each region is pooled once.
    Reduces dimensions by factor of window_size.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(
        x,
        ws=(window_size, window_size),
        stride=(window_size, window_size),
        mode="max",
    )

    rng = np.random.default_rng(42)
    # Make input size divisible by window_size
    size = window_size * 4
    x_val = rng.normal(size=(2, 3, size, size)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


@pytest.mark.parametrize("ws, stride", [(3, 1), (3, 2), (5, 2)])
def test_maxpool_stride_less_than_window(ws, stride):
    """
    Test MaxPool with stride < window size (overlapping pools).

    Overlapping pools provide more detailed feature maps.
    Common in deeper CNN architectures for fine-grained features.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(ws, ws), stride=(stride, stride), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


def test_maxpool_stride_greater_than_window():
    """
    Test MaxPool with stride > window size (sparse sampling).

    This skips regions between pools, aggressively downsampling.
    Less common but valid configuration.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), stride=(3, 3), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


@pytest.mark.parametrize("ws", [(2, 3), (3, 2), (4, 2)])
def test_maxpool_asymmetric_window(ws):
    """
    Test MaxPool with asymmetric window (different H and W).

    Useful for inputs with different spatial characteristics
    or aspect ratios (e.g., wide images, time-frequency domains).
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=ws, mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 12, 12)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


@pytest.mark.parametrize("stride", [(1, 2), (2, 1)])
def test_maxpool_asymmetric_stride(stride):
    """
    Test MaxPool with asymmetric stride (different H and W strides).

    Downsamples dimensions independently, useful for anisotropic data.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), stride=stride, mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


# ============================================================================
# Test Category 3: Edge Cases
# ============================================================================


def test_maxpool_1x1_window():
    """
    Test MaxPool with 1x1 window (identity operation).

    Should return input unchanged. Tests edge case of minimal pooling.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(1, 1), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


def test_maxpool_large_window():
    """
    Test MaxPool with window >= input size (global pooling).

    Reduces entire spatial dimensions to 1x1 per channel.
    Equivalent to global max pooling.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(8, 8), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


def test_maxpool_all_negative_values():
    """
    Test MaxPool with all negative input values.

    Verifies that max operation works correctly (should pick
    least negative, not zero or positive value).
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    # All negative values
    rng = np.random.default_rng(42)
    x_val = -np.abs(rng.normal(size=(2, 3, 8, 8))).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


def test_maxpool_with_inf_values():
    """
    Test MaxPool with infinity values in input.

    Verifies that +inf and -inf are handled correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    # Add some infinity values
    x_val[0, 0, 0, 0] = np.inf
    x_val[0, 1, 2, 2] = -np.inf

    compare_jax_and_py([x], [out], [x_val])


def test_maxpool_single_channel():
    """
    Test MaxPool with single channel (C=1).

    Ensures channel dimension is handled correctly.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 1, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


def test_maxpool_many_channels():
    """
    Test MaxPool with many channels (C=512).

    Verifies pooling scales to deeper network layers.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 512, 8, 8)).astype("float32")

    compare_jax_and_py([x], [out], [x_val])


# ============================================================================
# Test Category 4: Gradient Tests
# ============================================================================


def test_maxpool_gradient_single_max():
    """
    Test MaxPoolGrad routes gradient to max position.

    MaxPool gradient should only flow to the position that had
    the maximum value in each pool region.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")
    loss = out.sum()

    # Compute gradient
    grad_x = grad(loss, x)

    # Compile with JAX mode
    f_jax = function([x], [grad_x], mode="JAX")

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    grad_x_jax = f_jax(x_val)[0]

    # Compare with Python backend
    f_py = function([x], [grad_x], mode="FAST_RUN")
    grad_x_py = f_py(x_val)[0]

    np.testing.assert_allclose(grad_x_jax, grad_x_py, rtol=RTOL, atol=ATOL)

    # Verify gradient properties:
    # 1. Gradient should be non-zero
    assert np.abs(grad_x_jax).sum() > 0

    # 2. Gradient should only be at max positions (0 or 1)
    # (Each pool region has exactly one max that gets gradient=1)
    unique_vals = np.unique(grad_x_jax)
    assert (
        len(unique_vals) <= 3
    )  # Should be mostly 0 and 1 (maybe some duplicates get 0.5)


def test_maxpool_gradient_tied_values():
    """
    Test MaxPoolGrad when multiple values tie for max.

    When multiple positions have the same max value, gradient
    should be split among them (PyTensor behavior).
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), mode="max")
    loss = out.sum()

    grad_x = grad(loss, x)

    # Create input with tied max values
    x_val = np.ones((1, 1, 4, 4), dtype="float32")  # All same value
    x_val[0, 0, 2:, 2:] = 2.0  # Different region

    # Compare JAX and Python backends
    f_jax = function([x], [grad_x], mode="JAX")
    f_py = function([x], [grad_x], mode="FAST_RUN")

    grad_jax = f_jax(x_val)[0]
    grad_py = f_py(x_val)[0]

    np.testing.assert_allclose(grad_jax, grad_py, rtol=RTOL, atol=ATOL)


def test_maxpool_gradient_with_padding():
    """
    Test MaxPoolGrad with padding.

    Padded regions (filled with -inf) should never receive gradients.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), padding=(1, 1), mode="max")
    loss = out.sum()

    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")

    # Compare backends
    compare_jax_and_py([x], [grad_x], [x_val])


@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 3)])
def test_maxpool_gradient_with_stride(stride):
    """
    Test MaxPoolGrad with various strides.

    Gradient routing should work correctly regardless of stride.
    """
    x = pt.tensor4("x", dtype="float32")
    out = pool_2d(x, ws=(2, 2), stride=stride, mode="max")
    loss = out.sum()

    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")

    compare_jax_and_py([x], [grad_x], [x_val])


# ============================================================================
# Test Category 5: Dtype Tests
# ============================================================================


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_maxpool_dtypes(dtype):
    """
    Test MaxPool with different dtypes.

    Ensures pooling works with both single and double precision.
    """
    x = pt.tensor4("x", dtype=dtype)
    out = pool_2d(x, ws=(2, 2), mode="max")

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype(dtype)

    # Adjust tolerance for float32
    rtol = 1e-3 if dtype == "float32" else 1e-6
    atol = 1e-3 if dtype == "float32" else 1e-6

    compare_jax_and_py(
        [x],
        [out],
        [x_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol),
    )


# ============================================================================
# Test Category 6: Integration Tests
# ============================================================================


def test_yolo_sppf_cascaded_pooling():
    """
    Test YOLO SPPF block pattern (cascaded pooling).

    SPPF: Spatial Pyramid Pooling - Fast
    Uses three sequential 5x5 poolings to achieve different receptive fields.
    """
    x = pt.tensor4("x", dtype="float32")

    # SPPF pattern: 3 cascaded 5x5 max pools with stride=1 and padding=2
    # This maintains spatial dimensions while increasing receptive field
    pool1 = pool_2d(x, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")

    # Typically concatenated: [x, pool1, pool2, pool3]
    # For this test, just verify all pools work
    outputs = [pool1, pool2, pool3]

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(1, 512, 20, 20)).astype("float32")

    for out in outputs:
        compare_jax_and_py([x], [out], [x_val])
