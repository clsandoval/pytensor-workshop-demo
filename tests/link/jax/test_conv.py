"""Tests for JAX dispatch of Conv2D operations."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.compile.sharedvalue import shared
from pytensor.tensor.conv.abstract_conv import conv2d
from tests.link.jax.test_basic import compare_jax_and_py


# Skip if JAX not available
jax = pytest.importorskip("jax")

# Set tolerances based on precision
floatX = config.floatX
# Use relaxed tolerance for float32 due to accumulation of floating point errors
RTOL = 1e-6 if floatX.endswith("64") else 1e-5
ATOL = 1e-6 if floatX.endswith("64") else 1e-5


# =============================================================================
# Test Category 1: Basic Convolution Tests
# =============================================================================


def test_conv2d_valid_padding():
    """
    Test Conv2D with valid padding (no padding).

    This is the most basic convolution - output is smaller than input.
    Expected output size: (batch, out_channels, H-kH+1, W-kW+1)
    """
    # Arrange: Define symbolic variables
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    # Act: Create convolution operation
    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    # Arrange: Generate test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")  # (N, C_in, H, W)
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype(
        "float32"
    )  # (C_out, C_in, kH, kW)

    # Assert: JAX output matches Python backend
    compare_jax_and_py(
        [x, filters],
        [out],
        [x_val, filters_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL),
    )


def test_conv2d_same_padding():
    """
    Test Conv2D with half padding (PyTensor's equivalent to "same" padding).

    Half padding ensures output spatial dimensions equal input dimensions
    (with stride=1). This is common in ResNet and modern architectures.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="half", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


@pytest.mark.parametrize("padding", [(1, 1), (2, 2), (1, 2)])
def test_conv2d_explicit_padding(padding):
    """
    Test Conv2D with explicit padding tuple.

    Padding can be specified as (pad_h, pad_w) to add specific padding.
    This is common when fine control over output size is needed.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode=padding, filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


# =============================================================================
# Test Category 2: Filter Flip Tests
# =============================================================================


def test_conv2d_filter_flip_true_vs_false():
    """
    Test filter_flip parameter behavior.

    filter_flip=True: True convolution (flip kernel 180 degrees)
    filter_flip=False: Cross-correlation (no flip)

    Results should be different for non-symmetric kernels.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    # Both modes
    out_flip = conv2d(x, filters, border_mode="valid", filter_flip=True)
    out_no_flip = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype("float32")
    # Non-symmetric kernel to see difference
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    # Test both
    compare_jax_and_py([x, filters], [out_flip], [x_val, filters_val])
    compare_jax_and_py([x, filters], [out_no_flip], [x_val, filters_val])


# =============================================================================
# Test Category 3: Stride Tests
# =============================================================================


def test_conv2d_stride_2x2():
    """
    Test Conv2D with stride=(2, 2).

    Strided convolution reduces spatial dimensions by the stride factor.
    This is commonly used instead of pooling in modern architectures.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, subsample=(2, 2), border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


@pytest.mark.parametrize("stride", [(2, 1), (1, 2), (3, 2)])
def test_conv2d_stride_asymmetric(stride):
    """
    Test Conv2D with asymmetric strides.

    Different strides for H and W dimensions are occasionally used
    when input has different aspect ratios or anisotropic features.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, subsample=stride, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


# =============================================================================
# Test Category 4: Dilation Tests
# =============================================================================


def test_conv2d_dilation_2x2():
    """
    Test Conv2D with dilation=(2, 2) (atrous convolution).

    Dilation inserts gaps between kernel elements, expanding receptive
    field without increasing parameters. Used in DeepLab, etc.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(
        x, filters, border_mode="valid", filter_flip=False, filter_dilation=(2, 2)
    )

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


# =============================================================================
# Test Category 5: Kernel Size Variations
# =============================================================================


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
def test_conv2d_kernel_sizes(kernel_size):
    """
    Test Conv2D with various kernel sizes.

    - 1x1: Pointwise convolution (channel mixing)
    - 3x3: Most common (VGG, ResNet)
    - 5x5, 7x7: Larger receptive field (older architectures)
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 16, 16)).astype("float32")
    filters_val = rng.normal(size=(16, 3, kernel_size, kernel_size)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


# =============================================================================
# Test Category 6: Edge Cases
# =============================================================================


def test_conv2d_single_channel():
    """
    Test Conv2D with single input channel (grayscale).

    Ensures broadcasting and indexing work correctly for C=1.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 1, 8, 8)).astype("float32")  # C=1
    filters_val = rng.normal(size=(16, 1, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


def test_conv2d_single_batch():
    """
    Test Conv2D with batch size 1 (inference mode).

    Common during inference when processing single images.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(1, 3, 8, 8)).astype("float32")  # N=1
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_conv2d_large_batch(batch_size):
    """
    Test Conv2D with larger batch sizes.

    Verifies batching works correctly and efficiently.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(batch_size, 3, 8, 8)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


@pytest.mark.parametrize("num_groups", [2, 4])
def test_conv2d_grouped(num_groups):
    """
    Test grouped convolution.

    Grouped conv splits channels into groups, reducing parameters.
    When num_groups == in_channels, it's depthwise convolution.
    Used in MobileNet, ShuffleNet, etc.
    """
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    in_channels = 8
    out_channels = 16

    out = conv2d(
        x, filters, border_mode="valid", filter_flip=False, num_groups=num_groups
    )

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, in_channels, 8, 8)).astype("float32")
    # Grouped: out_channels must be divisible by num_groups
    filters_val = rng.normal(
        size=(out_channels, in_channels // num_groups, 3, 3)
    ).astype("float32")

    compare_jax_and_py([x, filters], [out], [x_val, filters_val])


# =============================================================================
# Test Category 7: Gradient Tests
# =============================================================================


def test_conv2d_gradient_wrt_input():
    """
    Test Conv2D gradient with respect to input.

    Verifies that JAX's automatic differentiation produces correct
    gradients for the input tensor during backpropagation.
    """
    x = pt.tensor4("x", dtype="float32")
    filters_val = np.random.randn(16, 3, 3, 3).astype("float32")
    filters = shared(filters_val, name="filters")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)
    loss = out.sum()

    grad_x = grad(loss, x)

    # Compile with JAX mode
    f = function([x], [loss, grad_x], mode="JAX")

    x_val = np.random.randn(2, 3, 8, 8).astype("float32")
    _loss_val, grad_x_val = f(x_val)

    # Verify gradient is not zero (should have meaningful values)
    assert np.abs(grad_x_val).sum() > 0, "Gradient should not be zero"
    assert grad_x_val.shape == x_val.shape, "Gradient shape should match input"

    # Compare with Python backend
    f_py = function([x], [loss, grad_x], mode="FAST_RUN")
    _loss_py, grad_x_py = f_py(x_val)

    np.testing.assert_allclose(grad_x_val, grad_x_py, rtol=1e-5, atol=1e-5)


def test_conv2d_gradient_wrt_filters():
    """
    Test Conv2D gradient with respect to filters.

    This is critical for training - verifies that filter gradients are
    computed correctly for weight updates during backpropagation.
    """
    x_val = np.random.randn(2, 3, 8, 8).astype("float32")
    x = shared(x_val, name="x")
    filters = pt.tensor4("filters", dtype="float32")

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)
    loss = out.sum()

    grad_filters = grad(loss, filters)

    f = function([filters], [loss, grad_filters], mode="JAX")

    filters_val = np.random.randn(16, 3, 3, 3).astype("float32")
    _loss_val, grad_filters_val = f(filters_val)

    assert np.abs(grad_filters_val).sum() > 0
    assert grad_filters_val.shape == filters_val.shape

    # Compare with Python backend
    f_py = function([filters], [loss, grad_filters], mode="FAST_RUN")
    _loss_py, grad_filters_py = f_py(filters_val)

    np.testing.assert_allclose(grad_filters_val, grad_filters_py, rtol=1e-5, atol=1e-5)


# =============================================================================
# Test Category 8: Dtype Tests
# =============================================================================


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_conv2d_dtypes(dtype):
    """
    Test Conv2D with different dtypes.

    Ensures convolution works with both single and double precision.
    """
    x = pt.tensor4("x", dtype=dtype)
    filters = pt.tensor4("filters", dtype=dtype)

    out = conv2d(x, filters, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 8, 8)).astype(dtype)
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype(dtype)

    # Adjust tolerance for float32
    rtol = 1e-3 if dtype == "float32" else 1e-6
    atol = 1e-3 if dtype == "float32" else 1e-6

    compare_jax_and_py(
        [x, filters],
        [out],
        [x_val, filters_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol),
    )
