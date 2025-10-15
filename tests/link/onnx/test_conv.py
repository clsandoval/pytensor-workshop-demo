"""Tests for ONNX convolution operations."""

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from pytensor.tensor.conv.abstract_conv import conv2d
from tests.link.onnx.test_basic import compare_onnx_and_py


# ============================================================================
# Test Category 1: Basic Operation Tests
# ============================================================================


def test_conv2d_valid_single_channel(tmp_path):
    """
    Test basic 2D convolution with valid padding and single channel.

    This is the simplest convolution case - verifies:
    - Conv2D op is recognized and converted
    - Basic ONNX Conv node is created
    - Output shape is calculated correctly
    - Numerical results match PyTensor

    Configuration:
    - border_mode='valid' (no padding)
    - subsample=(1,1) (no stride)
    - filter_flip=False (cross-correlation, matches ONNX)
    - filter_dilation=(1,1) (no dilation)
    - num_groups=1 (standard convolution)
    """
    # Arrange: Create symbolic inputs
    x = pt.tensor4("x", dtype="float32")  # (batch, channels, height, width)
    kernel = pt.tensor4("kernel", dtype="float32")  # (filters, in_channels, kh, kw)

    # Define convolution operation
    y = conv2d(
        x,
        kernel,
        border_mode="valid",
        subsample=(1, 1),
        filter_flip=False,  # CRITICAL: Use cross-correlation to match ONNX
        filter_dilation=(1, 1),
        num_groups=1,
    )

    # Test data: Simple values for manual verification
    x_val = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ]
            ]
        ],
        dtype="float32",
    )

    kernel_val = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype="float32")

    # Act & Assert: Compare ONNX Runtime output with PyTensor
    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


@pytest.mark.parametrize(
    "input_shape,kernel_shape,expected_output_shape",
    [
        ((1, 1, 5, 5), (1, 1, 3, 3), (1, 1, 3, 3)),  # Valid padding
        ((1, 1, 10, 10), (1, 1, 5, 5), (1, 1, 6, 6)),  # Larger input
        ((2, 1, 7, 7), (3, 1, 3, 3), (2, 3, 5, 5)),  # Batch + multiple filters
    ],
)
def test_conv2d_output_shape(
    tmp_path, input_shape, kernel_shape, expected_output_shape
):
    """
    Test that Conv2D output shapes are calculated correctly.

    Output shape formula (valid padding):
    output_h = (input_h - kernel_h) + 1
    output_w = (input_w - kernel_w) + 1

    This test verifies ONNX Conv respects shape semantics.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random(input_shape).astype("float32")
    kernel_val = rng.random(kernel_shape).astype("float32")

    # Compare outputs
    __session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Verify output shape
    assert onnx_res[0].shape == expected_output_shape, (
        f"Expected shape {expected_output_shape}, got {onnx_res[0].shape}"
    )


# ============================================================================
# Test Category 2: CRITICAL - Filter Flipping Tests
# ============================================================================


def test_conv2d_filter_flip_false(tmp_path):
    """
    Test Conv2D with filter_flip=False (cross-correlation).

    When filter_flip=False:
    - PyTensor performs cross-correlation (no kernel flip)
    - ONNX Conv also performs cross-correlation (no flip)
    - Direct mapping should work correctly

    This is the simpler case and should work immediately.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


def test_conv2d_filter_flip_true_symmetric(tmp_path):
    """
    Test Conv2D with filter_flip=True and symmetric kernel.

    When kernel is symmetric (e.g., Gaussian blur), flipping doesn't change result.
    This test ensures filter_flip=True is recognized, even if flip is no-op.

    Note: This test will PASS even if flip logic is broken (symmetric kernel)!
    See test_conv2d_filter_flip_true_asymmetric for the critical test.
    """
    from pytensor import shared

    x = pt.tensor4("x", dtype="float32")

    # Symmetric Gaussian-like kernel
    kernel_val = (
        np.array([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype="float32") / 16.0
    )  # Normalized

    # Use shared variable for kernel (realistic for trained weights)
    kernel = shared(kernel_val, name="kernel")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=True)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_conv2d_filter_flip_true_asymmetric(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: Conv2D with filter_flip=True and ASYMMETRIC kernel.

    This is THE most important test for Conv2D correctness!

    When filter_flip=True:
    - PyTensor flips kernel (mathematical convolution)
    - ONNX Conv does NOT flip (cross-correlation)
    - We MUST flip the kernel before passing to ONNX

    Using Sobel edge detector (asymmetric):
    - If we DON'T flip: Wrong results (detects edges in wrong direction)
    - If we DO flip correctly: Results match PyTensor

    Failure modes:
    - Test passes with symmetric kernel but fails here: Flip not implemented!
    - Results don't match: Flip implemented incorrectly
    - Error: Flip not supported yet (acceptable for Phase 1)
    """
    from pytensor import shared

    x = pt.tensor4("x", dtype="float32")

    # Sobel X edge detector (ASYMMETRIC!)
    # Detects vertical edges (left-to-right transitions)
    sobel_x = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype="float32")

    # Use shared variable for kernel (realistic for trained weights)
    kernel = shared(sobel_x, name="kernel")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=True)

    # Test image with vertical edge
    # Left side: bright (1.0), right side: dark (0.0)
    x_val = np.array(
        [
            [
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype="float32",
    )

    # Expected: Strong response at the edge (column index 1-2)
    # If flip is wrong: Response will be inverted or at wrong location

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# ============================================================================
# Test Category 3: Padding Mode Tests
# ============================================================================


def test_conv2d_valid_padding(tmp_path):
    """
    Test Conv2D with 'valid' padding (no padding).

    Valid padding:
    - PyTensor: border_mode='valid'
    - ONNX: auto_pad='VALID' or pads=[0,0,0,0]
    - Output size: (input_size - kernel_size) + 1

    This is the default and simplest padding mode.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 8, 8)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Verify output shape: (8-3)+1 = 6
    assert onnx_res[0].shape == (1, 1, 6, 6)


def test_conv2d_same_padding(tmp_path):
    """
    Test Conv2D with 'same' padding.

    Same padding:
    - PyTensor: border_mode='half' (maintains input size)
    - ONNX: auto_pad='SAME_UPPER'
    - Output size: same as input (when stride=1)

    Padding amount: floor(kernel_size / 2) on each side
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="half", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 8, 8)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Verify output shape: same as input
    assert onnx_res[0].shape == (1, 1, 8, 8)


def test_conv2d_explicit_symmetric_padding(tmp_path):
    """
    Test Conv2D with explicit symmetric padding.

    Symmetric padding:
    - PyTensor: border_mode=(pad_h, pad_w)
    - ONNX: pads=[pad_h, pad_w, pad_h, pad_w]
    - Same padding on all sides

    Example: (1, 1) adds 1 pixel padding on all 4 sides
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Add 1 pixel padding on each side
    y = conv2d(x, kernel, border_mode=(1, 1), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Output size: (5 + 2*1 - 3) + 1 = 5 (same as input)
    assert onnx_res[0].shape == (1, 1, 5, 5)


def test_conv2d_explicit_asymmetric_padding(tmp_path):
    """
    Test Conv2D with explicit asymmetric padding.

    Asymmetric padding:
    - PyTensor: border_mode=((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
    - ONNX: pads=[pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]
    - Different padding on each side

    Example: ((1,2), (0,1)) adds:
    - 1 pixel top, 2 pixels bottom
    - 0 pixels left, 1 pixel right
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Asymmetric padding
    y = conv2d(x, kernel, border_mode=((1, 2), (0, 1)), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Output size:
    # height: (5 + 1 + 2 - 3) + 1 = 6
    # width: (5 + 0 + 1 - 3) + 1 = 4
    assert onnx_res[0].shape == (1, 1, 6, 4)


# ============================================================================
# Test Category 4: Stride Tests (subsample)
# ============================================================================


def test_conv2d_stride_2x2(tmp_path):
    """
    Test Conv2D with stride 2x2 (downsampling).

    Strided convolution:
    - PyTensor: subsample=(stride_h, stride_w)
    - ONNX: strides=[stride_h, stride_w]
    - Output size: floor((input_size - kernel_size) / stride) + 1

    Common in CNNs for downsampling instead of pooling.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", subsample=(2, 2), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 8, 8)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Output size: floor((8-3)/2) + 1 = 3
    assert onnx_res[0].shape == (1, 1, 3, 3)


def test_conv2d_asymmetric_stride(tmp_path):
    """
    Test Conv2D with asymmetric stride (stride_h != stride_w).

    Asymmetric stride:
    - PyTensor: subsample=(2, 1)
    - ONNX: strides=[2, 1]
    - Different downsampling factors for H and W

    Less common but valid configuration.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", subsample=(2, 1), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 10, 10)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Output size: (floor((10-3)/2)+1, floor((10-3)/1)+1) = (4, 8)
    assert onnx_res[0].shape == (1, 1, 4, 8)


# ============================================================================
# Test Category 5: Dilation Tests (Atrous Convolution)
# ============================================================================


def test_conv2d_dilation_2x2(tmp_path):
    """
    Test Conv2D with dilation 2x2 (atrous convolution).

    Dilated convolution:
    - PyTensor: filter_dilation=(dilation_h, dilation_w)
    - ONNX: dilations=[dilation_h, dilation_w]
    - Expands receptive field without increasing parameters
    - Effective kernel size: kernel_size + (kernel_size - 1) * (dilation - 1)

    Example: 3x3 kernel with dilation=2 has effective size 5x5
    Common in semantic segmentation (DeepLab, etc.)
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(
        x, kernel, border_mode="valid", filter_dilation=(2, 2), filter_flip=False
    )

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 10, 10)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Effective kernel: 3 + (3-1)*1 = 5
    # Output size: (10-5)+1 = 6
    assert onnx_res[0].shape == (1, 1, 6, 6)


# ============================================================================
# Test Category 6: Grouped Convolution Tests
# ============================================================================


def test_conv2d_grouped_convolution(tmp_path):
    """
    Test Conv2D with grouped convolution.

    Grouped convolution:
    - PyTensor: num_groups=2 (or other value)
    - ONNX: group=2
    - Divides input/output channels into groups
    - Each group processes independently
    - Reduces parameters and computation

    Example: 4 input channels, 8 output channels, 2 groups
    - Group 1: channels 0-1 → filters 0-3
    - Group 2: channels 2-3 → filters 4-7

    Common in efficient architectures (ResNeXt, ShuffleNet).
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", num_groups=2, filter_flip=False)

    rng = np.random.default_rng(42)
    # 4 input channels, 8 output filters, 2 groups
    x_val = rng.random((1, 4, 8, 8)).astype("float32")
    kernel_val = rng.random((8, 2, 3, 3)).astype(
        "float32"
    )  # 8 filters, 2 channels per group

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


def test_conv2d_depthwise_convolution(tmp_path):
    """
    Test Conv2D with depthwise convolution (special case of grouped).

    Depthwise convolution:
    - PyTensor: num_groups = num_input_channels
    - ONNX: group = num_input_channels
    - Each input channel has its own filter
    - Extremely parameter-efficient
    - Common in MobileNet, EfficientNet

    Example: 16 input channels, 16 groups → 1 filter per channel
    Usually followed by 1x1 convolution (pointwise) → "Depthwise Separable"
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    num_channels = 8
    y = conv2d(
        x, kernel, border_mode="valid", num_groups=num_channels, filter_flip=False
    )

    rng = np.random.default_rng(42)
    x_val = rng.random((1, num_channels, 8, 8)).astype("float32")
    # Depthwise: num_filters = num_channels, channels_per_filter = 1
    kernel_val = rng.random((num_channels, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


# ============================================================================
# Test Category 7: Multi-Channel Tests
# ============================================================================


def test_conv2d_rgb_input(tmp_path):
    """
    Test Conv2D with RGB-like 3-channel input.

    Multi-channel input:
    - Common for color images (RGB: 3 channels)
    - Kernel must have matching input channels
    - Each output filter convolves across ALL input channels

    Configuration:
    - Input: (batch, 3, H, W) - RGB image
    - Kernel: (num_filters, 3, kH, kW) - 3 input channels
    - Output: (batch, num_filters, H', W')
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 3, 8, 8)).astype("float32")  # batch=2, RGB
    kernel_val = rng.random((16, 3, 3, 3)).astype("float32")  # 16 filters

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


def test_conv2d_batch_processing(tmp_path):
    """
    Test Conv2D with batch processing.

    Batch processing:
    - Multiple samples processed in parallel
    - Batch dimension is independent
    - Common in training (batch_size = 32, 64, etc.)

    Configuration:
    - Input: (batch, channels, H, W)
    - Kernel: (filters, channels, kH, kW)
    - Output: (batch, filters, H', W')

    Each sample in batch is convolved independently with same kernel.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((8, 1, 5, 5)).astype("float32")  # batch=8
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


# ============================================================================
# Test Category 8: Integration Tests
# ============================================================================


def test_conv2d_with_bias(tmp_path):
    """
    Test Conv2D followed by bias addition.

    Typical CNN layer:
    - Convolution computes weighted sum
    - Bias added to each output channel
    - Pattern: y = conv(x, kernel) + bias

    ONNX Conv can include bias as third input, but PyTensor
    typically does this as separate Add operation.

    This tests that pattern works correctly.
    Future optimization: Fuse bias into Conv node.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")
    bias = pt.vector("bias", dtype="float32")

    # Conv + bias
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)
    y = conv_out + bias.dimshuffle("x", 0, "x", "x")  # Broadcast bias

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((8, 1, 3, 3)).astype("float32")  # 8 filters
    bias_val = rng.random(8).astype("float32")  # 8 biases

    compare_onnx_and_py(
        [x, kernel, bias], y, [x_val, kernel_val, bias_val], tmp_path=tmp_path
    )


def test_conv2d_relu_pattern(tmp_path):
    """
    Test Conv2D followed by ReLU activation.

    Standard CNN layer pattern:
    - Convolution
    - ReLU activation (non-linearity)
    - Often followed by pooling (when available)

    Configuration: Conv → ReLU

    This tests that Conv integrates with existing activation converters.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Conv + ReLU
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)
    y = pt.maximum(conv_out, 0)  # ReLU

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((8, 1, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


def test_simple_cnn_block(tmp_path):
    """
    Test a simple CNN block: Conv → ReLU → Flatten.

    This simulates a typical CNN layer:
    1. Convolution extracts features
    2. ReLU adds non-linearity
    3. Flatten prepares for dense layer

    Integration test ensuring Conv works with rest of pipeline.
    """
    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # CNN block
    conv_out = conv2d(x, kernel, border_mode="valid", filter_flip=False)
    relu_out = pt.maximum(conv_out, 0)
    y = relu_out.flatten(2)  # Flatten spatial dimensions

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 1, 5, 5)).astype("float32")  # batch=2
    kernel_val = rng.random((4, 1, 3, 3)).astype("float32")  # 4 filters

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)
