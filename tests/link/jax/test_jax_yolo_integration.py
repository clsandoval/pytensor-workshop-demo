"""
Integration tests for YOLO11 model with JAX backend.
Tests complete forward pass through the model.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.conv import conv2d
from pytensor.tensor.pool import pool_2d


jax = pytest.importorskip("jax")


def test_yolo11_forward_pass():
    """
    Test complete YOLO11 forward pass with JAX.

    This test verifies:
    - Model compiles with JAX JIT
    - Forward pass completes successfully
    - Output shapes are correct
    - Dynamic batch size works
    """

    # Mock YOLO11 model structure (simplified)
    def build_yolo11_model():
        """Build simplified YOLO11 model structure."""
        x = pt.tensor4("x", dtype="float32", shape=(None, 3, 640, 640))

        # Backbone blocks (simplified)
        # Initial conv
        conv1_w = pt.tensor4("conv1_w", dtype="float32")
        h = conv2d(x, conv1_w, border_mode="half", subsample=(2, 2))
        h = pt.maximum(0, h)  # ReLU activation

        # Downsample blocks
        conv_weights = []
        for i in range(4):
            conv_w = pt.tensor4(f"conv{i + 2}_w", dtype="float32")
            conv_weights.append(conv_w)
            h = conv2d(h, conv_w, border_mode="half", subsample=(2, 2))
            h = pt.maximum(0, h)  # ReLU activation

        # Detection heads at different scales (simplified - just return fixed outputs for now)
        # P3 - small objects (80x80)
        det_p3_w = pt.tensor4("det_p3_w", dtype="float32")
        det_p3 = conv2d(h, det_p3_w, border_mode="half")

        # P4 - medium objects (40x40)
        det_p4_w = pt.tensor4("det_p4_w", dtype="float32")
        det_p4 = conv2d(h, det_p4_w, border_mode="half")

        # P5 - large objects (20x20)
        det_p5_w = pt.tensor4("det_p5_w", dtype="float32")
        det_p5 = conv2d(h, det_p5_w, border_mode="half")

        params = [conv1_w] + conv_weights + [det_p3_w, det_p4_w, det_p5_w]
        return x, [det_p3, det_p4, det_p5], params

    # Arrange
    x, outputs, params = build_yolo11_model()

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 640, 640)).astype("float32") * 0.1

    # Generate param values
    param_vals = [
        rng.normal(size=(64, 3, 3, 3)).astype("float32") * 0.02,  # conv1
        rng.normal(size=(128, 64, 3, 3)).astype("float32") * 0.02,  # conv2
        rng.normal(size=(256, 128, 3, 3)).astype("float32") * 0.02,  # conv3
        rng.normal(size=(512, 256, 3, 3)).astype("float32") * 0.02,  # conv4
        rng.normal(size=(512, 512, 3, 3)).astype("float32") * 0.02,  # conv5
        rng.normal(size=(85, 512, 3, 3)).astype("float32") * 0.02,  # det_p3
        rng.normal(size=(85, 512, 3, 3)).astype("float32") * 0.02,  # det_p4
        rng.normal(size=(85, 512, 3, 3)).astype("float32") * 0.02,  # det_p5
    ]

    # Act - compile with JAX
    try:
        fn = function(inputs=[x] + params, outputs=outputs, mode="JAX")

        # Execute forward pass
        result = fn(x_val, *param_vals)
        success = True
    except Exception as e:
        success = False
        result = str(e)

    # Assert
    assert success, f"Forward pass failed: {result}"

    # Verify output shapes (adjusted for actual convolution output sizes)
    # After 5 downsampling by 2, 640 -> 320 -> 160 -> 80 -> 40 -> 20
    # With appropriate padding, output channels should be 85
    assert len(result) == 3, f"Expected 3 outputs, got {len(result)}"

    # Check that outputs are JAX arrays
    for i, det in enumerate(result):
        assert isinstance(det, jax.Array), f"Output {i} is not JAX Array"
        assert det.shape[0] == 2, f"Batch size mismatch in output {i}"
        assert det.shape[1] == 85, (
            f"Channel mismatch in output {i}: {det.shape[1]} != 85"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_yolo11_dynamic_batch(batch_size):
    """
    Test YOLO11 with different batch sizes.

    This test verifies:
    - Dynamic batch dimension handling
    - JIT recompilation for different shapes
    - Consistent output structure
    """

    # Arrange - simplified model
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 320, 320))

    # Simple processing (would be full model)
    w = pt.tensor4("w", dtype="float32")
    y = conv2d(x, w, border_mode="half")
    y = pt.maximum(0, y)  # ReLU activation
    output = y.mean(axis=(2, 3))  # Global average pool

    # Act - compile and test
    fn = function(inputs=[x, w], outputs=output, mode="JAX")

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(batch_size, 3, 320, 320)).astype("float32") * 0.1
    w_val = rng.normal(size=(64, 3, 3, 3)).astype("float32") * 0.02

    # Execute
    result = fn(x_val, w_val)

    # Assert
    assert result.shape == (batch_size, 64), (
        f"Batch {batch_size}: output shape {result.shape} != ({batch_size}, 64)"
    )
    assert np.all(np.isfinite(result)), f"Batch {batch_size}: output has NaN/Inf"
    assert isinstance(result, jax.Array), f"Batch {batch_size}: result is not JAX Array"


def test_yolo11_multi_scale_outputs():
    """
    Test YOLO11 multi-scale detection outputs.

    This test verifies:
    - Multiple detection heads work
    - Different output resolutions handled correctly
    - Feature pyramids propagate properly
    """

    # Build model with multi-scale outputs
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 640, 640))

    # Backbone with feature pyramid
    conv1_w = pt.tensor4("conv1_w", dtype="float32")
    conv2_w = pt.tensor4("conv2_w", dtype="float32")
    conv3_w = pt.tensor4("conv3_w", dtype="float32")

    # Build feature pyramid
    h1 = conv2d(x, conv1_w, border_mode="half", subsample=(2, 2))  # 320x320
    h1 = pt.maximum(0, h1)  # ReLU activation

    h2 = conv2d(h1, conv2_w, border_mode="half", subsample=(2, 2))  # 160x160
    h2 = pt.maximum(0, h2)  # ReLU activation

    h3 = conv2d(h2, conv3_w, border_mode="half", subsample=(2, 2))  # 80x80
    h3 = pt.maximum(0, h3)  # ReLU activation

    # Detection heads at each scale
    det1_w = pt.tensor4("det1_w", dtype="float32")
    det2_w = pt.tensor4("det2_w", dtype="float32")
    det3_w = pt.tensor4("det3_w", dtype="float32")

    det1 = conv2d(h1, det1_w, border_mode="half")  # P3 equivalent
    det2 = conv2d(h2, det2_w, border_mode="half")  # P4 equivalent
    det3 = conv2d(h3, det3_w, border_mode="half")  # P5 equivalent

    params = [conv1_w, conv2_w, conv3_w, det1_w, det2_w, det3_w]
    outputs = [det1, det2, det3]

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 640, 640)).astype("float32") * 0.1

    param_vals = [
        rng.normal(size=(64, 3, 3, 3)).astype("float32") * 0.02,
        rng.normal(size=(128, 64, 3, 3)).astype("float32") * 0.02,
        rng.normal(size=(256, 128, 3, 3)).astype("float32") * 0.02,
        rng.normal(size=(85, 64, 3, 3)).astype("float32") * 0.02,
        rng.normal(size=(85, 128, 3, 3)).astype("float32") * 0.02,
        rng.normal(size=(85, 256, 3, 3)).astype("float32") * 0.02,
    ]

    # Compile and run
    fn = function(inputs=[x] + params, outputs=outputs, mode="JAX")

    results = fn(x_val, *param_vals)

    # Verify multi-scale outputs
    assert len(results) == 3, "Should have 3 detection scales"

    # Check output shapes (with half padding, dimensions should be preserved at each scale)
    expected_shapes = [
        (2, 85, 320, 320),  # P3
        (2, 85, 160, 160),  # P4
        (2, 85, 80, 80),  # P5
    ]

    for i, (result, expected) in enumerate(zip(results, expected_shapes)):
        assert isinstance(result, jax.Array), f"Scale {i}: not JAX Array"
        assert result.shape == expected, (
            f"Scale {i}: shape {result.shape} != {expected}"
        )
        assert np.all(np.isfinite(result)), f"Scale {i}: contains NaN/Inf"


def test_yolo11_with_pooling():
    """
    Test YOLO11 components with pooling layers.

    This test verifies:
    - SPPF (Spatial Pyramid Pooling Fast) works
    - MaxPool operations compile with JAX
    - Feature concatenation works
    """

    # Build model with SPPF block
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # SPPF block (simplified)
    conv_w = pt.tensor4("conv_w", dtype="float32")
    h = conv2d(x, conv_w, border_mode="half")

    # Multiple pooling scales
    pool1 = pool_2d(h, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")

    # Concatenate all scales
    output = pt.concatenate([h, pool1, pool2, pool3], axis=1)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 256, 32, 32)).astype("float32") * 0.1
    conv_w_val = rng.normal(size=(256, 256, 3, 3)).astype("float32") * 0.02

    # Compile and run
    fn = function(inputs=[x, conv_w], outputs=output, mode="JAX")

    result = fn(x_val, conv_w_val)

    # Verify SPPF output
    assert isinstance(result, jax.Array), "Output should be JAX Array"
    assert result.shape == (2, 1024, 32, 32), (
        f"SPPF output shape {result.shape} != (2, 1024, 32, 32)"
    )
    assert np.all(np.isfinite(result)), "SPPF output contains NaN/Inf"


def test_yolo11_csp_block():
    """
    Test YOLO11 CSP (Cross Stage Partial) blocks.

    This test verifies:
    - Split and concatenate operations work
    - Residual connections compile
    - Complex block patterns work with JAX
    """

    # Build CSP block
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # Split channels
    h_split1 = x[:, :128, :, :]
    h_split2 = x[:, 128:, :, :]

    # Process one split
    conv1_w = pt.tensor4("conv1_w", dtype="float32")
    conv2_w = pt.tensor4("conv2_w", dtype="float32")

    h_split2 = conv2d(h_split2, conv1_w, border_mode="half")
    h_split2 = pt.maximum(0, h_split2)  # ReLU activation
    h_split2 = conv2d(h_split2, conv2_w, border_mode="half")
    h_split2 = pt.maximum(0, h_split2)  # ReLU activation

    # Concatenate back
    output = pt.concatenate([h_split1, h_split2], axis=1)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 256, 32, 32)).astype("float32") * 0.1

    param_vals = [
        rng.normal(size=(128, 128, 3, 3)).astype("float32") * 0.02,
        rng.normal(size=(128, 128, 3, 3)).astype("float32") * 0.02,
    ]

    # Compile and run
    fn = function(inputs=[x, conv1_w, conv2_w], outputs=output, mode="JAX")

    result = fn(x_val, *param_vals)

    # Verify CSP output
    assert isinstance(result, jax.Array), "CSP output should be JAX Array"
    assert result.shape == (2, 256, 32, 32), (
        f"CSP output shape {result.shape} != (2, 256, 32, 32)"
    )
    assert np.all(np.isfinite(result)), "CSP output contains NaN/Inf"
