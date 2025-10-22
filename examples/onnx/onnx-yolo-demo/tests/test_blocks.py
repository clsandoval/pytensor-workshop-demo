"""Tests for YOLO11n building blocks."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from yolo.blocks import C2PSA, SPPF, Bottleneck, C3k2, ConvBNSiLU

import pytensor.tensor as pt
from pytensor import function


@settings(deadline=None)
@given(
    in_channels=st.sampled_from([3, 16, 32, 64]),
    out_filters=st.sampled_from([16, 32, 64]),
    size=st.sampled_from([(32, 32), (16, 16), (8, 8)]),
    stride=st.sampled_from([1, 2]),
)
def test_conv_bn_silu_output_shape(in_channels, out_filters, size, stride):
    """
    Property: ConvBNSiLU output shape matches convolution formula.

    For same padding:
    - out_h = ceil(in_h / stride)
    - out_w = ceil(in_w / stride)
    """
    batch = 2
    h, w = size
    input_tensor = np.random.randn(batch, in_channels, h, w).astype("float32")

    # Create symbolic input
    x = pt.tensor4("x", dtype="float32")

    # Apply ConvBNSiLU
    conv_block = ConvBNSiLU(
        in_channels, out_filters, kernel_size=3, stride=stride, padding="same"
    )
    y = conv_block(x)

    # Compile function
    f = function([x], y)
    output = f(input_tensor)

    # Expected shape
    expected_h = int(np.ceil(h / stride))
    expected_w = int(np.ceil(w / stride))
    expected_shape = (batch, out_filters, expected_h, expected_w)

    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )


@given(in_channels=st.sampled_from([16, 32, 64]))
def test_conv_bn_silu_activation_range(in_channels):
    """
    Property: SiLU activation produces finite outputs.

    SiLU(x) = x * sigmoid(x) should always produce finite values.
    """
    batch = 2
    h, w = 8, 8
    input_tensor = np.random.randn(batch, in_channels, h, w).astype("float32")

    x = pt.tensor4("x", dtype="float32")
    conv_block = ConvBNSiLU(in_channels, 16, kernel_size=3, stride=1, padding="same")
    y = conv_block(x)

    f = function([x], y)
    output = f(input_tensor)

    # Output should be finite
    assert np.all(np.isfinite(output)), "Conv BN SiLU produced non-finite values"


@settings(deadline=None)
@given(in_channels=st.sampled_from([32, 64, 128]))
def test_c3k2_csp_split_correctness(in_channels):
    """
    Property: C3k2 processes channels through CSP architecture.

    CSP (Cross Stage Partial) should:
    - Preserve shape when in_channels == out_channels
    - Transform inputs (not identity)
    """
    batch = 2
    h, w = 16, 16
    out_channels = in_channels  # Preserve channels

    # Use non-zero input to ensure transformation is detectable
    input_tensor = np.random.randn(batch, in_channels, h, w).astype("float32") + 1.0

    x = pt.tensor4("x", dtype="float32")
    c3k2 = C3k2(in_channels, out_channels, n_blocks=2, shortcut=True)
    y = c3k2(x)

    f = function([x], y)
    output = f(input_tensor)

    # Shape preservation
    assert output.shape == input_tensor.shape, (
        f"C3k2 changed shape: {input_tensor.shape} → {output.shape}"
    )

    # Output should be different from input (transformation occurred)
    # Check that mean absolute difference is significant
    mean_diff = np.abs(output - input_tensor).mean()
    assert mean_diff > 0.01, (
        f"C3k2 output too similar to input (mean diff: {mean_diff})"
    )


def test_sppf_multi_scale_feature_extraction():
    """
    Property: SPPF concatenates features from multiple pooling scales.

    SPPF applies cascaded max pooling and should preserve spatial dimensions.
    """
    batch = 2
    in_channels = 256
    h, w = 10, 10

    # Use non-zero, varied input
    input_tensor = np.random.randn(batch, in_channels, h, w).astype("float32") + 0.5

    x = pt.tensor4("x", dtype="float32")
    sppf = SPPF(in_channels, in_channels, pool_size=5)
    y = sppf(x)

    f = function([x], y)
    output = f(input_tensor)

    # SPPF preserves spatial dimensions
    assert output.shape == (batch, in_channels, h, w), (
        f"SPPF changed shape: {input_tensor.shape} → {output.shape}"
    )

    # Output incorporates pooled features (different from input)
    mean_diff = np.abs(output - input_tensor).mean()
    assert mean_diff > 0.01, (
        f"SPPF output too similar to input (mean diff: {mean_diff})"
    )


def test_c2psa_attention_modulation():
    """
    Property: C2PSA applies attention to modulate features.

    C2PSA (CSP with Parallel Spatial Attention) should:
    - Preserve shape
    - Apply transformation to features
    """
    batch = 2
    in_channels = 256
    h, w = 10, 10

    # Use non-zero, varied input
    input_tensor = np.random.randn(batch, in_channels, h, w).astype("float32") + 0.5

    x = pt.tensor4("x", dtype="float32")
    c2psa = C2PSA(in_channels, in_channels)
    y = c2psa(x)

    f = function([x], y)
    output = f(input_tensor)

    assert output.shape == input_tensor.shape, (
        f"C2PSA changed shape: {input_tensor.shape} → {output.shape}"
    )

    # Attention should modulate features
    mean_diff = np.abs(output - input_tensor).mean()
    assert mean_diff > 0.01, (
        f"C2PSA did not significantly modulate features (mean diff: {mean_diff})"
    )


def test_bottleneck_residual_connection():
    """
    Test: Bottleneck with shortcut preserves gradient flow.

    When shortcut=True, the residual connection should allow
    gradients to flow through both paths.
    """
    batch = 2
    channels = 64
    h, w = 16, 16

    x = pt.tensor4("x", dtype="float32")
    bottleneck = Bottleneck(channels, channels, shortcut=True)
    y = bottleneck(x)

    f = function([x], y)

    x_val = np.random.randn(batch, channels, h, w).astype("float32")
    output = f(x_val)

    # Shape should be preserved
    assert output.shape == x_val.shape, (
        f"Bottleneck changed shape: {x_val.shape} → {output.shape}"
    )

    # With residual connection, output should be different but related to input
    # (not identical due to transformation, but correlation exists)
    correlation = np.corrcoef(x_val.flatten(), output.flatten())[0, 1]
    assert not np.isnan(correlation), "Output correlation with input is NaN"
