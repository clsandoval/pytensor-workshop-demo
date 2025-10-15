"""Tests for Resize operations in PyTensor.

Tests the Resize operation independently of ONNX export.
"""

import numpy as np

import pytensor
import pytensor.tensor as pt


def test_resize_nearest_2x():
    """
    Test nearest neighbor resizing with 2x scale factor.

    Nearest neighbor:
    - Each pixel is duplicated
    - No interpolation
    - Fast but creates blocky output

    Configuration:
    - Mode: nearest
    - Scale: 2x (both H and W)
    """
    from pytensor.tensor.resize import resize  # Function we'll create

    x = pt.tensor4("x", dtype="float32")

    # Resize with 2x nearest neighbor
    y = resize(x, scale_factor=(2, 2), mode="nearest")

    f = pytensor.function([x], y)

    # Test data: 2x2 input
    x_val = np.array([[[[1, 2], [3, 4]]]], dtype="float32")

    # Expected: 4x4 output, each pixel duplicated
    # [[1, 1, 2, 2],
    #  [1, 1, 2, 2],
    #  [3, 3, 4, 4],
    #  [3, 3, 4, 4]]
    expected = np.array(
        [[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]], dtype="float32"
    )

    result = f(x_val)

    np.testing.assert_allclose(result, expected)


def test_resize_bilinear_2x():
    """
    Test bilinear resizing with 2x scale factor.

    Bilinear interpolation:
    - Smooth interpolation between pixels
    - Creates intermediate values
    - Higher quality than nearest neighbor
    """
    from pytensor.tensor.resize import resize

    x = pt.tensor4("x", dtype="float32")

    # Resize with 2x bilinear interpolation
    y = resize(x, scale_factor=(2, 2), mode="linear")

    f = pytensor.function([x], y)

    # Simple test case
    x_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype="float32")

    result = f(x_val)

    # Output should be (1, 1, 4, 4) with interpolated values
    assert result.shape == (1, 1, 4, 4)

    # Check corners match input
    np.testing.assert_allclose(result[0, 0, 0, 0], 1.0, rtol=1e-3)
    np.testing.assert_allclose(result[0, 0, -1, -1], 4.0, rtol=1e-3)


def test_resize_fractional_scale():
    """
    Test resize with non-integer scale factor.

    Example: 1.5x upsampling (6x6 -> 9x9)
    """
    from pytensor.tensor.resize import resize

    x = pt.tensor4("x", dtype="float32")

    # Resize with 1.5x scale
    y = resize(x, scale_factor=(1.5, 1.5), mode="nearest")

    f = pytensor.function([x], y)

    x_val = np.random.rand(1, 3, 6, 6).astype("float32")

    result = f(x_val)

    # Expected shape: (1, 3, 9, 9)
    assert result.shape == (1, 3, 9, 9)
