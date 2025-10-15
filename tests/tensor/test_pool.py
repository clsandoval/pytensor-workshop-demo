"""Tests for MaxPool operations in PyTensor.

Tests the MaxPool operation independently of ONNX export.
"""

import numpy as np

import pytensor
import pytensor.tensor as pt


def test_maxpool2d_basic():
    """
    Test basic MaxPool2D operation in PyTensor.

    Configuration:
    - 4D input: (batch, channels, height, width)
    - Kernel size: 2x2
    - Stride: 2 (default, same as kernel size)
    - No padding
    """
    from pytensor.tensor.pool import pool_2d  # Function we'll create

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with 2x2 kernel
    y = pool_2d(x, ws=(2, 2), mode="max")

    # Compile PyTensor function
    f = pytensor.function([x], y)

    # Test data: 4x4 input
    x_val = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype="float32",
    )

    # Expected: 2x2 output with max of each 2x2 region
    # [[6, 8],
    #  [14, 16]]
    expected = np.array([[[[6, 8], [14, 16]]]], dtype="float32")

    result = f(x_val)

    np.testing.assert_allclose(result, expected)


def test_maxpool2d_stride():
    """
    Test MaxPool2D with stride different from kernel size.

    Configuration:
    - Kernel: 3x3
    - Stride: 1 (overlapping pools)
    - Verifies stride parameter works independently
    """
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with 3x3 kernel, stride 1
    y = pool_2d(x, ws=(3, 3), stride=(1, 1), mode="max")

    f = pytensor.function([x], y)

    # 5x5 input
    x_val = np.arange(25, dtype="float32").reshape(1, 1, 5, 5)

    result = f(x_val)

    # Expected shape: (1, 1, 3, 3) with stride 1
    assert result.shape == (1, 1, 3, 3)


def test_maxpool2d_padding():
    """
    Test MaxPool2D with padding.

    Configuration:
    - Kernel: 2x2
    - Padding: (1, 1) - add 1 pixel border
    - Padding value: -inf (or very negative) so max ignores it
    """
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with padding
    y = pool_2d(x, ws=(2, 2), padding=(1, 1), mode="max")

    f = pytensor.function([x], y)

    x_val = np.ones((1, 1, 4, 4), dtype="float32")

    result = f(x_val)

    # With padding (1,1), output should be larger
    assert result.shape == (1, 1, 3, 3)
