"""Tests for MaxPool ONNX conversion.

Tests the conversion of PyTensor's MaxPool operation to ONNX MaxPool.
"""

import numpy as np
import pytest


# Skip entire module if ONNX not available
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py


def test_maxpool2d_onnx_basic(tmp_path):
    """
    Test MaxPool2D exports to ONNX and produces same results.

    This is THE fundamental test - verifies:
    - MaxPool op is recognized by ONNX converter
    - Kernel size is correctly converted
    - Numerical results match PyTensor
    """
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # MaxPool with 2x2 kernel
    y = pool_2d(x, ws=(2, 2), mode="max")

    # Test data
    x_val = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype="float32",
    )

    # Compare ONNX and PyTensor outputs
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_maxpool2d_onnx_3x3_kernel(tmp_path):
    """Test MaxPool with 3x3 kernel (different from 2x2)."""
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(3, 3), mode="max")

    x_val = np.random.rand(1, 1, 10, 10).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_maxpool2d_onnx_stride(tmp_path):
    """
    Test MaxPool with stride parameter in ONNX.

    ONNX MaxPool has 'strides' attribute that must match PyTensor stride.
    """
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(2, 2), stride=(2, 2), mode="max")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_maxpool2d_onnx_multiple_channels(tmp_path):
    """
    Test MaxPool with multiple channels (typical CNN scenario).

    MaxPool operates independently on each channel.
    """
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(2, 2), mode="max")

    # Batch=2, Channels=16, 10x10 spatial
    x_val = np.random.rand(2, 16, 10, 10).astype("float32")

    _session, onnx_res = compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Verify output shape: (2, 16, 5, 5)
    assert onnx_res[0].shape == (2, 16, 5, 5)


def test_maxpool2d_onnx_yolo_sppf_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: SPPF pattern from YOLO11n.

    SPPF (Spatial Pyramid Pooling Fast):
    - Apply MaxPool multiple times with same kernel
    - Concatenate all intermediate results
    - Creates multi-scale features

    Pattern:
    x → MaxPool → MaxPool → MaxPool
    └─────┴─────────┴─────────┴──> Concat all 4
    """
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # SPPF pattern: cascade of 5x5 MaxPool
    pool1 = pool_2d(x, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))

    # Concatenate original + all pooled versions
    result = pt.join(1, x, pool1, pool2, pool3)

    # Test with YOLO-like feature map
    x_val = np.random.rand(1, 256, 20, 20).astype("float32")

    compare_onnx_and_py([x], result, [x_val], tmp_path=tmp_path)


def test_maxpool2d_1x1_kernel(tmp_path):
    """Test MaxPool with 1x1 kernel (identity operation)."""
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")
    y = pool_2d(x, ws=(1, 1), mode="max")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    # Output should equal input (1x1 max pool is identity)
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_maxpool2d_large_kernel(tmp_path):
    """Test MaxPool with kernel larger than input (global pooling)."""
    from pytensor.tensor.pool import pool_2d

    x = pt.tensor4("x", dtype="float32")

    # 8x8 kernel on 8x8 input = global max pooling
    y = pool_2d(x, ws=(8, 8), mode="max")

    x_val = np.random.rand(1, 3, 8, 8).astype("float32")

    _session, onnx_res = compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Output should be (1, 3, 1, 1) - single value per channel
    assert onnx_res[0].shape == (1, 3, 1, 1)
