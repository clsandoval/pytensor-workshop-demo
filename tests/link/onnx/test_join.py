"""Tests for Join (Concat) ONNX conversion.

Tests the conversion of PyTensor's Join operation to ONNX Concat.
"""

import numpy as np
import pytest


# Skip entire module if ONNX not available
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py


def test_join_axis0_two_tensors(tmp_path):
    """
    Test Join along axis 0 (row concatenation) with two 2D tensors.

    This is the simplest join case - verifies:
    - Join op is recognized and converted to ONNX Concat
    - Axis parameter is correctly passed
    - Output shape is calculated correctly ([3+2, 4] = [5, 4])
    - Numerical results match PyTensor

    Configuration:
    - axis=0 (concatenate rows)
    - 2 input tensors
    - Same shape except axis 0: (3,4) and (2,4)
    """
    # Arrange: Create symbolic inputs
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    # Define join operation
    z = pt.join(0, x, y)  # Concatenate along axis 0

    # Test data: Simple values for manual verification
    x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype="float32")

    y_val = np.array([[13, 14, 15, 16], [17, 18, 19, 20]], dtype="float32")

    # Expected output (manual verification):
    # [[1, 2, 3, 4],
    #  [5, 6, 7, 8],
    #  [9, 10, 11, 12],
    #  [13, 14, 15, 16],
    #  [17, 18, 19, 20]]

    # Act & Assert
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_axis1_two_tensors(tmp_path):
    """
    Test Join along axis 1 (column concatenation) with two 2D tensors.

    Verifies axis parameter handling - same operation, different axis.

    Configuration:
    - axis=1 (concatenate columns)
    - 2 input tensors
    - Same shape except axis 1: (3,2) and (3,3)
    """
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    z = pt.join(1, x, y)  # Concatenate along axis 1

    x_val = np.array([[1, 2], [3, 4], [5, 6]], dtype="float32")

    y_val = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype="float32")

    # Expected: (3, 5) output
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_three_tensors(tmp_path):
    """
    Test Join with three input tensors.

    Verifies:
    - ONNX Concat supports variable number of inputs (not just 2)
    - Multiple inputs are concatenated in correct order

    Configuration:
    - axis=0
    - 3 input tensors
    """
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.matrix("z", dtype="float32")

    result = pt.join(0, x, y, z)

    x_val = np.array([[1, 2]], dtype="float32")
    y_val = np.array([[3, 4]], dtype="float32")
    z_val = np.array([[5, 6]], dtype="float32")

    # Expected: [[1,2], [3,4], [5,6]]
    compare_onnx_and_py([x, y, z], result, [x_val, y_val, z_val], tmp_path=tmp_path)


def test_join_float64(tmp_path):
    """Test Join with float64 dtype."""
    x = pt.matrix("x", dtype="float64")
    y = pt.matrix("y", dtype="float64")

    z = pt.join(0, x, y)

    x_val = np.array([[1.5, 2.5]], dtype="float64")
    y_val = np.array([[3.5, 4.5]], dtype="float64")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_int32(tmp_path):
    """Test Join with int32 dtype."""
    x = pt.matrix("x", dtype="int32")
    y = pt.matrix("y", dtype="int32")

    z = pt.join(0, x, y)

    x_val = np.array([[1, 2]], dtype="int32")
    y_val = np.array([[3, 4]], dtype="int32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_vectors_axis0(tmp_path):
    """Test Join with 1D vectors."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")

    z = pt.join(0, x, y)

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5], dtype="float32")

    # Expected: [1, 2, 3, 4, 5]
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_4d_tensors_axis1(tmp_path):
    """
    Test Join with 4D tensors (NCHW format, typical for CNNs).

    This is THE critical test for YOLO11n - skip connections join
    feature maps from different layers along the channel dimension.

    Configuration:
    - 4D tensors: (batch, channels, height, width)
    - axis=1 (channel dimension)
    - Simulates skip connection in FPN head
    """
    x = pt.tensor4("x", dtype="float32")
    y = pt.tensor4("y", dtype="float32")

    z = pt.join(1, x, y)  # Concatenate along channel axis

    # Batch=1, different channels, same H and W
    x_val = np.random.rand(1, 3, 8, 8).astype("float32")
    y_val = np.random.rand(1, 5, 8, 8).astype("float32")

    # Expected output shape: (1, 8, 8, 8)
    _session, onnx_res = compare_onnx_and_py(
        [x, y], z, [x_val, y_val], tmp_path=tmp_path
    )

    assert onnx_res[0].shape == (1, 8, 8, 8), (
        f"Expected shape (1, 8, 8, 8), got {onnx_res[0].shape}"
    )


def test_join_negative_axis(tmp_path):
    """
    Test Join with negative axis indexing.

    ONNX Concat supports negative axes (e.g., axis=-1 for last dimension).
    Verify PyTensor's negative axis is correctly converted.
    """
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    z = pt.join(-1, x, y)  # axis=-1 means last axis (columns for 2D)

    x_val = np.array([[1], [2]], dtype="float32")
    y_val = np.array([[3], [4]], dtype="float32")

    # Expected: [[1, 3], [2, 4]]
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_single_element_tensors(tmp_path):
    """Test Join with tensors containing single elements."""
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")

    z = pt.join(0, x, y)

    x_val = np.array([[1.0]], dtype="float32")
    y_val = np.array([[2.0]], dtype="float32")

    # Expected: [[1.0], [2.0]]
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_join_after_conv2d(tmp_path):
    """
    Test Join combined with Conv2D (typical YOLO11n pattern).

    Pattern:
    - Two parallel convolution paths
    - Concatenate outputs along channel axis
    - This is the C3k2 block pattern
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel1 = pt.tensor4("kernel1", dtype="float32")
    kernel2 = pt.tensor4("kernel2", dtype="float32")

    # Two conv paths
    conv1 = conv2d(x, kernel1, border_mode="valid", filter_flip=False)
    conv2 = conv2d(x, kernel2, border_mode="valid", filter_flip=False)

    # Concatenate along channel axis
    result = pt.join(1, conv1, conv2)

    x_val = np.random.rand(1, 3, 10, 10).astype("float32")
    kernel1_val = np.random.rand(4, 3, 3, 3).astype("float32")
    kernel2_val = np.random.rand(8, 3, 3, 3).astype("float32")

    # Expected: (1, 12, 8, 8) - 4+8 channels
    compare_onnx_and_py(
        [x, kernel1, kernel2],
        result,
        [x_val, kernel1_val, kernel2_val],
        tmp_path=tmp_path,
    )
