"""Regression tests for specific ONNX bugs.

These tests document specific bugs that were found and fixed.
They serve as fast smoke tests and documentation of edge cases.

DO NOT add routine tests here - use property tests in test_properties.py instead.
Only add tests for:
1. Specific bugs that were fixed
2. Edge cases that broke in production
3. Cases that took significant debugging to identify
"""

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from tests.link.onnx.test_basic import (
    compare_onnx_and_py,
    validate_onnx_graph_structure,
)


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


# ============================================================================
# DimShuffle Regressions
# ============================================================================


def test_dimshuffle_transpose_and_unsqueeze_regression(tmp_path):
    """
    Regression: DimShuffle incorrectly used Identity for transpose+unsqueeze.

    Bug: Pattern (1, 'x', 0) on shape (2,3) would incorrectly use Identity
    node, producing shape (2,3) instead of correct (3,1,2).

    Fixed: Added proper Squeeze→Transpose→Unsqueeze decomposition
    Reference: pytensor/link/onnx/dispatch/shape.py (DimShuffle conversion)
    """
    x = pt.matrix("x", dtype="float32")
    y = x.dimshuffle(1, "x", 0)  # (2,3) → (3,1,2)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

    # Should produce (3,1,2) shape, not (2,3)
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)

    # Verify correct ONNX structure includes Transpose or Unsqueeze
    from pytensor.link.onnx import export_onnx

    f = pytensor.function([x], y)
    model = export_onnx(f, tmp_path / "dimshuffle.onnx")

    structure = validate_onnx_graph_structure(model)
    # The graph should use Transpose or Unsqueeze for the complex pattern
    assert (
        "Transpose" in structure["node_types"] or "Unsqueeze" in structure["node_types"]
    ), "DimShuffle should use Transpose or Unsqueeze nodes"


def test_dimshuffle_squeeze_and_transpose_regression(tmp_path):
    """
    Regression: DimShuffle pattern (2, 0) on (2,1,3) incorrectly matched Case 3.

    Bug: Case 3 (pure transpose) didn't check for axes_to_add, so it matched
    patterns that also needed squeeze operations.

    Fixed: Added proper condition checking for squeeze operations
    Reference: pytensor/link/onnx/dispatch/shape.py (DimShuffle conversion)
    """
    x = pt.tensor(dtype="float32", shape=(2, 1, 3), name="x")
    y = x.dimshuffle(2, 0)  # (2,1,3) → (3,2)

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 1, 3)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# ============================================================================
# Composite Operation Regressions
# ============================================================================


def test_cast_in_composite_regression(tmp_path):
    """
    Regression: Cast operation not supported in Composite decomposition.

    Bug: decompose_composite_elemwise() didn't handle scalar.Cast operations.
    When PyTensor's optimizer fused Cast into a Composite, export would fail.

    Fixed: Added Cast handling in decompose_composite_elemwise
    Reference: pytensor/link/onnx/dispatch/elemwise.py (Composite decomposition)
    """
    x = pt.vector("x", dtype="int32")

    # This creates a Composite with Cast in FAST_RUN mode
    x_float = pt.cast(x, "float32")
    y_float = x_float * 2.5 + 1.0
    y = pt.cast(y_float, "int32")

    x_val = np.array([1, 2, 3, 4, 5], dtype="int32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_sqr_in_composite_regression(tmp_path):
    """
    Regression: Sqr scalar operation not in SCALAR_OP_TO_ONNX mapping.

    Bug: Expression x**2 creates scalar.Sqr op, which wasn't mapped to ONNX.

    Fixed: Added scalar.Sqr: "Mul" and special x*x handling
    Reference: pytensor/link/onnx/dispatch/elemwise.py (scalar op mapping)
    """
    x = pt.vector("x", dtype="float32")

    # Expression with x^2 that becomes Composite with Sqr
    y = x**2 * 2 + x

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# ============================================================================
# Structure Validation Tests
# ============================================================================


def test_cast_generates_correct_onnx_node(tmp_path):
    """Validate that Cast generates ONNX Cast node with correct 'to' attribute."""
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.cast(x, "int32")

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_cast.onnx"
    model = export_onnx(f, model_path)

    validate_onnx_graph_structure(
        model,
        expected_node_types=["Cast"],
        expected_node_count=1,
    )

    # Verify Cast node has correct 'to' attribute
    cast_node = model.graph.node[0]
    assert cast_node.op_type == "Cast"
    to_attr = next(attr for attr in cast_node.attribute if attr.name == "to")
    assert to_attr.i == 6, "Cast to int32 should have TensorProto.INT32 = 6"


def test_deep_copy_generates_identity(tmp_path):
    """Validate that DeepCopyOp generates ONNX Identity node."""
    from pytensor.compile.ops import DeepCopyOp
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    deep_copy_op = DeepCopyOp()
    y = deep_copy_op(x)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_deep_copy.onnx"
    model = export_onnx(f, model_path)

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Identity"],
        expected_node_count=1,
    )

    assert structure["node_types"] == ["Identity"]


# ============================================================================
# Known Edge Cases
# ============================================================================


def test_alloc_empty_with_shape_from_tensor(tmp_path):
    """Test AllocEmpty with dimensions extracted from another tensor's shape."""
    from pytensor.tensor.basic import AllocEmpty

    x = pt.matrix("x", dtype="float32")
    dim0 = x.shape[0]
    dim1 = x.shape[1]

    alloc_op = AllocEmpty(dtype="float32")
    y = alloc_op(dim0, dim1)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

    from pytensor.link.onnx import export_onnx

    f = pytensor.function([x], y)
    model_path = tmp_path / "test_alloc_empty.onnx"
    model = export_onnx(f, model_path)

    onnx.checker.check_model(model)

    # Run and verify shape matches
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {onnx_inputs[0].name: x_val}
    onnx_res = session.run(None, input_feed)

    assert onnx_res[0].shape == x_val.shape


# ============================================================================
# Conv2D Regressions ✨
# ============================================================================


def test_conv2d_filter_flip_true_asymmetric_regression(tmp_path):
    """
    ⭐⭐⭐ CRITICAL REGRESSION: Conv2D with filter_flip=True and asymmetric kernel.

    This is THE most important Conv2D correctness test!

    When filter_flip=True:
    - PyTensor flips kernel (mathematical convolution)
    - ONNX Conv does NOT flip (cross-correlation)
    - We MUST flip the kernel before passing to ONNX

    Using Sobel edge detector (asymmetric):
    - If we DON'T flip: Wrong results (detects edges in wrong direction)
    - If we DO flip correctly: Results match PyTensor

    This test ensures the filter flipping logic remains correct.
    Reference: pytensor/link/onnx/dispatch/conv.py (filter flip handling)
    """
    from pytensor import shared
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")

    # Sobel X edge detector (ASYMMETRIC!)
    sobel_x = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype="float32")

    kernel = shared(sobel_x, name="kernel")
    y = conv2d(x, kernel, border_mode="valid", filter_flip=True)

    # Test image with vertical edge
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

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_conv2d_explicit_asymmetric_padding_regression(tmp_path):
    """
    Regression: Conv2D with asymmetric padding mapping to ONNX.

    Asymmetric padding is less common but critical for certain architectures.
    ONNX format: pads=[pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]

    This test ensures the padding order and values are correctly mapped.
    Reference: pytensor/link/onnx/dispatch/conv.py (padding conversion)
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    # Asymmetric padding: different on each side
    y = conv2d(x, kernel, border_mode=((1, 2), (0, 1)), filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 1, 5, 5)).astype("float32")
    kernel_val = rng.random((1, 1, 3, 3)).astype("float32")

    _session, onnx_res = compare_onnx_and_py(
        [x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path
    )

    # Verify output shape matches expected
    # height: (5 + 1 + 2 - 3) + 1 = 6
    # width: (5 + 0 + 1 - 3) + 1 = 4
    assert onnx_res[0].shape == (1, 1, 6, 4)


def test_conv2d_grouped_convolution_regression(tmp_path):
    """
    Regression: Grouped convolution channel dimension handling.

    Grouped convolution divides channels into independent groups.
    Critical for efficient architectures (ResNeXt, etc.).

    This test ensures the num_groups parameter is correctly passed to ONNX.
    Reference: pytensor/link/onnx/dispatch/conv.py (group parameter)
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

    x = pt.tensor4("x", dtype="float32")
    kernel = pt.tensor4("kernel", dtype="float32")

    y = conv2d(x, kernel, border_mode="valid", num_groups=2, filter_flip=False)

    rng = np.random.default_rng(42)
    x_val = rng.random((1, 4, 8, 8)).astype("float32")
    # 8 filters, 2 channels per group (4 input channels / 2 groups)
    kernel_val = rng.random((8, 2, 3, 3)).astype("float32")

    compare_onnx_and_py([x, kernel], y, [x_val, kernel_val], tmp_path=tmp_path)


def test_conv2d_dilation_regression(tmp_path):
    """
    Regression: Dilated convolution (atrous) output shape.

    Dilation expands the receptive field without adding parameters.
    Common in semantic segmentation (DeepLab, etc.).

    Effective kernel size: kernel_size + (kernel_size - 1) * (dilation - 1)
    This test ensures dilation is correctly passed to ONNX.
    Reference: pytensor/link/onnx/dispatch/conv.py (dilation parameter)
    """
    from pytensor.tensor.conv.abstract_conv import conv2d

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
