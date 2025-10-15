"""Tests for ONNX elemwise operations."""

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_add(tmp_path):
    """Test addition operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_mul(tmp_path):
    """Test multiplication operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x * y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_sub(tmp_path):
    """Test subtraction operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x - y

    x_val = np.array([5, 6, 7], dtype="float32")
    y_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_div(tmp_path):
    """Test division operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x / y

    x_val = np.array([4, 9, 16], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_neg(tmp_path):
    """Test negation operation."""
    x = pt.vector("x", dtype="float32")
    z = -x

    x_val = np.array([1, -2, 3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_exp(tmp_path):
    """Test exponential operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.exp(x)

    x_val = np.array([0, 1, 2], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_log(tmp_path):
    """Test logarithm operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.log(x)

    x_val = np.array([1, 2.718, 7.389], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_sqrt(tmp_path):
    """Test square root operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.sqrt(x)

    x_val = np.array([1, 4, 9], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


def test_pow(tmp_path):
    """Test power operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x**y

    x_val = np.array([2, 3, 4], dtype="float32")
    y_val = np.array([2, 2, 2], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_abs(tmp_path):
    """Test absolute value operation."""
    x = pt.vector("x", dtype="float32")
    z = pt.abs(x)

    x_val = np.array([-1, 2, -3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize(
    "shape",
    [
        (3,),  # vector
        (2, 3),  # matrix
        (2, 3, 4),  # 3D tensor
    ],
)
def test_add_different_shapes(tmp_path, shape):
    """Test addition with different tensor shapes."""
    x = pt.tensor("x", dtype="float32", shape=shape)
    y = pt.tensor("y", dtype="float32", shape=shape)
    z = x + y

    rng = np.random.default_rng(42)
    x_val = rng.random(shape).astype("float32")
    y_val = rng.random(shape).astype("float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_chained_operations(tmp_path):
    """Test multiple operations chained together."""
    x = pt.vector("x", dtype="float32")
    # (x * 2 + 3) / 4
    z = ((x * 2) + 3) / 4

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    [
        ("float32", "float64"),
        ("float32", "int32"),
        ("float32", "int64"),
        ("int32", "float32"),
        ("int32", "int64"),
        ("int64", "float32"),
        ("float64", "float32"),
    ],
)
def test_cast_dtypes(tmp_path, from_dtype, to_dtype):
    """Test Cast operation with various dtype conversions."""
    x = pt.vector("x", dtype=from_dtype)
    y = pt.cast(x, to_dtype)

    rng = np.random.default_rng(42)
    if from_dtype.startswith("float"):
        x_val = rng.random(5).astype(from_dtype)
    else:
        x_val = rng.integers(-10, 10, size=5).astype(from_dtype)

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_cast_in_computation(tmp_path):
    """Test Cast used within a computation graph."""
    x = pt.vector("x", dtype="int32")
    # Convert to float, do computation, convert back
    x_float = pt.cast(x, "float32")
    y_float = x_float * 2.5 + 1.0
    y = pt.cast(y_float, "int32")

    x_val = np.array([1, 2, 3, 4, 5], dtype="int32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_cast_structure(tmp_path):
    """Test that Cast generates correct ONNX node."""
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.cast(x, "int32")

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_cast.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    validate_onnx_graph_structure(
        model,
        expected_node_types=["Cast"],
        expected_node_count=1,
    )

    # Check Cast node has 'to' attribute
    cast_node = model.graph.node[0]
    assert cast_node.op_type == "Cast"
    to_attr = next(attr for attr in cast_node.attribute if attr.name == "to")
    assert to_attr.i == 6  # TensorProto.INT32


def test_composite_scalar_op(tmp_path):
    """Test Composite scalar op decomposition.

    PyTensor's optimizer often fuses multiple scalar ops into a Composite.
    We need to decompose this back into individual ONNX nodes.
    """
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")

    # Create a computation that PyTensor might fuse into a Composite
    # (x * 2 + y) * 3
    z = (x * 2 + y) * 3

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    # Test execution
    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_composite_with_constants(tmp_path):
    """Test Composite that includes constant folding."""
    x = pt.vector("x", dtype="float32")

    # Expression with constants: x * 2.0 + 3.0
    y = x * 2.0 + 3.0

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_composite_complex_expression(tmp_path):
    """Test complex expression that becomes Composite."""
    x = pt.vector("x", dtype="float32")

    # Complex expression: x^2 * 2 + x
    y = x**2 * 2 + x

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


# Sigmoid tests
def test_sigmoid_basic(tmp_path):
    """
    Test basic sigmoid activation exports to ONNX.

    This is the fundamental test - verifies:
    - Sigmoid scalar op is recognized by ONNX converter
    - Output matches PyTensor sigmoid
    - Numerical stability for positive and negative values

    Sigmoid formula: y = 1 / (1 + exp(-x))
    - Maps any value to (0, 1) range
    - Used in attention mechanisms and gates
    """
    from functools import partial

    x = pt.vector("x", dtype="float32")

    # Apply sigmoid
    y = pt.sigmoid(x)

    # Test data covering different ranges
    x_val = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype="float32")

    # Expected (manual calculation):
    # sigmoid(-10) ≈ 0.0000454
    # sigmoid(-1) ≈ 0.268941
    # sigmoid(0) = 0.5
    # sigmoid(1) ≈ 0.731059
    # sigmoid(10) ≈ 0.9999546

    # Use absolute tolerance for small values near sigmoid saturation
    compare_onnx_and_py(
        [x],
        y,
        [x_val],
        tmp_path=tmp_path,
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-7),
    )


def test_sigmoid_matrix(tmp_path):
    """Test sigmoid on 2D matrix."""
    x = pt.matrix("x", dtype="float32")
    y = pt.sigmoid(x)

    x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_sigmoid_4d_tensor(tmp_path):
    """
    Test sigmoid on 4D tensor (CNN feature maps).

    Used in attention mechanisms like C2PSA in YOLO11n.
    """
    x = pt.tensor4("x", dtype="float32")
    y = pt.sigmoid(x)

    # Typical CNN feature map
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((2, 64, 16, 16)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_sigmoid_numerical_stability(tmp_path):
    """
    Test sigmoid with extreme values (numerical stability).

    Sigmoid should:
    - Not overflow for large positive values (→ 1.0)
    - Not underflow for large negative values (→ 0.0)
    - Handle values near zero correctly
    """
    from functools import partial

    x = pt.vector("x", dtype="float32")
    y = pt.sigmoid(x)

    # Extreme values
    x_val = np.array([-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0], dtype="float32")

    # Use absolute tolerance for extreme values that saturate to 0 or 1
    compare_onnx_and_py(
        [x],
        y,
        [x_val],
        tmp_path=tmp_path,
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-7),
    )


def test_sigmoid_in_attention_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: Sigmoid in attention mechanism (C2PSA pattern).

    Attention pattern:
    1. Compute attention scores
    2. Apply sigmoid to get attention weights (0 to 1)
    3. Multiply features by attention weights

    This is how C2PSA blocks use sigmoid in YOLO11n.
    """
    # Feature maps
    features = pt.tensor4("features", dtype="float32")
    # Attention scores (computed by some network)
    attention_scores = pt.tensor4("attention_scores", dtype="float32")

    # Apply sigmoid to attention scores
    attention_weights = pt.sigmoid(attention_scores)

    # Weighted features
    weighted_features = features * attention_weights

    # Test data
    rng = np.random.default_rng(42)
    features_val = rng.standard_normal((1, 256, 20, 20)).astype("float32")
    attention_scores_val = rng.standard_normal((1, 256, 20, 20)).astype("float32")

    compare_onnx_and_py(
        [features, attention_scores],
        weighted_features,
        [features_val, attention_scores_val],
        tmp_path=tmp_path,
    )


# SiLU/Swish activation tests
def test_silu_basic(tmp_path):
    """
    Test basic SiLU activation exports to ONNX.

    SiLU formula: y = x * sigmoid(x) = x / (1 + exp(-x))

    Since ONNX doesn't have a native SiLU operator, we decompose it into:
    1. Sigmoid node
    2. Mul node (x * sigmoid(x))

    This test verifies:
    - SiLU decomposition works correctly
    - Output matches PyTensor SiLU
    - Works for various input ranges
    """
    x = pt.vector("x", dtype="float32")
    y = pt.silu(x)

    # Test data covering different ranges
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")

    # Expected (manual calculation):
    # silu(-2) = -2 / (1 + exp(2)) ≈ -0.238
    # silu(-1) ≈ -0.269
    # silu(0) = 0.0
    # silu(1) ≈ 0.731
    # silu(2) ≈ 1.762

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_silu_swish_alias(tmp_path):
    """
    Test that swish (alias for silu) exports correctly.

    Both pt.silu(x) and pt.swish(x) should produce identical ONNX.
    """
    x = pt.vector("x", dtype="float32")

    # Test with silu
    y_silu = pt.silu(x)

    # Test with swish
    y_swish = pt.swish(x)

    x_val = np.array([0.5, 1.0, 1.5], dtype="float32")

    # Both should produce identical results
    _, onnx_res_silu = compare_onnx_and_py([x], y_silu, [x_val], tmp_path=tmp_path)
    _, onnx_res_swish = compare_onnx_and_py([x], y_swish, [x_val], tmp_path=tmp_path)

    np.testing.assert_allclose(onnx_res_silu[0], onnx_res_swish[0], rtol=1e-7)


def test_silu_4d_tensor(tmp_path):
    """
    Test SiLU on 4D tensor (CNN feature maps).

    Used in modern CNNs like EfficientNet, MobileNetV3, YOLO.
    """
    x = pt.tensor4("x", dtype="float32")
    y = pt.silu(x)

    # Typical CNN feature map
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((2, 64, 16, 16)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_silu_in_activation_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: SiLU in typical CNN activation pattern (C3k2 pattern).

    Typical pattern in YOLO11n:
    1. Convolution (produces feature maps)
    2. Batch normalization (normalize features)
    3. SiLU activation (non-linearity)

    This test verifies SiLU works correctly with 4D tensors that would
    come from convolutional layers in real neural networks.
    """
    # Input: (batch, channels, height, width) - typical conv output
    x = pt.tensor4("x", dtype="float32")
    y = pt.tensor4("y", dtype="float32")

    # Simulate pattern: add two feature maps, then apply SiLU
    # (This is similar to residual connections + activation)
    combined = x + y
    activated = pt.silu(combined)

    # Test data - simulates output from convolutional layers
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((2, 16, 16, 16)).astype("float32")
    y_val = rng.standard_normal((2, 16, 16, 16)).astype("float32")

    compare_onnx_and_py([x, y], activated, [x_val, y_val], tmp_path=tmp_path)


def test_silu_decomposition_structure(tmp_path):
    """
    Test that SiLU generates correct multi-node ONNX decomposition.

    Expected ONNX structure:
    - Input x
    - Sigmoid(x) → sig_x
    - Mul(x, sig_x) → output

    This verifies the decomposition strategy works correctly.
    """
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.silu(x)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_silu.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Sigmoid", "Mul"],
        expected_node_count=2,
    )

    # Verify we have exactly one Sigmoid and one Mul
    assert structure["node_types"].count("Sigmoid") == 1, (
        f"Expected 1 Sigmoid node, got {structure['node_types'].count('Sigmoid')}"
    )
    assert structure["node_types"].count("Mul") == 1, (
        f"Expected 1 Mul node, got {structure['node_types'].count('Mul')}"
    )
