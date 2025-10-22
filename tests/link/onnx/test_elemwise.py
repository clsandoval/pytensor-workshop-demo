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


# IntDiv (Floor Division) tests
def test_floor_div_yolo_padding_pattern(tmp_path):
    """Test floor division in YOLO padding calculation pattern.

    Pattern: (kernel_size - 1) // 2
    This is the specific usage that triggered the need for IntDiv support.
    """
    kernel_size = pt.lscalar("kernel_size")
    padding = (kernel_size - 1) // 2

    kernel_size_val = np.array(5, dtype="int64")
    # Expected: (5 - 1) // 2 = 2

    compare_onnx_and_py([kernel_size], padding, [kernel_size_val], tmp_path=tmp_path)


def test_floor_div_negative_integers(tmp_path):
    """Test that floor division (not truncation) is used for negative integers.

    Floor division:    -7 // 2 = -4 (toward -∞)
    Truncation:        -7 / 2 = -3 (toward 0)  ← wrong!

    This test locks in the floor division behavior.
    """
    x = pt.lvector("x")
    y = pt.lvector("y")
    z = x // y

    # Test cases that distinguish floor from truncation
    x_val = np.array([-7, -6, 7], dtype="int64")
    y_val = np.array([2, 2, -2], dtype="int64")
    # Floor division: [-4, -3, -4]
    # Truncation:     [-3, -3, -3] ← wrong

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_floor_div_scalar_tensors(tmp_path):
    """Test floor division on 0-dimensional (scalar) tensors.

    YOLO uses IntDiv on scalar tensors:
    - Pattern: Int_div(Sub.0, 2)
    - Input types: TensorType(int64, shape=())
    """
    x = pt.lscalar("x")
    y = pt.lscalar("y")
    z = x // y

    x_val = np.array(9, dtype="int64")
    y_val = np.array(2, dtype="int64")
    # Expected: 4

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


# ============================================================================
# Switch (Conditional Selection) Operation Tests
# ============================================================================


def test_switch_all_true_condition(tmp_path):
    """Test Switch operation with all-true condition.

    Switch(True, x, y) should return x.

    This test verifies:
    - Switch recognizes True condition
    - Then-branch (x) is selected
    - Else-branch (y) is ignored
    - ONNX Where matches PyTensor Switch
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, True, True], dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([9.0, 9.0, 9.0], dtype="float32")
    # Expected: [1.0, 2.0, 3.0] (x values)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_all_false_condition(tmp_path):
    """Test Switch operation with all-false condition.

    Switch(False, x, y) should return y.

    This test verifies:
    - Switch recognizes False condition
    - Else-branch (y) is selected
    - Then-branch (x) is ignored
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([False, False, False], dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([9.0, 8.0, 7.0], dtype="float32")
    # Expected: [9.0, 8.0, 7.0] (y values)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_mixed_condition(tmp_path):
    """Test Switch operation with mixed True/False condition.

    This is the most common case - element-wise conditional selection.

    Condition: [True, False, True, False]
    X:         [1,    2,     3,    4]
    Y:         [10,   20,    30,   40]
    Output:    [1,    20,    3,    40]
                ↑ from X     ↑ from X
                      ↑ from Y     ↑ from Y
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, False, True, False], dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    y_val = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    # Expected: [1.0, 20.0, 3.0, 40.0]

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_integers(tmp_path):
    """Test Switch operation on integer tensors.

    Integer selection is used in YOLO for dimension calculations.
    This verifies Switch works with int32/int64 dtypes.
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.lvector("x")  # int64
    y = pt.lvector("y")
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, False, True], dtype="bool")
    x_val = np.array([10, 20, 30], dtype="int64")
    y_val = np.array([100, 200, 300], dtype="int64")
    # Expected: [10, 200, 30]

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_scalar_condition(tmp_path):
    """Test Switch with scalar condition (broadcasts to all elements).

    If condition is scalar True, all elements come from x.
    If condition is scalar False, all elements come from y.
    """
    condition = pt.scalar("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    # Test with True condition
    condition_val = np.array(True, dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([9.0, 9.0, 9.0], dtype="float32")
    # Expected: [1.0, 2.0, 3.0] (all from x)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_scalar_tensors(tmp_path):
    """Test Switch on 0-dimensional (scalar) tensors.

    YOLO uses Switch on scalar tensors:
    - Pattern: Switch(Eq.0, Int_div.0, Sub.0)
    - Input types: TensorType(int64, shape=())

    This is critical for YOLO's dynamic dimension logic.
    """
    condition = pt.scalar("condition", dtype="bool")
    x = pt.lscalar("x")
    y = pt.lscalar("y")
    z = pt.switch(condition, x, y)

    # Test true condition
    condition_val = np.array(True, dtype="bool")
    x_val = np.array(10, dtype="int64")
    y_val = np.array(20, dtype="int64")
    # Expected: 10 (from x)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64", "int32", "int64"],
)
def test_switch_multiple_dtypes(tmp_path, dtype):
    """Test Switch operation with all supported dtypes.

    ONNX Where supports: float32, float64, int32, int64, bool
    This test verifies all numeric dtypes work correctly.
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, False, True, False], dtype="bool")
    rng = np.random.default_rng(42)
    if dtype.startswith("float"):
        x_val = rng.random(4).astype(dtype) * 10
        y_val = rng.random(4).astype(dtype) * 100
    else:
        x_val = rng.integers(0, 10, size=4).astype(dtype)
        y_val = rng.integers(100, 200, size=4).astype(dtype)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


@pytest.mark.parametrize(
    "shape",
    [
        (5,),  # vector
        (3, 4),  # matrix
        (2, 3, 4),  # 3D tensor
        (2, 3, 4, 5),  # 4D tensor (CNN feature maps)
    ],
)
def test_switch_different_shapes(tmp_path, shape):
    """Test Switch operation with different tensor shapes.

    Switch should work for any tensor shape (element-wise operation).
    """
    condition = pt.tensor("condition", dtype="bool", shape=shape)
    x = pt.tensor("x", dtype="float32", shape=shape)
    y = pt.tensor("y", dtype="float32", shape=shape)
    z = pt.switch(condition, x, y)

    rng = np.random.default_rng(42)
    condition_val = rng.random(shape) > 0.5  # Random True/False
    x_val = rng.random(shape).astype("float32")
    y_val = rng.random(shape).astype("float32") + 10.0  # Different values

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_broadcast_condition_scalar(tmp_path):
    """Test Switch with scalar condition broadcasting.

    Broadcasting: condition (scalar) broadcasts to x and y (vectors).
    Result shape: same as x and y.
    """
    condition = pt.scalar("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    # True condition → all from x
    condition_val = np.array(True, dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    y_val = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    # Expected: [1.0, 2.0, 3.0, 4.0]

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


@pytest.mark.skip(
    reason="PyTensor has strict broadcasting rules that don't allow this pattern"
)
def test_switch_broadcast_dimensions(tmp_path):
    """Test Switch with three-way dimension broadcasting.

    Broadcasting example:
    - condition: shape (3, 4)
    - x: shape (3, 4)
    - y: shape (1, 4)
    - Output: shape (3, 4)

    Note: PyTensor has strict broadcasting rules that prevent (1, 4) from
    broadcasting with (3, 4) in some contexts. ONNX supports this pattern,
    but we can't test it through PyTensor.
    """
    condition = pt.matrix("condition", dtype="bool")
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
        ],
        dtype="bool",
    )  # (3, 4)
    x_val = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype="float32",
    )  # (3, 4)
    y_val = np.array([[10.0, 20.0, 30.0, 40.0]], dtype="float32")  # (1, 4)

    # Broadcasts to (3, 4)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_broadcast_complex(tmp_path):
    """Test Switch with complex three-way broadcasting.

    This tests the full generality of ONNX Where broadcasting.
    """
    condition = pt.tensor("condition", dtype="bool", shape=(2, 1, 4))
    x = pt.tensor("x", dtype="float32", shape=(1, 3, 1))
    y = pt.tensor("y", dtype="float32", shape=(2, 3, 4))
    z = pt.switch(condition, x, y)

    rng = np.random.default_rng(42)
    condition_val = rng.random((2, 1, 4)) > 0.5
    x_val = rng.random((1, 3, 1)).astype("float32")
    y_val = rng.random((2, 3, 4)).astype("float32") + 10.0

    # Broadcasts to (2, 3, 4)

    compare_onnx_and_py(
        [condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path
    )


def test_switch_onnx_structure(tmp_path):
    """Test that Switch generates correct ONNX Where node.

    Verifies:
    - One Where node created
    - Node has three inputs (condition, X, Y)
    - Node has one output
    - No unnecessary nodes
    """
    from pytensor.link.onnx import export_onnx

    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    f = pytensor.function([condition, x, y], z)

    model_path = tmp_path / "test_switch.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    validate_onnx_graph_structure(
        model,
        expected_node_types=["Where"],
        expected_node_count=1,
    )

    # Check Where node structure
    where_node = model.graph.node[0]
    assert where_node.op_type == "Where"
    assert len(where_node.input) == 3  # condition, X, Y
    assert len(where_node.output) == 1


def test_switch_yolo_dimension_calculation_pattern(tmp_path):
    """Test Switch in YOLO dimension calculation pattern.

    YOLO pattern (from model.py:249-286):
    - Compare dimensions with EQ: is_match = Eq(dim1, dim2)
    - Conditionally select dimension: Switch(is_match, calc1, calc2)

    This simulates: if dims match, use one value, else use another.
    Pattern: Switch(Eq.0, value1, value2)
    """
    # Inputs: dimension values
    dim1 = pt.lscalar("dim1")
    dim2 = pt.lscalar("dim2")
    alt_value = pt.lscalar("alt_value")

    # Compare dimensions
    is_match = pt.eq(dim1, dim2)

    # Switch based on condition - use dim1 if match, alt_value otherwise
    result_dim = pt.switch(is_match, dim1, alt_value)

    # Test case 1: dims match (20 == 20)
    dim1_val = np.array(20, dtype="int64")
    dim2_val = np.array(20, dtype="int64")
    alt_value_val = np.array(5, dtype="int64")
    # is_match = True → use dim1: 20

    compare_onnx_and_py(
        [dim1, dim2, alt_value],
        result_dim,
        [dim1_val, dim2_val, alt_value_val],
        tmp_path=tmp_path,
    )


def test_switch_yolo_negative_one_pattern(tmp_path):
    """Test Switch for -1 (dynamic dimension) handling.

    YOLO uses -1 to indicate dynamic dimensions:
    - Check if dimension is dynamic: is_dynamic = Eq(dim, -1)
    - Use different calculation: Switch(is_dynamic, dynamic_calc, static_calc)

    Pattern: Switch(Eq(dim, -1), calc_dynamic, calc_static)
    """
    dim = pt.lscalar("dim")
    neg_one = pt.lscalar("neg_one")

    # Check if dynamic
    is_dynamic = pt.eq(dim, neg_one)

    # Conditional calculations
    # If dynamic (-1), use special value (e.g., 0 or another marker)
    # If static, use the actual dimension
    dynamic_value = pt.lscalar("dynamic_value")
    static_value = dim

    result = pt.switch(is_dynamic, dynamic_value, static_value)

    # Test case: static dimension (not -1)
    dim_val = np.array(10, dtype="int64")
    neg_one_val = np.array(-1, dtype="int64")
    dynamic_value_val = np.array(0, dtype="int64")
    # is_dynamic = False → use static_value: 10

    compare_onnx_and_py(
        [dim, neg_one, dynamic_value],
        result,
        [dim_val, neg_one_val, dynamic_value_val],
        tmp_path=tmp_path,
    )
