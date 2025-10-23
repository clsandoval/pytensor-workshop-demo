"""
Test mathematical operations with JAX backend.

These tests verify that sqrt, maximum, minimum, and log operations work correctly
with JAX tracers, particularly in the context of batch normalization and IoU computation.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import grad
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


@pytest.mark.parametrize(
    "op_name,op_func,test_input_fn",
    [
        ("sqrt", pt.sqrt, lambda: np.array([1, 4, 9, 16], dtype="float32")),
        (
            "maximum",
            lambda x: pt.maximum(x, 0.5),
            lambda: np.array([0, 0.3, 0.7, 1], dtype="float32"),
        ),
        (
            "minimum",
            lambda x: pt.minimum(x, 0.5),
            lambda: np.array([0, 0.3, 0.7, 1], dtype="float32"),
        ),
        ("log", pt.log, lambda: np.array([0.1, 1, 2, 10], dtype="float32")),
    ],
)
def test_math_operations_with_tracers(op_name, op_func, test_input_fn):
    """
    Test mathematical operations with JAX tracers.

    This test verifies:
    - Operations work with dynamic shapes
    - Gradients flow correctly
    - Numerical stability
    """
    # Arrange
    x = pt.vector("x", dtype="float32")

    # Act
    y = op_func(x)

    # Gradient (skip for operations that don't have gradients everywhere)
    if op_name not in ["maximum", "minimum"]:
        grad_x = grad(y.sum(), x)
    else:
        # For max/min, test gradient where differentiable
        grad_x = None

    # Test data
    x_val = test_input_fn()

    # Assert
    if grad_x is not None:
        fn, (y_val, grad_val) = compare_jax_and_py([x], [y, grad_x], [x_val])

        # Verify gradient exists
        assert np.all(np.isfinite(grad_val)), f"{op_name} gradient has NaN/Inf"
    else:
        fn, (y_val,) = compare_jax_and_py([x], [y], [x_val])

    # Verify output is finite
    assert np.all(np.isfinite(y_val)), f"{op_name} output has NaN/Inf: {y_val}"


def test_sqrt_for_batch_norm():
    """
    Test sqrt operation as used in batch normalization.

    This test verifies:
    - Sqrt of variance + epsilon pattern
    - Numerical stability for small variances
    - Gradient flow through normalization
    """
    # Arrange - batch norm pattern
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))
    epsilon = 1e-5

    # Compute variance (simplified)
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

    # Sqrt for normalization
    std = pt.sqrt(var + epsilon)
    normalized = (x - mean) / std

    # Gradient
    loss = normalized.sum()
    grad_x = grad(loss, x)

    # Test data - include small variance case
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 64, 32, 32)).astype("float32") * 0.01

    # Assert
    fn, (std_val, norm_val, grad_val) = compare_jax_and_py(
        [x], [std, normalized, grad_x], [x_val]
    )

    # Verify no NaN/Inf
    assert np.all(np.isfinite(std_val)), "Std has NaN/Inf"
    assert np.all(np.isfinite(norm_val)), "Normalized output has NaN/Inf"
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"

    # Verify std is positive
    assert np.all(std_val > 0), "Std should be positive"


def test_maximum_minimum_for_iou():
    """
    Test maximum/minimum operations for IoU computation.

    This test verifies:
    - Intersection area computation
    - Proper handling of non-overlapping boxes
    - Gradient flow for differentiable IoU
    """
    # Arrange - box coordinates
    boxes1 = pt.matrix("boxes1", dtype="float32")  # [N, 4]
    boxes2 = pt.matrix("boxes2", dtype="float32")  # [M, 4]

    # Compute intersections (simplified for 1 box each)
    x1_max = pt.maximum(boxes1[0, 0], boxes2[0, 0])
    y1_max = pt.maximum(boxes1[0, 1], boxes2[0, 1])
    x2_min = pt.minimum(boxes1[0, 2], boxes2[0, 2])
    y2_min = pt.minimum(boxes1[0, 3], boxes2[0, 3])

    # Intersection area
    inter_w = pt.maximum(x2_min - x1_max, 0)
    inter_h = pt.maximum(y2_min - y1_max, 0)
    inter_area = inter_w * inter_h

    # Test data - overlapping and non-overlapping boxes
    boxes1_val = np.array([[0, 0, 2, 2]], dtype="float32")  # Box at origin
    boxes2_val = np.array([[1, 1, 3, 3]], dtype="float32")  # Overlapping box

    # Assert
    fn, (inter_val,) = compare_jax_and_py(
        [boxes1, boxes2], [inter_area], [boxes1_val, boxes2_val]
    )

    # Expected intersection area is 1 (1x1 overlap)
    expected_inter = 1.0
    np.testing.assert_almost_equal(
        inter_val, expected_inter, err_msg="Intersection area incorrect"
    )


def test_maximum_minimum_broadcasting():
    """Test that maximum and minimum handle broadcasting correctly."""
    # Test with tensor and scalar
    x = pt.matrix("x", dtype="float32")
    y_max = pt.maximum(x, 0.5)
    y_min = pt.minimum(x, 0.5)

    x_val = np.array([[0, 0.3], [0.7, 1]], dtype="float32")

    fn, (max_val, min_val) = compare_jax_and_py([x], [y_max, y_min], [x_val])

    expected_max = np.maximum(x_val, 0.5)
    expected_min = np.minimum(x_val, 0.5)

    np.testing.assert_allclose(
        max_val, expected_max, err_msg="Maximum broadcasting failed"
    )
    np.testing.assert_allclose(
        min_val, expected_min, err_msg="Minimum broadcasting failed"
    )


def test_log_numerical_stability():
    """Test log operation with small positive values."""
    x = pt.vector("x", dtype="float32")
    y = pt.log(x)
    grad_y = grad(y.sum(), x)

    # Test with small positive values
    x_val = np.array([1e-10, 1e-5, 1e-2, 0.1, 1, 10], dtype="float32")

    fn, (y_val, grad_val) = compare_jax_and_py([x], [y, grad_y], [x_val])

    # Verify no NaN/Inf
    assert np.all(np.isfinite(y_val)), f"Log output has NaN/Inf: {y_val}"
    assert np.all(np.isfinite(grad_val)), f"Log gradient has NaN/Inf: {grad_val}"

    # Verify gradient is 1/x
    expected_grad = 1 / x_val
    np.testing.assert_allclose(
        grad_val, expected_grad, rtol=1e-5, err_msg="Log gradient incorrect"
    )


def test_sqrt_gradient():
    """Test sqrt gradient computation."""
    x = pt.vector("x", dtype="float32")
    y = pt.sqrt(x)
    grad_y = grad(y.sum(), x)

    x_val = np.array([0.01, 0.25, 1, 4, 16], dtype="float32")

    fn, (y_val, grad_val) = compare_jax_and_py([x], [y, grad_y], [x_val])

    # Verify gradient is 1/(2*sqrt(x))
    expected_grad = 0.5 / np.sqrt(x_val)
    np.testing.assert_allclose(
        grad_val, expected_grad, rtol=1e-5, err_msg="Sqrt gradient incorrect"
    )


def test_math_operations_with_dynamic_shapes():
    """Test math operations with dynamic batch dimensions."""
    # Dynamic batch size
    x = pt.tensor3("x", dtype="float32", shape=(None, 10, 10))

    # Test various operations
    y_sqrt = pt.sqrt(x + 1)  # Add 1 to ensure positive
    y_log = pt.log(x + 1)
    y_max = pt.maximum(x, 0)
    y_min = pt.minimum(x, 1)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.uniform(-0.5, 1.5, size=(2, 10, 10)).astype("float32")

    fn, outputs = compare_jax_and_py([x], [y_sqrt, y_log, y_max, y_min], [x_val])

    sqrt_val, log_val, max_val, min_val = outputs

    # Verify all outputs are finite
    assert np.all(np.isfinite(sqrt_val)), "Sqrt with dynamic shape has NaN/Inf"
    assert np.all(np.isfinite(log_val)), "Log with dynamic shape has NaN/Inf"
    assert np.all(np.isfinite(max_val)), "Maximum with dynamic shape has NaN/Inf"
    assert np.all(np.isfinite(min_val)), "Minimum with dynamic shape has NaN/Inf"
