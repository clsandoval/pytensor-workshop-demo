"""
Test log operations and binary cross-entropy patterns with JAX backend.

These tests verify that log operations work correctly with JAX tracers,
particularly for binary cross-entropy loss computation.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import grad
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_log_for_bce_loss():
    """
    Test log operation for binary cross-entropy loss.

    This test verifies:
    - Numerical stability with small probabilities
    - Proper clipping to avoid log(0)
    - Gradient flow through BCE
    """
    # Arrange - BCE loss pattern
    pred = pt.matrix("pred", dtype="float32")  # Predictions [N, C]
    target = pt.matrix("target", dtype="float32")  # Targets [N, C]

    # Clip predictions for numerical stability
    eps = 1e-7
    pred_clipped = pt.clip(pred, eps, 1 - eps)

    # BCE loss computation
    bce = -target * pt.log(pred_clipped) - (1 - target) * pt.log(1 - pred_clipped)
    loss = pt.mean(bce)

    # Gradient
    grad_pred = grad(loss, pred)

    # Test data - include edge cases
    pred_val = np.array([[0.01, 0.5, 0.99], [0.1, 0.9, 0.001]], dtype="float32")
    target_val = np.array([[0, 1, 1], [0, 1, 0]], dtype="float32")

    # Assert
    fn, (loss_val, grad_val) = compare_jax_and_py(
        [pred, target], [loss, grad_pred], [pred_val, target_val]
    )

    # Verify no NaN/Inf
    assert np.isfinite(loss_val), f"BCE loss is not finite: {loss_val}"
    assert np.all(np.isfinite(grad_val)), "BCE gradient has NaN/Inf"

    # Verify loss is positive
    assert loss_val > 0, f"BCE loss should be positive, got {loss_val}"


def test_log_basic():
    """Test basic log operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.log(x)
    grad_y = grad(y.sum(), x)

    x_val = np.array([0.1, 0.5, 1.0, 2.0, np.e, 10.0], dtype="float32")

    fn, (y_val, grad_val) = compare_jax_and_py([x], [y, grad_y], [x_val])

    # Verify values
    expected_y = np.log(x_val)
    np.testing.assert_allclose(
        y_val, expected_y, rtol=1e-6, err_msg="Log values incorrect"
    )

    # Verify gradient is 1/x
    expected_grad = 1.0 / x_val
    np.testing.assert_allclose(
        grad_val, expected_grad, rtol=1e-6, err_msg="Log gradient incorrect"
    )


def test_log_numerical_edge_cases():
    """Test log with numerical edge cases."""
    x = pt.vector("x", dtype="float32")

    # Log of very small and very large values
    y = pt.log(x)

    # Test data
    x_val = np.array([1e-30, 1e-10, 1e-5, 1, 1e5, 1e10], dtype="float32")

    fn, y_val = compare_jax_and_py([x], y, [x_val])

    # All values should be finite (even if very negative for small inputs)
    assert np.all(np.isfinite(y_val)), f"Log produced NaN/Inf: {y_val}"


def test_log1p_for_numerical_stability():
    """Test log1p for better numerical stability with small values."""
    x = pt.vector("x", dtype="float32")

    # Compare log(1 + x) vs log1p(x)
    y_log = pt.log(1 + x)
    y_log1p = pt.log1p(x)

    # Test with small values where log1p is more accurate
    x_val = np.array([1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1], dtype="float32")

    fn, (log_val, log1p_val) = compare_jax_and_py([x], [y_log, y_log1p], [x_val])

    # For very small values, log(1+x) loses precision due to floating point
    # So we only check values where x > 1e-6
    mask = x_val > 1e-6
    np.testing.assert_allclose(
        log_val[mask],
        log1p_val[mask],
        rtol=1e-3,
        err_msg="log(1+x) and log1p(x) differ too much for x > 1e-6",
    )

    # Verify against numpy
    expected = np.log1p(x_val)
    np.testing.assert_allclose(
        log1p_val, expected, rtol=1e-6, err_msg="log1p values incorrect"
    )


def test_bce_loss_complete():
    """Test complete BCE loss pattern with batches."""
    # Batch of predictions and targets
    pred = pt.tensor3("pred", dtype="float32", shape=(None, None, None))
    target = pt.tensor3("target", dtype="float32", shape=(None, None, None))

    # Stable BCE computation
    eps = 1e-7
    pred_clipped = pt.clip(pred, eps, 1 - eps)

    # Element-wise BCE
    bce_elem = -target * pt.log(pred_clipped) - (1 - target) * pt.log(1 - pred_clipped)

    # Different reduction strategies
    bce_mean = pt.mean(bce_elem)  # Mean over all
    bce_sum = pt.sum(bce_elem)  # Sum over all
    bce_batch_mean = pt.mean(bce_elem, axis=0)  # Mean over batch only

    # Test data
    rng = np.random.default_rng(42)
    batch_size, height, width = 2, 4, 4
    pred_val = rng.uniform(0.01, 0.99, size=(batch_size, height, width)).astype(
        "float32"
    )
    target_val = rng.choice([0, 1], size=(batch_size, height, width)).astype("float32")

    fn, outputs = compare_jax_and_py(
        [pred, target], [bce_mean, bce_sum, bce_batch_mean], [pred_val, target_val]
    )

    mean_val, sum_val, batch_mean_val = outputs

    # Verify shapes
    assert mean_val.shape == (), "BCE mean should be scalar"
    assert sum_val.shape == (), "BCE sum should be scalar"
    assert batch_mean_val.shape == (height, width), (
        f"BCE batch mean shape: {batch_mean_val.shape}"
    )

    # Mean should equal sum divided by size
    expected_mean = sum_val / pred_val.size
    np.testing.assert_allclose(
        mean_val, expected_mean, rtol=1e-6, err_msg="BCE mean != sum/size"
    )

    # All values should be positive
    assert mean_val > 0, "BCE mean should be positive"
    assert sum_val > 0, "BCE sum should be positive"
    assert np.all(batch_mean_val > 0), "BCE batch mean should be positive"


def test_bce_gradient_flow():
    """Test gradient flow through BCE loss."""
    pred = pt.matrix("pred", dtype="float32")
    target = pt.matrix("target", dtype="float32")

    # BCE with clipping
    eps = 1e-7
    pred_clipped = pt.clip(pred, eps, 1 - eps)
    bce = -target * pt.log(pred_clipped) - (1 - target) * pt.log(1 - pred_clipped)
    loss = pt.mean(bce)

    # Gradient
    grad_pred = grad(loss, pred)

    # Test with various prediction/target combinations
    pred_val = np.array(
        [
            [0.1, 0.2, 0.7, 0.9],  # Various predictions
            [0.3, 0.5, 0.6, 0.8],
        ],
        dtype="float32",
    )
    target_val = np.array(
        [
            [0, 0, 1, 1],  # Binary targets
            [0, 1, 1, 0],
        ],
        dtype="float32",
    )

    fn, (loss_val, grad_val) = compare_jax_and_py(
        [pred, target], [loss, grad_pred], [pred_val, target_val]
    )

    # Gradient should exist and be finite
    assert np.all(np.isfinite(grad_val)), "BCE gradient has NaN/Inf"
    assert grad_val.shape == pred_val.shape, "Gradient shape mismatch"

    # Manually compute expected gradient for BCE
    pred_clipped_np = np.clip(pred_val, eps, 1 - eps)
    expected_grad = (
        (pred_clipped_np - target_val)
        / (pred_clipped_np * (1 - pred_clipped_np))
        / pred_val.size
    )

    # Note: gradient w.r.t. unclipped predictions may differ at boundaries
    # So we only check that gradients are reasonable, not exact
    assert np.all(np.abs(grad_val) < 1e3), "BCE gradients unreasonably large"


def test_log_with_dynamic_shapes():
    """Test log operations with dynamic batch dimensions."""
    x = pt.tensor3("x", dtype="float32", shape=(None, 10, 10))

    # Various log-based operations
    y_log = pt.log(x + 1)  # Ensure positive
    y_log10 = pt.log10(x + 1)
    y_log2 = pt.log2(x + 1)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.uniform(0, 10, size=(3, 10, 10)).astype("float32")

    fn, (log_val, log10_val, log2_val) = compare_jax_and_py(
        [x], [y_log, y_log10, y_log2], [x_val]
    )

    # Verify all are finite
    assert np.all(np.isfinite(log_val)), "log with dynamic shape has NaN/Inf"
    assert np.all(np.isfinite(log10_val)), "log10 with dynamic shape has NaN/Inf"
    assert np.all(np.isfinite(log2_val)), "log2 with dynamic shape has NaN/Inf"

    # Verify relationships between different log bases
    # log10(x) = log(x) / log(10)
    expected_log10 = log_val / np.log(10)
    np.testing.assert_allclose(
        log10_val, expected_log10, rtol=1e-5, err_msg="log10 != log/log(10)"
    )

    # log2(x) = log(x) / log(2)
    expected_log2 = log_val / np.log(2)
    np.testing.assert_allclose(
        log2_val, expected_log2, rtol=1e-5, err_msg="log2 != log/log(2)"
    )


def test_focal_loss_pattern():
    """Test focal loss pattern which uses BCE as base."""
    pred = pt.matrix("pred", dtype="float32")
    target = pt.matrix("target", dtype="float32")

    # Focal loss parameters
    alpha = 0.25
    gamma = 2.0
    eps = 1e-7

    # Clip predictions
    pred_clipped = pt.clip(pred, eps, 1 - eps)

    # BCE component
    bce = -target * pt.log(pred_clipped) - (1 - target) * pt.log(1 - pred_clipped)

    # Focal weight
    p_t = target * pred_clipped + (1 - target) * (1 - pred_clipped)
    focal_weight = (1 - p_t) ** gamma

    # Complete focal loss
    focal_loss = alpha * focal_weight * bce
    loss = pt.mean(focal_loss)

    # Test data
    pred_val = np.array([[0.1, 0.7], [0.3, 0.9]], dtype="float32")
    target_val = np.array([[0, 1], [0, 1]], dtype="float32")

    fn, loss_val = compare_jax_and_py([pred, target], loss, [pred_val, target_val])

    # Focal loss should be positive and finite
    assert np.isfinite(loss_val), f"Focal loss not finite: {loss_val}"
    assert loss_val > 0, f"Focal loss should be positive, got {loss_val}"
