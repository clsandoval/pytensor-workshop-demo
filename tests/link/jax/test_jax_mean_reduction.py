"""
Test mean reduction operations with JAX backend.

These tests verify that mean reduction works correctly with JAX tracers,
particularly for loss aggregation patterns in neural networks.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import grad
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_mean_reduction_with_axes():
    """
    Test mean reduction over specific axes with dynamic shapes.

    This test verifies:
    - Reduction over dynamic batch dimension
    - Keepdims parameter handling
    - Multiple axis reduction
    """
    # Arrange - dynamic batch
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))

    # Act - various mean reductions
    mean_batch = pt.mean(x, axis=0)  # Average over batch
    mean_spatial = pt.mean(x, axis=(2, 3), keepdims=True)  # Spatial average
    mean_all = pt.mean(x)  # Global average

    # Gradients
    loss = mean_all
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(3, 64, 32, 32)).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x], [mean_batch, mean_spatial, mean_all, grad_x], [x_val]
    )

    mean_batch_val, mean_spatial_val, mean_all_val, grad_val = outputs

    # Verify shapes
    assert mean_batch_val.shape == (64, 32, 32), (
        f"Batch mean shape {mean_batch_val.shape} != (64, 32, 32)"
    )
    assert mean_spatial_val.shape == (3, 64, 1, 1), (
        f"Spatial mean shape {mean_spatial_val.shape} != (3, 64, 1, 1)"
    )
    assert mean_all_val.shape == (), (
        f"Global mean should be scalar, got shape {mean_all_val.shape}"
    )

    # Verify gradient
    expected_grad = np.ones_like(x_val) / x_val.size
    np.testing.assert_allclose(
        grad_val, expected_grad, rtol=1e-6, err_msg="Mean gradient incorrect"
    )


def test_mean_for_loss_aggregation():
    """
    Test mean operation for loss aggregation patterns.

    This test verifies:
    - Mean over batch dimension
    - Mean over spatial dimensions
    - Weighted mean computation
    """
    # Arrange - typical loss tensors
    per_pixel_loss = pt.tensor4(
        "per_pixel_loss", dtype="float32", shape=(None, 1, 32, 32)
    )
    weights = pt.tensor4("weights", dtype="float32", shape=(None, 1, 32, 32))

    # Act - different aggregation patterns
    # Pattern 1: Simple mean over all dimensions
    loss1 = pt.mean(per_pixel_loss)

    # Pattern 2: Mean over spatial, then batch
    loss2 = pt.mean(pt.mean(per_pixel_loss, axis=(2, 3)))

    # Pattern 3: Weighted mean
    weighted_loss = per_pixel_loss * weights
    loss3 = pt.sum(weighted_loss) / pt.sum(weights)

    # Test data
    rng = np.random.default_rng(42)
    loss_val = rng.uniform(0, 1, size=(2, 1, 32, 32)).astype("float32")
    weights_val = rng.uniform(0.5, 1.5, size=(2, 1, 32, 32)).astype("float32")

    # Assert
    fn, (l1, l2, l3) = compare_jax_and_py(
        [per_pixel_loss, weights], [loss1, loss2, loss3], [loss_val, weights_val]
    )

    # All aggregations should produce scalars
    assert l1.shape == (), "Loss1 should be scalar"
    assert l2.shape == (), "Loss2 should be scalar"
    assert l3.shape == (), "Loss3 should be scalar"

    # Pattern 1 and 2 should be equal
    np.testing.assert_almost_equal(
        l1, l2, decimal=6, err_msg="Different mean patterns gave different results"
    )


def test_mean_single_axis():
    """Test mean reduction over a single axis."""
    x = pt.tensor3("x", dtype="float32", shape=(None, 10, 20))

    # Mean over different axes
    mean_axis0 = pt.mean(x, axis=0)
    mean_axis1 = pt.mean(x, axis=1)
    mean_axis2 = pt.mean(x, axis=2)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(5, 10, 20)).astype("float32")

    fn, (m0, m1, m2) = compare_jax_and_py(
        [x], [mean_axis0, mean_axis1, mean_axis2], [x_val]
    )

    # Check shapes
    assert m0.shape == (10, 20), f"Mean over axis 0 shape: {m0.shape}"
    assert m1.shape == (5, 20), f"Mean over axis 1 shape: {m1.shape}"
    assert m2.shape == (5, 10), f"Mean over axis 2 shape: {m2.shape}"

    # Verify against numpy
    np.testing.assert_allclose(m0, np.mean(x_val, axis=0), rtol=1e-6)
    np.testing.assert_allclose(m1, np.mean(x_val, axis=1), rtol=1e-6)
    np.testing.assert_allclose(m2, np.mean(x_val, axis=2), rtol=1e-6)


def test_mean_keepdims():
    """Test mean reduction with keepdims parameter."""
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 32, 32))

    # Mean with keepdims=True
    mean_keepdims = pt.mean(x, axis=(2, 3), keepdims=True)
    # Mean with keepdims=False (default)
    mean_no_keepdims = pt.mean(x, axis=(2, 3), keepdims=False)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 32, 32)).astype("float32")

    fn, (mk, mnk) = compare_jax_and_py([x], [mean_keepdims, mean_no_keepdims], [x_val])

    # Check shapes
    assert mk.shape == (2, 3, 1, 1), f"Mean with keepdims shape: {mk.shape}"
    assert mnk.shape == (2, 3), f"Mean without keepdims shape: {mnk.shape}"

    # Values should be the same after squeezing
    np.testing.assert_allclose(mk.squeeze(), mnk, rtol=1e-6)


def test_mean_gradient_flow():
    """Test gradient flow through mean operations."""
    x = pt.matrix("x", dtype="float32")

    # Create a loss that involves mean
    y = x - pt.mean(x, axis=1, keepdims=True)
    loss = pt.mean(y**2)
    grad_x = grad(loss, x)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 5)).astype("float32")

    fn, (loss_val, grad_val) = compare_jax_and_py([x], [loss, grad_x], [x_val])

    # Verify gradient exists and is finite
    assert np.all(np.isfinite(grad_val)), "Gradient through mean has NaN/Inf"
    assert grad_val.shape == x_val.shape, "Gradient shape mismatch"

    # Loss should be positive (variance-like)
    assert loss_val > 0, f"Loss should be positive, got {loss_val}"


def test_mean_empty_axes():
    """Test mean with empty axes (should compute mean over all axes)."""
    x = pt.tensor3("x", dtype="float32", shape=(None, 4, 5))

    # Mean with axis=None should reduce everything
    mean_all = pt.mean(x)
    # Mean with empty tuple should also reduce everything
    mean_empty = pt.mean(x, axis=())

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 4, 5)).astype("float32")

    fn, (ma, me) = compare_jax_and_py([x], [mean_all, mean_empty], [x_val])

    # Both should be scalars
    assert ma.shape == (), f"Mean all shape: {ma.shape}"
    assert me.shape == x_val.shape, (
        f"Mean with empty axes should not reduce: {me.shape}"
    )


def test_mean_negative_axes():
    """Test mean with negative axis indices."""
    x = pt.tensor3("x", dtype="float32", shape=(None, 10, 20))

    # Negative indices
    mean_neg1 = pt.mean(x, axis=-1)  # Last axis
    mean_neg2 = pt.mean(x, axis=-2)  # Second to last

    # Equivalent positive indices
    mean_pos1 = pt.mean(x, axis=2)
    mean_pos2 = pt.mean(x, axis=1)

    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(3, 10, 20)).astype("float32")

    fn, outputs = compare_jax_and_py(
        [x], [mean_neg1, mean_neg2, mean_pos1, mean_pos2], [x_val]
    )

    mn1, mn2, mp1, mp2 = outputs

    # Negative and positive indices should give same results
    np.testing.assert_allclose(
        mn1, mp1, rtol=1e-6, err_msg="Negative axis -1 != positive axis 2"
    )
    np.testing.assert_allclose(
        mn2, mp2, rtol=1e-6, err_msg="Negative axis -2 != positive axis 1"
    )


def test_mean_scalar():
    """Test mean of a scalar (should return the scalar itself)."""
    x = pt.scalar("x", dtype="float32")
    mean_x = pt.mean(x)

    x_val = np.float32(3.14)

    fn, mean_val = compare_jax_and_py([x], mean_x, [x_val])

    # Mean of scalar should be the scalar itself
    np.testing.assert_almost_equal(
        mean_val, x_val, err_msg="Mean of scalar should return the scalar"
    )


def test_mean_large_reduction():
    """Test mean with large tensors to check numerical stability."""
    x = pt.tensor4("x", dtype="float32", shape=(None, 128, 64, 64))

    # Mean over large dimensions
    mean_val = pt.mean(x)
    grad_x = grad(mean_val, x)

    # Use smaller values to avoid numerical issues
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 128, 64, 64)).astype("float32") * 0.1

    fn, (m, g) = compare_jax_and_py([x], [mean_val, grad_x], [x_val])

    # Verify results are finite
    assert np.isfinite(m), f"Mean of large tensor not finite: {m}"
    assert np.all(np.isfinite(g)), "Gradient through mean of large tensor has NaN/Inf"

    # Gradient should be uniform
    expected_grad = np.ones_like(x_val) / x_val.size
    np.testing.assert_allclose(
        g, expected_grad, rtol=1e-5, err_msg="Large tensor mean gradient incorrect"
    )
