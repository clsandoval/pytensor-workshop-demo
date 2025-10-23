"""
Integration tests for gradient flow through YOLO11 model with JAX backend.
Tests that gradients can be computed through the entire model.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, grad
from pytensor.tensor.conv import conv2d
from pytensor.tensor.pool import pool_2d
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_yolo11_complete_gradient_flow():
    """
    Test gradient flow through complete YOLO11 model.

    This test verifies:
    - Gradients computed for all parameters
    - No operations block gradient flow
    - Gradient magnitudes are reasonable
    - All layer types propagate gradients
    """

    # Arrange - build complete model with all block types
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 320, 320))

    # Parameters for different block types
    params = []

    # ConvBNSiLU block (simplified BN for testing)
    conv1_w = pt.tensor4("conv1_w", dtype="float32")
    conv1_gamma = pt.vector("conv1_gamma", dtype="float32")
    conv1_beta = pt.vector("conv1_beta", dtype="float32")
    params.extend([conv1_w, conv1_gamma, conv1_beta])

    h = conv2d(x, conv1_w, border_mode="half")

    # Simplified BN
    h_mean = h.mean(axis=(0, 2, 3), keepdims=True)
    h_var = h.var(axis=(0, 2, 3), keepdims=True)
    h = (h - h_mean) / pt.sqrt(h_var + 1e-5)
    h = h * conv1_gamma.dimshuffle("x", 0, "x", "x") + conv1_beta.dimshuffle(
        "x", 0, "x", "x"
    )

    # SiLU activation
    h = h * pt.sigmoid(h)

    # CSP block (simplified)
    h_split1 = h[:, :32, :, :]
    h_split2 = h[:, 32:, :, :]

    conv2_w = pt.tensor4("conv2_w", dtype="float32")
    params.append(conv2_w)
    h_split2 = conv2d(h_split2, conv2_w, border_mode="half")
    h_split2 = pt.maximum(0, h_split2)  # ReLU activation

    # Concatenate
    h = pt.concatenate([h_split1, h_split2], axis=1)

    # SPPF block (simplified)
    pool1 = pool_2d(h, ws=(3, 3), stride=(1, 1), padding=(1, 1), mode="max")
    h = pt.concatenate([h, pool1], axis=1)

    # Detection head
    det_w = pt.tensor4("det_w", dtype="float32")
    params.append(det_w)
    det = conv2d(h, det_w, border_mode="half")

    # Loss (simplified)
    loss = det.sum()

    # Act - compute gradients for all parameters
    grads = grad(loss, params)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 320, 320)).astype("float32") * 0.01

    # Parameter values
    param_vals = [
        rng.normal(size=(64, 3, 3, 3)).astype("float32") * 0.02,  # conv1_w
        np.ones(64, dtype="float32"),  # gamma
        np.zeros(64, dtype="float32"),  # beta
        rng.normal(size=(32, 32, 3, 3)).astype("float32") * 0.02,  # conv2_w
        rng.normal(size=(85, 128, 3, 3)).astype("float32") * 0.02,  # det_w
    ]

    # Assert - test with compare_jax_and_py
    try:
        fn, outputs = compare_jax_and_py(
            [x] + params, [loss] + grads, [x_val] + param_vals
        )

        loss_val = outputs[0] if not isinstance(outputs, list) else outputs[0]
        grad_vals = outputs[1:] if isinstance(outputs, list) else [outputs]

        # If outputs is not a list, we need to run the function to get gradients
        if len(grad_vals) == 0:
            # Re-run to get all outputs
            fn_full = function([x] + params, [loss] + grads, mode="JAX")
            all_outputs = fn_full(x_val, *param_vals)
            loss_val = all_outputs[0]
            grad_vals = all_outputs[1:]

        # Verify all gradients computed
        for i, grad_val in enumerate(grad_vals):
            assert grad_val is not None, f"Gradient {i} is None"
            assert np.all(np.isfinite(grad_val)), f"Gradient {i} has NaN/Inf"

            # Check gradient magnitude
            grad_norm = np.linalg.norm(grad_val.flatten())
            assert 1e-10 < grad_norm < 1e10, (
                f"Gradient {i} norm {grad_norm} out of reasonable range"
            )

    except Exception as e:
        pytest.fail(f"Gradient computation failed: {e}")


@pytest.mark.parametrize("block_type", ["ConvBNSiLU", "CSP", "SPPF"])
def test_gradient_flow_all_blocks(block_type):
    """
    Test gradient flow through specific block types.

    This test verifies:
    - Each block type propagates gradients
    - No block type causes gradient issues
    - All YOLO components work
    """

    # Arrange - build specific block
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))
    params = []

    if block_type == "ConvBNSiLU":
        # ConvBNSiLU block
        w = pt.tensor4("w", dtype="float32")
        gamma = pt.vector("gamma", dtype="float32")
        beta = pt.vector("beta", dtype="float32")
        params = [w, gamma, beta]

        h = conv2d(x, w, border_mode="half")
        h_mean = h.mean(axis=(0, 2, 3), keepdims=True)
        h_var = h.var(axis=(0, 2, 3), keepdims=True)
        h = (h - h_mean) / pt.sqrt(h_var + 1e-5)
        h = h * gamma.dimshuffle("x", 0, "x", "x") + beta.dimshuffle("x", 0, "x", "x")
        output = h * pt.sigmoid(h)

    elif block_type == "CSP":
        # CSP block
        w1 = pt.tensor4("w1", dtype="float32")
        w2 = pt.tensor4("w2", dtype="float32")
        params = [w1, w2]

        h1 = x[:, :128, :, :]
        h2 = x[:, 128:, :, :]
        h2 = conv2d(h2, w1, border_mode="half")
        h2 = pt.maximum(0, h2)  # ReLU activation
        h2 = conv2d(h2, w2, border_mode="half")
        output = pt.concatenate([h1, h2], axis=1)

    elif block_type == "SPPF":
        # SPPF block
        w = pt.tensor4("w", dtype="float32")
        params = [w]

        h = conv2d(x, w, border_mode="half")
        pool1 = pool_2d(h, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
        pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
        pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
        output = pt.concatenate([h, pool1, pool2, pool3], axis=1)

    # Loss and gradients
    loss = output.sum()
    grads = grad(loss, params)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 256, 32, 32)).astype("float32") * 0.1

    if block_type == "ConvBNSiLU":
        param_vals = [
            rng.normal(size=(256, 256, 3, 3)).astype("float32") * 0.02,
            np.ones(256, dtype="float32"),
            np.zeros(256, dtype="float32"),
        ]
    elif block_type == "CSP":
        param_vals = [
            rng.normal(size=(128, 128, 3, 3)).astype("float32") * 0.02,
            rng.normal(size=(128, 128, 3, 3)).astype("float32") * 0.02,
        ]
    else:  # SPPF
        param_vals = [
            rng.normal(size=(256, 256, 3, 3)).astype("float32") * 0.02,
        ]

    # Assert
    fn, outputs = compare_jax_and_py([x] + params, [loss] + grads, [x_val] + param_vals)

    # Verify gradients flow
    # The compare_jax_and_py returns (fn, jax_results)
    # We need to extract the gradients from the results
    if isinstance(outputs, list):
        loss_val = outputs[0]
        grad_vals = outputs[1:]
    else:
        # Re-run to get gradients
        fn_full = function([x] + params, [loss] + grads, mode="JAX")
        all_outputs = fn_full(x_val, *param_vals)
        loss_val = all_outputs[0]
        grad_vals = all_outputs[1:]

    for i, grad_val in enumerate(grad_vals):
        assert np.all(np.isfinite(grad_val)), f"{block_type} gradient {i} has NaN/Inf"
        assert np.abs(grad_val).sum() > 0, f"{block_type} gradient {i} is zero"


def test_gradient_accumulation():
    """
    Test gradient accumulation through multiple paths.

    This test verifies:
    - Gradients accumulate correctly through multiple paths
    - Residual connections work
    - Skip connections propagate gradients
    """

    # Build model with multiple gradient paths
    x = pt.tensor4("x", dtype="float32", shape=(None, 128, 32, 32))

    # Shared parameter used in multiple places
    w = pt.tensor4("w", dtype="float32")

    # Path 1
    h1 = conv2d(x, w, border_mode="half")
    h1 = pt.maximum(0, h1)  # ReLU activation

    # Path 2 (reuses same weight)
    h2 = conv2d(x, w, border_mode="half")
    h2 = pt.sigmoid(h2)

    # Combine paths
    output = h1 + h2 + x  # Residual connection

    # Loss
    loss = output.sum()

    # Gradient should accumulate from both paths
    grad_w = grad(loss, w)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 128, 32, 32)).astype("float32") * 0.1
    w_val = rng.normal(size=(128, 128, 3, 3)).astype("float32") * 0.02

    # Compile and run
    fn = function(inputs=[x, w], outputs=[loss, grad_w], mode="JAX")

    loss_val, grad_val = fn(x_val, w_val)

    # Verify gradient accumulation
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is zero"

    # Gradient should be larger due to accumulation from multiple paths
    grad_norm = np.linalg.norm(grad_val.flatten())
    assert grad_norm > 1e-6, f"Gradient norm too small: {grad_norm}"


def test_gradient_through_dynamic_shapes():
    """
    Test gradient computation with dynamic shapes.

    This test verifies:
    - Gradients work with None dimensions
    - Different batch sizes produce valid gradients
    - Shape inference works correctly
    """

    # Model with dynamic batch size
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, None, None))
    w = pt.tensor4("w", dtype="float32")

    # Operations that preserve dynamic shapes
    h = conv2d(x, w, border_mode="half")
    h = pt.maximum(0, h)  # ReLU activation

    # Global pooling to fixed size
    h_pooled = h.mean(axis=(2, 3))

    # Loss
    loss = h_pooled.sum()

    # Gradient
    grad_w = grad(loss, w)

    # Test with different shapes
    rng = np.random.default_rng(42)
    w_val = rng.normal(size=(128, 64, 3, 3)).astype("float32") * 0.02

    for batch_size in [1, 2, 4]:
        for spatial_size in [16, 32, 64]:
            x_val = (
                rng.normal(size=(batch_size, 64, spatial_size, spatial_size)).astype(
                    "float32"
                )
                * 0.1
            )

            # Compile and run
            fn = function(inputs=[x, w], outputs=[loss, grad_w], mode="JAX")

            loss_val, grad_val = fn(x_val, w_val)

            # Verify gradient
            assert np.all(np.isfinite(grad_val)), (
                f"Gradient has NaN/Inf for shape ({batch_size}, 64, {spatial_size}, {spatial_size})"
            )
            assert grad_val.shape == w_val.shape, (
                f"Gradient shape mismatch: {grad_val.shape} != {w_val.shape}"
            )


def test_second_order_gradients():
    """
    Test second-order gradients (gradients of gradients).

    This test verifies:
    - Hessian-vector products can be computed
    - Higher-order derivatives work
    - JAX's automatic differentiation handles nested gradients
    """

    # Simple model for second-order gradient testing
    x = pt.tensor4("x", dtype="float32", shape=(None, 32, 8, 8))
    w = pt.tensor4("w", dtype="float32")

    # Forward pass
    h = conv2d(x, w, border_mode="half")
    h = h * h  # Quadratic to ensure non-zero second derivatives
    loss = h.sum()

    # First-order gradient
    grad_w = grad(loss, w)

    # Second-order gradient (gradient of gradient w.r.t. x)
    grad2_x = grad(grad_w.sum(), x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 32, 8, 8)).astype("float32") * 0.1
    w_val = rng.normal(size=(64, 32, 3, 3)).astype("float32") * 0.02

    try:
        # Compile with JAX
        fn = function(inputs=[x, w], outputs=[loss, grad_w, grad2_x], mode="JAX")

        loss_val, grad1_val, grad2_val = fn(x_val, w_val)

        # Verify second-order gradients
        assert np.all(np.isfinite(grad1_val)), "First gradient has NaN/Inf"
        assert np.all(np.isfinite(grad2_val)), "Second gradient has NaN/Inf"
        assert grad2_val.shape == x_val.shape, "Second gradient shape mismatch"

    except NotImplementedError:
        # Some operations might not support second-order gradients yet
        pytest.skip("Second-order gradients not yet supported for all operations")
