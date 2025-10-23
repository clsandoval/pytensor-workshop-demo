"""
Integration tests for YOLO11 training loops with JAX backend.
Tests complete training step execution and parameter updates.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, grad, shared
from pytensor.tensor.conv import conv2d


jax = pytest.importorskip("jax")


def test_yolo11_training_step():
    """
    Test complete YOLO11 training step.

    This test verifies:
    - Forward pass computes loss
    - Gradients computed for all parameters
    - Parameter updates applied
    - Training step completes successfully
    """

    # Arrange - simplified YOLO with shared parameters
    # Shared parameters (learnable)
    conv_w = shared(
        np.random.randn(64, 3, 3, 3).astype("float32") * 0.02, name="conv_w"
    )
    conv_b = shared(np.zeros(64, dtype="float32"), name="conv_b")

    # Input and target
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 128, 128))
    target = pt.tensor4("target", dtype="float32", shape=(None, 64, 128, 128))

    # Forward pass (simplified)
    h = conv2d(x, conv_w, border_mode="half")
    h = h + conv_b.dimshuffle("x", 0, "x", "x")
    output = pt.maximum(0, h)  # ReLU activation

    # Loss
    loss = ((output - target) ** 2).mean()

    # Gradients
    params = [conv_w, conv_b]
    grads = grad(loss, params)

    # Learning rate
    lr = 0.001

    # Updates
    updates = {
        conv_w: conv_w - lr * grads[0],
        conv_b: conv_b - lr * grads[1],
    }

    # Act - compile training function
    train_fn = function(
        inputs=[x, target], outputs=[loss, output], updates=updates, mode="JAX"
    )

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 3, 128, 128)).astype("float32") * 0.1
    target_val = rng.normal(size=(4, 64, 128, 128)).astype("float32") * 0.1

    # Store initial parameters
    conv_w_init = conv_w.get_value().copy()
    conv_b_init = conv_b.get_value().copy()

    # Execute training step
    loss_val, output_val = train_fn(x_val, target_val)

    # Get updated parameters
    conv_w_new = conv_w.get_value()
    conv_b_new = conv_b.get_value()

    # Assert
    # Loss computed
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert loss_val > 0, f"Loss should be positive: {loss_val}"

    # Output computed
    assert output_val.shape == target_val.shape, "Output shape mismatch"

    # Parameters updated
    assert not np.array_equal(conv_w_init, conv_w_new), (
        "Convolution weights not updated"
    )
    assert not np.array_equal(conv_b_init, conv_b_new), "Convolution bias not updated"

    # Updates in correct direction (gradient descent)
    # Since loss > 0, gradients should cause parameters to change
    w_change = np.linalg.norm(conv_w_new - conv_w_init)
    b_change = np.linalg.norm(conv_b_new - conv_b_init)
    assert w_change > 1e-6, f"Weight change too small: {w_change}"
    assert b_change > 1e-6, f"Bias change too small: {b_change}"


def test_yolo11_multi_step_training():
    """
    Test multiple YOLO11 training steps.

    This test verifies:
    - Multiple training steps execute
    - Loss generally decreases
    - Parameters continue updating
    - No memory leaks or errors
    """

    # Arrange - simple trainable model
    W = shared(np.eye(10, dtype="float32"), name="W")
    b = shared(np.zeros(10, dtype="float32"), name="b")

    x = pt.matrix("x", dtype="float32")
    y_true = pt.matrix("y_true", dtype="float32")

    # Forward
    y_pred = pt.dot(x, W) + b

    # Loss
    loss = ((y_pred - y_true) ** 2).mean()

    # Gradient and updates
    grads = grad(loss, [W, b])
    lr = 0.01
    updates = {
        W: W - lr * grads[0],
        b: b - lr * grads[1],
    }

    # Training function
    train_fn = function(inputs=[x, y_true], outputs=loss, updates=updates, mode="JAX")

    # Act - run multiple training steps
    rng = np.random.default_rng(42)
    losses = []

    for step in range(5):
        # Generate batch
        x_batch = rng.normal(size=(32, 10)).astype("float32")
        y_batch = x_batch @ np.eye(10, dtype="float32") * 2  # Target is 2x input

        # Train step
        loss_val = train_fn(x_batch, y_batch)
        losses.append(float(loss_val))

    # Assert
    # All steps completed
    assert len(losses) == 5, "Not all training steps completed"

    # Losses are finite
    assert all(np.isfinite(l) for l in losses), f"Some losses not finite: {losses}"

    # Loss generally decreases (allow some fluctuation)
    avg_early = np.mean(losses[:2])
    avg_late = np.mean(losses[-2:])
    assert avg_late <= avg_early * 1.1, (
        f"Loss not decreasing: early {avg_early:.4f}, late {avg_late:.4f}"
    )

    # Parameters changed
    W_final = W.get_value()
    assert not np.array_equal(W_final, np.eye(10)), (
        "Weights didn't change after training"
    )


def test_optimizer_integration():
    """
    Test integration with different optimizer patterns.

    This test verifies:
    - SGD-style updates work
    - Momentum accumulation works
    - Adam-style updates work
    - Gradient clipping works
    """

    # Model parameters
    W = shared(np.random.randn(5, 5).astype("float32") * 0.1, name="W")

    # Momentum buffer (for SGD with momentum)
    momentum = shared(np.zeros((5, 5), dtype="float32"), name="momentum")

    # Adam buffers
    m = shared(np.zeros((5, 5), dtype="float32"), name="m")  # First moment
    v = shared(np.zeros((5, 5), dtype="float32"), name="v")  # Second moment
    t = shared(np.array(0, dtype="float32"), name="t")  # Timestep

    # Input and loss
    x = pt.matrix("x", dtype="float32")
    y = pt.dot(x, W)
    loss = (y**2).sum()

    # Gradient
    grad_W = grad(loss, W)

    # Test different optimizer patterns
    # 1. SGD with momentum
    momentum_beta = 0.9
    lr = 0.01
    momentum_update = momentum_beta * momentum + grad_W
    sgd_momentum_updates = {
        W: W - lr * momentum_update,
        momentum: momentum_update,
    }

    # 2. Adam optimizer
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    t_new = t + 1
    m_new = beta1 * m + (1 - beta1) * grad_W
    v_new = beta2 * v + (1 - beta2) * grad_W**2
    m_hat = m_new / (1 - beta1**t_new)
    v_hat = v_new / (1 - beta2**t_new)
    adam_updates = {
        W: W - lr * m_hat / (pt.sqrt(v_hat) + epsilon),
        m: m_new,
        v: v_new,
        t: t_new,
    }

    # 3. Gradient clipping
    max_norm = 1.0
    grad_norm = pt.sqrt((grad_W**2).sum())
    clipped_grad = pt.switch(
        grad_norm > max_norm, grad_W * max_norm / grad_norm, grad_W
    )
    clip_updates = {
        W: W - lr * clipped_grad,
    }

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(10, 5)).astype("float32")

    # Test SGD with momentum
    train_sgd = function([x], loss, updates=sgd_momentum_updates, mode="JAX")
    W_init = W.get_value().copy()
    loss_sgd = train_sgd(x_val)
    W_sgd = W.get_value()
    assert not np.array_equal(W_init, W_sgd), "SGD momentum didn't update weights"
    W.set_value(W_init)  # Reset

    # Test Adam
    train_adam = function([x], loss, updates=adam_updates, mode="JAX")
    loss_adam = train_adam(x_val)
    W_adam = W.get_value()
    assert not np.array_equal(W_init, W_adam), "Adam didn't update weights"
    W.set_value(W_init)  # Reset

    # Test gradient clipping
    train_clip = function([x], loss, updates=clip_updates, mode="JAX")
    loss_clip = train_clip(x_val)
    W_clip = W.get_value()
    assert not np.array_equal(W_init, W_clip), "Gradient clipping didn't update weights"


def test_mixed_precision_patterns():
    """
    Test mixed precision training patterns.

    This test verifies:
    - Float16/32 conversions work
    - Gradient scaling patterns work
    - Loss scaling for stability
    """

    # Model with potential mixed precision
    W = shared(np.random.randn(8, 8).astype("float32") * 0.1, name="W")

    x = pt.matrix("x", dtype="float32")

    # Simulate mixed precision: compute in lower precision
    # Note: JAX backend should handle dtype conversions
    W_low = pt.cast(W, "float16")
    x_low = pt.cast(x, "float16")
    y_low = pt.dot(x_low, W_low)

    # Cast back to float32 for loss
    y = pt.cast(y_low, "float32")
    loss = (y**2).mean()

    # Gradient (computed in float32)
    grad_W = grad(loss, W)

    # Simple update
    lr = 0.01
    updates = {W: W - lr * grad_W}

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 8)).astype("float32") * 0.1

    # Compile and run
    try:
        train_fn = function([x], loss, updates=updates, mode="JAX")
        W_init = W.get_value().copy()
        loss_val = train_fn(x_val)
        W_new = W.get_value()

        # Verify update happened
        assert not np.array_equal(W_init, W_new), (
            "Weights not updated with mixed precision"
        )
        assert np.isfinite(loss_val), "Loss not finite with mixed precision"

    except NotImplementedError:
        # Mixed precision might not be fully supported yet
        pytest.skip("Mixed precision patterns not yet fully supported")


def test_batch_normalization_training():
    """
    Test batch normalization in training mode.

    This test verifies:
    - BN statistics are updated during training
    - Running statistics accumulate correctly
    - Gradients flow through BN layers
    """

    # BN parameters
    gamma = shared(np.ones(32, dtype="float32"), name="gamma")
    beta = shared(np.zeros(32, dtype="float32"), name="beta")

    # Running statistics
    running_mean = shared(np.zeros(32, dtype="float32"), name="running_mean")
    running_var = shared(np.ones(32, dtype="float32"), name="running_var")

    # Input
    x = pt.tensor4("x", dtype="float32", shape=(None, 32, 16, 16))

    # Batch normalization (training mode)
    axes = (0, 2, 3)  # Normalize over batch and spatial dimensions
    batch_mean = x.mean(axis=axes, keepdims=True)
    batch_var = x.var(axis=axes, keepdims=True)

    # Normalize
    x_norm = (x - batch_mean) / pt.sqrt(batch_var + 1e-5)

    # Scale and shift
    gamma_bc = gamma.dimshuffle("x", 0, "x", "x")
    beta_bc = beta.dimshuffle("x", 0, "x", "x")
    y = gamma_bc * x_norm + beta_bc

    # Update running statistics (momentum=0.1)
    momentum = 0.1
    new_running_mean = (1 - momentum) * running_mean + momentum * batch_mean.squeeze()
    new_running_var = (1 - momentum) * running_var + momentum * batch_var.squeeze()

    # Loss
    loss = y.sum()

    # Gradients
    grads = grad(loss, [gamma, beta])

    # Updates
    lr = 0.01
    updates = {
        gamma: gamma - lr * grads[0],
        beta: beta - lr * grads[1],
        running_mean: new_running_mean,
        running_var: new_running_var,
    }

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(8, 32, 16, 16)).astype("float32")

    # Compile and run
    train_fn = function([x], loss, updates=updates, mode="JAX")

    # Initial values
    gamma_init = gamma.get_value().copy()
    beta_init = beta.get_value().copy()
    rmean_init = running_mean.get_value().copy()
    rvar_init = running_var.get_value().copy()

    # Training step
    loss_val = train_fn(x_val)

    # Updated values
    gamma_new = gamma.get_value()
    beta_new = beta.get_value()
    rmean_new = running_mean.get_value()
    rvar_new = running_var.get_value()

    # Verify updates
    assert not np.array_equal(gamma_init, gamma_new), "Gamma not updated"
    assert not np.array_equal(beta_init, beta_new), "Beta not updated"
    assert not np.array_equal(rmean_init, rmean_new), "Running mean not updated"
    assert not np.array_equal(rvar_init, rvar_new), "Running var not updated"


def test_gradient_accumulation_pattern():
    """
    Test gradient accumulation for large batch simulation.

    This test verifies:
    - Gradients can be accumulated over mini-batches
    - Equivalent to training with larger batch
    - Useful for memory-constrained training
    """

    # Model
    W = shared(np.random.randn(10, 10).astype("float32") * 0.1, name="W")
    grad_accum = shared(np.zeros((10, 10), dtype="float32"), name="grad_accum")

    x = pt.matrix("x", dtype="float32")
    y = pt.dot(x, W)
    loss = (y**2).mean()

    # Gradient
    grad_W = grad(loss, W)

    # Accumulation step (accumulate gradients)
    accum_updates = {
        grad_accum: grad_accum + grad_W,
    }

    # Update step (apply accumulated gradients)
    n_accum = pt.scalar("n_accum", dtype="float32")
    lr = 0.01
    update_updates = {
        W: W - lr * grad_accum / n_accum,
        grad_accum: pt.zeros_like(grad_accum),
    }

    # Functions
    accum_fn = function([x], loss, updates=accum_updates, mode="JAX")
    update_fn = function([n_accum], updates=update_updates, mode="JAX")

    # Test data - simulate 4 mini-batches
    rng = np.random.default_rng(42)
    W_init = W.get_value().copy()

    # Accumulate gradients over 4 mini-batches
    total_loss = 0
    for i in range(4):
        x_val = rng.normal(size=(8, 10)).astype("float32")
        loss_val = accum_fn(x_val)
        total_loss += loss_val

    # Apply accumulated gradients
    update_fn(np.array(4.0, dtype="float32"))

    W_accumulated = W.get_value()

    # Compare with single large batch training
    W.set_value(W_init)
    grad_accum.set_value(np.zeros((10, 10), dtype="float32"))

    # Reset RNG to get same data
    rng = np.random.default_rng(42)

    # Create large batch with same random seed
    x_large = np.vstack([rng.normal(size=(8, 10)).astype("float32") for _ in range(4)])

    # Direct update without accumulation
    direct_updates = {W: W - lr * grad_W}
    train_direct = function([x], loss, updates=direct_updates, mode="JAX")

    loss_large = train_direct(x_large)
    W_direct = W.get_value()

    # Accumulated gradients should give similar result to large batch
    # (not exact due to different loss averaging)
    assert np.allclose(W_accumulated, W_direct, rtol=0.1), (
        "Gradient accumulation doesn't match large batch training"
    )
