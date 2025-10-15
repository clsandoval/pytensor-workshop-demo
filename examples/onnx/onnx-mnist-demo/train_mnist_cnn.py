"""
Train a Conv2D CNN on MNIST and export to ONNX for WebGPU inference.

This script:
1. Loads MNIST dataset (28x28 standard size)
2. Trains a small CNN with proper convergence
3. Monitors training progress with loss and accuracy
4. Exports trained model to ONNX format
5. Verifies the export works correctly

Architecture:
- Input: (batch, 1, 28, 28) - grayscale images
- Conv2D: 8 filters, 3x3 kernel
- ReLU activation
- Conv2D: 16 filters, 3x3 kernel
- ReLU activation
- Flatten
- Dense: 128 units
- ReLU activation
- Dense: 10 units (logits)
- Softmax (for classification)

The model is designed to:
- Converge quickly (5-10 epochs)
- Be small enough for browser deployment
- Work with the existing WebGPU demo infrastructure
- Match the input size expected by cnn_model_webgpu_demo.html (28x28)
"""

import os

import numpy as np


# Configure PyTensor before importing
os.environ.setdefault(
    "PYTENSOR_FLAGS", "optimizer=fast_compile,exception_verbosity=high"
)

import pytensor
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.conv.abstract_conv import conv2d
from pytensor.tensor.math import log
from pytensor.tensor.pool import pool_2d
from pytensor.tensor.special import softmax


# ============================================================
# 1. Data Loading
# ============================================================


def load_mnist_data(n_train=50000, n_test=10000):
    """
    Load MNIST dataset (standard 28x28 size).

    Args:
        n_train: Number of training samples
        n_test: Number of test samples

    Returns:
        X_train, y_train, X_test, y_test
    """
    print("\n" + "=" * 60)
    print("Loading MNIST Dataset")
    print("=" * 60)

    try:
        from sklearn.datasets import fetch_openml

        print("Fetching MNIST from OpenML...")
        mnist = fetch_openml("mnist_784", version=1, parser="liac-arff", as_frame=False)

        X = mnist.data.astype("float32")  # Shape: (70000, 784)
        y = mnist.target.astype("int32")  # Shape: (70000,)

        print(f"  Original data shape: {X.shape}")
        print(f"  Original labels shape: {y.shape}")

        # Shuffle data
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Split train/test
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train : n_train + n_test]
        y_test = y[n_train : n_train + n_test]

        # Reshape from (samples, 784) to (samples, 1, 28, 28)
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

        # Normalize to [0, 1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        print("\n✓ MNIST loaded successfully:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Image shape: {X_train.shape[1:]}")
        print(f"  Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  Classes: {np.unique(y_train)}")

        return X_train, y_train, X_test, y_test

    except ImportError as e:
        print("\n✗ Error: Missing required library")
        print(f"  {e}")
        print("\nPlease install:")
        print("  pip install scikit-learn")
        raise


# ============================================================
# 2. Model Architecture
# ============================================================


def build_mnist_cnn(learning_rate=0.01):
    """
    Build CNN architecture for MNIST classification (28x28 input).

    Architecture with MaxPooling for efficiency:
    - Conv2D (1→8, 3x3) → ReLU → MaxPool2D(2x2)
    - Conv2D (8→16, 3x3) → ReLU → MaxPool2D(2x2)
    - Flatten
    - Dense (400 → 64) → ReLU
    - Dense (64 → 10) → Softmax

    This produces 10 class probabilities for digit classification.

    Args:
        learning_rate: Learning rate for SGD with momentum

    Returns:
        x, y, predictions, loss, accuracy, updates, params
    """
    print("\n" + "=" * 60)
    print("Building CNN Model")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Input: batch of 28x28 grayscale images
    x = pt.tensor4("x", dtype="float32")  # (batch, 1, 28, 28)
    y = pt.ivector("y")  # (batch,) - integer labels

    # He initialization scale
    def he_init(shape, fan_in):
        """He initialization for ReLU networks"""
        return rng.normal(0, np.sqrt(2.0 / fan_in), shape).astype("float32")

    # Layer 1: Conv2D (1 -> 8 channels, 3x3 kernel)
    # Output: (batch, 8, 26, 26) → MaxPool → (batch, 8, 13, 13)
    W1 = shared(he_init((8, 1, 3, 3), 1 * 3 * 3), name="conv1_weights", borrow=True)
    b1 = shared(np.zeros(8, dtype="float32"), name="conv1_bias", borrow=True)

    conv1 = conv2d(x, W1, border_mode="valid", filter_flip=False)
    conv1_out = conv1 + b1.dimshuffle("x", 0, "x", "x")
    relu1 = pt.maximum(conv1_out, 0)
    pool1 = pool_2d(relu1, ws=(2, 2), stride=(2, 2), mode="max")

    print("  Layer 1: Conv2D(1→8, 3x3) + ReLU + MaxPool2D(2x2)")
    print("    Output: (batch, 8, 13, 13)")

    # Layer 2: Conv2D (8 -> 16 channels, 3x3 kernel)
    # Output: (batch, 16, 11, 11) → MaxPool → (batch, 16, 5, 5)
    W2 = shared(he_init((16, 8, 3, 3), 8 * 3 * 3), name="conv2_weights", borrow=True)
    b2 = shared(np.zeros(16, dtype="float32"), name="conv2_bias", borrow=True)

    conv2 = conv2d(pool1, W2, border_mode="valid", filter_flip=False)
    conv2_out = conv2 + b2.dimshuffle("x", 0, "x", "x")
    relu2 = pt.maximum(conv2_out, 0)
    pool2 = pool_2d(relu2, ws=(2, 2), stride=(2, 2), mode="max")

    print("  Layer 2: Conv2D(8→16, 3x3) + ReLU + MaxPool2D(2x2)")
    print("    Output: (batch, 16, 5, 5)")

    # Flatten: (batch, 16, 5, 5) -> (batch, 400)
    flat = pool2.flatten(2)
    flat_size = 16 * 5 * 5
    print("  Layer 3: Flatten")
    print(f"    Output: (batch, {flat_size})")

    # Dense layer 1: 400 -> 64
    W3 = shared(he_init((flat_size, 64), flat_size), name="fc1_weights", borrow=True)
    b3 = shared(np.zeros(64, dtype="float32"), name="fc1_bias", borrow=True)

    fc1 = pt.dot(flat, W3) + b3
    relu3 = pt.maximum(fc1, 0)
    print(f"  Layer 4: Dense({flat_size}→64) + ReLU")
    print("    Output: (batch, 64)")

    # Output layer: 64 -> 10
    W4 = shared(
        rng.normal(0, 0.01, (64, 10)).astype("float32"),
        name="fc2_weights",
        borrow=True,
    )
    b4 = shared(np.zeros(10, dtype="float32"), name="fc2_bias", borrow=True)

    logits = pt.dot(relu3, W4) + b4
    predictions = softmax(logits)
    print("  Layer 5: Dense(64→10) + Softmax")
    print("    Output: (batch, 10)")

    # Loss: Cross-entropy (manual implementation)
    # categorical_crossentropy: -log(p[y]) for each sample
    loss = -log(predictions[pt.arange(y.shape[0]), y]).mean()

    # Accuracy
    predicted_class = pt.argmax(predictions, axis=1)
    accuracy = pt.eq(predicted_class, y).mean()

    # Collect parameters
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    # Compute gradients
    grads = pytensor.grad(loss, params)

    # SGD with momentum (simple and effective)
    momentum = 0.9

    # Create velocity variables for momentum
    velocities = []
    for param in params:
        velocity = shared(
            np.zeros_like(param.get_value()), name=f"v_{param.name}", borrow=True
        )
        velocities.append(velocity)

    # Update rules: momentum SGD
    updates = []
    for param, grad, velocity in zip(params, grads, velocities):
        # v_t = momentum * v_{t-1} - lr * grad
        new_velocity = momentum * velocity - learning_rate * grad
        # param_t = param_{t-1} + v_t
        new_param = param + new_velocity

        # Cast to correct dtype to match parameter type
        new_velocity = new_velocity.astype(param.dtype)
        new_param = new_param.astype(param.dtype)

        updates.append((velocity, new_velocity))
        updates.append((param, new_param))

    print("\nOptimization:")
    print("  Method: SGD with momentum")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Momentum: {momentum}")
    print(f"  Parameters: {sum(p.get_value().size for p in params):,}")

    return x, y, predictions, loss, accuracy, updates, params


# ============================================================
# 3. Training Functions
# ============================================================


def compile_functions(x, y, predictions, loss, accuracy, updates):
    """
    Compile PyTensor functions for training and evaluation.

    Returns:
        train_fn, eval_fn, predict_fn
    """
    print("\n" + "=" * 60)
    print("Compiling PyTensor Functions")
    print("=" * 60)

    print("  Compiling training function...")
    train_fn = pytensor.function(
        inputs=[x, y], outputs=[loss, accuracy], updates=updates, name="train_function"
    )

    print("  Compiling evaluation function...")
    eval_fn = pytensor.function(
        inputs=[x, y], outputs=[loss, accuracy], name="eval_function"
    )

    print("  Compiling prediction function...")
    predict_fn = pytensor.function(
        inputs=[x], outputs=predictions, name="predict_function"
    )

    print("✓ Functions compiled successfully")

    return train_fn, eval_fn, predict_fn


def train_model(
    train_fn, eval_fn, X_train, y_train, X_test, y_test, epochs=10, batch_size=64
):
    """
    Train the CNN model.

    Args:
        train_fn: Training function (updates parameters)
        eval_fn: Evaluation function (no updates)
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Mini-batch size

    Returns:
        History dictionary with training metrics
    """
    print("\n" + "=" * 60)
    print(f"Training: {epochs} epochs, batch size {batch_size}")
    print("=" * 60)

    n_train_batches = len(X_train) // batch_size
    n_test_batches = len(X_test) // batch_size

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        # Shuffle training data each epoch
        rng = np.random.default_rng(epoch)
        indices = rng.permutation(len(X_train))

        epoch_losses = []
        epoch_accs = []

        # Training loop
        for batch_idx in range(n_train_batches):
            # Get batch
            batch_indices = indices[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Train on batch
            loss, acc = train_fn(X_batch, y_batch)
            epoch_losses.append(loss)
            epoch_accs.append(acc)

            # Progress indicator every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                avg_acc = np.mean(epoch_accs[-100:])
                print(
                    f"  Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{n_train_batches}: "
                    f"Loss={avg_loss:.4f}, Acc={avg_acc * 100:.2f}%"
                )

        # Compute epoch training statistics
        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_accs)

        # Evaluate on test set
        test_losses = []
        test_accs = []

        for batch_idx in range(n_test_batches):
            X_batch = X_test[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            y_batch = y_test[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            loss, acc = eval_fn(X_batch, y_batch)
            test_losses.append(loss)
            test_accs.append(acc)

        test_loss = np.mean(test_losses)
        test_acc = np.mean(test_accs)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Print epoch summary
        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Train - Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%")
        print(f"    Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
        print()

        # Early stopping if perfect accuracy
        if train_acc >= 0.99 and test_acc >= 0.95:
            print("  ✓ Early stopping: Model converged!")
            break

    print("=" * 60)
    print("✓ Training complete!")
    print("=" * 60)

    return history


# ============================================================
# 4. ONNX Export
# ============================================================


def export_to_onnx(predict_fn, output_path="cnn_model.onnx"):
    """
    Export trained model to ONNX format.

    Args:
        predict_fn: Prediction function to export
        output_path: Path to save ONNX model

    Returns:
        Path to saved model
    """
    from pytensor.link.onnx import export_onnx

    print("\n" + "=" * 60)
    print(f"Exporting to ONNX: {output_path}")
    print("=" * 60)

    try:
        model = export_onnx(predict_fn, output_path)

        print("✓ Export successful!")
        print(f"  Opset version: {model.opset_import[0].version}")
        print(f"  Inputs: {len(model.graph.input)}")
        print(f"  Outputs: {len(model.graph.output)}")
        print(f"  Nodes: {len(model.graph.node)}")
        print(f"  Initializers: {len(model.graph.initializer)}")

        # Show input/output info
        input_info = model.graph.input[0]
        output_info = model.graph.output[0]

        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]

        print(f"\n  Input '{input_info.name}':")
        print(f"    Shape: {input_shape}")
        print(f"    Type: {input_info.type.tensor_type.elem_type}")

        print(f"\n  Output '{output_info.name}':")
        print(f"    Shape: {output_shape}")
        print(f"    Type: {output_info.type.tensor_type.elem_type}")

        # Show node types
        node_types = [node.op_type for node in model.graph.node]
        print(f"\n  Operations: {', '.join(sorted(set(node_types)))}")

        return output_path

    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise


def verify_onnx_export(predict_fn, onnx_path, test_input):
    """
    Verify ONNX export produces same results as PyTensor.

    Args:
        predict_fn: PyTensor prediction function
        onnx_path: Path to ONNX model
        test_input: Test input to use for verification

    Returns:
        True if verification passes
    """
    print("\n" + "=" * 60)
    print("Verifying ONNX Export")
    print("=" * 60)

    try:
        import onnxruntime as ort

        # Get PyTensor output
        print("  Running PyTensor inference...")
        pytensor_output = predict_fn(test_input)

        # Get ONNX output
        print("  Running ONNX Runtime inference...")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        onnx_output = session.run([output_name], {input_name: test_input})[0]

        # Compare
        diff = np.abs(pytensor_output - onnx_output).max()
        mean_diff = np.abs(pytensor_output - onnx_output).mean()

        print("\n  Comparison:")
        print(f"    PyTensor output shape: {pytensor_output.shape}")
        print(f"    ONNX output shape: {onnx_output.shape}")
        print(f"    Max absolute difference: {diff:.2e}")
        print(f"    Mean absolute difference: {mean_diff:.2e}")

        tolerance = 1e-4
        if diff < tolerance:
            print(f"    ✓ Outputs match! (tolerance: {tolerance})")
            return True
        else:
            print("    ✗ Outputs differ significantly!")
            return False

    except ImportError:
        print("  ⚠ onnxruntime not installed, skipping verification")
        print("    Install with: pip install onnxruntime")
        return None


# ============================================================
# 5. Main Training Pipeline
# ============================================================


def main():
    """Main training and export pipeline."""

    print("\n" + "=" * 70)
    print(" " * 15 + "MNIST CNN Training & ONNX Export")
    print("=" * 70)

    # Configuration - Optimized for faster convergence
    LEARNING_RATE = 0.1  # Increased from 0.01 for faster convergence
    EPOCHS = 5
    BATCH_SIZE = 64  # Increased from 32 for more stable gradients
    N_TRAIN = 50000
    N_TEST = 10000

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data(n_train=N_TRAIN, n_test=N_TEST)

    # Build model
    x, y, predictions, loss, accuracy, updates, _params = build_mnist_cnn(
        learning_rate=LEARNING_RATE
    )

    # Compile functions
    train_fn, eval_fn, predict_fn = compile_functions(
        x, y, predictions, loss, accuracy, updates
    )

    # Train model
    history = train_model(
        train_fn,
        eval_fn,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"  Final Train Accuracy: {history['train_acc'][-1] * 100:.2f}%")
    print(f"  Final Test Accuracy:  {history['test_acc'][-1] * 100:.2f}%")
    print(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}")
    print(f"  Final Test Loss:      {history['test_loss'][-1]:.4f}")

    # Export to ONNX
    onnx_path = export_to_onnx(predict_fn, "cnn_model.onnx")

    # Verify export
    test_input = X_test[:5]  # Use first 5 test images
    verify_onnx_export(predict_fn, onnx_path, test_input)

    # Final summary
    print("\n" + "=" * 70)
    print("✓ Pipeline Complete!")
    print("=" * 70)
    print(f"\nTrained model saved to: {onnx_path}")
    print("\nTo use in browser:")
    print("  1. Model is already named 'cnn_model.onnx' (matches demo)")
    print("  2. Start server: python -m http.server 8000")
    print("  3. Open: http://localhost:8000/cnn_model_webgpu_demo.html")
    print("  4. Draw digits and run inference in your browser!")
    print()


if __name__ == "__main__":
    main()
