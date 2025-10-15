# MNIST CNN Classification with PyTensor and ONNX Export

**Complete end-to-end guide**: Build, train, export, and deploy a Convolutional Neural Network for MNIST digit classification using PyTensor and ONNX.

## Overview

This guide demonstrates how to:
1. Build a CNN in PyTensor for MNIST digit classification
2. Train the model (or use random weights for demo)
3. Export the model to ONNX format
4. Deploy to multiple platforms (browser, mobile, edge devices)

**What you'll build:**
- Simple CNN: Conv → ReLU → Conv → ReLU → Dense → Softmax
- Classifies handwritten digits (0-9)
- Exports to ONNX for cross-platform deployment
- Runs at GPU speed in browsers via WebGPU

## Prerequisites

### Python Environment

```bash
# Install PyTensor with ONNX support
pip install pytensor[onnx]

# Or using uv (faster)
uv pip install pytensor[onnx]

# Additional dependencies for training (optional)
pip install scikit-learn  # For loading MNIST
pip install matplotlib    # For visualization
```

### Verify Installation

```python
# Test imports
import pytensor
import pytensor.tensor as pt
from pytensor.tensor.nnet import conv2d
from pytensor.link.onnx import export_onnx
import onnxruntime as ort

print("✓ All dependencies installed!")
```

---

## Part 1: Build the CNN Model

### Complete Model Architecture

```python
"""
mnist_cnn_model.py - MNIST CNN Model Definition
"""
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.nnet import conv2d


def build_mnist_cnn():
    """
    Build a Convolutional Neural Network for MNIST digit classification.

    Architecture:
    ============
    Input: (batch_size, 1, 28, 28) - Grayscale 28x28 images

    Layer 1: Convolutional
    - 32 filters, 3x3 kernel, valid padding
    - Output: (batch, 32, 26, 26)
    - Activation: ReLU

    Layer 2: Convolutional
    - 64 filters, 3x3 kernel, valid padding
    - Output: (batch, 64, 24, 24)
    - Activation: ReLU

    Layer 3: Flatten
    - Output: (batch, 64*24*24 = 36864)

    Layer 4: Fully Connected (Dense)
    - 128 units
    - Activation: ReLU

    Layer 5: Output (Dense)
    - 10 units (one per digit 0-9)
    - Activation: Softmax

    Returns
    -------
    x : TensorVariable
        Input tensor (batch, 1, 28, 28)
    predictions : TensorVariable
        Output predictions (batch, 10)
    params : list
        List of all trainable parameters
    """

    # ============================================================
    # Input Layer
    # ============================================================
    # Symbolic input: (batch_size, channels=1, height=28, width=28)
    x = pt.tensor4('x', dtype='float32')

    # ============================================================
    # Layer 1: Convolutional Layer (1 → 32 filters)
    # ============================================================
    print("Building Layer 1: Conv2D (1 → 32 filters, 3x3)...")

    # Weights: (output_filters=32, input_channels=1, height=3, width=3)
    W_conv1 = shared(
        value=np.random.randn(32, 1, 3, 3).astype('float32') * 0.1,
        name='W_conv1',
        borrow=True
    )

    # Bias: one per output filter
    b_conv1 = shared(
        value=np.zeros(32, dtype='float32'),
        name='b_conv1',
        borrow=True
    )

    # Apply convolution
    # border_mode='valid': No padding, output size = (28-3+1) = 26
    # filter_flip=False: Use cross-correlation (matches ONNX Conv)
    conv1_out = conv2d(
        input=x,
        filters=W_conv1,
        border_mode='valid',
        subsample=(1, 1),
        filter_flip=False
    )

    # Add bias (broadcast across spatial dimensions)
    conv1_out = conv1_out + b_conv1.dimshuffle('x', 0, 'x', 'x')

    # Apply ReLU activation
    conv1_relu = pt.maximum(conv1_out, 0)

    print(f"  Output shape: (batch, 32, 26, 26)")

    # ============================================================
    # Layer 2: Convolutional Layer (32 → 64 filters)
    # ============================================================
    print("Building Layer 2: Conv2D (32 → 64 filters, 3x3)...")

    # Weights: (64, 32, 3, 3)
    W_conv2 = shared(
        value=np.random.randn(64, 32, 3, 3).astype('float32') * 0.1,
        name='W_conv2',
        borrow=True
    )

    # Bias: one per output filter
    b_conv2 = shared(
        value=np.zeros(64, dtype='float32'),
        name='b_conv2',
        borrow=True
    )

    # Apply convolution: (26-3+1) = 24
    conv2_out = conv2d(
        input=conv1_relu,
        filters=W_conv2,
        border_mode='valid',
        subsample=(1, 1),
        filter_flip=False
    )

    # Add bias and activation
    conv2_out = conv2_out + b_conv2.dimshuffle('x', 0, 'x', 'x')
    conv2_relu = pt.maximum(conv2_out, 0)

    print(f"  Output shape: (batch, 64, 24, 24)")

    # ============================================================
    # Layer 3: Flatten
    # ============================================================
    print("Building Layer 3: Flatten...")

    # Flatten: (batch, 64, 24, 24) → (batch, 64*24*24=36864)
    flat = conv2_relu.flatten(2)

    print(f"  Output shape: (batch, 36864)")

    # ============================================================
    # Layer 4: Fully Connected Layer (36864 → 128)
    # ============================================================
    print("Building Layer 4: Dense (36864 → 128)...")

    # Weights
    W_fc1 = shared(
        value=np.random.randn(64 * 24 * 24, 128).astype('float32') * 0.01,
        name='W_fc1',
        borrow=True
    )

    # Bias
    b_fc1 = shared(
        value=np.zeros(128, dtype='float32'),
        name='b_fc1',
        borrow=True
    )

    # Dense layer: matrix multiplication + bias
    fc1_out = pt.dot(flat, W_fc1) + b_fc1

    # ReLU activation
    fc1_relu = pt.maximum(fc1_out, 0)

    print(f"  Output shape: (batch, 128)")

    # ============================================================
    # Layer 5: Output Layer (128 → 10)
    # ============================================================
    print("Building Layer 5: Output Dense (128 → 10)...")

    # Weights
    W_fc2 = shared(
        value=np.random.randn(128, 10).astype('float32') * 0.01,
        name='W_fc2',
        borrow=True
    )

    # Bias
    b_fc2 = shared(
        value=np.zeros(10, dtype='float32'),
        name='b_fc2',
        borrow=True
    )

    # Output logits
    logits = pt.dot(fc1_relu, W_fc2) + b_fc2

    # Softmax activation for probabilities
    predictions = pt.nnet.softmax(logits)

    print(f"  Output shape: (batch, 10)")

    # ============================================================
    # Collect Parameters
    # ============================================================
    params = [
        W_conv1, b_conv1,
        W_conv2, b_conv2,
        W_fc1, b_fc1,
        W_fc2, b_fc2
    ]

    print(f"\n✓ Model built with {sum(p.get_value().size for p in params):,} parameters")

    return x, predictions, params


# Build the model
if __name__ == '__main__':
    x, predictions, params = build_mnist_cnn()
    print("\n✓ CNN architecture defined successfully!")
```

---

## Part 2: Training (Optional)

### Training Functions

```python
"""
mnist_cnn_training.py - Training Functions
"""
import numpy as np
import pytensor
import pytensor.tensor as pt


def create_training_functions(x, predictions, params):
    """
    Create training and evaluation functions.

    Parameters
    ----------
    x : TensorVariable
        Input tensor
    predictions : TensorVariable
        Model predictions
    params : list
        Trainable parameters

    Returns
    -------
    train_fn : Function
        Training function (updates parameters)
    eval_fn : Function
        Evaluation function (no updates)
    predict_fn : Function
        Prediction function (returns probabilities)
    """

    print("\nCreating training functions...")

    # Target labels: (batch_size,) - integer class indices 0-9
    y = pt.ivector('y')

    # ============================================================
    # Loss Function: Cross-Entropy
    # ============================================================
    # categorical_crossentropy expects (predictions, targets)
    loss = pt.nnet.categorical_crossentropy(predictions, y).mean()

    # ============================================================
    # Accuracy Metric
    # ============================================================
    predicted_class = pt.argmax(predictions, axis=1)
    accuracy = pt.eq(predicted_class, y).mean()

    # ============================================================
    # Gradients and Parameter Updates (SGD)
    # ============================================================
    learning_rate = 0.01

    # Compute gradients
    grads = pytensor.grad(loss, params)

    # SGD updates: param = param - learning_rate * gradient
    updates = [
        (param, param - learning_rate * grad)
        for param, grad in zip(params, grads)
    ]

    # ============================================================
    # Compile Functions
    # ============================================================
    print("  Compiling training function...")
    train_fn = pytensor.function(
        inputs=[x, y],
        outputs=[loss, accuracy],
        updates=updates,
        name='train_function'
    )

    print("  Compiling evaluation function...")
    eval_fn = pytensor.function(
        inputs=[x, y],
        outputs=[loss, accuracy],
        name='eval_function'
    )

    print("  Compiling prediction function...")
    predict_fn = pytensor.function(
        inputs=[x],
        outputs=predictions,
        name='predict_function'
    )

    print("✓ Functions compiled successfully!")

    return train_fn, eval_fn, predict_fn


def load_mnist_data():
    """
    Load MNIST dataset using scikit-learn.

    Returns
    -------
    X_train, X_test : ndarray
        Training and test images (float32, normalized to [0,1])
    y_train, y_test : ndarray
        Training and test labels (int32)
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    print("\nLoading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')

    # Normalize pixel values to [0, 1]
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int32')

    # Reshape from (samples, 784) to (samples, 1, 28, 28)
    X = X.reshape(-1, 1, 28, 28)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ MNIST loaded:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Image shape: {X_train.shape[1:]}")

    return X_train, X_test, y_train, y_test


def train_model(train_fn, eval_fn, X_train, y_train, X_test, y_test,
                epochs=5, batch_size=32):
    """
    Train the CNN model.

    Parameters
    ----------
    train_fn : Function
        Training function
    eval_fn : Function
        Evaluation function
    X_train, y_train : ndarray
        Training data
    X_test, y_test : ndarray
        Test data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    """

    print(f"\n{'='*60}")
    print(f"Starting Training: {epochs} epochs, batch size {batch_size}")
    print(f"{'='*60}\n")

    n_train_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))

        epoch_loss = 0
        epoch_acc = 0

        # Training loop
        for batch_idx in range(n_train_batches):
            # Get batch
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Train on batch
            loss, acc = train_fn(X_batch, y_batch)
            epoch_loss += loss
            epoch_acc += acc

            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, "
                      f"Batch {batch_idx+1}/{n_train_batches}: "
                      f"Loss={loss:.4f}, Acc={acc:.4f}")

        # Epoch statistics
        avg_loss = epoch_loss / n_train_batches
        avg_acc = epoch_acc / n_train_batches

        # Evaluate on test set
        n_test_batches = len(X_test) // batch_size
        test_loss = 0
        test_acc = 0

        for batch_idx in range(n_test_batches):
            X_batch = X_test[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            y_batch = y_test[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            loss, acc = eval_fn(X_batch, y_batch)
            test_loss += loss
            test_acc += acc

        test_loss /= n_test_batches
        test_acc /= n_test_batches

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        print(f"    Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print()

    print(f"{'='*60}")
    print("✓ Training complete!")
    print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    from mnist_cnn_model import build_mnist_cnn

    # Build model
    x, predictions, params = build_mnist_cnn()

    # Create training functions
    train_fn, eval_fn, predict_fn = create_training_functions(x, predictions, params)

    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()

    # Train model (or skip for demo)
    # train_model(train_fn, eval_fn, X_train, y_train, X_test, y_test, epochs=5)

    print("Note: Uncomment train_model() call above to actually train the network.")
    print("For demo purposes, we can skip training and use random weights.")
```

---

## Part 3: Export to ONNX

### Export Script

```python
"""
mnist_cnn_export.py - Export Model to ONNX
"""
import numpy as np
from mnist_cnn_model import build_mnist_cnn


def export_to_onnx(output_path='mnist_cnn.onnx'):
    """
    Export the trained PyTensor CNN model to ONNX format.

    Parameters
    ----------
    output_path : str
        Path where ONNX model will be saved

    Returns
    -------
    inference_fn : Function
        PyTensor inference function
    onnx_path : str
        Path to exported ONNX model
    """

    print(f"\n{'='*60}")
    print("Exporting PyTensor CNN to ONNX")
    print(f"{'='*60}\n")

    # ============================================================
    # 1. Build Model
    # ============================================================
    print("Step 1: Building model...")
    x, predictions, params = build_mnist_cnn()

    # ============================================================
    # 2. Compile Inference Function
    # ============================================================
    print("\nStep 2: Compiling inference function...")
    import pytensor

    inference_fn = pytensor.function(
        inputs=[x],
        outputs=predictions,
        name='inference'
    )

    print("✓ Inference function compiled")

    # ============================================================
    # 3. Test with Sample Input
    # ============================================================
    print("\nStep 3: Testing with sample input...")

    # Create random test image
    test_input = np.random.randn(1, 1, 28, 28).astype('float32')
    pytensor_output = inference_fn(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {pytensor_output.shape}")
    print(f"  Predicted digit: {np.argmax(pytensor_output[0])}")
    print(f"  Confidence: {np.max(pytensor_output[0]):.2%}")

    # ============================================================
    # 4. Export to ONNX
    # ============================================================
    print(f"\nStep 4: Exporting to ONNX...")

    from pytensor.link.onnx import export_onnx

    model = export_onnx(
        pytensor_function=inference_fn,
        output_path=output_path,
        model_name='mnist_cnn'
    )

    print(f"✓ Model exported to: {output_path}")

    # ============================================================
    # 5. Validate ONNX Model
    # ============================================================
    print("\nStep 5: Validating ONNX model...")

    import onnx

    # Load and check model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print("✓ ONNX model validation passed")

    # Print model info
    print(f"\nModel Information:")
    print(f"  IR Version: {onnx_model.ir_version}")
    print(f"  Opset Version: {onnx_model.opset_import[0].version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Graph Inputs: {len(onnx_model.graph.input)}")
    print(f"  Graph Outputs: {len(onnx_model.graph.output)}")
    print(f"  Graph Nodes: {len(onnx_model.graph.node)}")
    print(f"  Initializers: {len(onnx_model.graph.initializer)}")

    # ============================================================
    # 6. Test with ONNX Runtime
    # ============================================================
    print("\nStep 6: Testing with ONNX Runtime...")

    import onnxruntime as ort

    # Create inference session
    session = ort.InferenceSession(
        output_path,
        providers=['CPUExecutionProvider']
    )

    # Run inference
    onnx_output = session.run(None, {'x': test_input})[0]

    print(f"  ONNX output shape: {onnx_output.shape}")
    print(f"  ONNX predicted digit: {np.argmax(onnx_output[0])}")

    # Compare outputs
    matches = np.allclose(pytensor_output, onnx_output, rtol=1e-4)
    max_diff = np.abs(pytensor_output - onnx_output).max()

    print(f"\nVerification:")
    print(f"  Outputs match: {matches}")
    print(f"  Max difference: {max_diff:.2e}")

    if matches:
        print("  ✓ ONNX export successful!")
    else:
        print("  ⚠ Warning: Outputs differ!")

    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}\n")
    print(f"Your model is ready for deployment:")
    print(f"  • Browsers (ONNX Runtime Web + WebGPU)")
    print(f"  • Mobile (ONNX Runtime Mobile)")
    print(f"  • Edge devices (ONNX Runtime)")
    print(f"  • Any platform that supports ONNX!")

    return inference_fn, output_path


if __name__ == '__main__':
    inference_fn, onnx_path = export_to_onnx()
```

---

## Part 4: Browser Deployment

### HTML Demo Application

Create `mnist_demo.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST CNN Demo - PyTensor + ONNX</title>

    <!-- ONNX Runtime Web -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }

        h1 {
            color: #333;
            text-align: center;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        #canvas {
            border: 3px solid #333;
            cursor: crosshair;
            display: block;
            margin: 20px auto;
            background: white;
        }

        .controls {
            text-align: center;
            margin: 20px 0;
        }

        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 30px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }

        button:hover {
            background: #0056b3;
        }

        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        #result {
            text-align: center;
            margin-top: 20px;
            font-size: 24px;
            color: #333;
        }

        .digit {
            font-size: 64px;
            font-weight: bold;
            color: #007bff;
        }

        .confidence {
            font-size: 18px;
            color: #666;
        }

        .info {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #007bff;
        }

        .loading {
            text-align: center;
            color: #666;
            padding: 20px;
        }

        .probabilities {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-top: 20px;
        }

        .prob-bar {
            text-align: center;
        }

        .prob-label {
            font-weight: bold;
            margin-bottom: 5px;
        }

        .prob-value {
            height: 100px;
            background: #e0e0e0;
            border-radius: 5px;
            position: relative;
            overflow: hidden;
        }

        .prob-fill {
            position: absolute;
            bottom: 0;
            width: 100%;
            background: linear-gradient(to top, #007bff, #4dabf7);
            transition: height 0.3s ease;
        }

        .prob-text {
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 MNIST Digit Classifier</h1>
        <p style="text-align: center; color: #666;">
            Draw a digit (0-9) and let the CNN predict it!<br>
            <small>Powered by PyTensor + ONNX Runtime Web</small>
        </p>

        <div class="info">
            <strong>ℹ️ How it works:</strong>
            <ul style="margin: 10px 0 0 20px;">
                <li>Model trained in PyTensor (Python)</li>
                <li>Exported to ONNX format</li>
                <li>Running in your browser with ONNX Runtime Web</li>
                <li>Uses WebGPU for acceleration (if available)</li>
            </ul>
        </div>

        <div id="loading" class="loading">
            <p>Loading model...</p>
        </div>

        <div id="app" style="display: none;">
            <canvas id="canvas" width="280" height="280"></canvas>

            <div class="controls">
                <button onclick="predict()" id="predictBtn">🔮 Predict</button>
                <button onclick="clearCanvas()">🗑️ Clear</button>
            </div>

            <div id="result"></div>
        </div>
    </div>

    <script>
        let session;
        let canvas, ctx;
        let isDrawing = false;
        let lastX, lastY;

        // Initialize on page load
        window.onload = async function() {
            await loadModel();
            setupCanvas();
        };

        async function loadModel() {
            try {
                console.log('Loading ONNX model...');

                // Try to use WebGPU first, fall back to WASM
                const providers = ['webgpu', 'wasm'];

                session = await ort.InferenceSession.create('mnist_cnn.onnx', {
                    executionProviders: providers
                });

                console.log('✓ Model loaded successfully!');
                console.log('Provider:', session.handler._ep);

                // Hide loading, show app
                document.getElementById('loading').style.display = 'none';
                document.getElementById('app').style.display = 'block';

            } catch (error) {
                console.error('Error loading model:', error);
                document.getElementById('loading').innerHTML =
                    '<p style="color: red;">❌ Error loading model. Make sure mnist_cnn.onnx is in the same directory.</p>';
            }
        }

        function setupCanvas() {
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');

            // Fill with white background
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, 280, 280);

            // Drawing settings
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 20;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            // Mouse events
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);

            // Touch events for mobile
            canvas.addEventListener('touchstart', handleTouch);
            canvas.addEventListener('touchmove', handleTouch);
            canvas.addEventListener('touchend', stopDrawing);
        }

        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
        }

        function draw(e) {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();

            lastX = x;
            lastY = y;
        }

        function stopDrawing() {
            isDrawing = false;
        }

        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }

        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, 280, 280);
            document.getElementById('result').innerHTML = '';
        }

        async function predict() {
            const predictBtn = document.getElementById('predictBtn');
            predictBtn.disabled = true;
            predictBtn.textContent = '⏳ Predicting...';

            try {
                // Get image data from canvas
                const imageData = ctx.getImageData(0, 0, 280, 280);

                // Downsample to 28x28
                const resized = downsampleImage(imageData.data, 280, 280, 28, 28);

                // Prepare input tensor: (1, 1, 28, 28)
                const input = new Float32Array(1 * 1 * 28 * 28);

                for (let i = 0; i < 28 * 28; i++) {
                    // Invert colors (canvas has black on white, model expects white on black)
                    // and normalize to [0, 1]
                    input[i] = 1.0 - (resized[i * 4] / 255.0);
                }

                // Create ONNX tensor
                const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);

                // Run inference
                const startTime = performance.now();
                const results = await session.run({ x: tensor });
                const inferenceTime = performance.now() - startTime;

                // Get predictions
                const predictions = results.output.data;

                // Find predicted digit
                let maxIdx = 0;
                let maxVal = predictions[0];
                for (let i = 1; i < 10; i++) {
                    if (predictions[i] > maxVal) {
                        maxVal = predictions[i];
                        maxIdx = i;
                    }
                }

                // Display result
                displayResult(maxIdx, maxVal, predictions, inferenceTime);

            } catch (error) {
                console.error('Prediction error:', error);
                document.getElementById('result').innerHTML =
                    '<p style="color: red;">❌ Prediction failed. See console for details.</p>';
            } finally {
                predictBtn.disabled = false;
                predictBtn.textContent = '🔮 Predict';
            }
        }

        function displayResult(digit, confidence, allPredictions, inferenceTime) {
            let html = `
                <div class="digit">${digit}</div>
                <div class="confidence">Confidence: ${(confidence * 100).toFixed(1)}%</div>
                <div class="confidence">Inference time: ${inferenceTime.toFixed(1)}ms</div>

                <div class="probabilities">
            `;

            for (let i = 0; i < 10; i++) {
                const prob = allPredictions[i];
                const height = (prob * 100);
                html += `
                    <div class="prob-bar">
                        <div class="prob-label">${i}</div>
                        <div class="prob-value">
                            <div class="prob-fill" style="height: ${height}%"></div>
                        </div>
                        <div class="prob-text">${(prob * 100).toFixed(1)}%</div>
                    </div>
                `;
            }

            html += '</div>';

            document.getElementById('result').innerHTML = html;
        }

        function downsampleImage(imageData, srcWidth, srcHeight, dstWidth, dstHeight) {
            const result = new Uint8Array(dstWidth * dstHeight * 4);
            const xRatio = srcWidth / dstWidth;
            const yRatio = srcHeight / dstHeight;

            for (let y = 0; y < dstHeight; y++) {
                for (let x = 0; x < dstWidth; x++) {
                    const srcX = Math.floor(x * xRatio);
                    const srcY = Math.floor(y * yRatio);
                    const srcIdx = (srcY * srcWidth + srcX) * 4;
                    const dstIdx = (y * dstWidth + x) * 4;

                    result[dstIdx] = imageData[srcIdx];         // R
                    result[dstIdx + 1] = imageData[srcIdx + 1]; // G
                    result[dstIdx + 2] = imageData[srcIdx + 2]; // B
                    result[dstIdx + 3] = imageData[srcIdx + 3]; // A
                }
            }

            return result;
        }
    </script>
</body>
</html>
```

### Run the Demo

```bash
# Make sure mnist_cnn.onnx is in the same directory as mnist_demo.html

# Serve the files
python -m http.server 8000

# Open browser
# Navigate to: http://localhost:8000/mnist_demo.html

# Draw a digit and click Predict!
```

---

## Part 5: Complete Python Script

**`mnist_complete.py`** - All-in-one script:

```python
"""
Complete MNIST CNN Pipeline: Build → Train → Export → Test
"""
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.nnet import conv2d


def main():
    print("="*70)
    print("MNIST CNN: PyTensor → ONNX Pipeline")
    print("="*70)

    # ============================================================
    # Step 1: Build Model
    # ============================================================
    print("\n[1/5] Building CNN model...")

    x = pt.tensor4('x', dtype='float32')

    # Conv1: 1→32
    W1 = shared(np.random.randn(32, 1, 3, 3).astype('float32') * 0.1, name='W1')
    b1 = shared(np.zeros(32, dtype='float32'), name='b1')
    conv1 = conv2d(x, W1, border_mode='valid', filter_flip=False)
    conv1 = pt.maximum(conv1 + b1.dimshuffle('x', 0, 'x', 'x'), 0)

    # Conv2: 32→64
    W2 = shared(np.random.randn(64, 32, 3, 3).astype('float32') * 0.1, name='W2')
    b2 = shared(np.zeros(64, dtype='float32'), name='b2')
    conv2 = conv2d(conv1, W2, border_mode='valid', filter_flip=False)
    conv2 = pt.maximum(conv2 + b2.dimshuffle('x', 0, 'x', 'x'), 0)

    # Flatten + Dense
    flat = conv2.flatten(2)
    W3 = shared(np.random.randn(64*24*24, 128).astype('float32') * 0.01, name='W3')
    b3 = shared(np.zeros(128, dtype='float32'), name='b3')
    fc1 = pt.maximum(pt.dot(flat, W3) + b3, 0)

    # Output
    W4 = shared(np.random.randn(128, 10).astype('float32') * 0.01, name='W4')
    b4 = shared(np.zeros(10, dtype='float32'), name='b4')
    predictions = pt.nnet.softmax(pt.dot(fc1, W4) + b4)

    print("✓ Model built (using random weights for demo)")

    # ============================================================
    # Step 2: Compile Inference Function
    # ============================================================
    print("\n[2/5] Compiling inference function...")
    inference_fn = pytensor.function([x], predictions)
    print("✓ Function compiled")

    # ============================================================
    # Step 3: Test with Random Input
    # ============================================================
    print("\n[3/5] Testing with random input...")
    test_input = np.random.randn(1, 1, 28, 28).astype('float32')
    pytensor_output = inference_fn(test_input)
    print(f"✓ Output shape: {pytensor_output.shape}")
    print(f"  Predicted digit: {np.argmax(pytensor_output[0])}")
    print(f"  Confidence: {np.max(pytensor_output[0]):.2%}")

    # ============================================================
    # Step 4: Export to ONNX
    # ============================================================
    print("\n[4/5] Exporting to ONNX...")
    from pytensor.link.onnx import export_onnx

    export_onnx(inference_fn, 'mnist_cnn.onnx')
    print("✓ Exported to: mnist_cnn.onnx")

    # ============================================================
    # Step 5: Verify with ONNX Runtime
    # ============================================================
    print("\n[5/5] Verifying with ONNX Runtime...")
    import onnxruntime as ort

    session = ort.InferenceSession('mnist_cnn.onnx',
                                    providers=['CPUExecutionProvider'])
    onnx_output = session.run(None, {'x': test_input})[0]

    print(f"✓ ONNX output shape: {onnx_output.shape}")
    print(f"  ONNX predicted digit: {np.argmax(onnx_output[0])}")

    # Compare
    matches = np.allclose(pytensor_output, onnx_output, rtol=1e-4)
    max_diff = np.abs(pytensor_output - onnx_output).max()

    print(f"\nVerification:")
    print(f"  Outputs match: {'✓ YES' if matches else '✗ NO'}")
    print(f"  Max difference: {max_diff:.2e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("✓ COMPLETE! Model ready for deployment")
    print("="*70)
    print("\nYour model (mnist_cnn.onnx) can now run on:")
    print("  🌐 Browsers (via ONNX Runtime Web)")
    print("  📱 Mobile devices (iOS/Android)")
    print("  🔌 Edge devices (IoT, embedded systems)")
    print("  ☁️  Cloud servers (any platform)")
    print("\nNext steps:")
    print("  1. Open mnist_demo.html in a browser to test")
    print("  2. Or integrate into your application")
    print("  3. Enjoy GPU-accelerated inference! 🚀")


if __name__ == '__main__':
    main()
```

**Run it:**
```bash
python mnist_complete.py
```

---

## Deployment Options

### Browser (WebAssembly + WebGPU)

✅ **Advantages:**
- No server needed
- GPU acceleration via WebGPU
- Privacy (client-side inference)
- Low latency

📦 **Requirements:**
- ONNX Runtime Web
- Modern browser (Chrome 113+)

### Mobile (iOS/Android)

✅ **Advantages:**
- Native performance
- Offline capability
- Low power consumption

📦 **Requirements:**
- ONNX Runtime Mobile
- Xcode/Android Studio

### Edge Devices

✅ **Advantages:**
- Real-time inference
- Low latency
- Privacy

📦 **Requirements:**
- ONNX Runtime for embedded
- Compatible hardware

---

## Troubleshooting

### ONNX Export Issues

**Error: `NotImplementedError: No ONNX conversion available for: AbstractConv2d`**

**Solution:** Make sure you've implemented the Conv2D converter (see TDD plan).

### Browser Demo Issues

**Model not loading**

**Solution:**
- Check that `mnist_cnn.onnx` is in the same directory
- Check browser console for errors
- Try serving with `python -m http.server`

**WebGPU not available**

**Solution:**
- Use Chrome 113+ or Edge 113+
- Falls back to WebAssembly automatically

### Performance Issues

**Slow inference**

**Solution:**
- Check if WebGPU is being used (see browser console)
- Try reducing model size
- Ensure ONNX Runtime Web is loaded correctly

---

## Next Steps

1. **Train on real MNIST data** - Uncomment training code
2. **Improve accuracy** - Add more layers, tune hyperparameters
3. **Deploy to production** - Integrate into your application
4. **Optimize model** - Quantization, pruning, etc.
5. **Scale up** - Try larger datasets (CIFAR-10, ImageNet)

---

## Resources

- **PyTensor Documentation:** https://pytensor.readthedocs.io/
- **ONNX Documentation:** https://onnx.ai/
- **ONNX Runtime Web:** https://onnxruntime.ai/docs/tutorials/web/
- **WebGPU:** https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API

---

## License

This example is provided as-is for educational purposes.

**Happy coding! 🚀**
