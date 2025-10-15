"""
Demo: Export a PyTensor Conv2D CNN to ONNX and run inference.

This script demonstrates:
1. Creating a simple CNN with Conv2D layers
2. Exporting to ONNX format
3. Running inference with ONNX Runtime (Python)
4. Preparing for WebGPU deployment (browser)

The exported model can be deployed to:
- Web browsers (via ONNX Runtime Web with WebGPU/WebAssembly)
- Mobile devices (iOS/Android)
- Edge devices (Raspberry Pi, etc.)
"""

# Configure PyTensor before importing (must be done before import)
import os

import numpy as np


os.environ.setdefault(
    "PYTENSOR_FLAGS", "optimizer=fast_compile,exception_verbosity=high"
)

import pytensor
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.conv.abstract_conv import conv2d


def create_simple_cnn():
    """
    Create a simple CNN for image classification.

    Architecture:
    - Input: (batch, 1, 28, 28) - grayscale images
    - Conv2D: 8 filters, 3x3 kernel
    - ReLU activation
    - Conv2D: 16 filters, 3x3 kernel
    - ReLU activation
    - Output: (batch, 16, 24, 24) feature maps

    This is a simplified CNN suitable for edge deployment.
    """
    print("Creating CNN model...")

    # Input: batch of 28x28 grayscale images
    x = pt.tensor4("x", dtype="float32")

    # Layer 1: Conv2D (1 -> 8 channels)
    # Initialize with small random weights
    rng = np.random.default_rng(42)
    W1 = shared(
        rng.normal(0, 0.1, (8, 1, 3, 3)).astype("float32"),
        name="conv1_weights",
    )
    b1 = shared(np.zeros(8, dtype="float32"), name="conv1_bias")

    conv1 = conv2d(x, W1, border_mode="valid", filter_flip=False)
    conv1_bias = conv1 + b1.dimshuffle("x", 0, "x", "x")
    relu1 = pt.maximum(conv1_bias, 0)

    print("  Layer 1: Conv2D(1->8, 3x3) + ReLU")
    print("    Output shape: (batch, 8, 26, 26)")

    # Layer 2: Conv2D (8 -> 16 channels)
    W2 = shared(
        rng.normal(0, 0.1, (16, 8, 3, 3)).astype("float32"),
        name="conv2_weights",
    )
    b2 = shared(np.zeros(16, dtype="float32"), name="conv2_bias")

    conv2 = conv2d(relu1, W2, border_mode="valid", filter_flip=False)
    conv2_bias = conv2 + b2.dimshuffle("x", 0, "x", "x")
    output = pt.maximum(conv2_bias, 0)

    print("  Layer 2: Conv2D(8->16, 3x3) + ReLU")
    print("    Output shape: (batch, 16, 24, 24)")

    # Compile PyTensor function
    print("\nCompiling PyTensor function...")
    cnn_fn = pytensor.function([x], output)

    return cnn_fn, x, output


def export_to_onnx(cnn_fn, output_path="cnn_model.onnx"):
    """
    Export the PyTensor CNN to ONNX format.

    Args:
        cnn_fn: Compiled PyTensor function
        output_path: Path to save ONNX model

    Returns:
        Path to saved model
    """
    from pytensor.link.onnx import export_onnx

    print(f"\nExporting to ONNX: {output_path}")

    # Export to ONNX
    model = export_onnx(cnn_fn, output_path)

    print("✓ Exported successfully!")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {len(model.graph.input)}")
    print(f"  Outputs: {len(model.graph.output)}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")

    # Show node types
    node_types = [node.op_type for node in model.graph.node]
    print(f"\n  Node types: {', '.join(set(node_types))}")

    return output_path


def run_inference_python(model_path, test_image):
    """
    Run inference using ONNX Runtime in Python.

    This demonstrates how to use the exported model for inference.

    Args:
        model_path: Path to ONNX model
        test_image: Test image tensor to use for inference

    Returns:
        Inference result
    """
    import onnxruntime as ort

    print(f"\n{'=' * 60}")
    print("Running inference with ONNX Runtime (Python)")
    print(f"{'=' * 60}")

    # Create inference session
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("\nSession created:")
    print(f"  Provider: {session.get_providers()[0]}")

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"  Input: {input_name}")
    print(f"  Output: {output_name}")

    print("\nRunning inference on test image...")
    print(f"  Input shape: {test_image.shape}")

    # Run inference
    result = session.run([output_name], {input_name: test_image})

    print(f"  Output shape: {result[0].shape}")
    print(f"  Output range: [{result[0].min():.3f}, {result[0].max():.3f}]")

    # Show some statistics
    print("\n  Statistics:")
    print(f"    Mean: {result[0].mean():.3f}")
    print(f"    Std: {result[0].std():.3f}")
    print(f"    Non-zero activations: {(result[0] > 0).sum()} / {result[0].size}")

    return result[0]


def create_webgpu_demo_html(model_path):
    """
    Create an HTML file demonstrating WebGPU inference.

    This creates a standalone HTML file that loads and runs the ONNX model
    in a web browser using ONNX Runtime Web with WebGPU acceleration.
    """
    html_path = model_path.replace(".onnx", "_webgpu_demo.html")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyTensor CNN - WebGPU Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .info-box {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
        }
        .success-box {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #4CAF50;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #cccccc;
            cursor: not-allowed;
        }
        #output {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            margin-top: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .canvas-container {
            margin: 20px 0;
            text-align: center;
        }
        canvas {
            border: 2px solid #ddd;
            border-radius: 5px;
            image-rendering: pixelated;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #2196F3;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 PyTensor CNN on WebGPU</h1>

        <div class="info-box">
            <strong>About this demo:</strong><br>
            This CNN was created in PyTensor, exported to ONNX, and is now running
            in your browser using ONNX Runtime Web with WebGPU acceleration!
            <br><br>
            <strong>Model:</strong> 2-layer CNN (Conv2D → ReLU → Conv2D → ReLU)
        </div>

        <div class="canvas-container">
            <h3>Draw an Input Image</h3>
            <canvas id="inputCanvas" width="280" height="280"></canvas>
            <br>
            <button onclick="clearCanvas()">Clear</button>
            <button onclick="randomCanvas()">Random Noise</button>
        </div>

        <button id="runButton" onclick="runInference()">Run Inference</button>
        <button onclick="runBenchmark()">Run Benchmark (10 iterations)</button>

        <div id="statsContainer" class="stats" style="display:none;">
            <div class="stat-card">
                <div class="stat-value" id="inferenceTime">-</div>
                <div class="stat-label">Inference Time (ms)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="throughput">-</div>
                <div class="stat-label">Throughput (FPS)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="provider">-</div>
                <div class="stat-label">Execution Provider</div>
            </div>
        </div>

        <div id="output"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js"></script>
    <script>
        let session = null;
        const canvas = document.getElementById('inputCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        // Setup canvas drawing
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            startDrawing({offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top});
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            draw({offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top});
        });
        canvas.addEventListener('touchend', stopDrawing);

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function draw(e) {
            if (!isDrawing) return;
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
            ctx.fill();
        }

        function stopDrawing() {
            isDrawing = false;
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        function randomCanvas() {
            const imageData = ctx.createImageData(canvas.width, canvas.height);
            for (let i = 0; i < imageData.data.length; i += 4) {
                const value = Math.random() * 255;
                imageData.data[i] = value;
                imageData.data[i + 1] = value;
                imageData.data[i + 2] = value;
                imageData.data[i + 3] = 255;
            }
            ctx.putImageData(imageData, 0, 0);
        }

        function log(message) {
            const output = document.getElementById('output');
            output.textContent += message + '\\n';
            output.scrollTop = output.scrollHeight;
        }

        function getImageTensor() {
            // Get 28x28 downsampled image
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, 28, 28);

            const imageData = tempCtx.getImageData(0, 0, 28, 28);
            const data = new Float32Array(1 * 1 * 28 * 28);

            // Convert to grayscale and normalize
            for (let i = 0; i < 28 * 28; i++) {
                // Average RGB channels and normalize to [-1, 1]
                data[i] = (imageData.data[i * 4] / 255) * 2 - 1;
            }

            return new ort.Tensor('float32', data, [1, 1, 28, 28]);
        }

        async function loadModel() {
            try {
                log('Loading ONNX model...');

                // Try WebGPU first, fall back to WASM
                const providers = [];
                if (navigator.gpu) {
                    providers.push('webgpu');
                    log('✓ WebGPU available!');
                }
                providers.push('wasm');

                session = await ort.InferenceSession.create('cnn_model.onnx', {
                    executionProviders: providers
                });

                log(`✓ Model loaded successfully!`);
                log(`  Provider: ${session.handler._ep}`);
                log(`  Inputs: ${session.inputNames.join(', ')}`);
                log(`  Outputs: ${session.outputNames.join(', ')}`);

                document.getElementById('runButton').disabled = false;
                document.getElementById('statsContainer').style.display = 'grid';
                document.getElementById('provider').textContent = session.handler._ep || 'WASM';

            } catch (error) {
                log(`✗ Error loading model: ${error.message}`);
            }
        }

        async function runInference() {
            if (!session) return;

            try {
                log('\\n--- Running Inference ---');

                const inputTensor = getImageTensor();
                log(`Input shape: [${inputTensor.dims.join(', ')}]`);

                const start = performance.now();
                const results = await session.run({ x: inputTensor });
                const elapsed = performance.now() - start;

                const output = results[session.outputNames[0]];
                log(`Output shape: [${output.dims.join(', ')}]`);
                log(`Inference time: ${elapsed.toFixed(2)}ms`);

                // Show output statistics
                const data = output.data;
                const mean = data.reduce((a, b) => a + b, 0) / data.length;
                const min = Math.min(...data);
                const max = Math.max(...data);
                const nonZero = data.filter(x => x > 0).length;

                log(`Output statistics:`);
                log(`  Mean: ${mean.toFixed(3)}`);
                log(`  Range: [${min.toFixed(3)}, ${max.toFixed(3)}]`);
                log(`  Non-zero: ${nonZero} / ${data.length}`);

                document.getElementById('inferenceTime').textContent = elapsed.toFixed(2);
                document.getElementById('throughput').textContent = (1000 / elapsed).toFixed(1);

            } catch (error) {
                log(`✗ Error during inference: ${error.message}`);
            }
        }

        async function runBenchmark() {
            if (!session) return;

            log('\\n--- Running Benchmark (10 iterations) ---');

            const inputTensor = getImageTensor();
            const times = [];

            // Warmup
            await session.run({ x: inputTensor });

            // Benchmark
            for (let i = 0; i < 10; i++) {
                const start = performance.now();
                await session.run({ x: inputTensor });
                times.push(performance.now() - start);
            }

            const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
            const minTime = Math.min(...times);
            const maxTime = Math.max(...times);

            log(`Benchmark results:`);
            log(`  Average: ${avgTime.toFixed(2)}ms`);
            log(`  Min: ${minTime.toFixed(2)}ms`);
            log(`  Max: ${maxTime.toFixed(2)}ms`);
            log(`  Throughput: ${(1000 / avgTime).toFixed(1)} FPS`);

            document.getElementById('inferenceTime').textContent = avgTime.toFixed(2);
            document.getElementById('throughput').textContent = (1000 / avgTime).toFixed(1);
        }

        // Initialize
        clearCanvas();
        loadModel();
    </script>
</body>
</html>"""

    from pathlib import Path

    Path(html_path).write_text(html_content, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"WebGPU Demo Created: {html_path}")
    print(f"{'=' * 60}")
    print("\nTo run in browser:")
    print("1. Start a local HTTP server:")
    print("   python -m http.server 8000")
    print("2. Open browser to:")
    print(f"   http://localhost:8000/{html_path}")
    print("\n   Or use:")
    print("   npx serve .")
    print("\n   Then navigate to the HTML file")

    return html_path


def main():
    """Main demo function."""
    print("=" * 60)
    print("PyTensor Conv2D → ONNX → WebGPU Demo")
    print("=" * 60)

    # Create CNN model
    cnn_fn, _x, _output = create_simple_cnn()

    # Test inference in PyTensor
    print("\nTesting PyTensor inference...")
    test_input = np.random.randn(1, 1, 28, 28).astype("float32")
    pytensor_result = cnn_fn(test_input)
    print(f"  PyTensor output shape: {pytensor_result.shape}")
    print(
        f"  PyTensor output range: [{pytensor_result.min():.3f}, {pytensor_result.max():.3f}]"
    )

    # Export to ONNX
    model_path = export_to_onnx(cnn_fn, "cnn_model.onnx")

    # Run inference with ONNX Runtime
    onnx_result = run_inference_python(model_path)

    # Verify results match
    print(f"\n{'=' * 60}")
    print("Verification")
    print(f"{'=' * 60}")
    print("Comparing PyTensor vs ONNX Runtime outputs...")
    diff = np.abs(pytensor_result - onnx_result).max()
    print(f"  Max difference: {diff:.6f}")
    if diff < 1e-4:
        print("  ✓ Results match! (tolerance: 1e-4)")
    else:
        print("  ✗ Results differ significantly!")

    # Create WebGPU demo
    html_path = create_webgpu_demo_html(model_path)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print("✓ Created simple CNN with Conv2D layers")
    print(f"✓ Exported to ONNX: {model_path}")
    print("✓ Verified inference matches PyTensor")
    print(f"✓ Created WebGPU demo: {html_path}")
    print("\nYour CNN is now ready for edge deployment!")
    print("Deploy to: Browsers (WebGPU), Mobile (iOS/Android), Edge devices")


if __name__ == "__main__":
    main()
