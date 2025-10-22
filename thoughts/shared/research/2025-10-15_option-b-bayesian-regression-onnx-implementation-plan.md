---
date: 2025-10-15T00:00:00-00:00
researcher: Claude (Sonnet 4.5)
git_commit: 4a64652942ffbef447d0ef180260333304ad5478
branch: onnx-workshop-demo
repository: pytensor-workshop-demo
topic: "Option B Implementation Plan: ADVI Bayesian Regression with Deterministic Uncertainty Export to ONNX"
tags: [research, implementation-plan, pymc, advi, onnx, bayesian-regression, uncertainty, option-b]
status: complete
last_updated: 2025-10-15
last_updated_by: Claude (Sonnet 4.5)
---

# Research: Option B Implementation Plan - ADVI Bayesian Regression with Deterministic Uncertainty Export to ONNX

**Date**: 2025-10-15T00:00:00-00:00
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: 4a64652942ffbef447d0ef180260333304ad5478
**Branch**: onnx-workshop-demo
**Repository**: pytensor-workshop-demo

## Research Question

**How do we implement Option B from the ADVI demo evaluation: A Bayesian regression model with ADVI that exports deterministic posterior predictive mean + uncertainty to ONNX for browser-based visualization?**

Specifically:
1. What is the complete implementation workflow?
2. What existing patterns can we reuse from MNIST/YOLO demos?
3. How do we visualize uncertainty (confidence intervals) in the browser?
4. What is the minimal viable implementation path?
5. What are the technical details and gotchas?

## Executive Summary

**Option B** is the **recommended starting point** for the ADVI PyMC demo because:
- ✅ **No new ops required** - Works with current ONNX backend (no random ops needed)
- ✅ **Fast implementation** - 2-3 days estimated effort
- ✅ **Reuses existing patterns** - Multi-output ONNX from YOLO, canvas visualization from MNIST
- ✅ **Shows uncertainty quantification** - Mean ± std demonstrates Bayesian inference value
- ✅ **Establishes PyMC→ONNX pipeline** - First probabilistic programming demo

**Key Insight**: Export **two deterministic outputs** (mean, std) as separate ONNX outputs, avoiding the need for random ops in the browser. All computations are deterministic functions of trained parameters.

**Recommended Model**: Simple Bayesian linear regression on toy data
- Training: PyMC ADVI (~30 seconds)
- Model size: <100KB ONNX
- Browser inference: <1ms per prediction
- Visualization: 2D plot with confidence bands

---

## Detailed Implementation Plan

### Phase 1: Training Pipeline (Python)

#### Step 1.1: Generate Toy Data
```python
import numpy as np

# Simple linear relationship with noise
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
true_alpha, true_beta, true_sigma = 1.0, 2.5, 0.5
y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, n)

# Save for browser visualization
np.savez('data/training_data.npz', x=x, y=y)
```

#### Step 1.2: Define PyMC Model with ADVI
```python
import pymc as pm

with pm.Model() as model:
    # Priors (weakly informative)
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Likelihood
    x_data = pm.MutableData('x', x)
    mu = alpha + beta * x_data
    y_obs = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

    # Run ADVI (Automatic Differentiation Variational Inference)
    print("Running ADVI...")
    approx = pm.fit(method='advi', n=10000, progressbar=True)

    # Extract variational parameters
    posterior_means = approx.mean.eval()
    posterior_stds = approx.std.eval()

    print(f"Posterior: alpha={posterior_means['alpha']:.3f} ± {posterior_stds['alpha']:.3f}")
    print(f"Posterior: beta={posterior_means['beta']:.3f} ± {posterior_stds['beta']:.3f}")
    print(f"Posterior: sigma={posterior_means['sigma']:.3f} ± {posterior_stds['sigma']:.3f}")
```

**Key API Points** (from PyMC integration research):
- `pm.fit(method='advi')` - Runs variational inference
- `approx.mean.eval()` - Extracts posterior means as Python dict
- `approx.std.eval()` - Extracts posterior standard deviations

#### Step 1.3: Create PyTensor Graph for Posterior Predictive
```python
import pytensor.tensor as pt

# Input variable for new predictions
x_new = pt.vector('x_new', dtype='float32')

# Bake posterior parameters as constants (deterministic)
alpha_mean = pt.constant(posterior_means['alpha'], dtype='float32')
beta_mean = pt.constant(posterior_means['beta'], dtype='float32')
sigma_mean = pt.constant(posterior_means['sigma'], dtype='float32')

# Posterior predictive mean: E[y|x] = alpha + beta * x
y_pred_mean = alpha_mean + beta_mean * x_new

# Posterior predictive std: sqrt(Var[y|x]) = sigma (aleatoric uncertainty)
# Note: This is simplified - only includes observational noise
# Full version would also include epistemic uncertainty from parameter uncertainty
y_pred_std = pt.ones_like(x_new, dtype='float32') * sigma_mean
```

**Key Operations Used**:
- `pt.constant()` - Embeds trained values into graph
- `pt.vector()` - Creates input placeholder
- `pt.ones_like()` - Creates tensor of ones with same shape
- Arithmetic ops (`+`, `*`) - All supported in ONNX backend

#### Step 1.4: Compile and Test in Python
```python
import pytensor

# Compile PyTensor function
predict_fn = pytensor.function([x_new], [y_pred_mean, y_pred_std])

# Test with sample inputs
x_test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float32')
mean, std = predict_fn(x_test)

print("\nTest predictions:")
for xi, mi, si in zip(x_test, mean, std):
    print(f"x={xi:.1f}: y={mi:.3f} ± {si:.3f} (95% CI: [{mi-2*si:.3f}, {mi+2*si:.3f}])")
```

#### Step 1.5: Export to ONNX
```python
from pytensor.link.onnx import export_onnx
from pathlib import Path

# Export with multi-output
onnx_path = Path("site/bayesian_regression.onnx")
model = export_onnx(predict_fn, str(onnx_path))

print(f"\nONNX export complete!")
print(f"  Model size: {onnx_path.stat().st_size / 1024:.1f} KB")
print(f"  Nodes: {len(model.graph.node)}")
print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
print(f"  Outputs: {[out.name for out in model.graph.output]}")
```

**Expected Output**:
- Model size: ~2-10 KB (tiny!)
- Nodes: ~8 (constant, mul, add, ones, mul for std)
- Inputs: `['x_new']`
- Outputs: `['y_pred_mean', 'y_pred_std']` or similar

#### Step 1.6: Verify with ONNX Runtime
```python
import onnxruntime as ort

session = ort.InferenceSession(str(onnx_path))

# Compare PyTensor vs ONNX Runtime
pytensor_mean, pytensor_std = predict_fn(x_test)
onnx_results = session.run(None, {'x_new': x_test})
onnx_mean, onnx_std = onnx_results[0], onnx_results[1]

# Check agreement
mean_diff = np.abs(pytensor_mean - onnx_mean).max()
std_diff = np.abs(pytensor_std - onnx_std).max()

print(f"\nVerification:")
print(f"  Max mean difference: {mean_diff:.2e}")
print(f"  Max std difference: {std_diff:.2e}")

if mean_diff < 1e-4 and std_diff < 1e-4:
    print("  ✓ ONNX export verified!")
else:
    print("  ⚠ Warning: Large differences detected")
```

**Pattern from existing demos** (`tests/link/onnx/test_basic.py:22-102`):
- Always verify ONNX output matches PyTensor
- Use `np.testing.assert_allclose()` for robust comparison

---

### Phase 2: Browser Demo (HTML/JavaScript)

#### Step 2.1: HTML Structure

**Adapted from MNIST demo** (`examples/onnx/onnx-mnist-demo/cnn_model_webgpu_demo.html`):

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bayesian Regression Demo - PyTensor + ONNX</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #667eea;
            margin: 0 0 10px 0;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .chart-container {
            position: relative;
            height: 400px;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #667eea;
            color: white;
        }
        button:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-label {
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
        }
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .console {
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎲 Bayesian Linear Regression with Uncertainty</h1>
            <div class="subtitle">
                PyMC ADVI → PyTensor → ONNX → Browser (WebGPU/WASM)
            </div>
        </div>

        <div class="info-box">
            <strong>Model:</strong> Bayesian linear regression (y = α + βx + ε)<br>
            <strong>Training:</strong> ADVI (Variational Inference) on 100 synthetic data points<br>
            <strong>Inference:</strong> Deterministic posterior mean ± std (uncertainty quantification)
        </div>

        <div class="main-content">
            <div>
                <h3>Predictions with Uncertainty</h3>
                <div class="chart-container">
                    <canvas id="regressionChart"></canvas>
                </div>
            </div>
            <div>
                <h3>Posterior Distributions</h3>
                <div class="chart-container">
                    <canvas id="posteriorChart"></canvas>
                </div>
            </div>
        </div>

        <div class="controls">
            <button id="runInferenceBtn" onclick="runInference()">🚀 Run Inference</button>
            <button id="benchmarkBtn" onclick="runBenchmark()">⏱️ Benchmark (100 iterations)</button>
            <button id="addPointBtn" onclick="enablePointAdding()">➕ Add Data Point</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Inference Time</div>
                <div class="stat-value" id="inferenceTime">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Execution Provider</div>
                <div class="stat-value" id="provider">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Predictions/sec</div>
                <div class="stat-value" id="throughput">-</div>
            </div>
        </div>

        <div class="console" id="console"></div>
    </div>

    <script>
        // JavaScript implementation in next section
    </script>
</body>
</html>
```

#### Step 2.2: JavaScript Implementation

```javascript
// Global variables
let session = null;
let providerName = 'Loading...';
let regressionChart = null;
let posteriorChart = null;
let trainingData = null;

// Logging utility
function log(message) {
    const consoleEl = document.getElementById('console');
    const timestamp = new Date().toLocaleTimeString();
    consoleEl.innerHTML += `[${timestamp}] ${message}\n`;
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

// Initialize ONNX Runtime Session
async function initializeModel() {
    try {
        log('Initializing ONNX Runtime...');

        // Try WebGPU first
        if (navigator.gpu) {
            log('✓ WebGPU detected! Attempting to use WebGPU...');
            try {
                session = await ort.InferenceSession.create('bayesian_regression.onnx', {
                    executionProviders: ['webgpu']
                });
                providerName = 'WebGPU';
                log('✓ WebGPU successfully initialized!');
            } catch (webgpuError) {
                log(`⚠ WebGPU failed: ${webgpuError.message}`);
                log('  Falling back to WASM...');
                session = await ort.InferenceSession.create('bayesian_regression.onnx', {
                    executionProviders: ['wasm']
                });
                providerName = 'WASM';
            }
        } else {
            log('ℹ WebGPU not available, using WASM');
            session = await ort.InferenceSession.create('bayesian_regression.onnx', {
                executionProviders: ['wasm']
            });
            providerName = 'WASM';
        }

        // Log model metadata
        log(`  Inputs: ${session.inputNames.join(', ')}`);
        log(`  Outputs: ${session.outputNames.join(', ')}`);

        document.getElementById('provider').textContent = providerName;

        // Warmup run
        log('Running warmup inference...');
        const warmupInput = new ort.Tensor('float32', new Float32Array([0, 5, 10]), [3]);
        await session.run({ [session.inputNames[0]]: warmupInput });
        log('✓ Model ready!');

        return true;
    } catch (error) {
        log(`❌ Error: ${error.message}`);
        return false;
    }
}

// Run inference
async function runInference() {
    if (!session) {
        log('❌ Model not loaded!');
        return;
    }

    try {
        log('\n--- Running Inference ---');

        // Create test inputs (x values from 0 to 10)
        const n = 50;
        const xValues = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            xValues[i] = (i / (n - 1)) * 10;  // 0 to 10
        }

        const inputTensor = new ort.Tensor('float32', xValues, [n]);
        log(`Input shape: [${inputTensor.dims.join(', ')}]`);

        // Run inference with timing
        const start = performance.now();
        const results = await session.run({
            [session.inputNames[0]]: inputTensor
        });
        const elapsed = performance.now() - start;

        // Extract outputs
        const meanOutput = results[session.outputNames[0]];
        const stdOutput = results[session.outputNames[1]];

        log(`Output shapes: [${meanOutput.dims.join(', ')}], [${stdOutput.dims.join(', ')}]`);
        log(`\n🎯 Inference completed in ${elapsed.toFixed(2)}ms`);

        // Update UI
        document.getElementById('inferenceTime').textContent = `${elapsed.toFixed(2)}ms`;
        document.getElementById('throughput').textContent = `${(1000 / elapsed).toFixed(0)}`;

        // Visualize results
        visualizeRegression(xValues, meanOutput.data, stdOutput.data);

        // Log sample predictions
        log('\nSample predictions:');
        for (let i = 0; i < Math.min(5, n); i++) {
            const x = xValues[i];
            const mean = meanOutput.data[i];
            const std = stdOutput.data[i];
            log(`  x=${x.toFixed(1)}: y=${mean.toFixed(3)} ± ${std.toFixed(3)} (95% CI: [${(mean-2*std).toFixed(3)}, ${(mean+2*std).toFixed(3)}])`);
        }

    } catch (error) {
        log(`❌ Inference error: ${error.message}`);
    }
}

// Visualize regression with uncertainty
function visualizeRegression(xData, meanData, stdData) {
    const ctx = document.getElementById('regressionChart');

    // Prepare datasets
    const meanDataset = Array.from(xData).map((x, i) => ({ x, y: meanData[i] }));
    const upperBound = Array.from(xData).map((x, i) => ({ x, y: meanData[i] + 2 * stdData[i] }));
    const lowerBound = Array.from(xData).map((x, i) => ({ x, y: meanData[i] - 2 * stdData[i] }));

    if (regressionChart) {
        regressionChart.destroy();
    }

    regressionChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Posterior Mean',
                    data: meanDataset,
                    borderColor: '#667eea',
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointRadius: 0,
                    order: 1
                },
                {
                    label: '95% Confidence Interval',
                    data: upperBound.concat(lowerBound.reverse()),
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    fill: true,
                    pointRadius: 0,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Bayesian Linear Regression Predictions'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                const idx = context.dataIndex;
                                const std = stdData[idx];
                                return `Mean: ${context.parsed.y.toFixed(3)} ± ${std.toFixed(3)}`;
                            }
                            return context.dataset.label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'x'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'y'
                    }
                }
            }
        }
    });
}

// Benchmark function
async function runBenchmark() {
    if (!session) {
        log('❌ Model not loaded!');
        return;
    }

    log('\n--- Running Benchmark (100 iterations) ---');

    const n = 50;
    const xValues = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        xValues[i] = (i / (n - 1)) * 10;
    }

    const inputTensor = new ort.Tensor('float32', xValues, [n]);
    const times = [];

    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        await session.run({ [session.inputNames[0]]: inputTensor });
        times.push(performance.now() - start);

        if (i % 10 === 0) {
            log(`  Completed ${i}/100 iterations...`);
        }
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);

    log(`\n📊 Benchmark Results:`);
    log(`  Average: ${avgTime.toFixed(2)}ms`);
    log(`  Min: ${minTime.toFixed(2)}ms`);
    log(`  Max: ${maxTime.toFixed(2)}ms`);
    log(`  Throughput: ${(1000 / avgTime).toFixed(0)} predictions/sec`);
}

// Initialize on page load
window.onload = async () => {
    log('🚀 Bayesian Regression Demo Starting...');
    const success = await initializeModel();

    if (success) {
        // Auto-run first inference
        await runInference();
    }
};
```

**Key Patterns Used** (from existing demos):
- **WebGPU detection and fallback** - From MNIST demo lines 327-348
- **Session creation pattern** - `ort.InferenceSession.create()` with execution providers
- **Warmup run** - From YOLO demo lines 430-435
- **Multi-output handling** - Access by output names from `session.outputNames`
- **Chart.js integration** - NEW: Using Chart.js for easier visualization than raw canvas

#### Step 2.3: Uncertainty Visualization Options

**Option A: Chart.js (Recommended - Easiest)**

Already shown above using Chart.js `fill` property for confidence bands.

**Option B: Raw Canvas 2D API** (If Chart.js not desired)

```javascript
function drawRegressionCanvas(xData, meanData, stdData) {
    const canvas = document.getElementById('regressionCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Scale data to canvas coordinates
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = Math.min(...meanData.map((m, i) => m - 2 * stdData[i]));
    const yMax = Math.max(...meanData.map((m, i) => m + 2 * stdData[i]));

    function scaleX(x) {
        return ((x - xMin) / (xMax - xMin)) * (width - 40) + 20;
    }

    function scaleY(y) {
        return height - (((y - yMin) / (yMax - yMin)) * (height - 40) + 20);
    }

    // Draw confidence band
    ctx.fillStyle = 'rgba(102, 126, 234, 0.2)';
    ctx.beginPath();
    // Upper bound
    for (let i = 0; i < xData.length; i++) {
        const x = scaleX(xData[i]);
        const y = scaleY(meanData[i] + 2 * stdData[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    // Lower bound (reverse)
    for (let i = xData.length - 1; i >= 0; i--) {
        const x = scaleX(xData[i]);
        const y = scaleY(meanData[i] - 2 * stdData[i]);
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();

    // Draw mean line
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i < xData.length; i++) {
        const x = scaleX(xData[i]);
        const y = scaleY(meanData[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, height - 20);
    ctx.lineTo(width - 20, height - 20);
    ctx.moveTo(20, 20);
    ctx.lineTo(20, height - 20);
    ctx.stroke();
}
```

**Pattern adapted from**: YOLO bounding box drawing (lines 767-799)

---

### Phase 3: Testing and Validation

#### Step 3.1: Unit Tests for PyTensor Graph

```python
import numpy as np
import pytest
import pytensor.tensor as pt
from pytensor import function

def test_posterior_predictive_graph():
    """Test that posterior predictive graph is correct."""
    # Create simple posterior parameters
    alpha = pt.constant(1.0, dtype='float32')
    beta = pt.constant(2.0, dtype='float32')
    sigma = pt.constant(0.5, dtype='float32')

    x = pt.vector('x', dtype='float32')
    y_mean = alpha + beta * x
    y_std = pt.ones_like(x) * sigma

    # Compile
    f = function([x], [y_mean, y_std])

    # Test
    x_test = np.array([0, 1, 2, 3, 4], dtype='float32')
    mean, std = f(x_test)

    # Verify
    expected_mean = 1.0 + 2.0 * x_test
    expected_std = np.full(5, 0.5, dtype='float32')

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(std, expected_std, rtol=1e-5)
```

#### Step 3.2: ONNX Export Verification

```python
def test_onnx_export():
    """Test ONNX export matches PyTensor."""
    import onnxruntime as ort
    from pytensor.link.onnx import export_onnx

    # Create model
    alpha = pt.constant(1.0, dtype='float32')
    beta = pt.constant(2.0, dtype='float32')
    sigma = pt.constant(0.5, dtype='float32')

    x = pt.vector('x', dtype='float32')
    y_mean = alpha + beta * x
    y_std = pt.ones_like(x) * sigma

    # Compile PyTensor
    pytensor_fn = function([x], [y_mean, y_std])

    # Export ONNX
    onnx_path = '/tmp/test_regression.onnx'
    export_onnx(pytensor_fn, onnx_path)

    # Load with ONNX Runtime
    session = ort.InferenceSession(onnx_path)

    # Compare outputs
    x_test = np.array([0, 1, 2, 3, 4], dtype='float32')

    pytensor_mean, pytensor_std = pytensor_fn(x_test)
    onnx_results = session.run(None, {session.inputNames[0]: x_test})
    onnx_mean, onnx_std = onnx_results[0], onnx_results[1]

    np.testing.assert_allclose(pytensor_mean, onnx_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(pytensor_std, onnx_std, rtol=1e-4, atol=1e-5)
```

**Pattern from**: `tests/link/onnx/test_basic.py:22-102`

#### Step 3.3: Browser Integration Test

```python
def test_browser_ready():
    """Test that exported model is browser-compatible."""
    import onnx

    # Load exported model
    model = onnx.load('site/bayesian_regression.onnx')

    # Check opset version (18 is widely supported)
    assert model.opset_import[0].version >= 14

    # Check inputs/outputs
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 2

    # Check input name is reasonable
    assert model.graph.input[0].name in ['x_new', 'x', 'input']

    # Check output names
    output_names = [out.name for out in model.graph.output]
    assert len(output_names) == 2

    print(f"✓ Model is browser-ready!")
    print(f"  Input: {model.graph.input[0].name}")
    print(f"  Outputs: {output_names}")
```

---

### Phase 4: Enhancement Ideas (Optional)

#### Enhancement 1: Interactive Data Addition

Allow users to click on canvas to add new data points and see posterior update:

```javascript
let userDataPoints = [];

function enablePointAdding() {
    const canvas = document.getElementById('regressionChart').canvas;
    canvas.style.cursor = 'crosshair';
    canvas.onclick = function(e) {
        const rect = canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 10;
        const y = ((rect.bottom - e.clientY) / rect.height) * 30;

        userDataPoints.push({x, y});
        log(`Added point: x=${x.toFixed(2)}, y=${y.toFixed(2)}`);

        // Re-visualize with new point
        updateVisualization();
    };
}
```

**Note**: This requires **online learning** or **re-running ADVI in browser** which is complex. Better to show as "what-if" visualization.

#### Enhancement 2: Prior/Posterior Comparison

Show posterior distributions for α, β, σ:

```javascript
function visualizePosteriorDistributions() {
    const ctx = document.getElementById('posteriorChart');

    // For this, we need to store posterior means/stds during export
    const posteriors = {
        alpha: { mean: 1.023, std: 0.101 },
        beta: { mean: 2.487, std: 0.016 },
        sigma: { mean: 0.512, std: 0.035 }
    };

    // Draw histograms or density plots
    // Use Chart.js bar chart with normal distribution samples
}
```

#### Enhancement 3: Multiple Models

Export multiple models (linear, quadratic, cubic) and allow switching:

```javascript
const models = {
    linear: 'bayesian_linear.onnx',
    quadratic: 'bayesian_quadratic.onnx',
    cubic: 'bayesian_cubic.onnx'
};

async function switchModel(modelType) {
    log(`Loading ${modelType} model...`);
    session = await ort.InferenceSession.create(models[modelType], {
        executionProviders: [providerName.toLowerCase()]
    });
    await runInference();
}
```

---

## Technical Details and Gotchas

### Detail 1: Float32 Configuration

**Critical**: PyTensor must be configured for float32 before any model building:

```python
import pytensor

if pytensor.config.floatX != "float32":
    raise RuntimeError(
        "PyTensor floatX must be 'float32' for ONNX export!\n"
        "Set: export PYTENSOR_FLAGS='floatX=float32'"
    )
```

**From**: `examples/onnx/onnx-yolo-demo/train.py:37-45`

### Detail 2: Multi-Output Naming

ONNX output names are auto-generated unless explicitly set:

```python
# PyTensor function outputs
predict_fn = function([x_new], [y_pred_mean, y_pred_std])

# ONNX outputs will be named like:
# - "output_0" or variable name if available
# - "output_1" or variable name if available

# Access in JavaScript:
const meanOutput = results[session.outputNames[0]];
const stdOutput = results[session.outputNames[1]];
```

### Detail 3: Shape Handling

Dynamic shapes are supported but named symbolically:

```python
x = pt.vector('x')  # Shape: [n] where n is symbolic

# In ONNX:
# - Input shape: ["unk__0"] or similar
# - Allows any length input at runtime
```

### Detail 4: Uncertainty Interpretation

The exported `y_pred_std` represents **aleatoric uncertainty** (observational noise):
- **Does NOT include epistemic uncertainty** (parameter uncertainty)
- Full uncertainty: `sqrt(sigma^2 + Var[f(x)])` where `Var[f(x)]` depends on parameter covariance

**Simplification for Option B**: Export only aleatoric uncertainty (constant `sigma`)

**Future work**: Export epistemic uncertainty requires:
1. Storing posterior covariance matrix
2. Computing `Var[alpha + beta*x]` = `Var[alpha] + x^2 * Var[beta] + 2*x*Cov[alpha, beta]`
3. Total std: `sqrt(sigma^2 + Var[alpha + beta*x])`

### Detail 5: Chart.js vs Raw Canvas

**Chart.js Pros**:
- ✅ Easy API for line charts with fill
- ✅ Built-in tooltips, legends, axes
- ✅ Responsive design
- ✅ Only ~70KB via CDN

**Raw Canvas Pros**:
- ✅ Full control over rendering
- ✅ No external dependency
- ✅ Can be faster for simple plots

**Recommendation**: Start with Chart.js, optimize later if needed.

### Detail 6: WebGPU Availability

**WebGPU support** (as of 2025):
- ✅ Chrome 113+
- ✅ Edge 113+
- ⚠️ Firefox: Behind flag
- ❌ Safari: Not yet

**Always provide WASM fallback** for compatibility.

---

## Code References

### Existing Infrastructure

**ONNX Export**:
- `pytensor/link/onnx/export.py:18-108` - Main `export_onnx()` function
- `pytensor/link/onnx/dispatch/basic.py:29-320` - Dispatcher and FunctionGraph converter
- `pytensor/link/onnx/dispatch/elemwise.py:17-289` - Elementwise ops (add, mul, etc.)

**Multi-Output Examples**:
- `examples/onnx/onnx-yolo-demo/train.py:519-531` - YOLO with 3 outputs
- `tests/link/onnx/test_basic.py:22-102` - Test pattern for multi-output verification

**Browser Demos**:
- `examples/onnx/onnx-mnist-demo/cnn_model_webgpu_demo.html:1-492` - Single-file demo template
- `examples/onnx/onnx-yolo-demo/site/webgpu_vs_wasm_benchmark.html:1-877` - Advanced visualization

**PyMC Integration**:
- `doc/gallery/applications/normalizing_flows_in_pytensor.ipynb:37-1002` - PyMC usage patterns
- `doc/gallery/applications/normalizing_flows_in_pytensor.ipynb:975-1002` - `pm.logp()` for inference graphs

**Research Foundation**:
- `thoughts/shared/research/2025-10-15_pymc-advi-onnx-demo-evaluation.md:1-833` - Feasibility study with Option B specification

---

## Architecture Insights

### PyMC → PyTensor → ONNX Pipeline

**Three-Stage Compilation**:

1. **PyMC Stage** (Training):
   - User defines probabilistic model with `pm.Model()`
   - PyMC translates to PyTensor random variable graph
   - ADVI optimizer updates variational parameters
   - Extract posterior means/stds as Python dicts

2. **PyTensor Stage** (Graph Construction):
   - Create new deterministic graph for inference
   - Bake trained parameters using `pt.constant()`
   - Define posterior predictive computation
   - Compile to PyTensor function

3. **ONNX Stage** (Export):
   - PyTensor function → FunctionGraph extraction
   - Traverse graph, dispatch each op to ONNX converter
   - Shared variables/constants → ONNX initializers
   - Save as `.onnx` file

**Key Insight**: Stages are **decoupled**. PyMC doesn't know about ONNX; PyTensor doesn't know about PyMC. The user bridges them by extracting parameters and building a new graph.

### Why Option B Works Without Random Ops

**Option B exports deterministic functions**:
```
f_mean(x) = alpha_mean + beta_mean * x
f_std(x) = sigma_mean * ones_like(x)
```

Both are **pure arithmetic ops** (add, mul, constant), all supported in ONNX backend.

**Option C would require sampling**:
```
f_samples(x) = [alpha_samples[i] + beta_samples[i] * x for i in range(100)]
```

This needs `RandomNormal` op in ONNX to generate `alpha_samples`, `beta_samples` - **not implemented yet**.

### Uncertainty Decomposition

**Total predictive uncertainty**:
```
Var[y_new | x_new, data] = E[Var[y|theta, x]] + Var[E[y|theta, x]]
                          = E[sigma^2]        + Var[alpha + beta*x]
                          = aleatoric         + epistemic
```

**Option B simplification**:
- Export only `E[sigma^2]` (aleatoric, observational noise)
- Ignore `Var[alpha + beta*x]` (epistemic, parameter uncertainty)
- **Justification**: For well-trained models with lots of data, epistemic uncertainty is small
- **Future enhancement**: Export both components (requires parameter covariance)

---

## Timeline and Effort Estimates

### Minimal Viable Implementation

**Total: 2-3 days**

| Task | Estimated Time | Details |
|------|---------------|---------|
| Setup project structure | 0.5 hours | Create directories, copy templates |
| Generate toy data | 0.5 hours | Simple linear data with noise |
| PyMC ADVI training | 1 hour | Model definition, ADVI fit, extract parameters |
| PyTensor graph construction | 1 hour | Build posterior predictive graph |
| ONNX export + verification | 1 hour | Export, test with ONNX Runtime |
| HTML structure | 2 hours | Adapt MNIST demo, add Chart.js |
| JavaScript implementation | 3 hours | Model loading, inference, visualization |
| Testing and debugging | 2 hours | Cross-browser testing, edge cases |
| Documentation | 1 hour | README, comments, usage guide |
| **Total** | **12-14 hours** | **~2 days** |

### With Enhancements

**Total: 4-5 days**

Additional tasks:
- Interactive data addition (4 hours)
- Posterior distribution plots (3 hours)
- Multiple model types (3 hours)
- Polished UI/UX (4 hours)
- Deployment setup (2 hours)

**Total additional**: ~16 hours (~2 days)

### Comparison to Option C

**Option C** (sampling-based, requires random ops):
- All of Option B work: 2-3 days
- Implement random ops in ONNX dispatch: 2-3 days
- Update demo for sampling: 1 day
- **Total: 5-7 days**

**Recommendation**: Start with Option B to validate pipeline, add Option C later if needed.

---

## Related Research

- `thoughts/shared/research/2025-10-15_pymc-advi-onnx-demo-evaluation.md` - Feasibility analysis comparing Options A, B, C
- `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md` - Backend implementation guide (if random ops needed later)
- `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md` - Testing strategy for new ops

---

## Open Questions

### Technical Questions

1. **Should we export parameter covariance for full uncertainty?**
   - Pros: More accurate uncertainty quantification
   - Cons: Larger model size, more complex computation
   - **Decision**: Start without, add as enhancement if needed

2. **Which chart library is best?**
   - Chart.js: Easy, 70KB
   - D3.js: Powerful, 100KB, steeper learning curve
   - Raw canvas: Lightweight, full control
   - **Decision**: Chart.js for MVP, can switch later

3. **How to handle model metadata (posterior parameters)?**
   - Bake into ONNX as constants (current approach)
   - Separate JSON file loaded by browser
   - Embed in HTML as JavaScript object
   - **Decision**: Bake into ONNX for simplicity

4. **Should we show training data in browser?**
   - Pros: Shows model fit, more educational
   - Cons: Larger file, exposes training data
   - **Decision**: Yes, save as small .npz file (~10KB for 100 points)

### Design Questions

1. **What toy data should we use?**
   - Linear with noise (simplest, most interpretable)
   - Nonlinear with linear approximation (more interesting)
   - Real dataset (more realistic, but complex)
   - **Recommendation**: Start with simple linear, document how to extend

2. **How to visualize uncertainty?**
   - Shaded confidence band (continuous, clear)
   - Error bars at sample points (discrete, traditional)
   - Multiple sample lines (spaghetti plot, shows variability)
   - **Recommendation**: Shaded band for MVP, add spaghetti option

3. **Interactive or static demo?**
   - Static: Just shows pre-trained model predictions
   - Interactive: Allows adding data points, comparing models
   - **Recommendation**: Static for MVP (simpler), add interactivity as Phase 4

### Strategic Questions

1. **Does this demo add value beyond YOLO?**
   - **Yes**: Different domain (probabilistic vs deterministic)
   - **Yes**: Shows uncertainty quantification (valuable for decision-making)
   - **Yes**: Smaller model (faster implementation, lighter deployment)
   - **Yes**: Educational (teaches Bayesian concepts)

2. **Should we implement Option C (sampling) eventually?**
   - If goal is to showcase PyTensor ONNX capabilities → **Yes** (demonstrates random ops)
   - If goal is practical demo for PyMC users → **Maybe** (Option B sufficient)
   - If goal is fast MVP → **No** (Option B is enough)
   - **Recommendation**: Build Option B first, evaluate response, then decide on Option C

3. **Who is the target audience?**
   - PyMC users wanting browser deployment → Show PyMC→ONNX workflow
   - ML engineers learning Bayesian methods → Educational demo with clear explanations
   - PyTensor developers → Technical showcase of ONNX backend
   - **Recommendation**: Design for PyMC users, document for all audiences

---

## Final Recommendations

### Implementation Strategy

1. **Build Option B first** (this document's plan)
   - 2-3 day implementation
   - No new ops required
   - Validates entire pipeline
   - Delivers working demo

2. **Evaluate and iterate**
   - Get feedback from users
   - Measure: Is uncertainty visualization valuable?
   - Measure: Do users want sampling (Option C)?
   - Measure: What enhancements are most requested?

3. **Consider Option C** if:
   - Users want full posterior predictive distributions
   - Showcasing random ops is valuable
   - 3-5 additional days is acceptable

### Success Criteria

**MVP (Option B) is successful if**:
- ✅ PyMC ADVI training completes (<1 minute)
- ✅ ONNX export succeeds (<100KB model)
- ✅ Browser inference runs (<5ms per prediction)
- ✅ Uncertainty visualization is clear and interpretable
- ✅ Demo works on WebGPU and WASM
- ✅ Documentation explains PyMC→PyTensor→ONNX workflow

**Documentation should include**:
- README with setup instructions
- Mathematical explanation of model
- Code comments explaining each step
- Discussion of uncertainty interpretation (aleatoric vs epistemic)
- Comparison to frequentist approach

### Next Steps

**Week 1**:
- Day 1: Project setup, toy data, PyMC training
- Day 2: PyTensor graph, ONNX export, verification
- Day 3: Browser demo, visualization, testing

**Week 2** (if needed):
- Polish UI/UX
- Add enhancements (interactivity, multiple models)
- Write documentation
- Deploy demo

**Future**:
- Implement Option C (random ops) if valuable
- Add more complex models (logistic regression, hierarchical models)
- Create tutorial series on Bayesian inference in browser

---

## Appendix: Complete File Structure

```
examples/onnx/onnx-pymc-demo/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies (pymc, pytensor, onnxruntime)
├── train.py                            # Training script (PyMC ADVI → ONNX export)
├── data/
│   └── training_data.npz              # Synthetic data (x, y) for visualization
├── site/
│   ├── index.html                     # Browser demo (single-file)
│   └── bayesian_regression.onnx       # Exported ONNX model
├── tests/
│   ├── test_model.py                  # Unit tests for PyTensor graph
│   └── test_onnx_export.py            # ONNX verification tests
└── infra/
    └── Dockerfile                      # Optional deployment
```

**Total project size**: ~50-100KB (tiny!)

---

## Conclusion

Option B provides a **fast, practical, and valuable** demonstration of PyMC→PyTensor→ONNX workflow for Bayesian inference in the browser. By exporting deterministic posterior predictive functions (mean + std), we avoid the need for random ops while still showcasing uncertainty quantification.

The implementation leverages existing patterns from MNIST and YOLO demos, requires no new ONNX ops, and can be completed in 2-3 days. This positions Option B as the **ideal MVP** for validating the PyMC→ONNX pipeline before considering more complex extensions like Option C (sampling-based uncertainty).

**Key Takeaway**: Start simple, validate quickly, iterate based on feedback.
