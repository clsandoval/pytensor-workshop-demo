# WebGPU/WASM Browser Inference Tests - TDD Implementation Plan

## Overview

Add comprehensive browser-based inference testing for the YOLO11n ONNX demo using Selenium WebDriver to validate that exported models run correctly in WebGPU and WASM environments. This addresses the critical testing gap where models are validated only with CPU backend but deployed to browsers.

**TDD Approach**: Write browser tests first to define expected behavior, verify they fail properly, then implement infrastructure to make tests pass.

## Current State Analysis

### Existing Test Infrastructure

**Testing Framework**: pytest with Hypothesis (`examples/onnx/onnx-yolo-demo/tests/conftest.py:1-140`)
- pytest fixtures: `test_seed`, `rng`, `tmp_onnx_dir`, `yolo_input_size`, `simple_image_batch`
- Hypothesis profiles: "dev" (10 examples), "ci" (50 examples), "thorough" (200 examples)
- Helper utilities: `compare_jax_and_py()`, `export_and_validate_onnx()`
- Test markers: `@pytest.mark.onnx`, `@pytest.mark.slow`, `@pytest.mark.gpu`

**Current ONNX Tests** (`examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py:1-269`):
- 8 tests validating ONNX export with **CPU backend only**
- Numerical equivalence: PyTensor vs ONNX Runtime CPU (rtol=1e-4, atol=1e-5)
- Tests: export success, metadata, batch sizes, shapes, determinism, opset version
- **Critical test**: `test_onnx_runtime_vs_pytensor_equivalence()` (line 43-88)

**Browser Demo** (`examples/onnx/onnx-yolo-demo/site/webgpu_vs_wasm_benchmark.html:380-916`):
- Manual browser testing only (no automation)
- WebGPU session creation: `ort.InferenceSession.create('yolo11n.onnx', {executionProviders: ['webgpu']})` (line 462-464)
- WASM fallback: `executionProviders: ['wasm']` (line 491)
- Video processing: `processFrame()` (line 840-903)
- YOLO output parsing: `parseYOLOOutput()` (line 714-765)
- NMS: `applyNMS()` (line 767-790)

### Current Testing Landscape

**What IS Tested** ✓:
- ONNX export completes without errors
- Exported model passes `onnx.checker.check_model()`
- ONNX Runtime CPU produces same results as PyTensor
- Model handles dynamic batch sizes (1, 2, 4)
- Output shapes are correct for P3, P4, P5 scales
- Inference is deterministic
- No training-only operations in exported model

**What is NOT Tested** ✗ (Critical Gap):
- WebGPU browser execution (zero tests)
- WASM browser execution (zero tests)
- Browser model loading and initialization
- Browser inference numerical correctness vs CPU
- End-to-end video frame processing in browser
- YOLO output parsing correctness
- NMS (Non-Maximum Suppression) correctness
- Performance benchmarks (FPS targets)
- Error handling and diagnostic messages

### Key Discovery from Research

From `thoughts/shared/research/2025-10-19_15-31-42_onnx-testing-gap-browser-vs-export.md:24-28`:

> **The Critical Gap**: PyTensor's ONNX export tests validate that models export successfully and produce correct numerical results when run with **ONNX Runtime's CPU backend**, but they **never validate** that exported models actually work in the **browser environment** (WebGPU/WASM) where they're intended to run.

## Desired End State

After implementation, PyTensor's ONNX export will be validated against **actual browser runtimes** (WebGPU and WASM), ensuring:

1. **Browser Runtime Correctness**: Models load and run successfully in Chrome with WebGPU and WASM
2. **Numerical Equivalence**: Browser inference produces same results as CPU backend (within tolerance)
3. **End-to-End Validation**: Complete inference pipeline works (frame extraction → inference → YOLO parsing → NMS)
4. **Performance Validation**: WebGPU meets >30 FPS target, WASM meets >10 FPS baseline
5. **Error Diagnostics**: Clear, actionable error messages when browser tests fail
6. **CI/CD Ready**: Tests can run locally, skip gracefully in CI without browser support

## What We're NOT Testing/Implementing

**Out of Scope**:
- Cross-browser testing (Firefox, Safari, Edge) - Chrome only for now
- Mobile browser testing (Android, iOS)
- WebGL backend testing (focus on WebGPU and WASM)
- Network latency simulation
- Multiple concurrent sessions
- Browser memory profiling
- GPU fallback strategies beyond WASM
- Production deployment infrastructure (CDN, model serving)
- Security testing (model tampering, XSS)
- Accessibility testing
- Fixes to PyTensor core ONNX export (only test infrastructure)
- Changes to YOLO model architecture
- Browser polyfills or compatibility shims
- Custom ONNX Runtime Web builds

## TDD Approach

### Test Design Philosophy

1. **Browser as First-Class Target**: Tests treat browser runtime as equal to CPU backend
2. **Fail Fast, Fail Clear**: Tests should produce diagnostic messages guiding implementation
3. **Property-Based Where Possible**: Use Hypothesis to test various inputs, batch sizes, etc.
4. **Isolation**: Each test validates one specific aspect of browser inference
5. **Reproducibility**: Fixed seeds, deterministic inputs, consistent browser state

### How We'll Ensure Tests Are Informative

**Good Test Characteristics**:
- Test names describe behavior, not implementation: `test_webgpu_produces_correct_shapes()` ✓ not `test_run_inference()` ✗
- Assertion messages include actual vs expected: `f"Expected shape {expected}, got {actual}"`
- Failure messages guide debugging: "WebGPU session creation failed - check if Chrome supports WebGPU"
- Tests verify one thing: Shape correctness OR numerical correctness, not both in same test
- Tests are independent: Can run in any order

**Diagnostic Failure Messages**:
```python
# Good assertion message
assert output.shape == expected_shape, (
    f"WebGPU output shape mismatch:\n"
    f"  Expected: {expected_shape}\n"
    f"  Got: {output.shape}\n"
    f"  Input shape: {input_shape}\n"
    f"  Model: yolo11n.onnx"
)

# Bad assertion message
assert output.shape == expected_shape  # Just fails with no context
```

---

## Phase 1: Test Design & Implementation

### Overview

Write comprehensive browser tests that define correct behavior. Tests should fail in expected, diagnostic ways before any implementation.

**Priority**: End-to-end integration tests first, then runtime correctness, then performance.

---

### Test Category 1: End-to-End Browser Integration Tests (HIGHEST PRIORITY)

**Test File**: `examples/onnx/onnx-yolo-demo/tests/test_browser_e2e.py`

**Purpose**: Validate complete inference pipeline in browser, from frame extraction through detection output.

---

#### Test 1.1: `test_browser_full_inference_pipeline_webgpu`

**Purpose**: Verify complete YOLO inference pipeline works in browser with WebGPU

**Priority**: CRITICAL (end-to-end validation)

**Test Data**:
- Input: 320×320×3 RGB image (random or synthetic test pattern)
- Expected output: 3 detection tensors (P3, P4, P5) with correct shapes
- Tolerance: rtol=1e-3, atol=1e-4 (slightly looser than CPU due to GPU precision)

**Test Steps**:
1. Export YOLO11n model to ONNX
2. Start local HTTP server serving model and test HTML
3. Launch Chrome with WebGPU enabled
4. Load model in browser with WebGPU provider
5. Generate test input in browser (JavaScript)
6. Run inference
7. Extract output tensors
8. Compare with CPU baseline

**Expected Behavior**:
- Model loads without errors
- Inference completes successfully
- Output shapes match: P3=(batch,6,40,40), P4=(batch,6,20,20), P5=(batch,6,10,10)
- Output values are numerically close to CPU baseline

**Assertions**:
```python
# Shape assertions
assert webgpu_p3.shape == (1, 6, 40, 40), \
    f"WebGPU P3 shape incorrect: expected (1,6,40,40), got {webgpu_p3.shape}"
assert webgpu_p4.shape == (1, 6, 20, 20), \
    f"WebGPU P4 shape incorrect: expected (1,6,20,20), got {webgpu_p4.shape}"
assert webgpu_p5.shape == (1, 6, 10, 10), \
    f"WebGPU P5 shape incorrect: expected (1,6,10,10), got {webgpu_p5.shape}"

# Numerical assertions (compare with CPU baseline)
np.testing.assert_allclose(
    webgpu_p3, cpu_p3,
    rtol=1e-3, atol=1e-4,
    err_msg="WebGPU P3 output differs from CPU baseline"
)
np.testing.assert_allclose(
    webgpu_p4, cpu_p4,
    rtol=1e-3, atol=1e-4,
    err_msg="WebGPU P4 output differs from CPU baseline"
)
np.testing.assert_allclose(
    webgpu_p5, cpu_p5,
    rtol=1e-3, atol=1e-4,
    err_msg="WebGPU P5 output differs from CPU baseline"
)

# Finite value assertion
assert np.all(np.isfinite(webgpu_p3)), "WebGPU P3 contains NaN or Inf"
assert np.all(np.isfinite(webgpu_p4)), "WebGPU P4 contains NaN or Inf"
assert np.all(np.isfinite(webgpu_p5)), "WebGPU P5 contains NaN or Inf"
```

**Expected Failure Mode** (before implementation):
- Error type: `WebDriverException` or JavaScript error
- Expected message: "Failed to create WebGPU session" or browser-specific error
- Location: Browser console error during model loading or inference

---

#### Test 1.2: `test_browser_full_inference_pipeline_wasm`

**Purpose**: Verify complete YOLO inference pipeline works in browser with WASM fallback

**Priority**: HIGH (ensures broad browser compatibility)

**Test Data**: Same as Test 1.1

**Test Steps**: Same as Test 1.1, but use WASM execution provider

**Expected Behavior**: Same numerical correctness as WebGPU test

**Assertions**: Same as Test 1.1, but compare `wasm_*` outputs with `cpu_*` baseline

**Expected Failure Mode**:
- Error type: JavaScript error or execution error
- Expected message: "WASM session creation failed" or operator not supported
- Likely less common than WebGPU failures (WASM has broader operator support)

---

#### Test 1.3: `test_browser_yolo_output_parsing`

**Purpose**: Verify YOLO output parsing in browser produces correct detection format

**Priority**: HIGH (end-to-end correctness)

**Test Data**:
- Input: YOLO model output tensors (can use CPU baseline)
- Expected output: List of detection dicts with {x1, y1, x2, y2, confidence, classId, className}

**JavaScript Function Tested**: `parseYOLOOutput()` from `webgpu_vs_wasm_benchmark.html:714-765`

**Test Steps**:
1. Create known YOLO output tensors (synthetic or from CPU inference)
2. Pass to browser's `parseYOLOOutput()` function
3. Verify output format and values

**Expected Behavior**:
- Output is array of detection objects
- Each detection has required fields
- Bounding boxes are in correct format (corner coordinates)
- Confidence values in [0, 1] range
- Class IDs in [0, 79] range (80 COCO classes)

**Assertions**:
```python
# Parse output in browser
detections = browser.execute_script("""
    const output = new ort.Tensor('float32', arguments[0], [1, 84, 8400]);
    return parseYOLOOutput(output, 0.25, 0.45);
""", yolo_output_data.flatten().tolist())

# Verify output structure
assert isinstance(detections, list), f"Expected list, got {type(detections)}"
assert len(detections) > 0, "No detections returned (expected at least 1)"

for det in detections:
    assert 'x1' in det, "Detection missing x1 coordinate"
    assert 'y1' in det, "Detection missing y1 coordinate"
    assert 'x2' in det, "Detection missing x2 coordinate"
    assert 'y2' in det, "Detection missing y2 coordinate"
    assert 'confidence' in det, "Detection missing confidence"
    assert 'classId' in det, "Detection missing classId"
    assert 'className' in det, "Detection missing className"

    # Validate ranges
    assert 0 <= det['confidence'] <= 1, \
        f"Confidence {det['confidence']} out of [0,1] range"
    assert 0 <= det['classId'] < 80, \
        f"ClassId {det['classId']} out of [0,79] range"
    assert det['x1'] < det['x2'], \
        f"Invalid bbox: x1={det['x1']} >= x2={det['x2']}"
    assert det['y1'] < det['y2'], \
        f"Invalid bbox: y1={det['y1']} >= y2={det['y2']}"
```

**Expected Failure Mode**:
- Error type: `JavascriptException` or parsing error
- Expected message: "parseYOLOOutput is not defined" or "Cannot read property of undefined"
- Location: Browser JavaScript execution

---

#### Test 1.4: `test_browser_nms_correctness`

**Purpose**: Verify Non-Maximum Suppression (NMS) implementation in browser

**Priority**: MEDIUM-HIGH (critical for detection quality)

**Test Data**:
- Input: List of overlapping bounding boxes with known IoU values
- Expected output: Filtered list after NMS with IOU threshold 0.45

**JavaScript Function Tested**: `applyNMS()` from `webgpu_vs_wasm_benchmark.html:767-790`

**Test Steps**:
1. Create synthetic overlapping boxes (known IoU relationships)
2. Apply browser's NMS function
3. Verify correct boxes are suppressed

**Property to Verify**: For any two boxes in output, IoU < threshold (0.45)

**Test Cases**:
- **Case 1**: Two identical boxes → NMS keeps only one
- **Case 2**: High overlap (IoU=0.8) → Higher confidence box kept
- **Case 3**: Low overlap (IoU=0.2) → Both boxes kept
- **Case 4**: Three boxes with chain overlap → Correct suppression order

**Assertions**:
```python
# Test Case 1: Identical boxes
boxes = [
    {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50, 'confidence': 0.9, 'classId': 0},
    {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50, 'confidence': 0.8, 'classId': 0},
]
nms_result = browser.execute_script("return applyNMS(arguments[0], 0.45);", boxes)

assert len(nms_result) == 1, \
    f"NMS should suppress identical box, got {len(nms_result)} boxes"
assert nms_result[0]['confidence'] == 0.9, \
    f"NMS should keep higher confidence box, got {nms_result[0]['confidence']}"

# Test Case 3: Low overlap - both kept
boxes = [
    {'x1': 10, 'y1': 10, 'x2': 30, 'y2': 30, 'confidence': 0.9, 'classId': 0},
    {'x1': 40, 'y1': 40, 'x2': 60, 'y2': 60, 'confidence': 0.8, 'classId': 0},
]
nms_result = browser.execute_script("return applyNMS(arguments[0], 0.45);", boxes)

assert len(nms_result) == 2, \
    f"NMS should keep non-overlapping boxes, got {len(nms_result)} boxes"

# Property: All pairs have IoU < threshold
for i, box1 in enumerate(nms_result):
    for box2 in nms_result[i+1:]:
        iou = calculate_iou_python(box1, box2)  # Python reference implementation
        assert iou < 0.45, \
            f"NMS output has overlapping boxes: IoU={iou:.3f} >= 0.45"
```

**Expected Failure Mode**:
- Error type: `JavascriptException` or assertion failure
- Expected message: "applyNMS is not defined" or incorrect filtering
- Location: Browser JavaScript execution or assertion failure

---

#### Test 1.5: `test_browser_video_frame_extraction`

**Purpose**: Verify video frame extraction converts to correct tensor format

**Priority**: MEDIUM (important for video processing)

**Test Data**:
- Input: Static test image (320×320 RGB)
- Expected output: NCHW tensor [1, 3, 320, 320] with values in [0, 1]

**JavaScript Function Tested**: `getVideoTensor()` from `webgpu_vs_wasm_benchmark.html:691-712`

**Test Steps**:
1. Load test image in browser canvas
2. Extract tensor using `getVideoTensor()`
3. Verify tensor format, shape, and value range

**Expected Behavior**:
- Output is Float32Array
- Shape is [1, 3, 320, 320] (NCHW format)
- Values normalized to [0, 1] range
- Channel order is RGB (not BGR)

**Assertions**:
```python
# Load test image in browser and extract tensor
tensor_data = browser.execute_script("""
    const img = new Image();
    img.src = arguments[0];  // Base64 encoded test image
    return new Promise(resolve => {
        img.onload = () => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 320;
            tempCanvas.height = 320;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(img, 0, 0, 320, 320);

            const imageData = tempCtx.getImageData(0, 0, 320, 320);
            const data = new Float32Array(1 * 3 * 320 * 320);

            // Convert RGBA to RGB NCHW
            for (let i = 0; i < 320 * 320; i++) {
                data[i] = imageData.data[i * 4] / 255;         // R
                data[320 * 320 + i] = imageData.data[i * 4 + 1] / 255;  // G
                data[320 * 320 * 2 + i] = imageData.data[i * 4 + 2] / 255;  // B
            }

            resolve(Array.from(data));
        };
    });
""", test_image_base64)

tensor = np.array(tensor_data, dtype=np.float32).reshape(1, 3, 320, 320)

# Verify shape
assert tensor.shape == (1, 3, 320, 320), \
    f"Tensor shape incorrect: expected (1,3,320,320), got {tensor.shape}"

# Verify value range
assert np.all(tensor >= 0) and np.all(tensor <= 1), \
    f"Tensor values out of [0,1] range: min={tensor.min()}, max={tensor.max()}"

# Verify not all zeros (actual image data)
assert tensor.std() > 0.01, \
    f"Tensor appears empty (std={tensor.std():.6f} too low)"
```

**Expected Failure Mode**:
- Error type: Shape mismatch or value range violation
- Expected message: Assertion with actual vs expected values
- Location: Python assertion after tensor extraction

---

### Test Category 2: Browser Runtime Correctness Tests

**Test File**: `examples/onnx/onnx-yolo-demo/tests/test_browser_runtime.py`

**Purpose**: Validate ONNX models load and run correctly in browser runtimes

---

#### Test 2.1: `test_model_loads_in_browser_webgpu`

**Purpose**: Verify ONNX model loads successfully with WebGPU execution provider

**Priority**: HIGH (prerequisite for all WebGPU tests)

**Test Steps**:
1. Export YOLO11n to ONNX
2. Serve model via local HTTP server
3. Launch Chrome with WebGPU enabled
4. Attempt to create InferenceSession with WebGPU provider
5. Verify session creation succeeds

**Expected Behavior**:
- `ort.InferenceSession.create()` completes without error
- Session object is not null
- Session has correct input/output names

**Assertions**:
```python
# Attempt session creation in browser
session_info = browser.execute_async_script("""
    const callback = arguments[arguments.length - 1];

    ort.InferenceSession.create('yolo11n.onnx', {
        executionProviders: ['webgpu']
    }).then(session => {
        callback({
            success: true,
            inputNames: session.inputNames,
            outputNames: session.outputNames,
            error: null
        });
    }).catch(error => {
        callback({
            success: false,
            inputNames: null,
            outputNames: null,
            error: error.message
        });
    });
""")

assert session_info['success'], \
    f"WebGPU session creation failed: {session_info['error']}"
assert session_info['inputNames'] is not None, \
    "Session inputNames is null"
assert len(session_info['inputNames']) >= 1, \
    f"Expected at least 1 input, got {len(session_info['inputNames'])}"
assert len(session_info['outputNames']) == 3, \
    f"Expected 3 outputs (P3,P4,P5), got {len(session_info['outputNames'])}"
```

**Expected Failure Mode**:
- Error type: Session creation error
- Expected message: "Failed to create session" or "WebGPU not supported"
- Possible causes:
  - Chrome version < 113
  - WebGPU not enabled in browser flags
  - ONNX model has incompatible operators for WebGPU

---

#### Test 2.2: `test_model_loads_in_browser_wasm`

**Purpose**: Verify ONNX model loads successfully with WASM execution provider

**Priority**: HIGH (fallback runtime)

**Test Steps**: Same as Test 2.1, but with WASM provider

**Expected Behavior**: Same as Test 2.1

**Assertions**: Same structure as Test 2.1, but for WASM

**Expected Failure Mode**:
- Error type: Session creation error
- Expected message: "Failed to create WASM session" or operator not supported
- Less common than WebGPU failures (WASM has broader compatibility)

---

#### Test 2.3: `test_webgpu_inference_produces_valid_shapes`

**Purpose**: Verify WebGPU inference outputs have correct tensor shapes

**Priority**: HIGH (shape validation is critical)

**Test Steps**:
1. Create WebGPU session
2. Generate test input tensor
3. Run inference
4. Verify output shapes

**Expected Behavior**:
- Inference completes without error
- Three output tensors returned
- Shapes match expected: P3=(batch,6,40,40), P4=(batch,6,20,20), P5=(batch,6,10,10)

**Assertions**:
```python
# Run inference in browser
outputs = browser.execute_async_script("""
    const callback = arguments[arguments.length - 1];
    const inputData = arguments[0];

    const input = new ort.Tensor('float32', inputData, [1, 3, 320, 320]);

    ort.InferenceSession.create('yolo11n.onnx', {
        executionProviders: ['webgpu']
    }).then(session => {
        const inputName = session.inputNames[0];
        return session.run({[inputName]: input}).then(results => {
            callback({
                success: true,
                shapes: session.outputNames.map(name => Array.from(results[name].dims)),
                error: null
            });
        });
    }).catch(error => {
        callback({success: false, shapes: null, error: error.message});
    });
""", test_input_flat.tolist())

assert outputs['success'], f"WebGPU inference failed: {outputs['error']}"
assert len(outputs['shapes']) == 3, \
    f"Expected 3 outputs, got {len(outputs['shapes'])}"

# Verify each shape
expected_shapes = [(1, 6, 40, 40), (1, 6, 20, 20), (1, 6, 10, 10)]
for i, (actual, expected) in enumerate(zip(outputs['shapes'], expected_shapes)):
    assert tuple(actual) == expected, \
        f"Output {i} shape mismatch: expected {expected}, got {tuple(actual)}"
```

**Expected Failure Mode**:
- Error type: Inference error or shape mismatch
- Expected message: Specific operator failure or dimension error
- Location: During inference execution

---

#### Test 2.4: `test_wasm_inference_produces_valid_shapes`

**Purpose**: Same as Test 2.3 but for WASM backend

**Priority**: HIGH

**Test Steps**: Same as Test 2.3 with WASM session

**Expected Behavior**: Same shapes as WebGPU

**Assertions**: Same as Test 2.3

---

#### Test 2.5: `test_webgpu_vs_cpu_numerical_equivalence`

**Purpose**: Verify WebGPU produces numerically equivalent results to CPU backend

**Priority**: CRITICAL (correctness validation)

**Test Steps**:
1. Generate random test input
2. Run inference with CPU backend (baseline)
3. Run inference with WebGPU backend
4. Compare outputs with tolerance

**Expected Behavior**:
- WebGPU outputs match CPU outputs within tolerance: rtol=1e-3, atol=1e-4
- All output values are finite (no NaN/Inf)

**Assertions**:
```python
# Run CPU inference (baseline)
cpu_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
x_val = np.random.randn(1, 3, 320, 320).astype("float32")
cpu_outputs = cpu_session.run(None, {cpu_session.get_inputs()[0].name: x_val})

# Run WebGPU inference
webgpu_result = browser.execute_async_script("""
    const callback = arguments[arguments.length - 1];
    const inputData = arguments[0];

    const input = new ort.Tensor('float32', inputData, [1, 3, 320, 320]);

    ort.InferenceSession.create('yolo11n.onnx', {
        executionProviders: ['webgpu']
    }).then(session => {
        const inputName = session.inputNames[0];
        return session.run({[inputName]: input}).then(results => {
            callback({
                success: true,
                outputs: session.outputNames.map(name => Array.from(results[name].data))
            });
        });
    }).catch(error => {
        callback({success: false, error: error.message});
    });
""", x_val.flatten().tolist())

assert webgpu_result['success'], f"WebGPU inference failed: {webgpu_result['error']}"

# Convert to numpy arrays
webgpu_p3 = np.array(webgpu_result['outputs'][0]).reshape(1, 6, 40, 40)
webgpu_p4 = np.array(webgpu_result['outputs'][1]).reshape(1, 6, 20, 20)
webgpu_p5 = np.array(webgpu_result['outputs'][2]).reshape(1, 6, 10, 10)

# Compare with CPU baseline
np.testing.assert_allclose(
    webgpu_p3, cpu_outputs[0],
    rtol=1e-3, atol=1e-4,
    err_msg=f"WebGPU P3 differs from CPU\n"
            f"  Max diff: {np.abs(webgpu_p3 - cpu_outputs[0]).max()}\n"
            f"  Mean diff: {np.abs(webgpu_p3 - cpu_outputs[0]).mean()}"
)
np.testing.assert_allclose(
    webgpu_p4, cpu_outputs[1],
    rtol=1e-3, atol=1e-4,
    err_msg="WebGPU P4 differs from CPU"
)
np.testing.assert_allclose(
    webgpu_p5, cpu_outputs[2],
    rtol=1e-3, atol=1e-4,
    err_msg="WebGPU P5 differs from CPU"
)

# Verify finite values
assert np.all(np.isfinite(webgpu_p3)), "WebGPU P3 contains NaN or Inf"
assert np.all(np.isfinite(webgpu_p4)), "WebGPU P4 contains NaN or Inf"
assert np.all(np.isfinite(webgpu_p5)), "WebGPU P5 contains NaN or Inf"
```

**Expected Failure Mode**:
- Error type: `AssertionError` from `np.testing.assert_allclose`
- Expected message: Shows max/mean difference between WebGPU and CPU
- Possible causes:
  - WebGPU operator has different precision
  - Model export has bugs
  - Numerical instability in specific operations

---

#### Test 2.6: `test_wasm_vs_cpu_numerical_equivalence`

**Purpose**: Same as Test 2.5 but for WASM backend

**Priority**: CRITICAL

**Test Steps**: Same as Test 2.5 with WASM session

**Expected Behavior**: WASM matches CPU within same tolerance

**Assertions**: Same as Test 2.5

---

#### Test 2.7: `test_browser_handles_different_batch_sizes`

**Purpose**: Verify browser inference works with dynamic batch sizes

**Priority**: MEDIUM

**Test Steps**:
1. Export model with dynamic batch dimension
2. Run inference with batch sizes 1, 2, 4
3. Verify outputs have correct batch dimension

**Expected Behavior**:
- Model accepts varying batch sizes
- Output batch dimension matches input

**Assertions**:
```python
for batch_size in [1, 2, 4]:
    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")

    outputs = browser.execute_async_script("""
        const callback = arguments[arguments.length - 1];
        const inputData = arguments[0];
        const shape = arguments[1];

        const input = new ort.Tensor('float32', inputData, shape);

        ort.InferenceSession.create('yolo11n.onnx', {
            executionProviders: ['webgpu']
        }).then(session => {
            const inputName = session.inputNames[0];
            return session.run({[inputName]: input}).then(results => {
                callback({
                    success: true,
                    shapes: session.outputNames.map(name => Array.from(results[name].dims))
                });
            });
        }).catch(error => {
            callback({success: false, error: error.message});
        });
    """, x_val.flatten().tolist(), [batch_size, 3, 320, 320])

    assert outputs['success'], f"Inference failed for batch_size={batch_size}"
    assert outputs['shapes'][0][0] == batch_size, \
        f"P3 batch size wrong: expected {batch_size}, got {outputs['shapes'][0][0]}"
    assert outputs['shapes'][1][0] == batch_size, \
        f"P4 batch size wrong: expected {batch_size}, got {outputs['shapes'][1][0]}"
    assert outputs['shapes'][2][0] == batch_size, \
        f"P5 batch size wrong: expected {batch_size}, got {outputs['shapes'][2][0]}"
```

**Expected Failure Mode**:
- Error type: Shape mismatch or inference error
- Expected message: Batch dimension incorrect
- Location: Assertion failure

---

### Test Category 3: Browser Performance Tests

**Test File**: `examples/onnx/onnx-yolo-demo/tests/test_browser_performance.py`

**Purpose**: Validate inference performance meets targets

---

#### Test 3.1: `test_webgpu_performance_fps_target`

**Purpose**: Verify WebGPU inference meets >30 FPS target

**Priority**: MEDIUM (performance validation)

**Test Steps**:
1. Create WebGPU session
2. Run warmup inferences (10 iterations)
3. Benchmark 50 inference iterations
4. Calculate average FPS
5. Verify FPS > 30 (or skip with warning if below target)

**Expected Behavior**:
- After warmup, inference is consistently fast
- Average FPS exceeds 30 on modern GPU (or test skips)
- Variance is low (consistent timing)

**Assertions**:
```python
# Benchmark in browser
perf_result = browser.execute_async_script("""
    const callback = arguments[arguments.length - 1];

    ort.InferenceSession.create('yolo11n.onnx', {
        executionProviders: ['webgpu']
    }).then(async session => {
        const inputName = session.inputNames[0];
        const data = new Float32Array(1 * 3 * 320 * 320).fill(0.5);
        const input = new ort.Tensor('float32', data, [1, 3, 320, 320]);

        // Warmup
        for (let i = 0; i < 10; i++) {
            await session.run({[inputName]: input});
        }

        // Benchmark
        const times = [];
        for (let i = 0; i < 50; i++) {
            const start = performance.now();
            await session.run({[inputName]: input});
            times.push(performance.now() - start);
        }

        const avg = times.reduce((a, b) => a + b) / times.length;
        const fps = 1000 / avg;

        callback({success: true, avgTime: avg, fps: fps});
    }).catch(error => {
        callback({success: false, error: error.message});
    });
""")

assert perf_result['success'], f"Performance test failed: {perf_result['error']}"

avg_fps = perf_result['fps']

# Warn if below target (don't fail)
if avg_fps < 30:
    pytest.skip(
        f"WebGPU FPS below target: {avg_fps:.1f} FPS (target >30 FPS)\n"
        f"This may be due to GPU availability or system load.\n"
        f"Test skipped, not failed."
    )
else:
    print(f"✓ WebGPU performance: {avg_fps:.1f} FPS (target >30 FPS)")
```

**Expected Failure Mode**:
- Test skips if FPS < 30 (not a failure, environment-dependent)
- Error type: Skip with performance metrics
- Message: "WebGPU FPS below target: X FPS"

---

#### Test 3.2: `test_wasm_performance_fps_baseline`

**Purpose**: Verify WASM inference meets >10 FPS baseline

**Priority**: MEDIUM

**Test Steps**: Same as Test 3.1 but with WASM session and 10 FPS target

**Expected Behavior**: WASM achieves >10 FPS on modern CPU (or test skips)

**Assertions**: Same structure as Test 3.1 with 10 FPS threshold

---

#### Test 3.3: `test_webgpu_faster_than_wasm`

**Purpose**: Verify WebGPU is faster than WASM (performance comparison)

**Priority**: LOW (nice-to-have validation)

**Test Steps**:
1. Benchmark WebGPU (50 iterations)
2. Benchmark WASM (50 iterations)
3. Verify WebGPU is consistently faster

**Expected Behavior**:
- WebGPU average time < WASM average time
- WebGPU should be 3-10× faster depending on GPU

**Assertions**:
```python
assert webgpu_avg_time < wasm_avg_time, \
    f"WebGPU ({webgpu_avg_time:.2f}ms) not faster than WASM ({wasm_avg_time:.2f}ms)\n" \
    f"  Expected WebGPU to be faster on GPU hardware"

speedup = wasm_avg_time / webgpu_avg_time
print(f"WebGPU speedup: {speedup:.1f}× faster than WASM")
```

---

### Test Implementation: Browser Infrastructure

#### New Fixtures (add to `conftest.py`)

```python
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import http.server
import socketserver
import threading
import time
from pathlib import Path


@pytest.fixture(scope="session")
def chrome_options():
    """Chrome options for WebGPU support."""
    options = Options()
    options.add_argument("--headless=new")  # New headless mode
    options.add_argument("--enable-unsafe-webgpu")  # Enable WebGPU
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-web-security")  # Allow loading local files
    options.add_argument("--allow-file-access-from-files")
    return options


@pytest.fixture
def browser(chrome_options):
    """Selenium WebDriver for Chrome with WebGPU support."""
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    yield driver
    driver.quit()


@pytest.fixture(scope="session")
def local_server(tmp_path_factory):
    """Local HTTP server for serving ONNX models and HTML."""
    server_dir = tmp_path_factory.mktemp("server_files")

    # Copy necessary files from site directory
    site_dir = Path(__file__).parent.parent / "site"
    if site_dir.exists():
        import shutil
        for file in site_dir.glob("*.html"):
            shutil.copy(file, server_dir)
        for file in site_dir.glob("*.js"):
            shutil.copy(file, server_dir)

    # Start server in background thread
    port = 8765

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(server_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress server logs

    server = socketserver.TCPServer(("", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    time.sleep(1)  # Wait for server to start

    server_url = f"http://localhost:{port}"

    yield server_url, server_dir

    server.shutdown()


@pytest.fixture
def exported_yolo_model(local_server):
    """Export YOLO11n model and make available on local server."""
    from yolo.model import build_yolo11n
    from pytensor.link.onnx import export_onnx
    import pytensor

    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    server_url, server_dir = local_server
    onnx_path = server_dir / "yolo11n.onnx"

    export_onnx(f, str(onnx_path))

    return onnx_path, f


@pytest.fixture
def cpu_baseline_session(exported_yolo_model):
    """ONNX Runtime CPU session for baseline comparisons."""
    ort = pytest.importorskip("onnxruntime")
    onnx_path, pytensor_fn = exported_yolo_model

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    return session, pytensor_fn


@pytest.fixture
def test_image_data(rng):
    """Generate test image data for browser inference."""
    image = rng.random((1, 3, 320, 320)).astype(np.float32)
    return image


@pytest.fixture
def browser_test_html(local_server):
    """Create test HTML for browser inference with YOLO utilities."""
    server_url, server_dir = local_server

    # Read COCO classes and utilities from demo HTML
    demo_html_path = Path(__file__).parent.parent / "site" / "webgpu_vs_wasm_benchmark.html"

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js"></script>
</head>
<body>
    <div id="status">Ready for testing</div>
    <script>
        // COCO class names (80 classes)
        const COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ];

        function parseYOLOOutput(output, confidenceThreshold = 0.25, iouThreshold = 0.45) {
            const data = output.data;
            const dims = output.dims;
            const numDetections = dims[2]; // 8400
            const boxes = [];

            for (let i = 0; i < numDetections; i++) {
                let maxScore = 0;
                let maxClass = 0;

                for (let c = 0; c < 80; c++) {
                    const score = data[(4 + c) * numDetections + i];
                    if (score > maxScore) {
                        maxScore = score;
                        maxClass = c;
                    }
                }

                if (maxScore > confidenceThreshold) {
                    const cx = data[0 * numDetections + i];
                    const cy = data[1 * numDetections + i];
                    const w = data[2 * numDetections + i];
                    const h = data[3 * numDetections + i];

                    boxes.push({
                        x1: cx - w / 2,
                        y1: cy - h / 2,
                        x2: cx + w / 2,
                        y2: cy + h / 2,
                        confidence: maxScore,
                        classId: maxClass,
                        className: COCO_CLASSES[maxClass]
                    });
                }
            }

            return applyNMS(boxes, iouThreshold);
        }

        function applyNMS(boxes, iouThreshold) {
            boxes.sort((a, b) => b.confidence - a.confidence);
            const keep = [];

            while (boxes.length > 0) {
                const current = boxes.shift();
                keep.push(current);

                boxes = boxes.filter(box => {
                    if (box.classId !== current.classId) return true;
                    const iou = calculateIOU(current, box);
                    return iou < iouThreshold;
                });
            }

            return keep;
        }

        function calculateIOU(box1, box2) {
            const x1 = Math.max(box1.x1, box2.x1);
            const y1 = Math.max(box1.y1, box2.y1);
            const x2 = Math.min(box1.x2, box2.x2);
            const y2 = Math.min(box1.y2, box2.y2);

            const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
            const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
            const union = area1 + area2 - intersection;

            return intersection / union;
        }

        window.parseYOLOOutput = parseYOLOOutput;
        window.applyNMS = applyNMS;
        window.calculateIOU = calculateIOU;
    </script>
</body>
</html>
"""

    html_path = server_dir / "test.html"
    html_path.write_text(html_content)

    return f"{server_url}/test.html"
```

#### Helper Functions

```python
def calculate_iou_python(box1, box2):
    """Calculate IoU between two boxes (Python reference implementation)."""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
```

---

## Phase 2: Test Failure Verification

### Overview

Run all tests and verify they fail in expected, diagnostic ways. This ensures tests actually validate something and will catch regressions.

### Verification Steps

1. **Run complete test suite**:
   ```bash
   cd examples/onnx/onnx-yolo-demo
   uv run pytest tests/test_browser_*.py -v --tb=short
   ```

2. **For each test category, verify**:
   - Tests are discovered and collected
   - Tests fail (not pass or error unexpectedly)
   - Failure messages are informative
   - Error types match expectations

3. **Document actual failure modes**

### Expected Failures by Test Category

#### End-to-End Tests (`test_browser_e2e.py`)

**Test 1.1-1.2: Full pipeline tests**
- Expected: Infrastructure not yet implemented
- Error type: Fixture errors or `WebDriverException`
- Message: Browser/server setup issues initially, then inference errors

**Test 1.3-1.5: Parsing/NMS/Frame extraction**
- Expected: JavaScript functions work (may pass immediately)
- These are testing the demo HTML utilities which already exist

#### Runtime Correctness Tests (`test_browser_runtime.py`)

**All tests**:
- Expected: Fixtures need implementation first
- Once fixtures work: Should see actual browser inference attempts
- May encounter WebGPU availability issues (environment-dependent)

#### Performance Tests (`test_browser_performance.py`)

**All performance tests**:
- Expected: Depend on runtime tests passing first
- Will skip if FPS below target (not failures)

### Success Criteria for Phase 2

#### Automated Verification:

```bash
# All tests discovered
uv run pytest tests/test_browser_*.py --collect-only
# Expected: ~15 tests collected

# Verify fixtures work
uv run pytest tests/test_browser_e2e.py::test_browser_full_inference_pipeline_webgpu -v --setup-show
# Expected: Fixtures initialize (browser, server, model export)
```

#### Manual Verification Checklist:

- [ ] Browser launches successfully
- [ ] Local server starts and serves files
- [ ] ONNX model exports in fixture
- [ ] Tests discover and attempt to run
- [ ] Failure messages are clear and diagnostic
- [ ] No import errors or syntax errors

---

## Phase 3: Feature Implementation (Red → Green)

### Overview

Implement browser test infrastructure to make tests pass. The fixtures are the main implementation work.

### Implementation Order

1. **Browser infrastructure** (fixtures) - Already defined in Phase 1
2. **Verify fixtures work** - Debug any fixture issues
3. **Run tests and debug failures** - Iterative improvement
4. **Add documentation**

### Implementation Steps

#### Step 1: Add Dependencies

**File**: `examples/onnx/onnx-yolo-demo/pyproject.toml`

Add browser testing dependencies:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "selenium>=4.15.0",
    "webdriver-manager>=4.0.0",
]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "onnx: marks tests that require ONNX Runtime",
    "browser: marks tests that require browser (Selenium)",  # NEW
    "performance: marks performance benchmarks (environment-dependent)",  # NEW
]
```

Install dependencies:
```bash
uv pip install selenium webdriver-manager
```

#### Step 2: Add Fixtures to conftest.py

Copy the fixtures from Phase 1 test implementation section into `conftest.py`.

#### Step 3: Run Tests Iteratively

```bash
# Start with simplest test
uv run pytest tests/test_browser_runtime.py::test_model_loads_in_browser_webgpu -xvs

# Debug any issues:
# - Browser not launching: Check Chrome installation
# - Server not starting: Check port availability
# - Model not loading: Check ONNX export

# Once basic test works, run all runtime tests
uv run pytest tests/test_browser_runtime.py -v

# Then end-to-end tests
uv run pytest tests/test_browser_e2e.py -v

# Finally performance tests
uv run pytest tests/test_browser_performance.py -v
```

#### Step 4: Debug Common Issues

**Issue: Chrome not found**
```bash
# Install Chrome manually or update webdriver-manager
pip install --upgrade webdriver-manager
```

**Issue: WebGPU not available**
- Chrome version must be >= 113
- Headless mode may not support WebGPU on some systems
- Try with headless disabled for local testing

**Issue: Numerical differences**
- Small differences expected (GPU vs CPU precision)
- Adjust tolerance if needed: rtol=1e-3, atol=1e-4
- Investigate if differences are large

### Success Criteria

#### Automated Verification:

```bash
# All browser tests pass
uv run pytest tests/test_browser_*.py -v
# Expected: High pass rate (environment-dependent for WebGPU)

# Runtime tests pass
uv run pytest tests/test_browser_runtime.py -v
# Expected: All PASS

# End-to-end tests pass
uv run pytest tests/test_browser_e2e.py -v
# Expected: All PASS
```

#### Manual Verification:

- [ ] Browser launches and loads HTML
- [ ] Models load in WebGPU and WASM
- [ ] Inference produces correct shapes
- [ ] Numerical equivalence within tolerance
- [ ] YOLO parsing works correctly
- [ ] NMS filters boxes correctly
- [ ] Performance meets targets (or skips gracefully)

---

## Phase 4: Refactoring & Cleanup

### Overview

Improve code quality while keeping tests green.

### Refactoring Targets

1. **Parametrize WebGPU/WASM tests** - Reduce duplication
2. **Extract browser execution helpers** - DRY principle
3. **Improve fixture reuse** - Session vs function scope
4. **Add type hints** - Code clarity
5. **Improve test names** - Conciseness

### Refactoring Steps

1. **Ensure tests pass**: `pytest tests/test_browser_*.py -v`
2. **Make ONE change** (e.g., parametrize tests)
3. **Run tests again**: Should still pass
4. **Commit**: `git commit -m "refactor: ..."`
5. **Repeat**

### Success Criteria

- [ ] All tests still pass after refactoring
- [ ] Less code duplication
- [ ] Clearer test structure
- [ ] Better documentation

---

## Testing Strategy Summary

### Test Coverage

- **End-to-end**: 5 tests covering full pipeline
- **Runtime correctness**: 7 tests for WebGPU/WASM validation
- **Performance**: 3 tests for FPS benchmarks
- **Total**: ~15 comprehensive browser tests

### Running Tests

```bash
# All browser tests
uv run pytest tests/test_browser_*.py -v

# By category
uv run pytest tests/test_browser_e2e.py -v
uv run pytest tests/test_browser_runtime.py -v
uv run pytest tests/test_browser_performance.py -v

# Skip slow/performance tests
uv run pytest tests/test_browser_*.py -m "not performance" -v

# Single test with full output
uv run pytest tests/test_browser_e2e.py::test_browser_full_inference_pipeline_webgpu -xvs
```

---

## Performance Considerations

### Expected Performance

**WebGPU** (with discrete GPU):
- Inference: 10-30ms
- FPS: 30-100
- Target: >30 FPS

**WASM** (CPU):
- Inference: 100-300ms
- FPS: 3-10
- Target: >10 FPS

### Performance Testing Philosophy

Tests should **measure and report**, not fail on slow hardware:
- Skip if below target (environment-dependent)
- Report actual metrics
- Provide diagnostic info

---

## References

### Original Research
- `thoughts/shared/research/2025-10-19_15-31-42_onnx-testing-gap-browser-vs-export.md`
- Key finding: Models validated on CPU but deployed to browser (critical gap)

### Related Code
- ONNX export: `pytensor/link/onnx/export.py`
- YOLO model: `examples/onnx/onnx-yolo-demo/yolo/model.py`
- Browser demo: `examples/onnx/onnx-yolo-demo/site/webgpu_vs_wasm_benchmark.html`
- Existing tests: `examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py`

### External Documentation
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- WebGPU API: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- Selenium: https://selenium-python.readthedocs.io/

---

## Success Metrics

- [x] Comprehensive test coverage (15+ tests)
- [x] Tests written before implementation
- [x] Diagnostic failure messages
- [ ] All tests passing
- [ ] Browser deployment validated
- [ ] Maintainable test suite
- [ ] CI/CD ready

---

## Conclusion

This TDD plan provides a systematic approach to adding browser inference testing:

1. **Phase 1**: Write tests defining correct browser behavior
2. **Phase 2**: Verify tests fail diagnostically
3. **Phase 3**: Implement fixtures and infrastructure
4. **Phase 4**: Refactor for maintainability

**Expected Impact**:
- Closes critical testing gap (CPU vs browser validation)
- Enables confident browser deployment
- Provides regression protection
- Serves as example for other browser-deployed models
