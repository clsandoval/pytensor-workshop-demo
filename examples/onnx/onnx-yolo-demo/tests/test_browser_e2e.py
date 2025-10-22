"""End-to-End Browser Integration Tests for YOLO11n ONNX Demo.

Tests complete inference pipeline in browser, from frame extraction through detection output.
"""

import numpy as np
import pytest


pytestmark = [pytest.mark.browser, pytest.mark.onnx]


@pytest.mark.slow
def test_browser_full_inference_pipeline_webgpu(
    browser,
    exported_yolo_model,
    browser_test_html,
    cpu_baseline_session,
    test_image_data,
):
    """Verify complete YOLO inference pipeline works in browser with WebGPU.

    Priority: CRITICAL (end-to-end validation)

    Tests:
    - Model loads without errors
    - Inference completes successfully
    - Output shapes match: P3=(batch,6,40,40), P4=(batch,6,20,20), P5=(batch,6,10,10)
    - Output values are numerically close to CPU baseline
    """
    # Get CPU baseline outputs
    cpu_session, _pytensor_fn = cpu_baseline_session
    input_name = cpu_session.get_inputs()[0].name
    cpu_outputs = cpu_session.run(None, {input_name: test_image_data})

    # Load test page
    browser.get(browser_test_html)

    # Run WebGPU inference in browser
    webgpu_result = browser.execute_async_script(
        """
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
                    outputs: session.outputNames.map(name => ({
                        name: name,
                        data: Array.from(results[name].data),
                        dims: Array.from(results[name].dims)
                    })),
                    error: null
                });
            });
        }).catch(error => {
            callback({
                success: false,
                outputs: null,
                error: error.message
            });
        });
    """,
        test_image_data.flatten().tolist(),
    )

    assert webgpu_result["success"], (
        f"WebGPU inference failed: {webgpu_result['error']}"
    )

    # Extract outputs
    outputs = webgpu_result["outputs"]
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    webgpu_p3 = np.array(outputs[0]["data"]).reshape(outputs[0]["dims"])
    webgpu_p4 = np.array(outputs[1]["data"]).reshape(outputs[1]["dims"])
    webgpu_p5 = np.array(outputs[2]["data"]).reshape(outputs[2]["dims"])

    # Shape assertions
    assert webgpu_p3.shape == (1, 6, 40, 40), (
        f"WebGPU P3 shape incorrect: expected (1,6,40,40), got {webgpu_p3.shape}"
    )
    assert webgpu_p4.shape == (1, 6, 20, 20), (
        f"WebGPU P4 shape incorrect: expected (1,6,20,20), got {webgpu_p4.shape}"
    )
    assert webgpu_p5.shape == (1, 6, 10, 10), (
        f"WebGPU P5 shape incorrect: expected (1,6,10,10), got {webgpu_p5.shape}"
    )

    # Numerical assertions (compare with CPU baseline)
    np.testing.assert_allclose(
        webgpu_p3,
        cpu_outputs[0],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WebGPU P3 output differs from CPU baseline",
    )
    np.testing.assert_allclose(
        webgpu_p4,
        cpu_outputs[1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WebGPU P4 output differs from CPU baseline",
    )
    np.testing.assert_allclose(
        webgpu_p5,
        cpu_outputs[2],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WebGPU P5 output differs from CPU baseline",
    )

    # Finite value assertion
    assert np.all(np.isfinite(webgpu_p3)), "WebGPU P3 contains NaN or Inf"
    assert np.all(np.isfinite(webgpu_p4)), "WebGPU P4 contains NaN or Inf"
    assert np.all(np.isfinite(webgpu_p5)), "WebGPU P5 contains NaN or Inf"


@pytest.mark.slow
def test_browser_full_inference_pipeline_wasm(
    browser,
    exported_yolo_model,
    browser_test_html,
    cpu_baseline_session,
    test_image_data,
):
    """Verify complete YOLO inference pipeline works in browser with WASM fallback.

    Priority: HIGH (ensures broad browser compatibility)

    Tests same numerical correctness as WebGPU test, but with WASM execution provider.
    """
    # Get CPU baseline outputs
    cpu_session, _pytensor_fn = cpu_baseline_session
    input_name = cpu_session.get_inputs()[0].name
    cpu_outputs = cpu_session.run(None, {input_name: test_image_data})

    # Load test page
    browser.get(browser_test_html)

    # Run WASM inference in browser
    wasm_result = browser.execute_async_script(
        """
        const callback = arguments[arguments.length - 1];
        const inputData = arguments[0];

        const input = new ort.Tensor('float32', inputData, [1, 3, 320, 320]);

        ort.InferenceSession.create('yolo11n.onnx', {
            executionProviders: ['wasm']
        }).then(session => {
            const inputName = session.inputNames[0];
            return session.run({[inputName]: input}).then(results => {
                callback({
                    success: true,
                    outputs: session.outputNames.map(name => ({
                        name: name,
                        data: Array.from(results[name].data),
                        dims: Array.from(results[name].dims)
                    })),
                    error: null
                });
            });
        }).catch(error => {
            callback({
                success: false,
                outputs: null,
                error: error.message
            });
        });
    """,
        test_image_data.flatten().tolist(),
    )

    assert wasm_result["success"], f"WASM inference failed: {wasm_result['error']}"

    # Extract outputs
    outputs = wasm_result["outputs"]
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    wasm_p3 = np.array(outputs[0]["data"]).reshape(outputs[0]["dims"])
    wasm_p4 = np.array(outputs[1]["data"]).reshape(outputs[1]["dims"])
    wasm_p5 = np.array(outputs[2]["data"]).reshape(outputs[2]["dims"])

    # Shape assertions
    assert wasm_p3.shape == (1, 6, 40, 40), (
        f"WASM P3 shape incorrect: expected (1,6,40,40), got {wasm_p3.shape}"
    )
    assert wasm_p4.shape == (1, 6, 20, 20), (
        f"WASM P4 shape incorrect: expected (1,6,20,20), got {wasm_p4.shape}"
    )
    assert wasm_p5.shape == (1, 6, 10, 10), (
        f"WASM P5 shape incorrect: expected (1,6,10,10), got {wasm_p5.shape}"
    )

    # Numerical assertions (compare with CPU baseline)
    np.testing.assert_allclose(
        wasm_p3,
        cpu_outputs[0],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WASM P3 output differs from CPU baseline",
    )
    np.testing.assert_allclose(
        wasm_p4,
        cpu_outputs[1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WASM P4 output differs from CPU baseline",
    )
    np.testing.assert_allclose(
        wasm_p5,
        cpu_outputs[2],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WASM P5 output differs from CPU baseline",
    )

    # Finite value assertion
    assert np.all(np.isfinite(wasm_p3)), "WASM P3 contains NaN or Inf"
    assert np.all(np.isfinite(wasm_p4)), "WASM P4 contains NaN or Inf"
    assert np.all(np.isfinite(wasm_p5)), "WASM P5 contains NaN or Inf"


def test_browser_yolo_output_parsing(browser, browser_test_html, rng):
    """Verify YOLO output parsing in browser produces correct detection format.

    Priority: HIGH (end-to-end correctness)

    Tests parseYOLOOutput() function from webgpu_vs_wasm_benchmark.html:714-765
    """
    # Create synthetic YOLO output with known detections
    # Format: [cx, cy, w, h, class0_score, ..., class79_score] for 8400 detections
    # Output shape: (1, 84, 8400)
    yolo_output_data = (
        rng.random((1, 84, 8400)).astype(np.float32) * 0.1
    )  # Low confidence baseline

    # Add a few high-confidence detections
    # Detection 0: person (class 0) with high confidence
    yolo_output_data[0, 0, 0] = 160.0  # cx (center of 320x320 image)
    yolo_output_data[0, 1, 0] = 160.0  # cy
    yolo_output_data[0, 2, 0] = 80.0  # w
    yolo_output_data[0, 3, 0] = 100.0  # h
    yolo_output_data[0, 4, 0] = 0.9  # person class score

    # Detection 1: car (class 2) with high confidence
    yolo_output_data[0, 0, 1] = 240.0  # cx
    yolo_output_data[0, 1, 1] = 180.0  # cy
    yolo_output_data[0, 2, 1] = 60.0  # w
    yolo_output_data[0, 3, 1] = 40.0  # h
    yolo_output_data[0, 6, 1] = 0.85  # car class score (index 4 + 2)

    # Load test page
    browser.get(browser_test_html)

    # Parse output in browser
    detections = browser.execute_script(
        """
        const outputData = arguments[0];
        const output = new ort.Tensor('float32', outputData, [1, 84, 8400]);
        return parseYOLOOutput(output, 0.25, 0.45);
    """,
        yolo_output_data.flatten().tolist(),
    )

    # Verify output structure
    assert isinstance(detections, list), f"Expected list, got {type(detections)}"
    assert len(detections) >= 2, (
        f"Expected at least 2 detections, got {len(detections)}"
    )

    for det in detections:
        assert "x1" in det, "Detection missing x1 coordinate"
        assert "y1" in det, "Detection missing y1 coordinate"
        assert "x2" in det, "Detection missing x2 coordinate"
        assert "y2" in det, "Detection missing y2 coordinate"
        assert "confidence" in det, "Detection missing confidence"
        assert "classId" in det, "Detection missing classId"
        assert "className" in det, "Detection missing className"

        # Validate ranges
        assert 0 <= det["confidence"] <= 1, (
            f"Confidence {det['confidence']} out of [0,1] range"
        )
        assert 0 <= det["classId"] < 80, f"ClassId {det['classId']} out of [0,79] range"
        assert det["x1"] < det["x2"], f"Invalid bbox: x1={det['x1']} >= x2={det['x2']}"
        assert det["y1"] < det["y2"], f"Invalid bbox: y1={det['y1']} >= y2={det['y2']}"


def test_browser_nms_correctness(browser, browser_test_html):
    """Verify Non-Maximum Suppression (NMS) implementation in browser.

    Priority: MEDIUM-HIGH (critical for detection quality)

    Tests applyNMS() function from webgpu_vs_wasm_benchmark.html:767-790
    """
    browser.get(browser_test_html)

    # Test Case 1: Identical boxes - NMS keeps only one
    boxes = [
        {
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
            "confidence": 0.9,
            "classId": 0,
            "className": "person",
        },
        {
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
            "confidence": 0.8,
            "classId": 0,
            "className": "person",
        },
    ]
    nms_result = browser.execute_script("return applyNMS(arguments[0], 0.45);", boxes)

    assert len(nms_result) == 1, (
        f"NMS should suppress identical box, got {len(nms_result)} boxes"
    )
    assert nms_result[0]["confidence"] == 0.9, (
        f"NMS should keep higher confidence box, got {nms_result[0]['confidence']}"
    )

    # Test Case 2: High overlap (IoU > 0.45) - Higher confidence box kept
    boxes = [
        {
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
            "confidence": 0.9,
            "classId": 0,
            "className": "person",
        },
        {
            "x1": 15,
            "y1": 15,
            "x2": 55,
            "y2": 55,
            "confidence": 0.8,
            "classId": 0,
            "className": "person",
        },
    ]
    nms_result = browser.execute_script("return applyNMS(arguments[0], 0.45);", boxes)

    assert len(nms_result) == 1, (
        f"NMS should suppress overlapping box, got {len(nms_result)} boxes"
    )
    assert nms_result[0]["confidence"] == 0.9, "NMS should keep higher confidence box"

    # Test Case 3: Low overlap - Both boxes kept
    boxes = [
        {
            "x1": 10,
            "y1": 10,
            "x2": 30,
            "y2": 30,
            "confidence": 0.9,
            "classId": 0,
            "className": "person",
        },
        {
            "x1": 40,
            "y1": 40,
            "x2": 60,
            "y2": 60,
            "confidence": 0.8,
            "classId": 0,
            "className": "person",
        },
    ]
    nms_result = browser.execute_script("return applyNMS(arguments[0], 0.45);", boxes)

    assert len(nms_result) == 2, (
        f"NMS should keep non-overlapping boxes, got {len(nms_result)} boxes"
    )

    # Test Case 4: Different classes - Both kept regardless of overlap
    boxes = [
        {
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
            "confidence": 0.9,
            "classId": 0,
            "className": "person",
        },
        {
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
            "confidence": 0.8,
            "classId": 2,
            "className": "car",
        },
    ]
    nms_result = browser.execute_script("return applyNMS(arguments[0], 0.45);", boxes)

    assert len(nms_result) == 2, (
        f"NMS should keep boxes of different classes, got {len(nms_result)} boxes"
    )


def test_browser_video_frame_extraction(browser, browser_test_html):
    """Verify video frame extraction converts to correct tensor format.

    Priority: MEDIUM (important for video processing)

    Tests frame extraction from canvas to NCHW tensor format.
    """
    browser.get(browser_test_html)

    # Create a test pattern with red/green/blue gradients
    # This is a simple 320x320 gradient image in base64
    # Since we're testing the extraction mechanism, we'll create the image directly in JS
    test_image_script = """
        // Create a test canvas with colorful pattern
        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 320;
        const ctx = canvas.getContext('2d');

        // Fill with gradient pattern
        const gradient = ctx.createLinearGradient(0, 0, 320, 320);
        gradient.addColorStop(0, 'red');
        gradient.addColorStop(0.5, 'green');
        gradient.addColorStop(1, 'blue');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 320, 320);

        return canvas.toDataURL();
    """
    test_image_base64 = browser.execute_script(test_image_script)

    # Load image and extract tensor
    tensor_data = browser.execute_async_script(
        """
        const callback = arguments[arguments.length - 1];
        const imgSrc = arguments[0];

        const img = new Image();
        img.src = imgSrc;

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
                data[i] = imageData.data[i * 4] / 255;                    // R
                data[320 * 320 + i] = imageData.data[i * 4 + 1] / 255;    // G
                data[320 * 320 * 2 + i] = imageData.data[i * 4 + 2] / 255; // B
            }

            callback(Array.from(data));
        };

        img.onerror = () => {
            callback(null);
        };
    """,
        test_image_base64,
    )

    assert tensor_data is not None, "Failed to extract tensor from image"

    tensor = np.array(tensor_data, dtype=np.float32).reshape(1, 3, 320, 320)

    # Verify shape
    assert tensor.shape == (1, 3, 320, 320), (
        f"Tensor shape incorrect: expected (1,3,320,320), got {tensor.shape}"
    )

    # Verify value range
    assert np.all(tensor >= 0) and np.all(tensor <= 1), (
        f"Tensor values out of [0,1] range: min={tensor.min()}, max={tensor.max()}"
    )

    # Verify not all zeros (actual image data)
    assert tensor.std() > 0.01, f"Tensor appears empty (std={tensor.std():.6f} too low)"
