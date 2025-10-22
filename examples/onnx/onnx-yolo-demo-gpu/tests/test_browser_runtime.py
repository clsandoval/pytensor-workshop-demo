"""Browser Runtime Correctness Tests for YOLO11n ONNX Demo.

Validates ONNX models load and run correctly in browser runtimes (WebGPU and WASM).
"""

import numpy as np
import pytest


pytestmark = [pytest.mark.browser, pytest.mark.onnx]


def test_model_loads_in_browser_webgpu(browser, exported_yolo_model, browser_test_html):
    """Verify ONNX model loads successfully with WebGPU execution provider.

    Priority: HIGH (prerequisite for all WebGPU tests)
    """
    browser.get(browser_test_html)

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

    assert session_info["success"], (
        f"WebGPU session creation failed: {session_info['error']}"
    )
    assert session_info["inputNames"] is not None, "Session inputNames is null"
    assert len(session_info["inputNames"]) >= 1, (
        f"Expected at least 1 input, got {len(session_info['inputNames'])}"
    )
    assert len(session_info["outputNames"]) == 3, (
        f"Expected 3 outputs (P3,P4,P5), got {len(session_info['outputNames'])}"
    )


def test_model_loads_in_browser_wasm(browser, exported_yolo_model, browser_test_html):
    """Verify ONNX model loads successfully with WASM execution provider.

    Priority: HIGH (fallback runtime)
    """
    browser.get(browser_test_html)

    # Attempt session creation in browser
    session_info = browser.execute_async_script("""
        const callback = arguments[arguments.length - 1];

        ort.InferenceSession.create('yolo11n.onnx', {
            executionProviders: ['wasm']
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

    assert session_info["success"], (
        f"WASM session creation failed: {session_info['error']}"
    )
    assert session_info["inputNames"] is not None, "Session inputNames is null"
    assert len(session_info["inputNames"]) >= 1, (
        f"Expected at least 1 input, got {len(session_info['inputNames'])}"
    )
    assert len(session_info["outputNames"]) == 3, (
        f"Expected 3 outputs (P3,P4,P5), got {len(session_info['outputNames'])}"
    )


def test_webgpu_inference_produces_valid_shapes(
    browser, exported_yolo_model, browser_test_html, test_image_data
):
    """Verify WebGPU inference outputs have correct tensor shapes.

    Priority: HIGH (shape validation is critical)
    """
    browser.get(browser_test_html)

    # Run inference in browser
    outputs = browser.execute_async_script(
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
                    shapes: session.outputNames.map(name => Array.from(results[name].dims)),
                    error: null
                });
            });
        }).catch(error => {
            callback({success: false, shapes: null, error: error.message});
        });
    """,
        test_image_data.flatten().tolist(),
    )

    assert outputs["success"], f"WebGPU inference failed: {outputs['error']}"
    assert len(outputs["shapes"]) == 3, (
        f"Expected 3 outputs, got {len(outputs['shapes'])}"
    )

    # Verify each shape
    expected_shapes = [(1, 6, 40, 40), (1, 6, 20, 20), (1, 6, 10, 10)]
    for i, (actual, expected) in enumerate(zip(outputs["shapes"], expected_shapes)):
        assert tuple(actual) == expected, (
            f"Output {i} shape mismatch: expected {expected}, got {tuple(actual)}"
        )


def test_wasm_inference_produces_valid_shapes(
    browser, exported_yolo_model, browser_test_html, test_image_data
):
    """Verify WASM inference outputs have correct tensor shapes.

    Priority: HIGH
    """
    browser.get(browser_test_html)

    # Run inference in browser
    outputs = browser.execute_async_script(
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
                    shapes: session.outputNames.map(name => Array.from(results[name].dims)),
                    error: null
                });
            });
        }).catch(error => {
            callback({success: false, shapes: null, error: error.message});
        });
    """,
        test_image_data.flatten().tolist(),
    )

    assert outputs["success"], f"WASM inference failed: {outputs['error']}"
    assert len(outputs["shapes"]) == 3, (
        f"Expected 3 outputs, got {len(outputs['shapes'])}"
    )

    # Verify each shape
    expected_shapes = [(1, 6, 40, 40), (1, 6, 20, 20), (1, 6, 10, 10)]
    for i, (actual, expected) in enumerate(zip(outputs["shapes"], expected_shapes)):
        assert tuple(actual) == expected, (
            f"Output {i} shape mismatch: expected {expected}, got {tuple(actual)}"
        )


@pytest.mark.slow
def test_webgpu_vs_cpu_numerical_equivalence(
    browser,
    exported_yolo_model,
    browser_test_html,
    cpu_baseline_session,
    test_image_data,
):
    """Verify WebGPU produces numerically equivalent results to CPU backend.

    Priority: CRITICAL (correctness validation)
    """
    # Run CPU inference (baseline)
    cpu_session, _pytensor_fn = cpu_baseline_session
    input_name = cpu_session.get_inputs()[0].name
    cpu_outputs = cpu_session.run(None, {input_name: test_image_data})

    browser.get(browser_test_html)

    # Run WebGPU inference
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
                        data: Array.from(results[name].data),
                        dims: Array.from(results[name].dims)
                    }))
                });
            });
        }).catch(error => {
            callback({success: false, error: error.message});
        });
    """,
        test_image_data.flatten().tolist(),
    )

    assert webgpu_result["success"], (
        f"WebGPU inference failed: {webgpu_result['error']}"
    )

    # Convert to numpy arrays
    webgpu_p3 = np.array(webgpu_result["outputs"][0]["data"]).reshape(
        webgpu_result["outputs"][0]["dims"]
    )
    webgpu_p4 = np.array(webgpu_result["outputs"][1]["data"]).reshape(
        webgpu_result["outputs"][1]["dims"]
    )
    webgpu_p5 = np.array(webgpu_result["outputs"][2]["data"]).reshape(
        webgpu_result["outputs"][2]["dims"]
    )

    # Compare with CPU baseline
    max_diff_p3 = np.abs(webgpu_p3 - cpu_outputs[0]).max()
    mean_diff_p3 = np.abs(webgpu_p3 - cpu_outputs[0]).mean()

    np.testing.assert_allclose(
        webgpu_p3,
        cpu_outputs[0],
        rtol=1e-3,
        atol=1e-4,
        err_msg=f"WebGPU P3 differs from CPU\n"
        f"  Max diff: {max_diff_p3}\n"
        f"  Mean diff: {mean_diff_p3}",
    )
    np.testing.assert_allclose(
        webgpu_p4,
        cpu_outputs[1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WebGPU P4 differs from CPU",
    )
    np.testing.assert_allclose(
        webgpu_p5,
        cpu_outputs[2],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WebGPU P5 differs from CPU",
    )

    # Verify finite values
    assert np.all(np.isfinite(webgpu_p3)), "WebGPU P3 contains NaN or Inf"
    assert np.all(np.isfinite(webgpu_p4)), "WebGPU P4 contains NaN or Inf"
    assert np.all(np.isfinite(webgpu_p5)), "WebGPU P5 contains NaN or Inf"


@pytest.mark.slow
def test_wasm_vs_cpu_numerical_equivalence(
    browser,
    exported_yolo_model,
    browser_test_html,
    cpu_baseline_session,
    test_image_data,
):
    """Verify WASM produces numerically equivalent results to CPU backend.

    Priority: CRITICAL
    """
    # Run CPU inference (baseline)
    cpu_session, _pytensor_fn = cpu_baseline_session
    input_name = cpu_session.get_inputs()[0].name
    cpu_outputs = cpu_session.run(None, {input_name: test_image_data})

    browser.get(browser_test_html)

    # Run WASM inference
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
                        data: Array.from(results[name].data),
                        dims: Array.from(results[name].dims)
                    }))
                });
            });
        }).catch(error => {
            callback({success: false, error: error.message});
        });
    """,
        test_image_data.flatten().tolist(),
    )

    assert wasm_result["success"], f"WASM inference failed: {wasm_result['error']}"

    # Convert to numpy arrays
    wasm_p3 = np.array(wasm_result["outputs"][0]["data"]).reshape(
        wasm_result["outputs"][0]["dims"]
    )
    wasm_p4 = np.array(wasm_result["outputs"][1]["data"]).reshape(
        wasm_result["outputs"][1]["dims"]
    )
    wasm_p5 = np.array(wasm_result["outputs"][2]["data"]).reshape(
        wasm_result["outputs"][2]["dims"]
    )

    # Compare with CPU baseline
    max_diff_p3 = np.abs(wasm_p3 - cpu_outputs[0]).max()
    mean_diff_p3 = np.abs(wasm_p3 - cpu_outputs[0]).mean()

    np.testing.assert_allclose(
        wasm_p3,
        cpu_outputs[0],
        rtol=1e-3,
        atol=1e-4,
        err_msg=f"WASM P3 differs from CPU\n"
        f"  Max diff: {max_diff_p3}\n"
        f"  Mean diff: {mean_diff_p3}",
    )
    np.testing.assert_allclose(
        wasm_p4,
        cpu_outputs[1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WASM P4 differs from CPU",
    )
    np.testing.assert_allclose(
        wasm_p5,
        cpu_outputs[2],
        rtol=1e-3,
        atol=1e-4,
        err_msg="WASM P5 differs from CPU",
    )

    # Verify finite values
    assert np.all(np.isfinite(wasm_p3)), "WASM P3 contains NaN or Inf"
    assert np.all(np.isfinite(wasm_p4)), "WASM P4 contains NaN or Inf"
    assert np.all(np.isfinite(wasm_p5)), "WASM P5 contains NaN or Inf"


def test_browser_handles_different_batch_sizes(
    browser, exported_yolo_model, browser_test_html, rng
):
    """Verify browser inference works with dynamic batch sizes.

    Priority: MEDIUM
    """
    browser.get(browser_test_html)

    for batch_size in [1, 2, 4]:
        x_val = rng.random((batch_size, 3, 320, 320)).astype(np.float32)

        outputs = browser.execute_async_script(
            """
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
        """,
            x_val.flatten().tolist(),
            [batch_size, 3, 320, 320],
        )

        assert outputs["success"], f"Inference failed for batch_size={batch_size}"
        assert outputs["shapes"][0][0] == batch_size, (
            f"P3 batch size wrong: expected {batch_size}, got {outputs['shapes'][0][0]}"
        )
        assert outputs["shapes"][1][0] == batch_size, (
            f"P4 batch size wrong: expected {batch_size}, got {outputs['shapes'][1][0]}"
        )
        assert outputs["shapes"][2][0] == batch_size, (
            f"P5 batch size wrong: expected {batch_size}, got {outputs['shapes'][2][0]}"
        )
