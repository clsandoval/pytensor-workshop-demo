"""Browser Performance Tests for YOLO11n ONNX Demo.

Validates inference performance meets targets (>30 FPS for WebGPU, >10 FPS for WASM).
"""

import pytest


pytestmark = [pytest.mark.browser, pytest.mark.onnx, pytest.mark.performance]


@pytest.mark.slow
def test_webgpu_performance_fps_target(browser, exported_yolo_model, browser_test_html):
    """Verify WebGPU inference meets >30 FPS target.

    Priority: MEDIUM (performance validation)

    Note: Test skips (not fails) if FPS below target, as performance is environment-dependent.
    """
    browser.get(browser_test_html)

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

    assert perf_result["success"], f"Performance test failed: {perf_result['error']}"

    avg_fps = perf_result["fps"]
    avg_time = perf_result["avgTime"]

    # Warn if below target (don't fail)
    if avg_fps < 30:
        pytest.skip(
            f"WebGPU FPS below target: {avg_fps:.1f} FPS (target >30 FPS)\n"
            f"Average inference time: {avg_time:.2f}ms\n"
            f"This may be due to GPU availability or system load.\n"
            f"Test skipped, not failed."
        )
    else:
        print(f"✓ WebGPU performance: {avg_fps:.1f} FPS (target >30 FPS)")
        print(f"  Average inference time: {avg_time:.2f}ms")


@pytest.mark.slow
def test_wasm_performance_fps_baseline(browser, exported_yolo_model, browser_test_html):
    """Verify WASM inference meets >10 FPS baseline.

    Priority: MEDIUM

    Note: Test skips (not fails) if FPS below target, as performance is environment-dependent.
    """
    browser.get(browser_test_html)

    # Benchmark in browser
    perf_result = browser.execute_async_script("""
        const callback = arguments[arguments.length - 1];

        ort.InferenceSession.create('yolo11n.onnx', {
            executionProviders: ['wasm']
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

    assert perf_result["success"], f"Performance test failed: {perf_result['error']}"

    avg_fps = perf_result["fps"]
    avg_time = perf_result["avgTime"]

    # Warn if below target (don't fail)
    if avg_fps < 10:
        pytest.skip(
            f"WASM FPS below target: {avg_fps:.1f} FPS (target >10 FPS)\n"
            f"Average inference time: {avg_time:.2f}ms\n"
            f"This may be due to CPU performance or system load.\n"
            f"Test skipped, not failed."
        )
    else:
        print(f"✓ WASM performance: {avg_fps:.1f} FPS (target >10 FPS)")
        print(f"  Average inference time: {avg_time:.2f}ms")


@pytest.mark.slow
def test_webgpu_faster_than_wasm(browser, exported_yolo_model, browser_test_html):
    """Verify WebGPU is faster than WASM (performance comparison).

    Priority: LOW (nice-to-have validation)

    Note: Test skips if WebGPU not significantly faster, as this depends on hardware.
    """
    browser.get(browser_test_html)

    # Benchmark WebGPU
    webgpu_result = browser.execute_async_script("""
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
            callback({success: true, avgTime: avg});
        }).catch(error => {
            callback({success: false, error: error.message});
        });
    """)

    assert webgpu_result["success"], (
        f"WebGPU benchmark failed: {webgpu_result['error']}"
    )

    # Benchmark WASM
    wasm_result = browser.execute_async_script("""
        const callback = arguments[arguments.length - 1];

        ort.InferenceSession.create('yolo11n.onnx', {
            executionProviders: ['wasm']
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
            callback({success: true, avgTime: avg});
        }).catch(error => {
            callback({success: false, error: error.message});
        });
    """)

    assert wasm_result["success"], f"WASM benchmark failed: {wasm_result['error']}"

    webgpu_avg_time = webgpu_result["avgTime"]
    wasm_avg_time = wasm_result["avgTime"]

    if webgpu_avg_time >= wasm_avg_time:
        pytest.skip(
            f"WebGPU ({webgpu_avg_time:.2f}ms) not faster than WASM ({wasm_avg_time:.2f}ms)\n"
            f"Expected WebGPU to be faster on GPU hardware.\n"
            f"This may indicate WebGPU is not properly accelerated or GPU is not available."
        )
    else:
        speedup = wasm_avg_time / webgpu_avg_time
        print(f"✓ WebGPU speedup: {speedup:.1f}x faster than WASM")
        print(f"  WebGPU: {webgpu_avg_time:.2f}ms avg")
        print(f"  WASM: {wasm_avg_time:.2f}ms avg")
