"""Pytest configuration for YOLO11n demo tests."""

import os
from datetime import timedelta

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, settings


# Set PyTensor flags for float32
os.environ.setdefault("PYTENSOR_FLAGS", "floatX=float32")


# Register Hypothesis profiles
settings.register_profile(
    "dev",
    max_examples=10,
    deadline=timedelta(milliseconds=500),
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    print_blob=False,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    derandomize=True,
    print_blob=True,
)

settings.register_profile(
    "thorough",
    max_examples=200,
    deadline=None,
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# === Random Seeds ===
@pytest.fixture
def test_seed():
    """Consistent seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(test_seed):
    """NumPy random generator with consistent seed."""
    return np.random.default_rng(test_seed)


# === Directories ===
@pytest.fixture
def tmp_onnx_dir(tmp_path):
    """Temporary directory for ONNX model files."""
    onnx_dir = tmp_path / "onnx_models"
    onnx_dir.mkdir()
    return onnx_dir


# === Test Data ===
@pytest.fixture
def yolo_input_size():
    """Standard YOLO11n input size."""
    return (320, 320)


@pytest.fixture
def simple_image_batch(rng, yolo_input_size):
    """Generate simple batch of images for testing."""
    batch_size = 2
    channels = 3
    height, width = yolo_input_size
    return rng.standard_normal((batch_size, channels, height, width)).astype(np.float32)


# === Utilities ===
def compare_jax_and_py(inputs, outputs, test_values, rtol=1e-4, atol=1e-5):
    """Compare JAX and Python backend outputs.

    Args:
        inputs: List of PyTensor symbolic inputs
        outputs: PyTensor symbolic output(s)
        test_values: List of numpy arrays for inputs
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (jax_function, jax_output)
    """
    pytest.importorskip("jax")

    import pytensor
    from pytensor.compile.mode import Mode

    f_jax = pytensor.function(inputs, outputs, mode="JAX")
    f_py = pytensor.function(inputs, outputs, mode=Mode(linker="py"))

    jax_out = f_jax(*test_values)
    py_out = f_py(*test_values)

    if isinstance(outputs, list):
        for jo, po in zip(jax_out, py_out):
            np.testing.assert_allclose(jo, po, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(jax_out, py_out, rtol=rtol, atol=atol)

    return f_jax, jax_out


def export_and_validate_onnx(pytensor_fn, onnx_path):
    """Export PyTensor function to ONNX and validate.

    Args:
        pytensor_fn: Compiled PyTensor function
        onnx_path: Path to save ONNX model

    Returns:
        ONNX ModelProto
    """
    import onnx

    from pytensor.link.onnx import export_onnx

    onnx_model = export_onnx(pytensor_fn, str(onnx_path))
    assert onnx_path.exists(), f"ONNX file not created: {onnx_path}"
    onnx.checker.check_model(onnx_model)
    return onnx_model


@pytest.fixture
def compile_and_run():
    """Helper to compile PyTensor function and run with test data."""
    from pytensor import function

    def _compile(inputs, outputs, test_values):
        f = function(inputs, outputs)
        return f(*test_values)

    return _compile


# === Browser Testing Fixtures ===
@pytest.fixture(scope="session")
def chrome_options():
    """Chrome options for WebGPU support."""
    pytest.importorskip("selenium")
    from selenium.webdriver.chrome.options import Options

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
    pytest.importorskip("selenium")
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    yield driver
    driver.quit()


@pytest.fixture(scope="session")
def local_server(tmp_path_factory):
    """Local HTTP server for serving ONNX models and HTML."""
    import http.server
    import shutil
    import socketserver
    import threading
    import time
    from pathlib import Path

    server_dir = tmp_path_factory.mktemp("server_files")

    # Copy necessary files from site directory
    site_dir = Path(__file__).parent.parent / "site"
    if site_dir.exists():
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

    import pytensor
    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    _server_url, server_dir = local_server
    onnx_path = server_dir / "yolo11n.onnx"

    export_onnx(f, str(onnx_path))

    return onnx_path, f


@pytest.fixture
def cpu_baseline_session(exported_yolo_model):
    """ONNX Runtime CPU session for baseline comparisons."""
    ort = pytest.importorskip("onnxruntime")
    onnx_path, pytensor_fn = exported_yolo_model

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

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
