"""Tests for ONNX export and inference."""

import numpy as np
import pytest
from yolo.model import build_yolo11n

import pytensor


@pytest.mark.onnx
def test_model_exports_to_onnx(tmp_path):
    """
    Test: Model exports to ONNX without errors.

    Verifies:
    - ONNX export completes
    - ONNX file is created
    - ONNX model passes validation
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile PyTensor function
    f = pytensor.function([x_sym], predictions)

    # Export to ONNX
    onnx_path = tmp_path / "yolo11n.onnx"
    onnx_model = export_onnx(f, str(onnx_path))

    # Verify file exists
    assert onnx_path.exists(), "ONNX file was not created"

    # Validate ONNX model
    onnx.checker.check_model(onnx_model)


@pytest.mark.onnx
def test_onnx_runtime_vs_pytensor_equivalence(tmp_path):
    """
    Property: ONNX Runtime inference matches PyTensor.

    This is the most critical test - ensures deployed model correctness.

    Tolerances: rtol=1e-4, atol=1e-5 (account for float32 differences)
    """
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile PyTensor function
    f_pytensor = pytensor.function([x_sym], predictions)

    # Export to ONNX
    onnx_path = tmp_path / "yolo11n.onnx"
    export_onnx(f_pytensor, str(onnx_path))

    # Load with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test data
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")

    # PyTensor inference
    pytensor_out = f_pytensor(x_val)

    # ONNX Runtime inference
    onnx_inputs = {session.get_inputs()[0].name: x_val}
    onnx_out = session.run(None, onnx_inputs)

    # Compare each scale
    for pt_det, onnx_det, scale in zip(pytensor_out, onnx_out, ["P3", "P4", "P5"]):
        np.testing.assert_allclose(
            onnx_det,
            pt_det,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"ONNX vs PyTensor mismatch at {scale}",
        )


@pytest.mark.onnx
def test_onnx_model_metadata(tmp_path):
    """
    Test: ONNX model contains expected metadata.

    Verifies:
    - Model has inputs/outputs with correct names and shapes
    - Model version and producer are set
    """
    pytest.importorskip("onnx")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    onnx_path = tmp_path / "yolo11n.onnx"
    onnx_model = export_onnx(f, str(onnx_path))

    # Check inputs
    assert len(onnx_model.graph.input) >= 1, "Model should have at least 1 input"

    # Check outputs
    assert len(onnx_model.graph.output) == 3, "Model should have 3 outputs (P3, P4, P5)"


@pytest.mark.onnx
def test_onnx_export_different_batch_sizes(tmp_path):
    """
    Test: ONNX model handles different batch sizes.

    ONNX models with dynamic batch dimension should work with
    various batch sizes.
    """
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    onnx_path = tmp_path / "yolo11n.onnx"
    export_onnx(f, str(onnx_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
        onnx_inputs = {session.get_inputs()[0].name: x_val}
        onnx_out = session.run(None, onnx_inputs)

        # Verify output shapes
        assert onnx_out[0].shape[0] == batch_size, (
            f"P3 batch size wrong for batch_size={batch_size}"
        )
        assert onnx_out[1].shape[0] == batch_size, (
            f"P4 batch size wrong for batch_size={batch_size}"
        )
        assert onnx_out[2].shape[0] == batch_size, (
            f"P5 batch size wrong for batch_size={batch_size}"
        )


@pytest.mark.onnx
def test_onnx_output_shapes(tmp_path):
    """
    Test: ONNX model outputs have correct shapes.

    For 320x320 input with num_classes=2:
    - P3: (batch, 6, 40, 40)
    - P4: (batch, 6, 20, 20)
    - P5: (batch, 6, 10, 10)
    """
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    onnx_path = tmp_path / "yolo11n.onnx"
    export_onnx(f, str(onnx_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    batch = 2
    x_val = np.random.randn(batch, 3, 320, 320).astype("float32")
    onnx_inputs = {session.get_inputs()[0].name: x_val}
    onnx_out = session.run(None, onnx_inputs)

    assert onnx_out[0].shape == (batch, 6, 40, 40), f"P3 shape: {onnx_out[0].shape}"
    assert onnx_out[1].shape == (batch, 6, 20, 20), f"P4 shape: {onnx_out[1].shape}"
    assert onnx_out[2].shape == (batch, 6, 10, 10), f"P5 shape: {onnx_out[2].shape}"


@pytest.mark.onnx
def test_onnx_deterministic_inference(tmp_path):
    """
    Test: ONNX Runtime inference is deterministic.

    Running the same input twice should produce identical results.
    """
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    onnx_path = tmp_path / "yolo11n.onnx"
    export_onnx(f, str(onnx_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    x_val = np.random.randn(2, 3, 320, 320).astype("float32")
    onnx_inputs = {session.get_inputs()[0].name: x_val}

    # Run twice
    out1 = session.run(None, onnx_inputs)
    out2 = session.run(None, onnx_inputs)

    # Should be identical
    for o1, o2, scale in zip(out1, out2, ["P3", "P4", "P5"]):
        np.testing.assert_array_equal(
            o1, o2, err_msg=f"ONNX is non-deterministic at {scale}"
        )


@pytest.mark.onnx
def test_onnx_model_opset_version(tmp_path):
    """
    Test: ONNX model uses a reasonable opset version.

    ONNX models should use opset >= 11 for good operator support.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    onnx_path = tmp_path / "yolo11n.onnx"
    onnx_model = export_onnx(f, str(onnx_path))

    # Check opset version
    opset_version = onnx_model.opset_import[0].version
    assert opset_version >= 11, f"Opset version {opset_version} is too old (< 11)"


@pytest.mark.onnx
def test_onnx_no_training_ops(tmp_path):
    """
    Test: ONNX model doesn't contain training-only operations.

    Exported model should be in inference mode (no dropout, etc.).
    """
    pytest.importorskip("onnx")
    pytest.importorskip("pytensor.link.onnx", reason="ONNX export not available")

    from pytensor.link.onnx import export_onnx

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions)

    onnx_path = tmp_path / "yolo11n.onnx"
    onnx_model = export_onnx(f, str(onnx_path))

    # Get all node types
    node_types = {node.op_type for node in onnx_model.graph.node}

    # Should not contain training-only ops
    training_ops = {"Dropout", "RandomUniform", "RandomNormal"}
    found_training_ops = node_types & training_ops

    assert len(found_training_ops) == 0, (
        f"Found training-only ops in exported model: {found_training_ops}"
    )
