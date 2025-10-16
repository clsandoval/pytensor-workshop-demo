"""
Tests for ONNX export, verification, and browser compatibility.

This module tests the ONNX export workflow including basic export,
ONNX Runtime execution, verification against PyTensor, browser
compatibility checks, and property-based testing.
"""

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# Import ONNX and ONNX Runtime (skip tests if not available)
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
from pytensor.link.onnx import export_onnx  # noqa: E402


# Import compare_onnx_and_py from pytensor tests
# Add tests directory to path to access test utilities
tests_dir = Path(__file__).parent.parent.parent.parent / "tests"
sys.path.insert(0, str(tests_dir))
from link.onnx.test_basic import compare_onnx_and_py  # noqa: E402


def test_onnx_export_basic(tmp_model_dir):
    """
    Test basic ONNX export of posterior predictive function.

    This test verifies:
    - export_onnx() completes without errors
    - ONNX file is created on disk
    - Model can be loaded and validated
    - Model has correct input/output count
    """
    # Create simple posterior predictive graph
    x_new = pt.vector("x_new", dtype="float32")
    alpha = pt.constant(1.0, dtype="float32")
    beta = pt.constant(2.5, dtype="float32")
    sigma = pt.constant(0.5, dtype="float32")

    y_mean = alpha + beta * x_new
    y_std = pt.ones_like(x_new) * sigma

    # Compile PyTensor function
    predict_fn = pytensor.function([x_new], [y_mean, y_std])

    # Export to ONNX
    onnx_path = tmp_model_dir / "test_basic.onnx"
    model = export_onnx(predict_fn, str(onnx_path))

    # Verify export succeeded
    assert model is not None, "export_onnx returned None"
    assert onnx_path.exists(), f"ONNX file not created at {onnx_path}"

    # Verify model structure
    assert isinstance(model, onnx.ModelProto), (
        f"Expected onnx.ModelProto, got {type(model)}"
    )

    # Verify input/output counts
    assert len(model.graph.input) == 1, (
        f"Expected 1 input, got {len(model.graph.input)}"
    )
    assert len(model.graph.output) == 2, (
        f"Expected 2 outputs (mean, std), got {len(model.graph.output)}"
    )

    # Validate ONNX model
    onnx.checker.check_model(model)


def test_onnx_runtime_execution(tmp_model_dir):
    """
    Test ONNX Runtime execution of exported model.

    This test verifies:
    - ONNX Runtime can load the model
    - Model can be executed with valid inputs
    - Outputs have correct shapes
    - Output values are reasonable
    """
    # Create and export model
    x_new = pt.vector("x_new", dtype="float32")
    alpha = pt.constant(1.0, dtype="float32")
    beta = pt.constant(2.5, dtype="float32")
    sigma = pt.constant(0.5, dtype="float32")

    y_mean = alpha + beta * x_new
    y_std = pt.ones_like(x_new) * sigma

    predict_fn = pytensor.function([x_new], [y_mean, y_std])
    onnx_path = tmp_model_dir / "test_runtime.onnx"
    _model = export_onnx(predict_fn, str(onnx_path))

    # Load with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Verify session loaded
    assert session is not None, "ONNX Runtime session creation failed"

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    assert len(output_names) == 2, f"Expected 2 outputs, got {len(output_names)}"

    # Run inference
    x_test = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype="float32")
    results = session.run(None, {input_name: x_test})

    # Verify results
    assert len(results) == 2, f"Expected 2 output arrays, got {len(results)}"

    mean_out, std_out = results

    assert mean_out.shape == x_test.shape, f"Mean shape: {mean_out.shape}"
    assert std_out.shape == x_test.shape, f"Std shape: {std_out.shape}"

    # Verify reasonable values
    assert not np.any(np.isnan(mean_out)), "Mean output contains NaN"
    assert not np.any(np.isnan(std_out)), "Std output contains NaN"
    assert np.all(std_out > 0), "Std should be positive"


def test_onnx_matches_pytensor(tmp_model_dir):
    """
    Test that ONNX Runtime output matches PyTensor output exactly.

    This test verifies:
    - Numerical outputs match within tolerance (rtol=1e-4, atol=1e-5)
    - Both mean and std outputs are consistent
    - Works across different input sizes

    **Pattern**: Uses compare_onnx_and_py() utility from tests/link/onnx/test_basic.py
    This is the gold standard for ONNX verification used throughout PyTensor.
    """
    # Create posterior predictive graph
    x_new = pt.vector("x_new", dtype="float32")
    alpha = pt.constant(1.0, dtype="float32")
    beta = pt.constant(2.5, dtype="float32")
    sigma = pt.constant(0.5, dtype="float32")

    y_mean = alpha + beta * x_new
    y_std = pt.ones_like(x_new) * sigma

    # Test data
    x_test = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype="float32")

    # Compare outputs
    compare_onnx_and_py(
        graph_inputs=[x_new],
        graph_outputs=[y_mean, y_std],
        test_inputs=[x_test],
        tmp_path=tmp_model_dir,
    )


def test_onnx_browser_compatibility(tmp_model_dir):
    """
    Test that exported ONNX model is browser-compatible.

    This test verifies:
    - Opset version is widely supported (>= 14)
    - No unsupported operations
    - Input/output shapes are reasonable
    - Model size is small (<100KB as planned)
    - Float32 dtypes (WebGPU/WASM support)

    **Browser Context**:
    - ONNX Runtime Web supports opset 14+ well
    - Float32 is standard for browser inference
    - Small models load faster

    Future work (out of scope):
    - Actual browser testing with onnxruntime-web
    - WebGPU vs WASM performance comparison
    """
    # Create and export model
    x_new = pt.vector("x_new", dtype="float32")
    alpha = pt.constant(1.0, dtype="float32")
    beta = pt.constant(2.5, dtype="float32")
    sigma = pt.constant(0.5, dtype="float32")

    y_mean = alpha + beta * x_new
    y_std = pt.ones_like(x_new) * sigma

    predict_fn = pytensor.function([x_new], [y_mean, y_std])
    onnx_path = tmp_model_dir / "test_browser.onnx"
    model = export_onnx(predict_fn, str(onnx_path))

    # Check opset version
    opset_version = model.opset_import[0].version
    assert opset_version >= 14, (
        f"Opset version {opset_version} may not be well-supported in browsers.\n"
        "Recommend opset >= 14 for ONNX Runtime Web."
    )

    # Check model size
    model_size_kb = onnx_path.stat().st_size / 1024
    assert model_size_kb < 100, (
        f"Model size {model_size_kb:.1f} KB exceeds target (<100KB).\n"
        "Large models slow down browser loading."
    )

    # Check float32 dtypes
    for inp in model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        assert elem_type == onnx.TensorProto.FLOAT, (
            f"Input {inp.name} has dtype {elem_type}, expected FLOAT (1).\n"
            "Browser inference typically requires float32."
        )

    for out in model.graph.output:
        elem_type = out.type.tensor_type.elem_type
        assert elem_type == onnx.TensorProto.FLOAT, (
            f"Output {out.name} has dtype {elem_type}, expected FLOAT (1).\n"
            "Browser inference typically requires float32."
        )

    # Check for unsupported ops (none expected for this simple model)
    op_types = [node.op_type for node in model.graph.node]
    unsupported_ops = {"RandomNormal", "RandomUniform"}  # Not in our model
    found_unsupported = set(op_types) & unsupported_ops

    assert not found_unsupported, (
        f"Found unsupported ops for browser: {found_unsupported}\nAll ops: {op_types}"
    )


@given(
    alpha=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    beta=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    sigma=st.floats(
        min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=10, deadline=None)
def test_hypothesis_onnx_export(tmp_model_dir, alpha, beta, sigma):
    """
    Property test: ONNX export works across wide parameter ranges.

    This test verifies:
    - Export succeeds for various parameter values
    - ONNX output matches PyTensor across parameter space
    - No numerical issues (NaN, Inf) in outputs
    """
    # Create graph with hypothesis-generated parameters
    x_new = pt.vector("x_new", dtype="float32")
    alpha_const = pt.constant(alpha, dtype="float32")
    beta_const = pt.constant(beta, dtype="float32")
    sigma_const = pt.constant(sigma, dtype="float32")

    y_mean = alpha_const + beta_const * x_new
    y_std = pt.ones_like(x_new) * sigma_const

    # Test data
    x_test = np.array([0.0, 1.0, 2.0, 5.0], dtype="float32")

    # Use a unique path for each test
    test_id = uuid.uuid4().hex[:8]
    test_path = tmp_model_dir / f"hypothesis_{test_id}"
    test_path.mkdir(exist_ok=True)

    # Compare outputs
    compare_onnx_and_py(
        graph_inputs=[x_new],
        graph_outputs=[y_mean, y_std],
        test_inputs=[x_test],
        tmp_path=test_path,
    )
