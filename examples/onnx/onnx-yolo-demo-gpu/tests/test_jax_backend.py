"""Tests for JAX backend compilation and execution."""

import numpy as np
import pytest
from yolo.model import YOLO11n, build_yolo11n

import pytensor
import pytensor.tensor as pt
from pytensor import grad
from pytensor.compile.mode import Mode


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_model_compiles_with_jax():
    """
    Test: Model compiles successfully with JAX backend.

    Verifies:
    - No shape tracing errors
    - No unsupported operations
    - Returns JAX DeviceArray (indicates GPU execution)
    """
    pytest.importorskip("jax")
    import jax

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile with JAX mode
    f = pytensor.function([x_sym], predictions, mode="JAX")

    # Test forward pass
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    det_p3, det_p4, det_p5 = f(x_val)

    # Verify outputs are JAX arrays
    assert isinstance(det_p3, jax.Array), "det_p3 is not JAX Array"
    assert isinstance(det_p4, jax.Array), "det_p4 is not JAX Array"
    assert isinstance(det_p5, jax.Array), "det_p5 is not JAX Array"


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_jax_vs_python_backend_numerical_equivalence():
    """
    Property: JAX backend produces numerically equivalent results to Python.

    Compares:
    - Forward pass outputs
    - Tolerances: rtol=1e-4, atol=1e-5 (float32 accumulation errors)
    """
    pytest.importorskip("jax")

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile with both backends
    f_jax = pytensor.function([x_sym], predictions, mode="JAX")
    f_py = pytensor.function([x_sym], predictions, mode=Mode(linker="py"))

    # Test data
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")

    # Run both
    jax_out = f_jax(x_val)
    py_out = f_py(x_val)

    # Compare each scale
    for jax_det, py_det, scale in zip(jax_out, py_out, ["P3", "P4", "P5"]):
        np.testing.assert_allclose(
            jax_det,
            py_det,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"JAX vs Python mismatch at {scale}",
        )


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_jax_gradient_flow():
    """
    Test: Gradients flow correctly through JAX-compiled model.

    Verifies:
    - Gradients w.r.t. model parameters are computed
    - Gradients are non-zero (model has learning signal)
    - Gradients are finite (no NaN/Inf)
    """
    pytest.importorskip("jax")

    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    det_p3, det_p4, det_p5 = model(x)

    # Compute scalar loss (sum of outputs)
    loss = det_p3.sum() + det_p4.sum() + det_p5.sum()

    # Get model parameters
    params = [
        p for p in model.backbone.params + model.head.params if hasattr(p, "name")
    ]

    if len(params) == 0:
        pytest.skip("No parameters found in model")

    # Compute gradient w.r.t. first parameter
    first_param = params[0]
    grad_param = grad(loss, first_param)

    # Compile with JAX
    f = pytensor.function([x], [loss, grad_param], mode="JAX")

    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    loss_val, grad_val = f(x_val)

    # Verify gradient properties
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert np.all(np.isfinite(grad_val)), "Gradient contains NaN or Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is all zeros"


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_jax_compilation_caching():
    """
    Test: JAX compilation is cached for repeated calls.

    Second call should be faster than first call due to JIT caching.
    """
    pytest.importorskip("jax")
    import time

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions, mode="JAX")

    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # First call (includes compilation)
    start1 = time.time()
    _ = f(x_val)
    time1 = time.time() - start1

    # Second call (should use cached compilation)
    start2 = time.time()
    _ = f(x_val)
    time2 = time.time() - start2

    # Second call should be faster (or at least not much slower)
    # We use a lenient check since timing can be noisy
    assert time2 < time1 * 2, (
        f"Second call ({time2:.3f}s) not faster than first ({time1:.3f}s)"
    )


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_jax_batch_processing():
    """
    Test: JAX handles different batch sizes correctly.

    JAX should compile different batch sizes without errors.
    """
    pytest.importorskip("jax")

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions, mode="JAX")

    # Test various batch sizes
    for batch_size in [1, 2, 4]:
        x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
        det_p3, det_p4, det_p5 = f(x_val)

        assert det_p3.shape[0] == batch_size, (
            f"Batch size {batch_size}: wrong P3 batch dim"
        )
        assert det_p4.shape[0] == batch_size, (
            f"Batch size {batch_size}: wrong P4 batch dim"
        )
        assert det_p5.shape[0] == batch_size, (
            f"Batch size {batch_size}: wrong P5 batch dim"
        )


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_jax_deterministic():
    """
    Test: JAX execution is deterministic.

    Running the same input twice should produce identical results.
    """
    pytest.importorskip("jax")

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions, mode="JAX")

    x_val = np.random.randn(2, 3, 320, 320).astype("float32")

    # Run twice
    out1 = f(x_val)
    out2 = f(x_val)

    # Should be identical
    for o1, o2, scale in zip(out1, out2, ["P3", "P4", "P5"]):
        np.testing.assert_array_equal(
            o1, o2, err_msg=f"JAX is non-deterministic at {scale}"
        )


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"),
    reason="JAX not installed",
)
def test_jax_handles_edge_cases():
    """
    Test: JAX handles edge cases gracefully.

    - Zero inputs
    - Very small inputs
    - Very large inputs (within reasonable bounds)
    """
    pytest.importorskip("jax")

    _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
    f = pytensor.function([x_sym], predictions, mode="JAX")

    # Test zero input
    x_zeros = np.zeros((1, 3, 320, 320), dtype="float32")
    out_zeros = f(x_zeros)
    for out in out_zeros:
        assert np.all(np.isfinite(out)), "Zero input produced non-finite output"

    # Test small input
    x_small = np.ones((1, 3, 320, 320), dtype="float32") * 0.01
    out_small = f(x_small)
    for out in out_small:
        assert np.all(np.isfinite(out)), "Small input produced non-finite output"

    # Test larger input
    x_large = np.ones((1, 3, 320, 320), dtype="float32") * 10.0
    out_large = f(x_large)
    for out in out_large:
        assert np.all(np.isfinite(out)), "Large input produced non-finite output"
