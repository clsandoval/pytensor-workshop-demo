"""Integration tests for full YOLO11n model."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from yolo.model import YOLO11n, build_yolo11n

import pytensor.tensor as pt
from pytensor import function


@given(batch_size=st.integers(1, 4))
@settings(deadline=None)
def test_yolo11n_forward_pass_deterministic(batch_size):
    """
    Property: Model produces deterministic outputs (no stochastic layers in eval).

    Running the same input twice should produce identical results.
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    det_p3, det_p4, det_p5 = model(x)

    f = function([x], [det_p3, det_p4, det_p5])

    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")

    # Run twice
    out1 = f(x_val)
    out2 = f(x_val)

    # Should be identical
    for o1, o2 in zip(out1, out2):
        np.testing.assert_array_equal(o1, o2, err_msg="Model is non-deterministic")


def test_yolo11n_output_tuple_structure():
    """
    Property: Model output is tuple (det_p3, det_p4, det_p5), not dict.

    This is required for:
    - Loss function compatibility (yolo_loss expects tuple unpacking)
    - ONNX export (easier to handle tuple than dict)
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    f = function([x], predictions)
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    result = f(x_val)

    # Verify it's a tuple of 3 arrays
    assert isinstance(result, tuple), f"Output is {type(result)}, not tuple"
    assert len(result) == 3, f"Output has {len(result)} elements, expected 3"

    # Verify each element is array
    for i, r in enumerate(result):
        assert isinstance(r, np.ndarray), f"Output[{i}] is {type(r)}, not ndarray"


def test_yolo11n_output_shapes():
    """
    Test: Full YOLO11n model produces correct output shapes.

    For 320x320 input with num_classes=2:
    - det_p3: (batch, 6, 40, 40)
    - det_p4: (batch, 6, 20, 20)
    - det_p5: (batch, 6, 10, 10)
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    det_p3, det_p4, det_p5 = model(x)

    f = function([x], [det_p3, det_p4, det_p5])

    batch = 2
    x_val = np.random.randn(batch, 3, 320, 320).astype("float32")
    det_p3_val, det_p4_val, det_p5_val = f(x_val)

    assert det_p3_val.shape == (batch, 6, 40, 40), f"det_p3 shape: {det_p3_val.shape}"
    assert det_p4_val.shape == (batch, 6, 20, 20), f"det_p4 shape: {det_p4_val.shape}"
    assert det_p5_val.shape == (batch, 6, 10, 10), f"det_p5 shape: {det_p5_val.shape}"


def test_yolo11n_different_num_classes():
    """
    Test: Model handles different numbers of classes correctly.

    Output channels should be 4 + num_classes.
    """
    test_cases = [
        (2, 6),  # 4 bbox + 2 classes
        (20, 24),  # 4 bbox + 20 classes
        (80, 84),  # 4 bbox + 80 classes (COCO)
    ]

    for num_classes, expected_channels in test_cases:
        x = pt.tensor4("x", dtype="float32")
        model = YOLO11n(num_classes=num_classes, input_size=320)
        det_p3, det_p4, det_p5 = model(x)

        f = function([x], [det_p3, det_p4, det_p5])

        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        det_p3_val, _det_p4_val, _det_p5_val = f(x_val)

        assert det_p3_val.shape[1] == expected_channels, (
            f"For {num_classes} classes, expected {expected_channels} channels, "
            f"got {det_p3_val.shape[1]}"
        )


def test_build_yolo11n_helper():
    """
    Test: build_yolo11n helper function works correctly.

    The helper should return (model, input_symbol, predictions).
    """
    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Verify return types
    assert isinstance(model, YOLO11n), f"model is {type(model)}, not YOLO11n"
    assert hasattr(x_sym, "type"), "x_sym is not a PyTensor variable"
    assert isinstance(predictions, tuple), "predictions is not a tuple"
    assert len(predictions) == 3, (
        f"predictions has {len(predictions)} elements, expected 3"
    )

    # Verify we can compile and run
    f = function([x_sym], predictions)
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    result = f(x_val)

    assert len(result) == 3, "Function execution failed"


def test_yolo11n_batch_size_flexibility():
    """
    Property: Model handles different batch sizes correctly.

    YOLO should work with batch sizes from 1 to larger values.
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    det_p3, det_p4, det_p5 = model(x)

    f = function([x], [det_p3, det_p4, det_p5])

    for batch_size in [1, 2, 4, 8]:
        x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
        det_p3_val, det_p4_val, det_p5_val = f(x_val)

        assert det_p3_val.shape[0] == batch_size, (
            f"Batch size {batch_size}: det_p3 batch dim is {det_p3_val.shape[0]}"
        )
        assert det_p4_val.shape[0] == batch_size, (
            f"Batch size {batch_size}: det_p4 batch dim is {det_p4_val.shape[0]}"
        )
        assert det_p5_val.shape[0] == batch_size, (
            f"Batch size {batch_size}: det_p5 batch dim is {det_p5_val.shape[0]}"
        )


@settings(deadline=None)
def test_yolo11n_different_input_sizes():
    """
    Test: Model works with different input sizes (multiples of 32).

    YOLO architecture should handle various input resolutions.
    """
    test_sizes = [(320, 320), (416, 416), (640, 640)]

    for input_size in test_sizes:
        x = pt.tensor4("x", dtype="float32")
        model = YOLO11n(num_classes=2, input_size=input_size[0])
        det_p3, det_p4, det_p5 = model(x)

        f = function([x], [det_p3, det_p4, det_p5])

        h, w = input_size
        x_val = np.random.randn(1, 3, h, w).astype("float32")
        det_p3_val, det_p4_val, det_p5_val = f(x_val)

        # Verify output spatial dimensions
        expected_p3_size = (h // 8, w // 8)
        expected_p4_size = (h // 16, w // 16)
        expected_p5_size = (h // 32, w // 32)

        assert det_p3_val.shape[2:] == expected_p3_size, (
            f"Input {input_size}: P3 spatial dims {det_p3_val.shape[2:]}, "
            f"expected {expected_p3_size}"
        )
        assert det_p4_val.shape[2:] == expected_p4_size, (
            f"Input {input_size}: P4 spatial dims {det_p4_val.shape[2:]}, "
            f"expected {expected_p4_size}"
        )
        assert det_p5_val.shape[2:] == expected_p5_size, (
            f"Input {input_size}: P5 spatial dims {det_p5_val.shape[2:]}, "
            f"expected {expected_p5_size}"
        )


def test_yolo11n_output_values_reasonable():
    """
    Property: Model outputs have reasonable value ranges.

    Detection outputs shouldn't contain NaN, Inf, or extreme values.
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    det_p3, det_p4, det_p5 = model(x)

    f = function([x], [det_p3, det_p4, det_p5])

    x_val = np.random.randn(2, 3, 320, 320).astype("float32")
    det_p3_val, det_p4_val, det_p5_val = f(x_val)

    for name, det_val in [("P3", det_p3_val), ("P4", det_p4_val), ("P5", det_p5_val)]:
        # Check for NaN/Inf
        assert np.all(np.isfinite(det_val)), f"{name} contains NaN or Inf"

        # Check for reasonable ranges (not too extreme)
        assert np.abs(det_val).max() < 1e6, (
            f"{name} has extreme values: {np.abs(det_val).max()}"
        )
