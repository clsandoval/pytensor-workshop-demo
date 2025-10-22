"""Tests for YOLO loss functions."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from yolo.loss import box_iou, yolo_loss
from yolo.model import YOLO11n

import pytensor.tensor as pt
from pytensor import function


@given(
    box=st.lists(
        st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        min_size=4,
        max_size=4,
    )
)
def test_box_iou_identity(box):
    """
    Property: IoU of a box with itself is 1.0.

    For any box [x_c, y_c, w, h], IoU(box, box) = 1.0
    """
    box1 = pt.vector("box1", dtype="float32")
    box2 = pt.vector("box2", dtype="float32")
    iou = box_iou(box1, box2)

    f = function([box1, box2], iou)

    box_arr = np.array(box, dtype="float32")
    iou_val = f(box_arr, box_arr)

    np.testing.assert_allclose(iou_val, 1.0, rtol=1e-6, err_msg="IoU(box, box) != 1.0")


@given(
    box1=st.lists(st.floats(0.01, 0.99, allow_nan=False), min_size=4, max_size=4),
    box2=st.lists(st.floats(0.01, 0.99, allow_nan=False), min_size=4, max_size=4),
)
def test_box_iou_symmetry(box1, box2):
    """
    Property: IoU is symmetric - IoU(A, B) = IoU(B, A).
    """
    b1 = pt.vector("box1", dtype="float32")
    b2 = pt.vector("box2", dtype="float32")
    iou_12 = box_iou(b1, b2)
    iou_21 = box_iou(b2, b1)

    f = function([b1, b2], [iou_12, iou_21])

    box1_arr = np.array(box1, dtype="float32")
    box2_arr = np.array(box2, dtype="float32")

    iou_12_val, iou_21_val = f(box1_arr, box2_arr)

    np.testing.assert_allclose(
        iou_12_val, iou_21_val, rtol=1e-6, err_msg="IoU not symmetric"
    )


@given(
    box1=st.lists(st.floats(0.01, 0.99, allow_nan=False), min_size=4, max_size=4),
    box2=st.lists(st.floats(0.01, 0.99, allow_nan=False), min_size=4, max_size=4),
)
def test_box_iou_bounds(box1, box2):
    """
    Property: IoU is bounded between 0 and 1.

    For any two boxes, 0 <= IoU(box1, box2) <= 1
    """
    b1 = pt.vector("box1", dtype="float32")
    b2 = pt.vector("box2", dtype="float32")
    iou = box_iou(b1, b2)

    f = function([b1, b2], iou)

    box1_arr = np.array(box1, dtype="float32")
    box2_arr = np.array(box2, dtype="float32")

    iou_val = f(box1_arr, box2_arr)

    assert 0.0 <= iou_val <= 1.0, f"IoU out of bounds: {iou_val}"


def test_box_iou_no_overlap():
    """
    Test: IoU is 0 for non-overlapping boxes.

    Two boxes that don't overlap should have IoU = 0.
    """
    b1 = pt.vector("box1", dtype="float32")
    b2 = pt.vector("box2", dtype="float32")
    iou = box_iou(b1, b2)

    f = function([b1, b2], iou)

    # Box 1: centered at (0.25, 0.25), size (0.2, 0.2)
    box1 = np.array([0.25, 0.25, 0.2, 0.2], dtype="float32")

    # Box 2: centered at (0.75, 0.75), size (0.2, 0.2) - far apart
    box2 = np.array([0.75, 0.75, 0.2, 0.2], dtype="float32")

    iou_val = f(box1, box2)

    np.testing.assert_allclose(
        iou_val, 0.0, atol=1e-6, err_msg="Non-overlapping boxes have non-zero IoU"
    )


def test_box_iou_partial_overlap():
    """
    Test: IoU for partially overlapping boxes is between 0 and 1.

    Two boxes with partial overlap should have 0 < IoU < 1.
    """
    b1 = pt.vector("box1", dtype="float32")
    b2 = pt.vector("box2", dtype="float32")
    iou = box_iou(b1, b2)

    f = function([b1, b2], iou)

    # Box 1: centered at (0.4, 0.4), size (0.4, 0.4)
    box1 = np.array([0.4, 0.4, 0.4, 0.4], dtype="float32")

    # Box 2: centered at (0.5, 0.5), size (0.4, 0.4) - overlapping
    box2 = np.array([0.5, 0.5, 0.4, 0.4], dtype="float32")

    iou_val = f(box1, box2)

    assert 0.0 < iou_val < 1.0, (
        f"Partial overlap IoU should be in (0, 1), got {iou_val}"
    )


@given(batch_size=st.integers(1, 4))
@settings(deadline=None)
def test_yolo_loss_non_negative(batch_size):
    """
    Property: YOLO loss is always non-negative.

    All loss components (bbox, classification) should be >= 0.
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    # Simplified targets (dummy)
    targets = {
        "boxes": pt.tensor3("boxes", dtype="float32"),
        "classes": pt.matrix("classes", dtype="int64"),
        "num_boxes": pt.vector("num_boxes", dtype="int64"),
    }

    total_loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)

    f = function(
        [x, targets["boxes"], targets["classes"], targets["num_boxes"]],
        [total_loss, loss_dict["box_loss"], loss_dict["cls_loss"]],
    )

    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
    boxes_val = np.zeros((batch_size, 10, 4), dtype="float32")
    classes_val = np.zeros((batch_size, 10), dtype="int64")
    num_boxes_val = np.array([0] * batch_size, dtype="int64")

    total, box_loss, cls_loss = f(x_val, boxes_val, classes_val, num_boxes_val)

    assert total >= 0, f"Total loss is negative: {total}"
    assert box_loss >= 0, f"Box loss is negative: {box_loss}"
    assert cls_loss >= 0, f"Cls loss is negative: {cls_loss}"


def test_yolo_loss_zero_targets():
    """
    Test: Loss with zero targets (no objects) should be finite.

    When there are no ground truth objects, the loss should still
    be computable and finite (classification loss only).
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    targets = {
        "boxes": pt.tensor3("boxes", dtype="float32"),
        "classes": pt.matrix("classes", dtype="int64"),
        "num_boxes": pt.vector("num_boxes", dtype="int64"),
    }

    total_loss, _loss_dict = yolo_loss(predictions, targets, num_classes=2)

    f = function(
        [x, targets["boxes"], targets["classes"], targets["num_boxes"]],
        total_loss,
    )

    batch_size = 2
    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
    boxes_val = np.zeros((batch_size, 10, 4), dtype="float32")
    classes_val = np.zeros((batch_size, 10), dtype="int64")
    num_boxes_val = np.array([0, 0], dtype="int64")  # No objects

    loss_val = f(x_val, boxes_val, classes_val, num_boxes_val)

    assert np.isfinite(loss_val), f"Loss with zero targets is not finite: {loss_val}"
    assert loss_val >= 0, f"Loss with zero targets is negative: {loss_val}"


def test_yolo_loss_with_targets():
    """
    Test: Loss with actual targets is finite and positive.

    When there are ground truth objects, the loss should be positive
    (model needs to learn to match the targets).
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    targets = {
        "boxes": pt.tensor3("boxes", dtype="float32"),
        "classes": pt.matrix("classes", dtype="int64"),
        "num_boxes": pt.vector("num_boxes", dtype="int64"),
    }

    total_loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)

    f = function(
        [x, targets["boxes"], targets["classes"], targets["num_boxes"]],
        [total_loss, loss_dict["box_loss"], loss_dict["cls_loss"]],
    )

    batch_size = 2
    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")

    # Create some dummy targets
    boxes_val = np.array(
        [
            [
                [0.5, 0.5, 0.3, 0.3],
                [0.2, 0.2, 0.1, 0.1],
                [0.0, 0.0, 0.0, 0.0],
            ],  # batch 0: 2 objects
            [
                [0.7, 0.7, 0.2, 0.2],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],  # batch 1: 1 object
        ],
        dtype="float32",
    )

    classes_val = np.array(
        [
            [0, 1, 0],  # batch 0: class 0, 1, padding
            [1, 0, 0],  # batch 1: class 1, padding, padding
        ],
        dtype="int64",
    )

    num_boxes_val = np.array([2, 1], dtype="int64")

    total, box_loss, cls_loss = f(x_val, boxes_val, classes_val, num_boxes_val)

    assert np.isfinite(total), f"Loss is not finite: {total}"
    assert np.isfinite(box_loss), f"Box loss is not finite: {box_loss}"
    assert np.isfinite(cls_loss), f"Cls loss is not finite: {cls_loss}"

    assert total > 0, f"Loss should be positive for untrained model: {total}"


def test_yolo_loss_components_sum():
    """
    Property: Total loss equals sum of component losses.

    total_loss = box_loss + cls_loss
    """
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    targets = {
        "boxes": pt.tensor3("boxes", dtype="float32"),
        "classes": pt.matrix("classes", dtype="int64"),
        "num_boxes": pt.vector("num_boxes", dtype="int64"),
    }

    total_loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)

    f = function(
        [x, targets["boxes"], targets["classes"], targets["num_boxes"]],
        [total_loss, loss_dict["box_loss"], loss_dict["cls_loss"]],
    )

    batch_size = 1
    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
    boxes_val = np.zeros((batch_size, 5, 4), dtype="float32")
    classes_val = np.zeros((batch_size, 5), dtype="int64")
    num_boxes_val = np.array([0], dtype="int64")

    total, box_loss, cls_loss = f(x_val, boxes_val, classes_val, num_boxes_val)

    expected_total = box_loss + cls_loss

    np.testing.assert_allclose(
        total,
        expected_total,
        rtol=1e-5,
        err_msg=f"Total loss {total} != sum of components {expected_total}",
    )
