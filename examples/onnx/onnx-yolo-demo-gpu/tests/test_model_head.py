"""Tests for YOLO11n Head (FPN-PAN architecture)."""

import numpy as np
import pytest
from yolo.model import YOLO11nHead

import pytensor.tensor as pt
from pytensor import function


def test_head_fpn_upsampling_correctness():
    """
    Property: FPN upsampling path doubles spatial dimensions correctly.

    FPN path:
    - P5 (10x10) → upsample 2x → (20x20) → concat with P4 → C3k2
    - P4_fused (20x20) → upsample 2x → (40x40) → concat with P3 → C3k2
    """
    # Create symbolic inputs matching backbone output shapes
    p3 = pt.tensor4("p3", dtype="float32")  # (batch, 64, 40, 40)
    p4 = pt.tensor4("p4", dtype="float32")  # (batch, 128, 20, 20)
    p5 = pt.tensor4("p5", dtype="float32")  # (batch, 256, 10, 10)

    head = YOLO11nHead(num_classes=2)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    # Test data
    batch = 2
    p3_val = np.random.randn(batch, 64, 40, 40).astype("float32")
    p4_val = np.random.randn(batch, 128, 20, 20).astype("float32")
    p5_val = np.random.randn(batch, 256, 10, 10).astype("float32")

    det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

    # Detection outputs: (batch, 4+num_classes, H, W)
    assert det_p3_val.shape == (batch, 6, 40, 40), f"det_p3 shape: {det_p3_val.shape}"
    assert det_p4_val.shape == (batch, 6, 20, 20), f"det_p4 shape: {det_p4_val.shape}"
    assert det_p5_val.shape == (batch, 6, 10, 10), f"det_p5 shape: {det_p5_val.shape}"


@pytest.mark.parametrize("num_classes", [2, 20, 80])
def test_head_detection_channel_format(num_classes):
    """
    Property: Detection heads output (4 + num_classes) channels.

    Channel format: [x_offset, y_offset, w, h, class_0, class_1, ..., class_N]
    """
    p3 = pt.tensor4("p3", dtype="float32")
    p4 = pt.tensor4("p4", dtype="float32")
    p5 = pt.tensor4("p5", dtype="float32")

    head = YOLO11nHead(num_classes=num_classes)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    batch = 1
    p3_val = np.random.randn(batch, 64, 40, 40).astype("float32")
    p4_val = np.random.randn(batch, 128, 20, 20).astype("float32")
    p5_val = np.random.randn(batch, 256, 10, 10).astype("float32")

    det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

    expected_channels = 4 + num_classes

    assert det_p3_val.shape[1] == expected_channels, (
        f"det_p3 channels: expected {expected_channels}, got {det_p3_val.shape[1]}"
    )
    assert det_p4_val.shape[1] == expected_channels, (
        f"det_p4 channels: expected {expected_channels}, got {det_p4_val.shape[1]}"
    )
    assert det_p5_val.shape[1] == expected_channels, (
        f"det_p5 channels: expected {expected_channels}, got {det_p5_val.shape[1]}"
    )


def test_head_spatial_dimensions_preserved():
    """
    Property: Head preserves spatial dimensions from backbone.

    If backbone outputs (40x40, 20x20, 10x10), head detection outputs
    should maintain these same spatial dimensions.
    """
    p3 = pt.tensor4("p3", dtype="float32")
    p4 = pt.tensor4("p4", dtype="float32")
    p5 = pt.tensor4("p5", dtype="float32")

    head = YOLO11nHead(num_classes=2)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    # Test with various spatial dimensions
    batch = 1
    p3_val = np.random.randn(batch, 64, 40, 40).astype("float32")
    p4_val = np.random.randn(batch, 128, 20, 20).astype("float32")
    p5_val = np.random.randn(batch, 256, 10, 10).astype("float32")

    det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

    # Spatial dimensions should match input feature maps
    assert det_p3_val.shape[2:] == (40, 40), "P3 spatial dims not preserved"
    assert det_p4_val.shape[2:] == (20, 20), "P4 spatial dims not preserved"
    assert det_p5_val.shape[2:] == (10, 10), "P5 spatial dims not preserved"


def test_head_deterministic():
    """
    Test: Head produces deterministic outputs.

    Running the same inputs twice should produce identical results.
    """
    p3 = pt.tensor4("p3", dtype="float32")
    p4 = pt.tensor4("p4", dtype="float32")
    p5 = pt.tensor4("p5", dtype="float32")

    head = YOLO11nHead(num_classes=2)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    batch = 2
    p3_val = np.random.randn(batch, 64, 40, 40).astype("float32")
    p4_val = np.random.randn(batch, 128, 20, 20).astype("float32")
    p5_val = np.random.randn(batch, 256, 10, 10).astype("float32")

    # Run twice
    out1 = f(p3_val, p4_val, p5_val)
    out2 = f(p3_val, p4_val, p5_val)

    # Should be identical
    for o1, o2 in zip(out1, out2):
        np.testing.assert_array_equal(o1, o2, err_msg="Head is non-deterministic")


def test_head_batch_independence():
    """
    Property: Head processes each batch element independently.

    Processing batch of 2 should give same results as processing
    each element separately.
    """
    p3 = pt.tensor4("p3", dtype="float32")
    p4 = pt.tensor4("p4", dtype="float32")
    p5 = pt.tensor4("p5", dtype="float32")

    head = YOLO11nHead(num_classes=2)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    # Create test data
    p3_val_batch = np.random.randn(2, 64, 40, 40).astype("float32")
    p4_val_batch = np.random.randn(2, 128, 20, 20).astype("float32")
    p5_val_batch = np.random.randn(2, 256, 10, 10).astype("float32")

    # Process as batch
    batch_out = f(p3_val_batch, p4_val_batch, p5_val_batch)

    # Process individually
    individual_outs = []
    for i in range(2):
        p3_single = p3_val_batch[i : i + 1]
        p4_single = p4_val_batch[i : i + 1]
        p5_single = p5_val_batch[i : i + 1]
        individual_out = f(p3_single, p4_single, p5_single)
        individual_outs.append(individual_out)

    # Compare
    for scale_idx in range(3):
        batch_result = batch_out[scale_idx]
        for batch_elem_idx in range(2):
            individual_result = individual_outs[batch_elem_idx][scale_idx]
            np.testing.assert_allclose(
                batch_result[batch_elem_idx : batch_elem_idx + 1],
                individual_result,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Batch element {batch_elem_idx} at scale {scale_idx} differs",
            )
