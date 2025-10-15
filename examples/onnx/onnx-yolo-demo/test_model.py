"""
Quick tests for YOLO11n model architecture.

Run: python test_model.py
"""

import sys

import numpy as np


# Add parent directory to path for imports
sys.path.insert(0, ".")

from blocks import SPPF, C3k2, ConvBNSiLU
from model import build_yolo11n

import pytensor
import pytensor.tensor as pt
from pytensor import function


def test_conv_bn_silu():
    """Test ConvBNSiLU block."""
    print("\n[Test 1/5] Testing ConvBNSiLU block...")

    # Create block
    conv = ConvBNSiLU(3, 16, kernel_size=3, stride=2, padding="same")

    # Input
    x = pt.tensor4("x", dtype="float32")
    y = conv(x)

    # Compile
    f = function([x], y)

    # Test
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    y_val = f(x_val)

    expected_shape = (1, 16, 160, 160)
    assert y_val.shape == expected_shape, (
        f"Expected {expected_shape}, got {y_val.shape}"
    )
    print(f"  ✓ ConvBNSiLU output shape: {y_val.shape}")


def test_c3k2():
    """Test C3k2 block."""
    print("\n[Test 2/5] Testing C3k2 block...")

    # Create block
    c3k2 = C3k2(64, 64, n_blocks=2)

    # Input
    x = pt.tensor4("x", dtype="float32")
    y = c3k2(x)

    # Compile
    f = function([x], y)

    # Test
    x_val = np.random.randn(1, 64, 40, 40).astype("float32")
    y_val = f(x_val)

    expected_shape = (1, 64, 40, 40)
    assert y_val.shape == expected_shape, (
        f"Expected {expected_shape}, got {y_val.shape}"
    )
    print(f"  ✓ C3k2 output shape: {y_val.shape}")


def test_sppf():
    """Test SPPF block."""
    print("\n[Test 3/5] Testing SPPF block...")

    # Create block
    sppf = SPPF(256, 256, pool_size=5)

    # Input
    x = pt.tensor4("x", dtype="float32")
    y = sppf(x)

    # Compile
    f = function([x], y)

    # Test
    x_val = np.random.randn(1, 256, 10, 10).astype("float32")
    y_val = f(x_val)

    expected_shape = (1, 256, 10, 10)
    assert y_val.shape == expected_shape, (
        f"Expected {expected_shape}, got {y_val.shape}"
    )
    print(f"  ✓ SPPF output shape: {y_val.shape}")


def test_yolo11n_forward():
    """Test full YOLO11n forward pass."""
    print("\n[Test 4/5] Testing full YOLO11n forward pass...")

    # Build model
    _model, x, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile forward pass
    f = function([x], [predictions["p3"], predictions["p4"], predictions["p5"]])

    # Test
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")
    p3_val, p4_val, p5_val = f(x_val)

    # Verify shapes (320x320 input → 40x40, 20x20, 10x10 outputs)
    # For 2 classes: 4 bbox coords + 2 classes = 6 channels
    assert p3_val.shape == (2, 6, 40, 40), f"P3 shape: {p3_val.shape}"
    assert p4_val.shape == (2, 6, 20, 20), f"P4 shape: {p4_val.shape}"
    assert p5_val.shape == (2, 6, 10, 10), f"P5 shape: {p5_val.shape}"

    print(f"  ✓ P3 output shape: {p3_val.shape}")
    print(f"  ✓ P4 output shape: {p4_val.shape}")
    print(f"  ✓ P5 output shape: {p5_val.shape}")


def test_yolo11n_gradients():
    """Test that gradients can be computed through the model."""
    print("\n[Test 5/5] Testing gradient computation...")

    # Build model
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Simple loss (just sum of outputs)
    loss = predictions["p3"].sum() + predictions["p4"].sum() + predictions["p5"].sum()

    # Compute gradients w.r.t. first few parameters
    test_params = model.params[:5]  # Just test first 5 params
    grads = pytensor.grad(loss, test_params)

    # Compile
    f = function([x], grads)

    # Test
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    grad_vals = f(x_val)

    # Verify gradients are non-zero and have correct shapes
    for i, (param, grad) in enumerate(zip(test_params, grad_vals)):
        assert grad.shape == param.get_value().shape, f"Gradient {i} shape mismatch"
        assert np.abs(grad).sum() > 0, f"Gradient {i} is all zeros"

    print(f"  ✓ Computed gradients for {len(test_params)} parameters")
    print("  ✓ All gradients are non-zero")


def main():
    """Run all tests."""
    print("=" * 70)
    print(" " * 20 + "YOLO11n Model Tests")
    print("=" * 70)

    try:
        test_conv_bn_silu()
        test_c3k2()
        test_sppf()
        test_yolo11n_forward()
        test_yolo11n_gradients()

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
