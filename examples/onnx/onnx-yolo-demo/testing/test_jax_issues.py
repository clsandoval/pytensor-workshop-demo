"""
Test individual JAX JIT compatibility issues in YOLO11n.

This script tests each potential problem in isolation to identify
exactly what breaks with JAX backend.
"""

import os


os.environ["PYTENSOR_FLAGS"] = "floatX=float32"

import sys

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor import function


# Try to set JAX mode
try:
    import jax

    pytensor.config.mode = "JAX"
    print(f"✓ JAX mode enabled: {pytensor.config.mode}")
    print(f"  JAX devices: {jax.devices()}")
except Exception as e:
    print(f"✗ Could not enable JAX mode: {e}")
    sys.exit(1)


def test_repeat_operation():
    """Test if pt.repeat works with JAX JIT."""
    print("\n[Test 1] pt.repeat operation")
    try:
        x = pt.tensor4("x", dtype="float32")
        # This is what _upsample does
        x_up = pt.repeat(x, 2, axis=2)
        x_up = pt.repeat(x_up, 2, axis=3)

        f = function([x], x_up)

        x_val = np.random.randn(1, 16, 10, 10).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ pt.repeat works!")
        return True
    except Exception as e:
        print(f"  ✗ pt.repeat FAILS: {e}")
        return False


def test_dictionary_return():
    """Test if dictionary returns work with JAX."""
    print("\n[Test 2] Dictionary return from function")
    try:
        x = pt.tensor4("x", dtype="float32")
        y1 = x * 2
        y2 = x * 3

        # Can't actually return dict from PyTensor function
        # but we can return tuple and check if that works
        f = function([x], [y1, y2])

        x_val = np.random.randn(1, 3, 10, 10).astype("float32")
        r1, r2 = f(x_val)

        print(f"  ✓ Multiple returns work (r1: {r1.shape}, r2: {r2.shape})")
        print("  Note: PyTensor functions can't return dicts anyway")
        return True
    except Exception as e:
        print(f"  ✗ Multiple returns FAIL: {e}")
        return False


def test_ellipsis_slicing():
    """Test if ellipsis slicing works with JAX."""
    print("\n[Test 3] Ellipsis slicing")
    try:
        x = pt.tensor4("x", dtype="float32")
        # This is what loss function does
        y1 = x[..., :4]

        f = function([x], y1)

        x_val = np.random.randn(2, 20, 20, 10).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ Ellipsis slicing works!")
        return True
    except Exception as e:
        print(f"  ✗ Ellipsis slicing FAILS: {e}")
        return False


def test_complex_model():
    """Test a simplified version of YOLO head with upsampling."""
    print("\n[Test 4] Simplified YOLO head with upsample")
    try:
        from pytensor import shared

        # Simplified head
        x = pt.tensor4("x", dtype="float32")

        # Simulate what head does
        # 1. Upsample (THIS IS THE PROBLEM)
        x_up = pt.repeat(x, 2, axis=2)
        x_up = pt.repeat(x_up, 2, axis=3)

        # 2. Conv (simplified as matrix multiply)
        W = shared(np.random.randn(16, 16, 3, 3).astype("float32"), name="W")
        y = pt.conv2d(x_up, W, border_mode="same")

        f = function([x], y)

        x_val = np.random.randn(1, 16, 10, 10).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print("  After upsample: (1, 16, 20, 20) [expected]")
        print(f"  Output shape: {result.shape}")
        print("  ✓ Simplified head works!")
        return True
    except Exception as e:
        print(f"  ✗ Simplified head FAILS: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_yolo_model():
    """Test actual YOLO11n model with JAX."""
    print("\n[Test 5] Full YOLO11n model")
    try:
        sys.path.insert(0, "..")
        from yolo.model import build_yolo11n

        _model, x, predictions = build_yolo11n(num_classes=2, input_size=320)

        # predictions is now a tuple (pred_p3, pred_p4, pred_p5)
        pred_p3, pred_p4, pred_p5 = predictions

        # Try to compile forward pass
        f = function([x], [pred_p3, pred_p4, pred_p5])

        print("  ✓ Model compilation started...")
        print("  (This may take several minutes with JAX JIT...)")

        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        p3, p4, p5 = f(x_val)

        print(f"  P3 shape: {p3.shape}")
        print(f"  P4 shape: {p4.shape}")
        print(f"  P5 shape: {p5.shape}")
        print("  ✓ Full YOLO11n works with JAX!")
        return True
    except Exception as e:
        print(f"  ✗ Full YOLO11n FAILS: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step with gradients and updates."""
    print("\n[Test 6] Training step with gradients")
    try:
        sys.path.insert(0, "..")
        from yolo.loss import yolo_loss
        from yolo.model import build_yolo11n

        from pytensor import shared

        # Build model
        model, x, predictions = build_yolo11n(num_classes=2, input_size=320)

        # Define loss
        loss, _loss_dict = yolo_loss(predictions, targets=None, num_classes=2)

        # Compute gradient for first parameter only (for speed)
        test_param = model.params[0]
        grad = pytensor.grad(loss, test_param)

        # Simple update
        momentum = pt.as_tensor_variable(np.float32(0.9))
        lr = pt.as_tensor_variable(np.float32(0.01))
        velocity = shared(
            np.zeros_like(test_param.get_value(), dtype="float32"), name="v"
        )

        v_new = momentum * velocity - lr * pt.cast(grad, "float32")
        p_new = test_param + v_new

        updates = [(velocity, v_new), (test_param, p_new)]

        # Compile
        print("  Compiling training function...")
        train_fn = function([x], loss, updates=updates, name="train_step")

        print("  Running training step...")
        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        loss_val = train_fn(x_val)

        print(f"  Loss: {loss_val:.6f}")
        print("  ✓ Training step works with JAX!")
        return True
    except Exception as e:
        print(f"  ✗ Training step FAILS: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("JAX JIT Compatibility Tests for YOLO11n")
    print("=" * 70)

    results = {}

    # Test each issue
    results["repeat"] = test_repeat_operation()
    results["dict_return"] = test_dictionary_return()
    results["ellipsis"] = test_ellipsis_slicing()
    results["complex_model"] = test_complex_model()
    results["full_model"] = test_full_yolo_model()
    results["training_step"] = test_training_step()

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")

    print("\n" + "=" * 70)

    all_passed = all(results.values())
    if all_passed:
        print("✓ All tests passed! JAX training should work.")
    else:
        print("✗ Some tests failed. JAX training will not work.")
        print("\nFailed tests indicate what needs to be fixed:")
        for name, passed in results.items():
            if not passed:
                print(f"  - {name}")

    print("=" * 70)


if __name__ == "__main__":
    main()
