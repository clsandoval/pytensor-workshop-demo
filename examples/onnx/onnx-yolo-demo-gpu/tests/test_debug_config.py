#!/usr/bin/env python
"""Debug test to verify PyTensor configuration for JAX."""

import os


# Set environment before any imports
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer_excluding=shape_unsafe"

import numpy as np

import pytensor
import pytensor.tensor as pt


def test_pytensor_config():
    """Verify PyTensor configuration is correct."""
    print("\n" + "=" * 70)
    print("PyTensor Configuration Check")
    print("=" * 70)

    print(f"floatX: {pytensor.config.floatX}")
    print(f"optimizer_excluding: {pytensor.config.optimizer_excluding}")
    print(f"mode: {pytensor.config.mode}")
    print(f"optimizer: {pytensor.config.optimizer}")

    assert pytensor.config.floatX == "float32", (
        f"Expected floatX=float32, got {pytensor.config.floatX}"
    )
    assert "shape_unsafe" in str(pytensor.config.optimizer_excluding), (
        f"Expected shape_unsafe in optimizer_excluding, got {pytensor.config.optimizer_excluding}"
    )

    print("\n✓ Configuration is correct")


def test_simple_jax_compile():
    """Test if a simple function compiles with JAX."""
    print("\n" + "=" * 70)
    print("Simple JAX Compilation Test")
    print("=" * 70)

    try:
        import jax

        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("JAX not available, skipping test")
        return

    # Simple test
    x = pt.matrix("x", dtype="float32")
    y = x * 2 + 1

    try:
        # Try to compile with JAX mode
        f = pytensor.function([x], y, mode="JAX")

        # Test with simple input
        test_input = np.array([[1, 2], [3, 4]], dtype="float32")
        result = f(test_input)

        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {result.shape}")
        print(f"Output type: {type(result)}")
        print("\n✓ Simple JAX compilation works")

    except Exception as e:
        print(f"\n✗ JAX compilation failed: {e}")
        raise


def test_model_compile_with_explicit_config():
    """Test model compilation with explicit config override."""
    print("\n" + "=" * 70)
    print("Model JAX Compilation Test (with explicit config)")
    print("=" * 70)

    try:
        import jax  # noqa: F401
    except ImportError:
        print("JAX not available, skipping test")
        return

    # Double-check and force configuration
    import pytensor.config

    pytensor.config.optimizer_excluding = "shape_unsafe"
    pytensor.config.floatX = "float32"

    print(f"Forced optimizer_excluding: {pytensor.config.optimizer_excluding}")

    # Import model AFTER setting config
    from yolo.model import build_yolo11n

    try:
        _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

        # Try compiling with JAX
        f = pytensor.function([x_sym], predictions, mode="JAX")

        # Test with small input
        test_input = np.random.randn(1, 3, 320, 320).astype("float32")
        outputs = f(test_input)

        print("Model compiled successfully with JAX!")
        print(f"Output shapes: {[o.shape for o in outputs]}")
        print("\n✓ Model JAX compilation works with explicit config")

    except Exception as e:
        print(f"\n✗ Model JAX compilation failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_pytensor_config()
    test_simple_jax_compile()
    test_model_compile_with_explicit_config()

    print("\n" + "=" * 70)
    print("All debug tests completed")
    print("=" * 70)
