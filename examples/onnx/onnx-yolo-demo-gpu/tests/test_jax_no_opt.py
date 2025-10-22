#!/usr/bin/env python
"""Test JAX backend with NO optimizations (nuclear option)."""

import os


# Set environment before any imports - DISABLE ALL OPTIMIZATIONS
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=None"

import numpy as np
import pytensor.config

import pytensor


# Force settings
pytensor.config.floatX = "float32"
pytensor.config.optimizer = "None"  # Disable ALL optimizations


def test_model_no_optimizations():
    """Test if model works with JAX when ALL optimizations are disabled."""
    print("\n" + "=" * 70)
    print("JAX Test with NO Optimizations")
    print("=" * 70)

    print(f"floatX: {pytensor.config.floatX}")
    print(f"optimizer: {pytensor.config.optimizer}")
    print(f"optimizer_excluding: {pytensor.config.optimizer_excluding}")

    try:
        import jax

        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("JAX not available")
        return False

    # Import model AFTER setting config
    from yolo.model import build_yolo11n

    try:
        print("\nBuilding model...")
        _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

        print("Compiling with JAX (no optimizations)...")
        f = pytensor.function([x_sym], predictions, mode="JAX")

        print("Testing forward pass...")
        test_input = np.random.randn(1, 3, 320, 320).astype("float32")
        outputs = f(test_input)

        print("\n✓ SUCCESS! Model works with JAX when optimizations are disabled")
        print(f"Output shapes: {[o.shape for o in outputs]}")
        return True

    except Exception as e:
        print(f"\n✗ FAILED even with no optimizations: {e}")
        return False


def test_gradual_optimization():
    """Test which optimizations cause the problem."""
    print("\n" + "=" * 70)
    print("Testing Different Optimization Levels")
    print("=" * 70)

    configs_to_test = [
        ("None", "No optimizations"),
        ("fast_compile", "Fast compile mode"),
        ("fast_run", "Fast run mode (default)"),
    ]

    from yolo.model import build_yolo11n

    for optimizer, description in configs_to_test:
        print(f"\nTesting: {description} (optimizer={optimizer})")

        # Reset and set config
        pytensor.config.optimizer = optimizer
        pytensor.config.floatX = "float32"

        if optimizer != "None":
            pytensor.config.optimizer_excluding = "shape_unsafe"

        try:
            _model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)
            f = pytensor.function([x_sym], predictions, mode="JAX")

            test_input = np.random.randn(1, 3, 320, 320).astype("float32")
            _ = f(test_input)

            print(f"  ✓ {description} works!")

        except Exception as e:
            print(f"  ✗ {description} failed: {type(e).__name__}")


if __name__ == "__main__":
    success = test_model_no_optimizations()

    if success:
        print(
            "\nSince no optimizations work, let's test which optimization level breaks it:"
        )
        test_gradual_optimization()

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)
