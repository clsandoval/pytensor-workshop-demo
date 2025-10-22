#!/usr/bin/env python
"""Test the JAX-fixed model."""

import os


# Set environment before imports
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=None"

import numpy as np

import pytensor


def test_jax_fixed_model():
    """Test if the fixed model works with JAX."""
    print("\n" + "=" * 70)
    print("Testing JAX-Fixed Model")
    print("=" * 70)

    print(
        f"Config: floatX={pytensor.config.floatX}, optimizer={pytensor.config.optimizer}"
    )

    try:
        import jax

        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("JAX not available")
        return False

    # Import the fixed model
    from yolo.model_jax_fixed import build_yolo11n_jax_fixed

    try:
        print("\nBuilding JAX-fixed model...")
        _model, x_sym, predictions = build_yolo11n_jax_fixed(
            num_classes=2, input_size=320
        )

        print("Compiling with JAX...")
        f = pytensor.function([x_sym], predictions, mode="JAX")

        print("Testing forward pass...")
        test_input = np.random.randn(1, 3, 320, 320).astype("float32")
        outputs = f(test_input)

        print("\n✓ SUCCESS! JAX-fixed model works!")
        print(f"Output shapes: {[o.shape for o in outputs]}")

        # Test with different batch sizes
        print("\nTesting different batch sizes...")
        for batch_size in [1, 2, 4]:
            test_input = np.random.randn(batch_size, 3, 320, 320).astype("float32")
            outputs = f(test_input)
            print(f"  Batch {batch_size}: ✓ {[o.shape for o in outputs]}")

        return True

    except Exception as e:
        print(f"\n✗ JAX-fixed model failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_jax_fixed_model()

    if success:
        print("\n" + "=" * 70)
        print("JAX-FIXED MODEL WORKS! 🎉")
        print("=" * 70)
        print("\nThe issue was in the upsampling implementation.")
        print("The fix: Use repeat() instead of reshape() with dynamic shapes.")
    else:
        print("\n" + "=" * 70)
        print("JAX-fixed model still has issues")
        print("=" * 70)
