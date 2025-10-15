#!/usr/bin/env python3
"""
Comprehensive JAX JIT compatibility test for YOLO11n components.

This script tests each component of the YOLO11n model independently
to identify which parts work with JAX JIT compilation.

Run on GPU server:
    PYTENSOR_FLAGS="floatX=float32" python test_jax_components.py

Or with JAX mode enabled:
    PYTENSOR_FLAGS="floatX=float32" python test_jax_components.py --jax
"""

import argparse
import os
import sys
import time

import numpy as np


# Set PyTensor flags before importing
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"

import pytensor
import pytensor.tensor as pt
from pytensor import function, shared


def setup_jax(enable_jax=False):
    """Setup JAX backend if requested."""
    if enable_jax:
        try:
            import jax

            devices = jax.devices()
            device_type = devices[0].platform if len(devices) > 0 else "none"

            if device_type == "gpu":
                pytensor.config.mode = "JAX"
                print(f"✓ JAX mode enabled: {pytensor.config.mode}")
                print(f"  JAX devices: {devices}")
                print(f"  Device type: {device_type}")
                return True
            else:
                print(f"⚠ JAX found but no GPU: {device_type}")
                print("  Using default CPU backend")
                return False
        except Exception as e:
            print(f"⚠ Could not enable JAX: {e}")
            print("  Using default CPU backend")
            return False
    else:
        print("[INFO] Running in default mode (CPU)")
        print(f"  PyTensor mode: {pytensor.config.mode}")
        return False


# =============================================================================
# Test 1: Basic Operations
# =============================================================================


def test_basic_ops():
    """Test basic PyTensor operations."""
    print("\n[Test 1] Basic Operations")
    print("-" * 70)

    try:
        x = pt.tensor4("x", dtype="float32")
        y = x * 2 + 1

        f = function([x], y)
        x_val = np.random.randn(1, 3, 10, 10).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ Basic operations work")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


# =============================================================================
# Test 2: Dimshuffle and Tile
# =============================================================================


def test_dimshuffle_tile():
    """Test dimshuffle and tile operations."""
    print("\n[Test 2] Dimshuffle and Tile")
    print("-" * 70)

    try:
        x = pt.tensor4("x", dtype="float32")

        # Add dimensions
        x_expanded = x.dimshuffle(0, 1, 2, "x", 3, "x")

        # Tile
        x_tiled = pt.tile(x_expanded, (1, 1, 1, 2, 1, 2))

        f = function([x], [x_expanded, x_tiled])
        x_val = np.random.randn(1, 16, 10, 10).astype("float32")
        expanded, tiled = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Expanded shape: {expanded.shape}")
        print(f"  Tiled shape: {tiled.shape}")
        print("  ✓ Dimshuffle and tile work")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


# =============================================================================
# Test 3: Upsampling (Full)
# =============================================================================


def test_upsampling():
    """Test the complete upsampling operation."""
    print("\n[Test 3] Upsampling Operation")
    print("-" * 70)

    try:
        x = pt.tensor4("x", dtype="float32")
        scale = 2

        # Get input shape
        input_shape = x.shape
        batch_size = input_shape[0]
        channels = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]

        # Upsample
        x_expanded = x.dimshuffle(0, 1, 2, "x", 3, "x")
        x_tiled = pt.tile(x_expanded, (1, 1, 1, scale, 1, scale))
        x_rearranged = x_tiled.dimshuffle(0, 1, 2, 4, 3, 5)

        out_height = height * scale
        out_width = width * scale
        x_upsampled = x_rearranged.reshape(
            (batch_size, channels, out_height, out_width)
        )

        f = function([x], x_upsampled)
        x_val = np.random.randn(1, 16, 10, 10).astype("float32")
        result = f(x_val)

        expected = (1, 16, 20, 20)
        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Expected shape: {expected}")

        if result.shape == expected:
            print("  ✓ Upsampling works correctly")
            return True
        else:
            print("  ✗ Shape mismatch!")
            return False

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 4: ConvBNSiLU Block
# =============================================================================


def test_conv_bn_silu():
    """Test ConvBNSiLU building block."""
    print("\n[Test 4] ConvBNSiLU Block")
    print("-" * 70)

    try:
        from blocks import ConvBNSiLU

        conv = ConvBNSiLU(3, 16, kernel_size=3, stride=1, padding="same")

        x = pt.tensor4("x", dtype="float32")
        y = conv(x)

        f = function([x], y)
        x_val = np.random.randn(1, 3, 32, 32).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ ConvBNSiLU works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 5: Bottleneck Block
# =============================================================================


def test_bottleneck():
    """Test Bottleneck block."""
    print("\n[Test 5] Bottleneck Block")
    print("-" * 70)

    try:
        from blocks import Bottleneck

        bottleneck = Bottleneck(32, 32, shortcut=True, name_prefix="test")

        x = pt.tensor4("x", dtype="float32")
        y = bottleneck(x)

        f = function([x], y)
        x_val = np.random.randn(1, 32, 40, 40).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ Bottleneck works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 6: C3k2 Block
# =============================================================================


def test_c3k2():
    """Test C3k2 block."""
    print("\n[Test 6] C3k2 Block")
    print("-" * 70)

    try:
        from blocks import C3k2

        c3k2 = C3k2(64, 64, n_blocks=1, name_prefix="test")

        x = pt.tensor4("x", dtype="float32")
        y = c3k2(x)

        f = function([x], y)
        x_val = np.random.randn(1, 64, 40, 40).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ C3k2 works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 7: SPPF Block
# =============================================================================


def test_sppf():
    """Test SPPF block."""
    print("\n[Test 7] SPPF Block")
    print("-" * 70)

    try:
        from blocks import SPPF

        sppf = SPPF(256, 256, pool_size=5, name_prefix="test")

        x = pt.tensor4("x", dtype="float32")
        y = sppf(x)

        f = function([x], y)
        x_val = np.random.randn(1, 256, 10, 10).astype("float32")
        result = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  Output shape: {result.shape}")
        print("  ✓ SPPF works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 8: Backbone
# =============================================================================


def test_backbone():
    """Test YOLO11n backbone."""
    print("\n[Test 8] YOLO11n Backbone")
    print("-" * 70)

    try:
        from model import YOLO11nBackbone

        print("  Building backbone...")
        backbone = YOLO11nBackbone()

        x = pt.tensor4("x", dtype="float32")
        p3, p4, p5 = backbone(x)

        print("  Compiling function...")
        f = function([x], [p3, p4, p5])

        print("  Running forward pass...")
        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        p3_val, p4_val, p5_val = f(x_val)

        print(f"  Input shape: {x_val.shape}")
        print(f"  P3 shape: {p3_val.shape}")
        print(f"  P4 shape: {p4_val.shape}")
        print(f"  P5 shape: {p5_val.shape}")
        print("  ✓ Backbone works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 9: Detection Head
# =============================================================================


def test_head():
    """Test YOLO11n detection head."""
    print("\n[Test 9] YOLO11n Detection Head")
    print("-" * 70)

    try:
        from model import YOLO11nHead

        print("  Building head...")
        head = YOLO11nHead(num_classes=2)

        # Create dummy backbone outputs
        p3 = pt.tensor4("p3", dtype="float32")
        p4 = pt.tensor4("p4", dtype="float32")
        p5 = pt.tensor4("p5", dtype="float32")

        det_p3, det_p4, det_p5 = head(p3, p4, p5)

        print("  Compiling function...")
        f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

        print("  Running forward pass...")
        p3_val = np.random.randn(1, 64, 40, 40).astype("float32")
        p4_val = np.random.randn(1, 128, 20, 20).astype("float32")
        p5_val = np.random.randn(1, 256, 10, 10).astype("float32")

        det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

        print(f"  P3 input: {p3_val.shape} -> output: {det_p3_val.shape}")
        print(f"  P4 input: {p4_val.shape} -> output: {det_p4_val.shape}")
        print(f"  P5 input: {p5_val.shape} -> output: {det_p5_val.shape}")
        print("  ✓ Detection head works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 10: Full Model
# =============================================================================


def test_full_model():
    """Test complete YOLO11n model."""
    print("\n[Test 10] Full YOLO11n Model")
    print("-" * 70)

    try:
        from model import build_yolo11n

        print("  Building full model...")
        start_time = time.time()
        _model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
        build_time = time.time() - start_time
        print(f"  Build time: {build_time:.2f}s")

        pred_p3, pred_p4, pred_p5 = predictions

        print("  Compiling function...")
        compile_start = time.time()
        f = function([x], [pred_p3, pred_p4, pred_p5])
        compile_time = time.time() - compile_start
        print(f"  Compile time: {compile_time:.2f}s")

        print("  Running forward pass...")
        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        start_time = time.time()
        p3_val, p4_val, p5_val = f(x_val)
        inference_time = time.time() - start_time

        print(f"  Input: {x_val.shape}")
        print(f"  P3 output: {p3_val.shape}")
        print(f"  P4 output: {p4_val.shape}")
        print(f"  P5 output: {p5_val.shape}")
        print(f"  Inference time: {inference_time * 1000:.2f}ms")
        print("  ✓ Full model works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 11: Loss Function
# =============================================================================


def test_loss():
    """Test loss function."""
    print("\n[Test 11] Loss Function")
    print("-" * 70)

    try:
        from loss import yolo_loss
        from model import build_yolo11n

        print("  Building model and loss...")
        _model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
        loss, loss_dict = yolo_loss(predictions, targets=None, num_classes=2)

        print("  Compiling loss function...")
        f = function([x], [loss, loss_dict["box_loss"], loss_dict["cls_loss"]])

        print("  Computing loss...")
        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        total_loss, box_loss, cls_loss = f(x_val)

        print(f"  Total loss: {total_loss:.6f}")
        print(f"  Box loss: {box_loss:.6f}")
        print(f"  Cls loss: {cls_loss:.6f}")
        print("  ✓ Loss function works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 12: Gradients
# =============================================================================


def test_gradients():
    """Test gradient computation."""
    print("\n[Test 12] Gradient Computation")
    print("-" * 70)

    try:
        from loss import yolo_loss
        from model import build_yolo11n

        print("  Building model and loss...")
        model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
        loss, _loss_dict = yolo_loss(predictions, targets=None, num_classes=2)

        print("  Computing gradients for first 5 parameters...")
        test_params = model.params[:5]
        grads = []
        for param in test_params:
            grad = pytensor.grad(loss, param, disconnected_inputs="ignore")
            grads.append(grad)

        print(f"  ✓ Computed {len(grads)} gradients")

        print("  Compiling gradient function...")
        f = function([x], grads)

        print("  Evaluating gradients...")
        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        grad_vals = f(x_val)

        for i, grad_val in enumerate(grad_vals):
            print(
                f"    Grad {i}: shape={grad_val.shape}, mean={np.mean(np.abs(grad_val)):.6f}"
            )

        print("  ✓ Gradients work")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 13: Training Step
# =============================================================================


def test_training_step():
    """Test a complete training step with updates."""
    print("\n[Test 13] Training Step with Updates")
    print("-" * 70)

    try:
        from loss import yolo_loss
        from model import build_yolo11n

        print("  Building model and loss...")
        model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
        loss, _loss_dict = yolo_loss(predictions, targets=None, num_classes=2)

        print("  Setting up optimizer for first parameter...")
        test_param = model.params[0]
        grad = pytensor.grad(loss, test_param)

        # Create optimizer state
        momentum = pt.as_tensor_variable(np.float32(0.9))
        lr = pt.as_tensor_variable(np.float32(0.01))
        velocity = shared(
            np.zeros_like(test_param.get_value(), dtype="float32"), name="v"
        )

        # Compute updates
        v_new = momentum * velocity - lr * pt.cast(grad, "float32")
        p_new = test_param + v_new

        updates = [(velocity, v_new), (test_param, p_new)]

        print("  Compiling training function...")
        train_fn = function([x], loss, updates=updates)

        print("  Running training step...")
        x_val = np.random.randn(1, 3, 320, 320).astype("float32")
        loss_val = train_fn(x_val)

        print(f"  Loss: {loss_val:.6f}")
        print("  ✓ Training step works")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test YOLO11n components with JAX JIT")
    parser.add_argument(
        "--jax", action="store_true", help="Enable JAX mode (requires GPU)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip slow tests)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("YOLO11n Component Tests".center(70))
    print("=" * 70)

    # Setup
    jax_enabled = setup_jax(args.jax)
    print()

    # Define all tests
    all_tests = [
        ("Basic Ops", test_basic_ops, False),
        ("Dimshuffle/Tile", test_dimshuffle_tile, False),
        ("Upsampling", test_upsampling, False),
        ("ConvBNSiLU", test_conv_bn_silu, False),
        ("Bottleneck", test_bottleneck, False),
        ("C3k2", test_c3k2, False),
        ("SPPF", test_sppf, False),
        ("Backbone", test_backbone, True),
        ("Head", test_head, True),
        ("Full Model", test_full_model, True),
        ("Loss", test_loss, True),
        ("Gradients", test_gradients, True),
        ("Training Step", test_training_step, True),
    ]

    # Run tests
    results = {}
    for name, test_func, is_slow in all_tests:
        if args.quick and is_slow:
            print(f"\n[Skipped] {name} (slow test)")
            continue

        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n  ✗ Unexpected error: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary".center(70))
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")

    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if jax_enabled:
        print("\n[INFO] JAX mode was enabled for these tests")
    else:
        print("\n[INFO] Tests ran in CPU mode")
        print("  To test with JAX: python test_jax_components.py --jax")

    print("=" * 70)

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
