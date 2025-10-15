#!/usr/bin/env python3
# ruff: noqa: E402
"""
Complete dtype diagnostic script.
Run this to find exactly where float64 is being introduced.
"""

import os
import sys


print("=" * 70)
print("PyTensor dtype Diagnostic")
print("=" * 70)
print()

# Step 1: Check environment
print("[1/8] Environment Variables")
print("-" * 70)
pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "NOT SET")
print(f"PYTENSOR_FLAGS: {pytensor_flags}")
print()

# Step 2: Import PyTensor and check config
print("[2/8] PyTensor Configuration")
print("-" * 70)
import pytensor


print(f"PyTensor version: {pytensor.__version__}")
print(f"config.floatX: {pytensor.config.floatX}")
print("Expected: float32")
if pytensor.config.floatX != "float32":
    print("❌ ERROR: floatX is not float32!")
    print("   This is the root cause of all dtype issues.")
    sys.exit(1)
else:
    print("✓ Config is correct")
print()

# Step 3: Test shared variable creation
print("[3/8] Shared Variable Creation")
print("-" * 70)
import numpy as np

from pytensor import shared


test_array = np.ones((2, 3), dtype="float32")
test_shared = shared(test_array, name="test_shared")
print(f"NumPy array dtype: {test_array.dtype}")
print(f"Shared variable dtype: {test_shared.dtype}")
print(f"Shared variable type: {test_shared.type}")
if str(test_shared.dtype) != "float32":
    print("❌ ERROR: Shared variables are not float32!")
    sys.exit(1)
else:
    print("✓ Shared variables are float32")
print()

# Step 4: Test scalar operations
print("[4/8] Scalar Operations")
print("-" * 70)
import pytensor.tensor as pt


scalar_np = np.float32(0.5)
scalar_pt = pt.as_tensor_variable(scalar_np)
print(f"NumPy scalar dtype: {scalar_np.dtype}")
print(f"PyTensor scalar dtype: {scalar_pt.dtype}")

# Test multiplication
result = scalar_pt * test_shared
print(f"Multiplication result dtype: {result.dtype}")
if str(result.dtype) != "float32":
    print("❌ ERROR: Operations promote to float64!")
    sys.exit(1)
else:
    print("✓ Operations stay float32")
print()

# Step 5: Test model weight creation
print("[5/8] Model Weight Initialization")
print("-" * 70)
from blocks import ConvBNSiLU


conv = ConvBNSiLU(3, 16, kernel_size=3, name_prefix="test")
print(f"Conv weight dtype: {conv.W.dtype}")
print(f"Conv weight type: {conv.W.type}")
print(f"Conv gamma dtype: {conv.gamma.dtype}")
print(f"Conv beta dtype: {conv.beta.dtype}")

if str(conv.W.dtype) != "float32":
    print("❌ ERROR: Model weights are not float32!")
    print("   Weights were created with wrong dtype despite config.")
    sys.exit(1)
else:
    print("✓ Model weights are float32")
print()

# Step 6: Test forward pass
print("[6/8] Forward Pass")
print("-" * 70)
x_input = pt.tensor4("x", dtype="float32")
x_test = np.random.randn(1, 3, 32, 32).astype("float32")

try:
    output = conv(x_input)
    print(f"Input dtype: {x_input.dtype}")
    print(f"Output dtype: {output.dtype}")

    if str(output.dtype) != "float32":
        print("❌ ERROR: Forward pass promotes to float64!")
        sys.exit(1)
    else:
        print("✓ Forward pass stays float32")
except Exception as e:
    print(f"❌ ERROR in forward pass: {e}")
    sys.exit(1)
print()

# Step 7: Test gradient computation
print("[7/8] Gradient Computation")
print("-" * 70)
loss = pt.sum(output**2)
print(f"Loss dtype: {loss.dtype}")

try:
    grad_W = pytensor.grad(loss, conv.W)
    print(f"Gradient dtype: {grad_W.dtype}")
    print(f"Gradient type: {grad_W.type}")

    if str(grad_W.dtype) != "float32":
        print("❌ ERROR: Gradients are float64!")
        print("   This is where the dtype mismatch originates.")
        print(f"   Parameter: {conv.W.dtype}")
        print(f"   Gradient:  {grad_W.dtype}")
        sys.exit(1)
    else:
        print("✓ Gradients are float32")
except Exception as e:
    print(f"❌ ERROR computing gradients: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

# Step 8: Test optimizer update
print("[8/8] Optimizer Update")
print("-" * 70)
velocity = shared(np.zeros_like(conv.W.get_value(), dtype="float32"), name="velocity")
lr = pt.as_tensor_variable(np.float32(0.01))
momentum = pt.as_tensor_variable(np.float32(0.9))

print(f"Velocity dtype: {velocity.dtype}")
print(f"LR dtype: {lr.dtype}")
print(f"Momentum dtype: {momentum.dtype}")

try:
    # Cast gradient explicitly
    grad_W_casted = pt.cast(grad_W, "float32")

    # Compute velocity update
    v_new = momentum * velocity - lr * grad_W_casted
    print(f"Velocity update dtype: {v_new.dtype}")
    print(f"Velocity update type: {v_new.type}")

    if str(v_new.dtype) != "float32":
        print("❌ ERROR: Velocity update is not float32!")
        print("   Component dtypes:")
        print(f"     momentum * velocity = {(momentum * velocity).dtype}")
        print(f"     lr * grad = {(lr * grad_W_casted).dtype}")
        sys.exit(1)
    else:
        print("✓ Optimizer update is float32")

    # Try to compile a function with updates
    from pytensor import function

    updates = [(velocity, v_new)]
    test_fn = function([x_input], loss, updates=updates)
    print("✓ Function compilation successful")

except Exception as e:
    print(f"❌ ERROR in optimizer update: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

# Final summary
print("=" * 70)
print("✓ ALL CHECKS PASSED - dtype configuration is correct!")
print("=" * 70)
print()
print("Summary:")
print("  - PyTensor config.floatX = float32")
print("  - Shared variables created as float32")
print("  - Model weights are float32")
print("  - Forward pass stays float32")
print("  - Gradients are float32")
print("  - Optimizer updates are float32")
print()
print("If this script passes but training still fails, the issue is")
print("elsewhere (likely in how train.py is run or imports are cached).")
