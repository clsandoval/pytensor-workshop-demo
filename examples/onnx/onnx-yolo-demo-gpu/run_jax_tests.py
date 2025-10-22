#!/usr/bin/env python
"""
Test runner that ensures PyTensor config is set BEFORE any imports.

This script MUST be run from a fresh Python process where PyTensor
has not been imported yet.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_test_with_config(test_file, config_string, description):
    """Run a test file with specific PyTensor configuration."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Config: {config_string}")
    print(f"Test: {test_file}")
    print("-" * 70)

    env = os.environ.copy()
    env["PYTENSOR_FLAGS"] = config_string

    # Run the test in a subprocess to ensure clean import
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-xvs", "--tb=short"],
        env=env,
        capture_output=False,
        text=True,
        cwd=Path(__file__).parent,
    )

    return result.returncode == 0


def main():
    """Run tests with different configurations."""
    print("JAX Backend Test Suite")
    print("=" * 70)

    test_configs = [
        # Try with no optimizations at all
        ("floatX=float32,optimizer=None", "No optimizations (nuclear option)"),
        # Try with fast_compile and exclusions
        (
            "floatX=float32,optimizer=fast_compile,optimizer_excluding=shape_unsafe",
            "Fast compile with shape_unsafe excluded",
        ),
        # Try with just float32
        ("floatX=float32", "Default optimizer with float32"),
    ]

    results = []

    # First, test a simple JAX compilation
    print("\nTesting basic JAX functionality...")
    env = os.environ.copy()
    env["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=None"

    test_code = """
import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=None'
import pytensor
import pytensor.tensor as pt
import numpy as np

print(f"Config: optimizer={pytensor.config.optimizer}, floatX={pytensor.config.floatX}")

try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Simple test
    x = pt.matrix('x', dtype='float32')
    y = x * 2 + 1
    f = pytensor.function([x], y, mode='JAX')

    test_input = np.array([[1, 2], [3, 4]], dtype='float32')
    result = f(test_input)
    print(f"Simple JAX test passed! Result shape: {result.shape}")

except Exception as e:
    print(f"Simple JAX test failed: {e}")
    import traceback
    traceback.print_exc()
"""

    _ = subprocess.run(
        [sys.executable, "-c", test_code], env=env, capture_output=False, text=True
    )

    print("\n" + "=" * 70)

    # Run actual tests with different configs
    for config, description in test_configs:
        success = run_test_with_config(
            "tests/test_jax_backend.py::test_jax_handles_edge_cases",
            config,
            description,
        )
        results.append((description, success))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for desc, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {desc}")

    print("\n" + "=" * 70)

    if not any(r[1] for r in results):
        print("\n⚠️  All configurations failed!")
        print("\nThis suggests the issue is NOT with the optimizer configuration,")
        print("but rather with the model operations themselves being incompatible")
        print("with JAX JIT compilation (likely the upsampling operations).")
    else:
        print("\n✓ At least one configuration works!")
        print("\nWorking configurations:")
        for desc, success in results:
            if success:
                print(f"  - {desc}")


if __name__ == "__main__":
    # Check if PyTensor has already been imported
    if "pytensor" in sys.modules:
        print("ERROR: PyTensor has already been imported!")
        print("This script must be run from a fresh Python process.")
        print("\nRun it directly: python run_jax_tests.py")
        sys.exit(1)

    main()
