#!/usr/bin/env python
"""Run tests with proper PyTensor configuration for JAX."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("Running YOLO11n GPU Tests with JAX".center(70))
    print("=" * 70)
    print()

    # CRITICAL: Set PyTensor flags for JAX compatibility
    # optimizer_excluding=shape_unsafe prevents graph rewrites that cause JAX JIT errors
    env = os.environ.copy()
    env["PYTENSOR_FLAGS"] = "floatX=float32,optimizer_excluding=shape_unsafe"
    env["JAX_PLATFORMS"] = "cuda"
    env["JAX_ENABLE_X64"] = "False"

    print("Configuration:")
    print(f"  PYTENSOR_FLAGS: {env['PYTENSOR_FLAGS']}")
    print(f"  JAX_PLATFORMS: {env['JAX_PLATFORMS']}")
    print()

    # Run the JAX backend tests
    print("Running JAX backend tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_jax_backend.py", "-xvs"],
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
    )

    print()
    print("=" * 70)
    if result.returncode == 0:
        print("All tests passed!".center(70))
    else:
        print(f"Tests failed with code {result.returncode}".center(70))
    print("=" * 70)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
