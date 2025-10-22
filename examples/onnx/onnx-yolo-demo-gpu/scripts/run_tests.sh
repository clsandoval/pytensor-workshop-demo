#!/bin/bash
# Script to run tests with proper PyTensor configuration for JAX

set -e

echo "=========================================="
echo "Running YOLO11n GPU Tests with JAX"
echo "=========================================="
echo ""

# CRITICAL: Set PyTensor flags for JAX compatibility
# optimizer_excluding=shape_unsafe prevents graph rewrites that cause JAX JIT errors
export PYTENSOR_FLAGS="floatX=float32,optimizer_excluding=shape_unsafe"
export JAX_PLATFORMS="cuda"
export JAX_ENABLE_X64=False

echo "Configuration:"
echo "  PYTENSOR_FLAGS: $PYTENSOR_FLAGS"
echo "  JAX_PLATFORMS: $JAX_PLATFORMS"
echo ""

# Change to project directory
cd "$(dirname "$0")/.."

# Run the JAX backend tests
echo "Running JAX backend tests..."
python -m pytest tests/test_jax_backend.py -xvs

echo ""
echo "=========================================="
echo "Test run complete!"
echo "=========================================="