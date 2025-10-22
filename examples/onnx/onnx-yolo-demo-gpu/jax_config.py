"""JAX-specific PyTensor configuration for YOLO model."""

import os


# Set environment BEFORE importing pytensor
os.environ["PYTENSOR_FLAGS"] = (
    "floatX=float32,"
    "optimizer=fast_compile,"  # Use fast_compile instead of fast_run
    "optimizer_excluding=shape_unsafe"
)

import pytensor


# Note: Config can't be changed after import - must be set via environment variable
print("JAX Configuration (read-only):")
print(f"  floatX: {pytensor.config.floatX}")
print(f"  optimizer: {pytensor.config.optimizer}")
print(f"  optimizer_excluding: {pytensor.config.optimizer_excluding}")
print(f"  mode: {pytensor.config.mode}")
