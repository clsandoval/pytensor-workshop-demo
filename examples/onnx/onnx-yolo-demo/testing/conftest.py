"""Pytest configuration and shared fixtures for YOLO tests."""

import os
import sys
from pathlib import Path

import pytest


# CRITICAL: Set environment variables BEFORE importing JAX or PyTensor
# These force JAX to prioritize GPU over CPU
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["PYTENSOR_FLAGS"] = (
    "floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"
)

import pytensor


@pytest.fixture(scope="session")
def jax_setup():
    """Setup JAX backend if available, forcing GPU usage."""
    try:
        import jax

        # Force JAX to use GPU by checking devices
        devices = jax.devices()
        device_type = devices[0].platform if len(devices) > 0 else "none"

        if device_type == "gpu":
            # Only enable JAX mode if GPU is available
            pytensor.config.mode = "JAX"
            pytensor.config.optimizer_excluding = "shape_unsafe"

            return {
                "jax_available": True,
                "devices": devices,
                "device_type": device_type,
                "mode": pytensor.config.mode,
            }
        else:
            # JAX available but no GPU - tests will skip
            return {
                "jax_available": False,
                "devices": devices,
                "device_type": device_type,
                "error": f"JAX found but no GPU detected. Platform: {device_type}",
                "mode": pytensor.config.mode,
            }
    except Exception as e:
        return {"jax_available": False, "error": str(e), "mode": pytensor.config.mode}


@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """Ensure parent directory is in Python path for imports."""
    parent_dir = str(Path(__file__).parent.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
