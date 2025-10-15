"""Pytest configuration for ONNX tests with Hypothesis."""

import os
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, Phase, settings


# Register Hypothesis profiles
settings.register_profile(
    "dev",
    max_examples=10,
    deadline=timedelta(milliseconds=500),
    phases=[Phase.explicit, Phase.reuse, Phase.generate],  # Skip shrinking in dev
    print_blob=False,
)

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,  # No deadline in CI
    derandomize=True,  # Deterministic for CI
    print_blob=True,  # Print failing examples for debugging
)

settings.register_profile(
    "thorough",
    max_examples=1000,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Suppress health checks that are problematic for ONNX operations
settings.register_profile(
    "onnx",
    suppress_health_check=[
        HealthCheck.too_slow,  # ONNX operations can be slow
        HealthCheck.filter_too_much,  # We filter invalid inputs aggressively
    ],
    max_examples=50,
    deadline=timedelta(seconds=5),  # Allow 5s per test
)

# Load profile from environment, default to 'dev'
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# Standard pytest fixture for tmp_path
@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")
