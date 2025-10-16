"""
Test configuration and fixtures for Bayesian Regression ONNX Demo.

This module provides:
- Shared pytest fixtures for test data
- Hypothesis strategies for property-based testing
- Test utilities and helpers
"""

import numpy as np
import pytest
from hypothesis import strategies as st


@pytest.fixture
def test_seed():
    """Reproducible random seed for testing."""
    return 42


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for ONNX models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def simple_linear_data(test_seed):
    """
    Simple linear regression data for deterministic tests.

    Ground truth: y = 1.0 + 2.5 * x + noise(0, 0.5)
    """
    rng = np.random.default_rng(test_seed)
    n = 100
    x = np.linspace(0, 10, n, dtype="float32")
    true_alpha, true_beta, true_sigma = 1.0, 2.5, 0.5
    y = (true_alpha + true_beta * x + rng.normal(0, true_sigma, n)).astype("float32")

    return {
        "x": x,
        "y": y,
        "true_alpha": true_alpha,
        "true_beta": true_beta,
        "true_sigma": true_sigma,
    }


@st.composite
def linear_regression_data(
    draw,
    n_points=None,
    x_range=None,
    alpha_range=None,
    beta_range=None,
    sigma_range=None,
    dtype="float32",
):
    """
    Generate synthetic linear regression data: y = alpha + beta * x + noise.

    Parameters
    ----------
    n_points : int or strategy, optional
        Number of data points (default: 20-200)
    x_range : tuple, optional
        Range for x values (default: (0, 10))
    alpha_range : tuple, optional
        Range for intercept (default: (-5, 5))
    beta_range : tuple, optional
        Range for slope (default: (-5, 5))
    sigma_range : tuple, optional
        Range for noise std (default: (0.1, 2.0))
    dtype : str
        NumPy dtype (default: "float32")

    Returns
    -------
    dict with keys: x, y, true_alpha, true_beta, true_sigma, n_points
    """
    # Draw parameters
    if n_points is None:
        n = draw(st.integers(min_value=20, max_value=200))
    else:
        n = draw(n_points) if isinstance(n_points, st.SearchStrategy) else n_points

    if x_range is None:
        x_min, x_max = 0.0, 10.0
    else:
        x_min, x_max = x_range

    if alpha_range is None:
        alpha = draw(
            st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
        )
    else:
        alpha = draw(
            st.floats(
                min_value=alpha_range[0],
                max_value=alpha_range[1],
                allow_nan=False,
                allow_infinity=False,
            )
        )

    if beta_range is None:
        beta = draw(
            st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
        )
    else:
        beta = draw(
            st.floats(
                min_value=beta_range[0],
                max_value=beta_range[1],
                allow_nan=False,
                allow_infinity=False,
            )
        )

    if sigma_range is None:
        sigma = draw(
            st.floats(
                min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False
            )
        )
    else:
        sigma = draw(
            st.floats(
                min_value=sigma_range[0],
                max_value=sigma_range[1],
                allow_nan=False,
                allow_infinity=False,
            )
        )

    # Generate data
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    x = np.linspace(x_min, x_max, n, dtype=dtype)
    noise = rng.normal(0, sigma, n).astype(dtype)
    y = (alpha + beta * x + noise).astype(dtype)

    return {
        "x": x,
        "y": y,
        "true_alpha": float(alpha),
        "true_beta": float(beta),
        "true_sigma": float(sigma),
        "n_points": n,
    }
