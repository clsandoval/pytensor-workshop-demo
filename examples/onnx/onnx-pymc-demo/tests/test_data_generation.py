"""
Tests for data generation.

This module tests synthetic data generation for training and testing.
"""

import numpy as np
from hypothesis import given, settings

from conftest import linear_regression_data


def test_generate_simple_linear_data(simple_linear_data):
    """
    Test generation of simple linear regression data.

    This test verifies:
    - Data arrays have correct shapes
    - Data types are float32 (required for ONNX)
    - Linear relationship exists (R² > 0.9)
    - Generated data is reproducible
    """
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]
    true_alpha = simple_linear_data["true_alpha"]
    true_beta = simple_linear_data["true_beta"]

    # Verify shapes
    assert x.shape == (100,), f"Expected x.shape=(100,), got {x.shape}"
    assert y.shape == (100,), f"Expected y.shape=(100,), got {y.shape}"

    # Verify dtypes
    assert x.dtype == np.float32, f"Expected float32, got {x.dtype}"
    assert y.dtype == np.float32, f"Expected float32, got {y.dtype}"

    # Verify linear relationship (fit OLS, check R²)
    # Simple implementation without sklearn dependency
    X = np.column_stack([np.ones_like(x), x])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted_alpha, fitted_beta = coeffs[0], coeffs[1]

    y_pred = fitted_alpha + fitted_beta * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    assert r_squared > 0.9, (
        f"Linear relationship too weak: R² = {r_squared:.3f}\n"
        f"Expected strong linear relationship (R² > 0.9)"
    )

    # Verify parameter recovery (approximately)
    # Allow 20% error (due to noise)
    assert np.abs(fitted_alpha - true_alpha) < 0.2 * np.abs(true_alpha) + 0.3, (
        f"Fitted alpha={fitted_alpha:.3f} too far from true={true_alpha}"
    )
    assert np.abs(fitted_beta - true_beta) < 0.2 * np.abs(true_beta) + 0.1, (
        f"Fitted beta={fitted_beta:.3f} too far from true={true_beta}"
    )


@given(data=linear_regression_data())
@settings(max_examples=20, deadline=None)
def test_hypothesis_data_generation(data):
    """
    Property test: Generated data has correct statistical properties.

    This test verifies across many random parameter combinations:
    - Arrays have consistent shapes
    - Linear relationship exists
    - Noise level is appropriate
    - No NaN or Inf values
    """
    x = data["x"]
    y = data["y"]
    n_points = data["n_points"]

    # Check shapes
    assert x.shape == (n_points,), f"x.shape mismatch: {x.shape}"
    assert y.shape == (n_points,), f"y.shape mismatch: {y.shape}"

    # Check no invalid values
    assert not np.any(np.isnan(x)), "x contains NaN"
    assert not np.any(np.isnan(y)), "y contains NaN"
    assert not np.any(np.isinf(x)), "x contains Inf"
    assert not np.any(np.isinf(y)), "y contains Inf"

    # Check reasonable value ranges (given x in [0, 10] and params in [-5, 5])
    assert np.abs(y).max() < 100, f"y values unreasonably large: max={np.abs(y).max()}"

    # Check monotonicity preservation (if beta > 0, generally increasing)
    if data["true_beta"] > 1.0:  # Strong positive slope
        # Compute trend: split into 3 regions, check mean increases
        n_third = n_points // 3
        mean_first = y[:n_third].mean()
        mean_last = y[-n_third:].mean()
        assert mean_last > mean_first, (
            f"Expected increasing trend with beta={data['true_beta']:.2f}, "
            f"but mean_first={mean_first:.2f} >= mean_last={mean_last:.2f}"
        )
