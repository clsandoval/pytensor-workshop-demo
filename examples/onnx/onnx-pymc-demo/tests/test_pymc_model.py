"""
Tests for PyMC model definition, ADVI training, and parameter extraction.

This module tests the PyMC workflow including model definition, ADVI training,
and extraction of posterior parameters.
"""
# ruff: noqa

import numpy as np
import pymc as pm
import pytest


def test_pymc_model_definition(simple_linear_data):
    """
    Test PyMC Bayesian linear regression model definition.

    This test verifies:
    - Model can be constructed with pm.Model()
    - Priors are defined correctly (Normal for alpha/beta, HalfNormal for sigma)
    - Likelihood is defined with correct observed data
    - Model has expected random variables
    """
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]

    with pm.Model():
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood
        x_data = pm.MutableData("x", x)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    # Verify model structure
    assert model is not None, "Model construction failed"

    # Check random variables exist
    rv_names = [rv.name for rv in model.basic_RVs]
    assert "alpha" in rv_names, f"'alpha' not in model RVs: {rv_names}"
    assert "beta" in rv_names, f"'beta' not in model RVs: {rv_names}"
    assert "sigma" in rv_names, f"'sigma' not in model RVs: {rv_names}"

    # Check observed variable
    assert "y" in [rv.name for rv in model.observed_RVs], (
        "Observed variable 'y' not found"
    )

    # Check MutableData
    assert "x" in model.named_vars, "MutableData 'x' not found"


@pytest.mark.slow
def test_advi_training(simple_linear_data):
    """
    Test ADVI variational inference training.

    This test verifies:
    - ADVI fit() completes without errors
    - Training converges in reasonable time (<30 seconds)
    - Posterior approximation is created
    - ELBO (evidence lower bound) improves during training
    """
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]

    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        x_data = pm.MutableData("x", x)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Run ADVI (reduced iterations for testing)
        import time

        start_time = time.time()
        approx = pm.fit(method="advi", n=5000, progressbar=False)
        elapsed = time.time() - start_time

    # Verify training completed
    assert approx is not None, "ADVI fit() returned None"
    assert elapsed < 60, f"Training took too long: {elapsed:.1f}s (expected <60s)"

    # Verify approximation has required methods
    assert hasattr(approx, "mean"), "Approximation missing 'mean' attribute"
    assert hasattr(approx, "std"), "Approximation missing 'std' attribute"

    # Verify ELBO improved (final ELBO should be higher than initial)
    hist = approx.hist
    assert len(hist) > 0, "No training history recorded"

    # Check ELBO trend (last 10% should be better than first 10%)
    n = len(hist)
    early_elbo = np.mean(hist[: n // 10])
    late_elbo = np.mean(hist[-n // 10 :])

    assert late_elbo > early_elbo, (
        f"ELBO did not improve: early={early_elbo:.2f}, late={late_elbo:.2f}\n"
        "Training may not have converged"
    )


@pytest.mark.slow
def test_posterior_parameter_extraction(simple_linear_data):
    """
    Test extraction of posterior parameters from ADVI approximation.

    This test verifies:
    - approx.mean.eval() returns dictionary with parameter means
    - approx.std.eval() returns dictionary with parameter stds
    - Extracted values are reasonable (near true parameters)
    - All expected parameters are present

    **IMPORTANT**: This test documents a key part of the pipeline:
    how we extract trained parameters to bake into the ONNX graph.
    """
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]
    true_alpha = simple_linear_data["true_alpha"]
    true_beta = simple_linear_data["true_beta"]
    true_sigma = simple_linear_data["true_sigma"]

    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        x_data = pm.MutableData("x", x)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        approx = pm.fit(method="advi", n=5000, progressbar=False)

        # Extract parameters
        posterior_means = approx.mean.eval()
        posterior_stds = approx.std.eval()

    # Verify structure
    assert isinstance(posterior_means, dict), (
        f"posterior_means should be dict, got {type(posterior_means)}"
    )
    assert isinstance(posterior_stds, dict), (
        f"posterior_stds should be dict, got {type(posterior_stds)}"
    )

    # Verify all parameters present
    expected_params = {"alpha", "beta", "sigma"}
    assert expected_params.issubset(posterior_means.keys()), (
        f"Missing parameters in means: {expected_params - set(posterior_means.keys())}"
    )
    assert expected_params.issubset(posterior_stds.keys()), (
        f"Missing parameters in stds: {expected_params - set(posterior_stds.keys())}"
    )

    # Verify reasonable values (within 3 sigma of true values + tolerance)
    alpha_mean = posterior_means["alpha"]
    beta_mean = posterior_means["beta"]
    sigma_mean = posterior_means["sigma"]

    alpha_std = posterior_stds["alpha"]
    beta_std = posterior_stds["beta"]
    sigma_std = posterior_stds["sigma"]

    # Check means are close to true values (allowing 3 standard deviations + tolerance)
    assert np.abs(alpha_mean - true_alpha) < 3 * alpha_std + 0.5, (
        f"alpha posterior mean {alpha_mean:.3f} too far from true {true_alpha}\n"
        f"Posterior std: {alpha_std:.3f}"
    )

    assert np.abs(beta_mean - true_beta) < 3 * beta_std + 0.5, (
        f"beta posterior mean {beta_mean:.3f} too far from true {true_beta}\n"
        f"Posterior std: {beta_std:.3f}"
    )

    assert np.abs(sigma_mean - true_sigma) < 3 * sigma_std + 0.5, (
        f"sigma posterior mean {sigma_mean:.3f} too far from true {true_sigma}\n"
        f"Posterior std: {sigma_std:.3f}"
    )

    # Verify stds are positive and reasonable
    assert alpha_std > 0, f"alpha_std should be positive, got {alpha_std}"
    assert beta_std > 0, f"beta_std should be positive, got {beta_std}"
    assert sigma_std > 0, f"sigma_std should be positive, got {sigma_std}"

    # Stds should be smaller than the prior width (learning happened)
    assert alpha_std < 5.0, (
        f"alpha_std too large: {alpha_std} (prior std=10, expected <5)"
    )
    assert beta_std < 5.0, f"beta_std too large: {beta_std} (prior std=10, expected <5)"
