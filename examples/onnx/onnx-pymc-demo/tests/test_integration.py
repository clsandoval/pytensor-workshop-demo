"""
End-to-end integration tests for the Bayesian Regression ONNX pipeline.

This module tests the complete workflow from PyMC ADVI training through
PyTensor graph construction to ONNX export and verification.
"""
# ruff: noqa

import numpy as np
import pymc as pm
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.mark.slow
def test_end_to_end_pipeline(simple_linear_data, tmp_model_dir):
    """
    End-to-end integration test: Complete Bayesian regression ONNX export pipeline.

    This test validates the complete workflow:
    1. Generate synthetic data
    2. Train PyMC model with ADVI
    3. Extract posterior parameters
    4. Build PyTensor posterior predictive graph
    5. Export to ONNX
    6. Verify with ONNX Runtime
    7. Check predictions are reasonable

    **This is the most important test**: It validates the entire pipeline works.
    """
    # Step 1: Get data
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]
    true_alpha = simple_linear_data["true_alpha"]
    true_beta = simple_linear_data["true_beta"]
    true_sigma = simple_linear_data["true_sigma"]

    # Step 2: Train PyMC model with ADVI
    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        x_data = pm.MutableData("x", x)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Train (reduced iterations for testing)
        approx = pm.fit(method="advi", n=5000, progressbar=False)

        # Step 3: Extract parameters
        posterior_means = approx.mean.eval()
        _posterior_stds = approx.std.eval()

    alpha_mean = posterior_means["alpha"]
    beta_mean = posterior_means["beta"]
    sigma_mean = posterior_means["sigma"]

    # Step 4: Build PyTensor posterior predictive graph
    x_new = pt.vector("x_new", dtype="float32")

    alpha_const = pt.constant(alpha_mean, dtype="float32")
    beta_const = pt.constant(beta_mean, dtype="float32")
    sigma_const = pt.constant(sigma_mean, dtype="float32")

    y_pred_mean = alpha_const + beta_const * x_new
    y_pred_std = pt.ones_like(x_new) * sigma_const

    # Compile PyTensor function
    predict_fn = pytensor.function([x_new], [y_pred_mean, y_pred_std])

    # Step 5: Export to ONNX
    onnx_path = tmp_model_dir / "end_to_end.onnx"
    model = export_onnx(predict_fn, str(onnx_path))

    # Verify export
    assert onnx_path.exists(), "ONNX export failed"
    onnx.checker.check_model(model)

    # Step 6: Verify with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test on new data
    x_test = np.linspace(0, 10, 20, dtype="float32")

    # Get PyTensor predictions
    pytensor_mean, pytensor_std = predict_fn(x_test)

    # Get ONNX Runtime predictions
    input_name = session.get_inputs()[0].name
    onnx_results = session.run(None, {input_name: x_test})
    onnx_mean, onnx_std = onnx_results[0], onnx_results[1]

    # Step 7: Verify outputs match
    np.testing.assert_allclose(
        onnx_mean,
        pytensor_mean,
        rtol=1e-4,
        atol=1e-5,
        err_msg="ONNX mean doesn't match PyTensor",
    )

    np.testing.assert_allclose(
        onnx_std,
        pytensor_std,
        rtol=1e-4,
        atol=1e-5,
        err_msg="ONNX std doesn't match PyTensor",
    )

    # Step 8: Verify predictions are reasonable
    # Predictions should be close to true linear relationship
    expected_mean = true_alpha + true_beta * x_test

    # Allow some error (model isn't perfect, just close)
    mean_error = np.abs(onnx_mean - expected_mean).mean()
    assert mean_error < 1.0, (
        f"Mean prediction error {mean_error:.3f} too large.\n"
        f"Predictions don't match true relationship well."
    )

    # Std should be close to true sigma
    std_error = np.abs(onnx_std.mean() - true_sigma)
    assert std_error < 0.5, (
        f"Std error {std_error:.3f} too large.\n"
        f"Predicted std={onnx_std.mean():.3f}, true sigma={true_sigma}"
    )


@pytest.mark.slow
def test_uncertainty_quantification(simple_linear_data, tmp_model_dir):
    """
    Test that uncertainty quantification is correctly implemented and documented.

    This test verifies:
    - Exported std represents aleatoric uncertainty
    - Std is approximately equal to learned sigma
    - Std is constant across x values (as expected for simplified model)
    - 95% confidence intervals have reasonable coverage

    **CRITICAL - Uncertainty Interpretation**:

    The exported uncertainty is ALEATORIC ONLY (observational noise).

    Total predictive uncertainty should include:
    1. Aleatoric: sigma² (observation noise)
    2. Epistemic: Var[alpha + beta*x] (parameter uncertainty)

    This implementation exports only (1), which is appropriate when:
    - Model is well-trained with sufficient data
    - Parameter uncertainty is small (tight posteriors)
    - We want to communicate irreducible prediction uncertainty

    Limitations:
    - Uncertainty does NOT increase in extrapolation regions
    - Uncertainty does NOT reflect parameter uncertainty
    - For sparse data, understates true uncertainty

    For full uncertainty, would need to export:
    - Posterior means: alpha_mean, beta_mean, sigma_mean
    - Posterior covariance: Cov[alpha, beta]
    - Compute: total_std² = sigma² + Var[alpha] + x²*Var[beta] + 2*x*Cov[alpha,beta]

    See: thoughts/shared/research/2025-10-15_option-b...:948-960
    """
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]
    true_sigma = simple_linear_data["true_sigma"]

    # Train model
    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        x_data = pm.MutableData("x", x)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        approx = pm.fit(method="advi", n=5000, progressbar=False)
        posterior_means = approx.mean.eval()

    sigma_mean = posterior_means["sigma"]

    # Build and export
    x_new = pt.vector("x_new", dtype="float32")
    alpha_const = pt.constant(posterior_means["alpha"], dtype="float32")
    beta_const = pt.constant(posterior_means["beta"], dtype="float32")
    sigma_const = pt.constant(sigma_mean, dtype="float32")

    y_pred_mean = alpha_const + beta_const * x_new
    y_pred_std = pt.ones_like(x_new) * sigma_const

    predict_fn = pytensor.function([x_new], [y_pred_mean, y_pred_std])
    onnx_path = tmp_model_dir / "uncertainty.onnx"
    export_onnx(predict_fn, str(onnx_path))

    # Test uncertainty properties
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    x_test = np.linspace(0, 10, 50, dtype="float32")
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: x_test})
    _mean_pred, std_pred = results  # noqa: F841[0], results[1]

    # 1. Verify std is close to learned sigma
    assert np.allclose(std_pred, sigma_mean, rtol=1e-5), (
        f"Exported std should equal sigma_mean={sigma_mean:.3f}\nGot: {std_pred[0]:.3f}"
    )

    # 2. Verify std is constant across x
    assert np.allclose(std_pred, std_pred[0], rtol=1e-7), (
        "Std should be constant across x values (aleatoric only)\n"
        f"Range: [{std_pred.min():.6f}, {std_pred.max():.6f}]"
    )

    # 3. Verify learned sigma is close to true sigma
    assert np.abs(sigma_mean - true_sigma) < 0.3, (
        f"Learned sigma={sigma_mean:.3f} far from true={true_sigma}\n"
        "Model may not have converged properly"
    )

    # 4. Test coverage (approximately 95% of points should be within 2 std)
    # Predict on training data
    results_train = session.run(None, {input_name: x})
    mean_train, std_train = results_train[0], results_train[1]

    # Compute z-scores
    z_scores = (y - mean_train) / std_train
    coverage = np.sum(np.abs(z_scores) <= 2) / len(z_scores)

    # Allow some flexibility (stochastic data)
    assert coverage > 0.85, (
        f"Coverage {coverage:.2%} too low (expected ~95% within 2 std)\n"
        "Uncertainty estimate may be incorrect"
    )
    assert coverage < 1.0, "100% coverage suggests uncertainty is overestimated"
