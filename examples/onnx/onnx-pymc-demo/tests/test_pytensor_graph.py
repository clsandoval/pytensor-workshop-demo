"""
Tests for PyTensor graph construction for posterior predictive inference.

This module tests the construction of deterministic PyTensor graphs that
compute posterior predictive mean and standard deviation.
"""

import numpy as np

import pytensor
import pytensor.tensor as pt


def test_posterior_predictive_mean_graph():
    """
    Test PyTensor graph for posterior predictive mean.

    This test verifies:
    - Graph can be constructed with pt.constant() for trained parameters
    - Input variable is created correctly
    - Mean computation: y_mean = alpha + beta * x
    - Graph can be compiled to a PyTensor function
    - Function produces correct numerical results

    **Uncertainty Note**: This graph computes E[y|x], the posterior predictive mean.
    It uses point estimates (posterior means) for alpha and beta.
    """
    # Simulate extracted posterior parameters
    alpha_mean = 1.0
    beta_mean = 2.5

    # Create PyTensor graph
    x_new = pt.vector("x_new", dtype="float32")
    alpha_const = pt.constant(alpha_mean, dtype="float32")
    beta_const = pt.constant(beta_mean, dtype="float32")

    y_pred_mean = alpha_const + beta_const * x_new

    # Compile function
    predict_mean = pytensor.function([x_new], y_pred_mean)

    # Test with known inputs
    x_test = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype="float32")
    y_test = predict_mean(x_test)

    # Verify output shape
    assert y_test.shape == x_test.shape, (
        f"Output shape mismatch: {y_test.shape} vs {x_test.shape}"
    )

    # Verify numerical correctness
    expected = alpha_mean + beta_mean * x_test
    np.testing.assert_allclose(
        y_test,
        expected,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Posterior predictive mean computation incorrect",
    )


def test_posterior_predictive_std_graph():
    """
    Test PyTensor graph for posterior predictive standard deviation.

    This test verifies:
    - Graph constructs constant std using pt.ones_like()
    - Std is broadcast to match input shape
    - Graph compiles and executes correctly

    **IMPORTANT - Uncertainty Simplification**:

    This implementation exports ONLY aleatoric uncertainty (observational noise).

    Full posterior predictive uncertainty decomposes as:
        Var[y|x, data] = E[sigma²] + Var[alpha + beta*x]
                       = aleatoric + epistemic

    Where:
    - Aleatoric (irreducible): Observation noise, sigma²
    - Epistemic (reducible): Parameter uncertainty, Var[alpha + beta*x]

    **Our Simplification**: We export only sigma (aleatoric), ignoring epistemic.

    Justification:
    - Simpler ONNX export (no covariance matrix needed)
    - Sufficient for well-trained models with lots of data
    - Epistemic uncertainty is small when posteriors are tight
    - Clearly communicates irreducible prediction uncertainty

    Future Enhancement:
    - Full epistemic uncertainty requires exporting posterior covariance
    - Total std: sqrt(sigma² + Var[alpha] + x²*Var[beta] + 2*x*Cov[alpha,beta])
    - See thoughts/shared/research/2025-10-15_option-b...:948-960 for details
    """
    # Simulate extracted posterior parameter
    sigma_mean = 0.5

    # Create PyTensor graph
    x_new = pt.vector("x_new", dtype="float32")
    sigma_const = pt.constant(sigma_mean, dtype="float32")

    # Create constant std (same for all x values in this simplified version)
    y_pred_std = pt.ones_like(x_new, dtype="float32") * sigma_const

    # Compile function
    predict_std = pytensor.function([x_new], y_pred_std)

    # Test with known inputs
    x_test = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype="float32")
    std_test = predict_std(x_test)

    # Verify output shape
    assert std_test.shape == x_test.shape, (
        f"Output shape mismatch: {std_test.shape} vs {x_test.shape}"
    )

    # Verify numerical correctness (all values should be sigma_mean)
    expected = np.full_like(x_test, sigma_mean, dtype="float32")
    np.testing.assert_allclose(
        std_test,
        expected,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Posterior predictive std computation incorrect",
    )

    # Verify all values are identical (constant uncertainty)
    assert np.all(std_test == std_test[0]), (
        f"Expected constant std across all x values\nGot: {std_test}"
    )


def test_multi_output_graph():
    """
    Test PyTensor graph with multiple outputs (mean, std).

    This test verifies:
    - Single function can return multiple outputs
    - Outputs are computed independently
    - Both outputs have correct shapes

    **Pattern**: This is the graph structure we'll export to ONNX.
    Multi-output ONNX is supported (see YOLO demo: 3 outputs).
    """
    # Simulate extracted parameters
    alpha_mean = 1.0
    beta_mean = 2.5
    sigma_mean = 0.5

    # Create PyTensor graph with multi-output
    x_new = pt.vector("x_new", dtype="float32")

    alpha_const = pt.constant(alpha_mean, dtype="float32")
    beta_const = pt.constant(beta_mean, dtype="float32")
    sigma_const = pt.constant(sigma_mean, dtype="float32")

    y_pred_mean = alpha_const + beta_const * x_new
    y_pred_std = pt.ones_like(x_new, dtype="float32") * sigma_const

    # Compile multi-output function
    predict_fn = pytensor.function([x_new], [y_pred_mean, y_pred_std])

    # Test
    x_test = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype="float32")
    mean_test, std_test = predict_fn(x_test)

    # Verify both outputs
    assert mean_test.shape == x_test.shape, f"Mean shape: {mean_test.shape}"
    assert std_test.shape == x_test.shape, f"Std shape: {std_test.shape}"

    # Verify numerical correctness
    expected_mean = alpha_mean + beta_mean * x_test
    expected_std = np.full_like(x_test, sigma_mean)

    np.testing.assert_allclose(mean_test, expected_mean, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(std_test, expected_std, rtol=1e-5, atol=1e-7)
