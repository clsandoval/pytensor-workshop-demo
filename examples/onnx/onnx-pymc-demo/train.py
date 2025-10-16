"""
Bayesian Linear Regression ONNX Export Demo

This script demonstrates the complete workflow:
1. Generate synthetic training data
2. Train a Bayesian linear regression model using PyMC ADVI
3. Extract posterior parameters
4. Build deterministic PyTensor graphs for posterior predictive inference
5. Export to ONNX for browser-based inference

Usage:
    python train.py --n-points 100 --advi-iterations 10000
"""
# ruff: noqa

import argparse
import time
from pathlib import Path

import numpy as np
import pymc as pm

import pytensor
import pytensor.tensor as pt
from pytensor.link.onnx import export_onnx


def generate_synthetic_data(n_points=100, seed=42):
    """
    Generate synthetic linear regression data: y = alpha + beta * x + noise.

    Ground truth: alpha=1.0, beta=2.5, sigma=0.5

    Parameters
    ----------
    n_points : int
        Number of data points
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - x: Input features (float32)
        - y: Target values (float32)
        - true_alpha: True intercept
        - true_beta: True slope
        - true_sigma: True noise std
    """
    rng = np.random.default_rng(seed)

    # Generate linear data with noise
    x = np.linspace(0, 10, n_points, dtype="float32")
    true_alpha, true_beta, true_sigma = 1.0, 2.5, 0.5
    noise = rng.normal(0, true_sigma, n_points).astype("float32")
    y = (true_alpha + true_beta * x + noise).astype("float32")

    print(f"Generated {n_points} data points")
    print(f"  Ground truth: y = {true_alpha} + {true_beta}*x + N(0, {true_sigma})")

    return {
        "x": x,
        "y": y,
        "true_alpha": true_alpha,
        "true_beta": true_beta,
        "true_sigma": true_sigma,
    }


def train_pymc_model(x, y, advi_iterations=10000):
    """
    Train Bayesian linear regression model using PyMC ADVI.

    Parameters
    ----------
    x : np.ndarray
        Input features
    y : np.ndarray
        Target values
    advi_iterations : int
        Number of ADVI iterations

    Returns
    -------
    dict with keys: posterior_means, posterior_stds, model, approx
    """
    print(f"\nTraining PyMC model with ADVI ({advi_iterations} iterations)...")

    with pm.Model() as model:
        # Priors (weakly informative)
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood
        x_data = pm.MutableData("x", x)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        # Run ADVI
        start_time = time.time()
        approx = pm.fit(method="advi", n=advi_iterations, progressbar=True)
        elapsed = time.time() - start_time

        # Extract parameters
        posterior_means = approx.mean.eval()
        posterior_stds = approx.std.eval()

    print(f"\nTraining completed in {elapsed:.1f}s")
    print(
        f"  Posterior: alpha = {posterior_means['alpha']:.3f} ± {posterior_stds['alpha']:.3f}"
    )
    print(
        f"  Posterior: beta  = {posterior_means['beta']:.3f} ± {posterior_stds['beta']:.3f}"
    )
    print(
        f"  Posterior: sigma = {posterior_means['sigma']:.3f} ± {posterior_stds['sigma']:.3f}"
    )

    return {
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "model": model,
        "approx": approx,
    }


def build_pytensor_graph(posterior_means):
    """
    Build PyTensor graph for posterior predictive inference.

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

    Parameters
    ----------
    posterior_means : dict
        Posterior parameter means from ADVI

    Returns
    -------
    tuple: (pytensor_function, input_variable)
    """
    print("\nBuilding PyTensor posterior predictive graph...")

    # Input variable for new predictions
    x_new = pt.vector("x_new", dtype="float32")

    # Bake posterior parameters as constants (deterministic)
    alpha_mean = pt.constant(posterior_means["alpha"], dtype="float32")
    beta_mean = pt.constant(posterior_means["beta"], dtype="float32")
    sigma_mean = pt.constant(posterior_means["sigma"], dtype="float32")

    # Posterior predictive mean: E[y|x] = alpha + beta * x
    y_pred_mean = alpha_mean + beta_mean * x_new

    # Posterior predictive std: sqrt(Var[y|x]) = sigma (aleatoric only)
    # Note: This is simplified - only includes observational noise
    y_pred_std = pt.ones_like(x_new, dtype="float32") * sigma_mean

    # Compile PyTensor function (multi-output)
    predict_fn = pytensor.function([x_new], [y_pred_mean, y_pred_std])

    print("  Graph compiled successfully")
    print("  Inputs: x_new (float32 vector)")
    print("  Outputs: y_mean, y_std (float32 vectors)")

    return predict_fn, x_new


def export_to_onnx(pytensor_function, onnx_path):
    """
    Export PyTensor function to ONNX.

    Parameters
    ----------
    pytensor_function : pytensor.compile.function.Function
        Compiled PyTensor function
    onnx_path : str or Path
        Output path for ONNX model

    Returns
    -------
    onnx.ModelProto
    """
    print(f"\nExporting to ONNX: {onnx_path}")

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Export with multi-output
    model = export_onnx(pytensor_function, str(onnx_path))

    # Get model info
    model_size_kb = onnx_path.stat().st_size / 1024
    opset_version = model.opset_import[0].version

    print("  ONNX export complete!")
    print(f"  Model size: {model_size_kb:.1f} KB")
    print(f"  Opset version: {opset_version}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")

    return model


def verify_onnx_export(pytensor_function, onnx_path, test_input):
    """
    Verify ONNX export matches PyTensor output.

    Parameters
    ----------
    pytensor_function : pytensor.compile.function.Function
        Original PyTensor function
    onnx_path : str or Path
        Path to ONNX model
    test_input : np.ndarray
        Test input for verification

    Returns
    -------
    bool: True if verification passes
    """
    print("\nVerifying ONNX export...")

    import onnx
    import onnxruntime as ort

    # Validate ONNX model
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("  ✓ ONNX model validation passed")

    # Load with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Compare PyTensor vs ONNX Runtime
    pytensor_mean, pytensor_std = pytensor_function(test_input)
    input_name = session.get_inputs()[0].name
    onnx_results = session.run(None, {input_name: test_input})
    onnx_mean, onnx_std = onnx_results[0], onnx_results[1]

    # Check agreement
    mean_diff = np.abs(pytensor_mean - onnx_mean).max()
    std_diff = np.abs(pytensor_std - onnx_std).max()

    print(f"  Max mean difference: {mean_diff:.2e}")
    print(f"  Max std difference: {std_diff:.2e}")

    if mean_diff < 1e-4 and std_diff < 1e-4:
        print("  ✓ ONNX export verified!")
        return True
    else:
        print("  ⚠ Warning: Large differences detected")
        return False


def save_training_data(data, output_path):
    """
    Save training data for browser visualization.

    Parameters
    ----------
    data : dict
        Training data dictionary
    output_path : str or Path
        Output path for .npz file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        x=data["x"],
        y=data["y"],
        true_alpha=data["true_alpha"],
        true_beta=data["true_beta"],
        true_sigma=data["true_sigma"],
    )

    print(f"\nTraining data saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main training and export workflow."""
    parser = argparse.ArgumentParser(
        description="Train and export Bayesian regression model"
    )
    parser.add_argument(
        "--n-points", type=int, default=100, help="Number of training points"
    )
    parser.add_argument(
        "--advi-iterations", type=int, default=10000, help="ADVI iterations"
    )
    parser.add_argument(
        "--output-dir", type=str, default="site", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Bayesian Linear Regression ONNX Export")
    print("=" * 70)

    # Step 1: Generate synthetic data
    data = generate_synthetic_data(n_points=args.n_points, seed=args.seed)

    # Step 2: Train PyMC model with ADVI
    training_result = train_pymc_model(
        data["x"], data["y"], advi_iterations=args.advi_iterations
    )

    # Step 3: Build PyTensor graph
    predict_fn, _x_new = build_pytensor_graph(training_result["posterior_means"])

    # Step 4: Test predictions in Python
    print("\nTesting predictions...")
    x_test = np.array([0, 2.5, 5.0, 7.5, 10.0], dtype="float32")
    mean, std = predict_fn(x_test)
    print("  Sample predictions:")
    for xi, mi, si in zip(x_test, mean, std):
        print(
            f"    x={xi:.1f}: y={mi:.3f} ± {si:.3f} (95% CI: [{mi - 2 * si:.3f}, {mi + 2 * si:.3f}])"
        )

    # Step 5: Export to ONNX
    onnx_path = Path(args.output_dir) / "bayesian_regression.onnx"
    _model = export_to_onnx(predict_fn, onnx_path)

    # Step 6: Verify ONNX export
    x_verify = np.linspace(0, 10, 20, dtype="float32")
    verify_onnx_export(predict_fn, onnx_path, x_verify)

    # Step 7: Save training data for browser
    data_path = Path(args.output_dir) / "training_data.npz"
    save_training_data(data, data_path)

    print("\n" + "=" * 70)
    print("✓ Training and export complete!")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Training data: {data_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
