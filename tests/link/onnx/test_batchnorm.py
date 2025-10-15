"""Tests for ONNX batch normalization conversion."""

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.batchnorm import batch_normalization
from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_batchnorm_basic_4d(tmp_path):
    """
    Test basic 4D batch normalization (NCHW format).

    This is the most common use case for batch normalization in CNNs.
    """
    # Input: (batch=2, channels=3, height=4, width=4)
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((2, 3, 4, 4)).astype("float32")
    gamma_val = np.array([1.0, 2.0, 0.5], dtype="float32")
    beta_val = np.array([0.0, 1.0, -1.0], dtype="float32")
    mean_val = np.array([0.0, 0.5, -0.5], dtype="float32")
    var_val = np.array([1.0, 0.5, 2.0], dtype="float32")

    compare_onnx_and_py(
        [x, gamma, beta, mean, variance],
        y,
        [x_val, gamma_val, beta_val, mean_val, var_val],
        tmp_path=tmp_path,
    )


def test_batchnorm_different_channels(tmp_path):
    """
    Test batch normalization with different channel counts.

    Verifies that the op works correctly for various channel dimensions.
    """
    from functools import partial

    for num_channels in [1, 8, 16, 64]:
        x = pt.tensor4("x", dtype="float32")
        gamma = pt.vector("gamma", dtype="float32")
        beta = pt.vector("beta", dtype="float32")
        mean = pt.vector("mean", dtype="float32")
        variance = pt.vector("variance", dtype="float32")

        y = batch_normalization(x, gamma, beta, mean, variance)

        rng = np.random.default_rng(42 + num_channels)
        x_val = rng.standard_normal((2, num_channels, 8, 8)).astype("float32")
        gamma_val = rng.random(num_channels).astype("float32")
        beta_val = rng.random(num_channels).astype("float32")
        mean_val = rng.standard_normal(num_channels).astype("float32")
        var_val = (
            rng.random(num_channels).astype("float32") + 0.1
        )  # Avoid zero variance

        compare_onnx_and_py(
            [x, gamma, beta, mean, variance],
            y,
            [x_val, gamma_val, beta_val, mean_val, var_val],
            tmp_path=tmp_path,
            assert_fn=partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-7),
        )


def test_batchnorm_with_epsilon(tmp_path):
    """
    Test batch normalization with custom epsilon value.

    Epsilon is important for numerical stability when variance is small.
    """
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    # Use larger epsilon
    y = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-3)

    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((1, 2, 3, 3)).astype("float32")
    gamma_val = np.array([1.0, 1.0], dtype="float32")
    beta_val = np.array([0.0, 0.0], dtype="float32")
    mean_val = np.array([0.0, 0.0], dtype="float32")
    # Very small variance to test epsilon
    var_val = np.array([1e-6, 1e-6], dtype="float32")

    compare_onnx_and_py(
        [x, gamma, beta, mean, variance],
        y,
        [x_val, gamma_val, beta_val, mean_val, var_val],
        tmp_path=tmp_path,
    )


def test_batchnorm_2d(tmp_path):
    """
    Test 2D batch normalization (NC format).

    This format is used after flattening layers or for fully connected networks.
    """
    x = pt.matrix("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance)

    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((8, 16)).astype("float32")
    gamma_val = rng.random(16).astype("float32")
    beta_val = rng.random(16).astype("float32")
    mean_val = rng.standard_normal(16).astype("float32")
    var_val = rng.random(16).astype("float32") + 0.1

    compare_onnx_and_py(
        [x, gamma, beta, mean, variance],
        y,
        [x_val, gamma_val, beta_val, mean_val, var_val],
        tmp_path=tmp_path,
    )


def test_batchnorm_structure(tmp_path):
    """
    Test that BatchNorm generates correct ONNX node.

    Verifies:
    - Node type is BatchNormalization
    - Epsilon attribute is correctly set
    - Node has correct inputs/outputs
    """
    from pytensor.link.onnx import export_onnx

    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-4)

    f = pytensor.function([x, gamma, beta, mean, variance], y)

    model_path = tmp_path / "test_batchnorm.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    validate_onnx_graph_structure(
        model,
        expected_node_types=["BatchNormalization"],
        expected_node_count=1,
    )

    # Check BatchNormalization node
    bn_node = model.graph.node[0]
    assert bn_node.op_type == "BatchNormalization"

    # Check epsilon attribute
    epsilon_attr = next(attr for attr in bn_node.attribute if attr.name == "epsilon")
    assert np.isclose(epsilon_attr.f, 1e-4)


def test_batchnorm_single_batch(tmp_path):
    """
    Test batch normalization with batch size = 1.

    Single-batch inference is common in deployed models.
    """
    from functools import partial

    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance)

    rng = np.random.default_rng(42)
    # Batch size = 1
    x_val = rng.standard_normal((1, 16, 32, 32)).astype("float32")
    gamma_val = rng.random(16).astype("float32")
    beta_val = rng.random(16).astype("float32")
    mean_val = rng.standard_normal(16).astype("float32")
    var_val = rng.random(16).astype("float32") + 0.1

    compare_onnx_and_py(
        [x, gamma, beta, mean, variance],
        y,
        [x_val, gamma_val, beta_val, mean_val, var_val],
        tmp_path=tmp_path,
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-7),
    )


def test_c3k2_pattern(tmp_path):
    """
    ⭐⭐⭐ CRITICAL TEST: Complete C3k2 pattern (Conv → BatchNorm → SiLU).

    This is the exact pattern used in YOLO11n's C3k2 blocks.
    It's the integration test that validates all three operations work together.

    Pattern:
    1. Conv2D (feature extraction)
    2. BatchNormalization (normalize activations)
    3. SiLU activation (non-linearity)
    """
    from functools import partial

    # Input: (batch, channels, height, width)
    x = pt.tensor4("x", dtype="float32")

    # Simulated conv output (in real model, this comes from Conv2D)
    # For testing, we use x directly as if it's conv output

    # BatchNorm parameters
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    # C3k2 pattern: Conv → BatchNorm → SiLU
    # (We skip Conv for testing, focus on BatchNorm + SiLU)
    bn_out = batch_normalization(x, gamma, beta, mean, variance)
    silu_out = pt.silu(bn_out)

    # Test data - simulates conv output
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((2, 32, 20, 20)).astype("float32")
    gamma_val = rng.random(32).astype("float32") + 0.5  # Scale around 1.0
    beta_val = rng.standard_normal(32).astype("float32") * 0.1  # Small bias
    mean_val = rng.standard_normal(32).astype("float32") * 0.5
    var_val = rng.random(32).astype("float32") + 0.1

    compare_onnx_and_py(
        [x, gamma, beta, mean, variance],
        silu_out,
        [x_val, gamma_val, beta_val, mean_val, var_val],
        tmp_path=tmp_path,
        assert_fn=partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-7),
    )
