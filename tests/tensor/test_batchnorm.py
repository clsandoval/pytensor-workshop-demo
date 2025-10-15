"""Tests for batch normalization operations."""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.batchnorm import BatchNormalization, batch_normalization


def test_batchnorm_basic_2d():
    """
    Test basic 2D batch normalization (NC format).

    This tests the simplest case: batch normalization on 2D tensors
    where normalization happens over the channel dimension.
    """
    # Input: (batch_size=3, channels=4)
    x = pt.matrix("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    f = function([x, gamma, beta, mean, variance], y)

    # Test data
    x_val = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype="float32",
    )

    gamma_val = np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")
    beta_val = np.array([0.0, 0.0, 0.0, 0.0], dtype="float32")
    mean_val = np.array([5.0, 6.0, 7.0, 8.0], dtype="float32")
    var_val = np.array([16.0, 16.0, 16.0, 16.0], dtype="float32")

    result = f(x_val, gamma_val, beta_val, mean_val, var_val)

    # Manual calculation:
    # For channel 0: x values are [1, 5, 9], mean=5, var=16, std=4
    # normalized = (x - 5) / 4 = [-1, 0, 1]
    # output = 1.0 * normalized + 0.0 = [-1, 0, 1]

    expected = np.array(
        [[-1.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        dtype="float32",
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_batchnorm_4d_cnn():
    """
    Test 4D batch normalization (NCHW format).

    This is the typical use case for CNNs where batch normalization
    is applied to convolutional feature maps.

    Format: (batch, channels, height, width)
    """
    # Input: (batch=2, channels=3, height=2, width=2)
    x = pt.tensor4("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    f = function([x, gamma, beta, mean, variance], y)

    # Test data: simple pattern to verify broadcasting
    rng = np.random.default_rng(42)
    x_val = rng.standard_normal((2, 3, 2, 2)).astype("float32")

    # Parameters (one per channel)
    gamma_val = np.array([1.0, 2.0, 0.5], dtype="float32")
    beta_val = np.array([0.0, 1.0, -1.0], dtype="float32")
    mean_val = np.array([0.0, 0.0, 0.0], dtype="float32")
    var_val = np.array([1.0, 1.0, 1.0], dtype="float32")

    result = f(x_val, gamma_val, beta_val, mean_val, var_val)

    # Manual calculation for channel 0:
    # normalized = (x - 0.0) / sqrt(1.0 + 1e-5) ≈ x
    # output = 1.0 * x + 0.0 = x
    expected_ch0 = x_val[:, 0, :, :]

    # Channel 1: output = 2.0 * x + 1.0
    expected_ch1 = 2.0 * x_val[:, 1, :, :] + 1.0

    # Channel 2: output = 0.5 * x - 1.0
    expected_ch2 = 0.5 * x_val[:, 2, :, :] - 1.0

    np.testing.assert_allclose(result[:, 0, :, :], expected_ch0, rtol=1e-4)
    np.testing.assert_allclose(result[:, 1, :, :], expected_ch1, rtol=1e-4)
    np.testing.assert_allclose(result[:, 2, :, :], expected_ch2, rtol=1e-4)


def test_batchnorm_with_scale_and_shift():
    """
    Test batch normalization with non-trivial scale (gamma) and shift (beta).

    This verifies that gamma and beta are applied correctly:
    y = gamma * normalized + beta
    """
    x = pt.vector("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5)

    f = function([x, gamma, beta, mean, variance], y)

    # Test data
    x_val = np.array([0.0, 1.0, 2.0, 3.0], dtype="float32")

    # Scale by 2, shift by 5
    gamma_val = np.array([2.0, 2.0, 2.0, 2.0], dtype="float32")
    beta_val = np.array([5.0, 5.0, 5.0, 5.0], dtype="float32")

    # Mean=1.5, Variance=1.25 => Std ≈ 1.118
    mean_val = np.array([1.5, 1.5, 1.5, 1.5], dtype="float32")
    var_val = np.array([1.25, 1.25, 1.25, 1.25], dtype="float32")

    result = f(x_val, gamma_val, beta_val, mean_val, var_val)

    # Manual calculation:
    # normalized = (x - 1.5) / sqrt(1.25 + 1e-5)
    # For x=[0, 1, 2, 3]: normalized ≈ [-1.342, -0.447, 0.447, 1.342]
    # output = 2.0 * normalized + 5.0
    std = np.sqrt(1.25 + 1e-5)
    normalized = (x_val - 1.5) / std
    expected = 2.0 * normalized + 5.0

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_batchnorm_op_properties():
    """
    Test BatchNormalization op properties.

    Verifies:
    - Op can be instantiated with different epsilon values
    - make_node creates correct Apply node
    - infer_shape works correctly
    """
    # Test epsilon parameter
    bn_op1 = BatchNormalization(epsilon=1e-5)
    bn_op2 = BatchNormalization(epsilon=1e-3)

    assert bn_op1.epsilon == 1e-5
    assert bn_op2.epsilon == 1e-3

    # Test make_node
    x = pt.matrix("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    node = bn_op1.make_node(x, gamma, beta, mean, variance)

    assert len(node.inputs) == 5
    assert len(node.outputs) == 1
    assert node.outputs[0].type == x.type

    # Test infer_shape
    from pytensor.graph.fg import FunctionGraph

    fg = FunctionGraph(
        [x, gamma, beta, mean, variance], [bn_op1(x, gamma, beta, mean, variance)]
    )

    input_shapes = [(10, 20), (20,), (20,), (20,), (20,)]
    output_shapes = bn_op1.infer_shape(fg, node, input_shapes)

    assert output_shapes == [(10, 20)]


def test_batchnorm_gradient_not_implemented():
    """
    Test that gradient raises NotImplementedError.

    We only implement inference mode, so gradients are not supported.
    """
    x = pt.vector("x", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")
    mean = pt.vector("mean", dtype="float32")
    variance = pt.vector("variance", dtype="float32")

    y = batch_normalization(x, gamma, beta, mean, variance)

    # Attempting to compute gradient should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="inference only"):
        pt.grad(y.sum(), x)
