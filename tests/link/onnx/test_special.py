"""Tests for ONNX special functions and activations."""

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from pytensor.tensor.special import softmax
from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_softmax(tmp_path):
    """Test softmax activation."""
    x = pt.matrix("x", dtype="float32")
    y = softmax(x)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 5)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_softmax_axis(tmp_path, axis):
    """Test softmax with different axes."""
    x = pt.matrix("x", dtype="float32")
    y = softmax(x, axis=axis)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 5)).astype("float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_relu_via_maximum(tmp_path):
    """Test ReLU implementation via maximum(x, 0)."""
    x = pt.vector("x", dtype="float32")
    y = pt.maximum(x, 0)

    x_val = np.array([-2, -1, 0, 1, 2], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_maximum(tmp_path):
    """Test maximum operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.maximum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_minimum(tmp_path):
    """Test minimum operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.minimum(x, y)

    x_val = np.array([1, 5, 3], dtype="float32")
    y_val = np.array([2, 3, 4], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_two_layer_network(tmp_path):
    """Test a complete 2-layer neural network with activations."""
    # Input
    x = pt.vector("x", dtype="float32")

    # Layer 1 weights (as inputs for now - will test shared vars in test_basic.py)
    W1 = pt.matrix("W1", dtype="float32")
    b1 = pt.vector("b1", dtype="float32")

    # Layer 1 forward pass
    h1 = pt.dot(W1, x) + b1
    h1_relu = pt.maximum(h1, 0)  # ReLU

    # Layer 2 weights
    W2 = pt.matrix("W2", dtype="float32")
    b2 = pt.vector("b2", dtype="float32")

    # Layer 2 forward pass
    y_logits = pt.dot(W2, h1_relu) + b2
    y_pred = softmax(y_logits.reshape((1, -1))).flatten()

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.random(4).astype("float32")
    W1_val = rng.random((8, 4)).astype("float32") * 0.1
    b1_val = np.zeros(8, dtype="float32")
    W2_val = rng.random((3, 8)).astype("float32") * 0.1
    b2_val = np.zeros(3, dtype="float32")

    compare_onnx_and_py(
        [x, W1, b1, W2, b2],
        y_pred,
        [x_val, W1_val, b1_val, W2_val, b2_val],
        tmp_path=tmp_path,
    )
