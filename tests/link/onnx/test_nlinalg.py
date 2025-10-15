"""Tests for ONNX linear algebra operations."""

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


def test_dot_vector_vector(tmp_path):
    """Test dot product of two vectors."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.dot(x, y)

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_dot_matrix_vector(tmp_path):
    """Test matrix-vector multiplication."""
    x = pt.matrix("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.dot(x, y)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 4)).astype("float32")
    y_val = rng.random(4).astype("float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_dot_matrix_matrix(tmp_path):
    """Test matrix-matrix multiplication."""
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.dot(x, y)

    rng = np.random.default_rng(42)
    x_val = rng.random((3, 4)).astype("float32")
    y_val = rng.random((4, 5)).astype("float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_simple_linear_layer(tmp_path):
    """Test a simple linear layer: W @ x + b."""
    x = pt.vector("x", dtype="float32")
    W = pt.matrix("W", dtype="float32")
    b = pt.vector("b", dtype="float32")

    # Linear layer
    y = pt.dot(W, x) + b

    rng = np.random.default_rng(42)
    x_val = rng.random(10).astype("float32")
    W_val = rng.random((5, 10)).astype("float32")
    b_val = rng.random(5).astype("float32")

    compare_onnx_and_py([x, W, b], y, [x_val, W_val, b_val], tmp_path=tmp_path)


def test_gemv_operation(tmp_path):
    """Test Gemv (general matrix-vector multiplication with scaling).

    Gemv computes: y = alpha * A @ x + beta * y_in
    """
    # Define inputs
    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha = pt.scalar("alpha", dtype="float32")
    beta = pt.scalar("beta", dtype="float32")

    # Import Gemv from blas
    from pytensor.tensor.blas import Gemv

    gemv_op = Gemv(inplace=False)

    # Create Gemv operation: y = alpha * A @ x + beta * y_in
    y = gemv_op(y_in, alpha, A, x, beta)

    # Test data
    rng = np.random.default_rng(42)
    A_val = rng.random((3, 4)).astype("float32")
    x_val = rng.random(4).astype("float32")
    y_in_val = rng.random(3).astype("float32")
    alpha_val = np.array(2.0, dtype="float32")
    beta_val = np.array(0.5, dtype="float32")

    compare_onnx_and_py(
        [y_in, alpha, A, x, beta],
        y,
        [y_in_val, alpha_val, A_val, x_val, beta_val],
        tmp_path=tmp_path,
    )


def test_gemv_structure(tmp_path):
    """Test that Gemv generates correct 4-node ONNX structure."""
    from pytensor.link.onnx import export_onnx
    from pytensor.tensor.blas import Gemv

    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha = pt.scalar("alpha", dtype="float32")
    beta = pt.scalar("beta", dtype="float32")

    gemv_op = Gemv(inplace=False)
    y = gemv_op(y_in, alpha, A, x, beta)

    f = pytensor.function([y_in, alpha, A, x, beta], y)

    # Export
    model_path = tmp_path / "test_gemv.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["MatMul", "Mul", "Mul", "Add"],
        expected_node_count=4,
    )

    # Verify the 4 nodes are: MatMul, Mul (alpha), Mul (beta), Add
    node_types = structure["node_types"]
    assert node_types.count("MatMul") == 1
    assert node_types.count("Mul") == 2
    assert node_types.count("Add") == 1


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (1.0, 0.0),  # Just A @ x
        (1.0, 1.0),  # A @ x + y
        (2.0, 0.5),  # Scaled
        (0.0, 1.0),  # Just beta * y
    ],
)
def test_gemv_scaling_factors(tmp_path, alpha, beta):
    """Test Gemv with different scaling factors."""
    from pytensor.tensor.blas import Gemv

    A = pt.matrix("A", dtype="float32")
    x = pt.vector("x", dtype="float32")
    y_in = pt.vector("y_in", dtype="float32")
    alpha_var = pt.scalar("alpha", dtype="float32")
    beta_var = pt.scalar("beta", dtype="float32")

    gemv_op = Gemv(inplace=False)
    y = gemv_op(y_in, alpha_var, A, x, beta_var)

    rng = np.random.default_rng(42)
    A_val = rng.random((3, 4)).astype("float32")
    x_val = rng.random(4).astype("float32")
    y_in_val = rng.random(3).astype("float32")
    alpha_val = np.array(alpha, dtype="float32")
    beta_val = np.array(beta, dtype="float32")

    compare_onnx_and_py(
        [y_in, alpha_var, A, x, beta_var],
        y,
        [y_in_val, alpha_val, A_val, x_val, beta_val],
        tmp_path=tmp_path,
    )
