"""Tests for ONNX shape operations."""

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


def test_dimshuffle_unsqueeze_start(tmp_path):
    """Test adding dimension at the start."""
    x = pt.vector("x", dtype="float32")
    y = x.dimshuffle("x", 0)  # (3,) -> (1, 3)

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_unsqueeze_end(tmp_path):
    """Test adding dimension at the end."""
    x = pt.vector("x", dtype="float32")
    y = x.dimshuffle(0, "x")  # (3,) -> (3, 1)

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_unsqueeze_multiple(tmp_path):
    """Test adding multiple dimensions."""
    x = pt.vector("x", dtype="float32")
    y = x.dimshuffle("x", 0, "x")  # (3,) -> (1, 3, 1)

    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_squeeze_first(tmp_path):
    """Test removing first dimension."""
    x = pt.tensor(dtype="float32", shape=(1, 3), name="x")
    y = x.dimshuffle(1)  # (1, 3) -> (3,)

    x_val = np.array([[1.0, 2.0, 3.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_squeeze_last(tmp_path):
    """Test removing last dimension."""
    x = pt.tensor(dtype="float32", shape=(3, 1), name="x")
    y = x.dimshuffle(0)  # (3, 1) -> (3,)

    x_val = np.array([[1.0], [2.0], [3.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_transpose(tmp_path):
    """Test transposing 2D matrix."""
    x = pt.matrix("x", dtype="float32")
    y = x.dimshuffle(1, 0)  # (2, 3) -> (3, 2)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_transpose_3d(tmp_path):
    """Test transposing 3D tensor."""
    x = pt.tensor(dtype="float32", shape=(2, 3, 4), name="x")
    y = x.dimshuffle(2, 0, 1)  # (2, 3, 4) -> (4, 2, 3)

    rng = np.random.default_rng(42)
    x_val = rng.random((2, 3, 4)).astype("float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_transpose_and_unsqueeze(tmp_path):
    """Test transpose combined with unsqueeze - currently FAILS (bug)."""
    x = pt.matrix("x", dtype="float32")
    # Input: (2, 3), Output: (3, 1, 2)
    # This requires: Transpose(1,0) → Unsqueeze(axis=1)
    y = x.dimshuffle(1, "x", 0)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_squeeze_and_transpose(tmp_path):
    """Test squeeze combined with transpose - currently FAILS (bug)."""
    x = pt.tensor(dtype="float32", shape=(2, 1, 3), name="x")
    # Input: (2, 1, 3), Output: (3, 2)
    # This requires: Squeeze(axis=1) → Transpose(1,0)
    y = x.dimshuffle(2, 0)

    x_val = np.random.default_rng(42).random((2, 1, 3)).astype("float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_dimshuffle_unsqueeze_and_transpose(tmp_path):
    """Test unsqueeze combined with transpose - currently FAILS (bug)."""
    x = pt.matrix("x", dtype="float32")
    # Input: (2, 3), Output: (1, 3, 2)
    # This requires: Transpose(1,0) → Unsqueeze(axis=0)
    y = x.dimshuffle("x", 1, 0)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize(
    "pattern,input_shape,expected_shape",
    [
        # (new_order, input_shape, expected_shape)
        ((1, "x", 0), (2, 3), (3, 1, 2)),  # transpose + unsqueeze
        ((2, 0), (2, 1, 3), (3, 2)),  # squeeze + transpose
        (("x", 1, 0), (2, 3), (1, 3, 2)),  # unsqueeze + transpose
        ((0, 2, "x"), (3, 1, 4), (3, 4, 1)),  # squeeze + unsqueeze
        ((2, "x", 0, 1), (2, 3, 4), (4, 1, 2, 3)),  # transpose + unsqueeze
        (("x", 2, 1, "x", 0), (2, 3, 4), (1, 4, 3, 1, 2)),  # complex
    ],
)
def test_dimshuffle_complex_patterns(tmp_path, pattern, input_shape, expected_shape):
    """Test various complex DimShuffle patterns that combine operations."""
    x = pt.tensor(dtype="float32", shape=input_shape, name="x")
    y = x.dimshuffle(*pattern)

    rng = np.random.default_rng(42)
    x_val = rng.random(input_shape).astype("float32")

    # Verify expected shape
    assert y.type.shape == expected_shape, (
        f"Shape mismatch: {y.type.shape} vs {expected_shape}"
    )

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_reshape_vector_to_matrix(tmp_path):
    """Test reshaping vector to matrix."""
    x = pt.vector("x", dtype="float32")
    y = x.reshape((2, 3))

    x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_reshape_with_minus_one(tmp_path):
    """Test reshape with inferred dimension (-1)."""
    x = pt.vector("x", dtype="float32")
    y = x.reshape((2, -1))

    x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_reshape_flatten(tmp_path):
    """Test flattening a matrix."""
    x = pt.matrix("x", dtype="float32")
    y = x.reshape((-1,))

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_flatten_method(tmp_path):
    """Test flatten() method."""
    x = pt.matrix("x", dtype="float32")
    y = x.flatten()

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_shape_i_get_dimension(tmp_path):
    """Test extracting specific dimensions with shape_i."""
    x = pt.matrix("x", dtype="float32")
    # Get the shape and use it in a computation
    dim0 = x.shape[0]
    # Cast to float32 to match tensor dtype
    dim0_float = pt.cast(dim0, "float32")
    # Create a computation that uses the shape
    y = x + dim0_float  # Broadcasting scalar with matrix

    x_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_combined_reshape_operations(tmp_path):
    """Test multiple reshape operations in sequence."""
    x = pt.vector("x", dtype="float32")
    y = x.reshape((2, 3))
    z = y.dimshuffle(1, 0)  # Transpose
    w = z.reshape((-1,))  # Flatten

    x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float32")
    compare_onnx_and_py([x], w, [x_val], tmp_path=tmp_path)


def test_alloc_empty_scalar_dims(tmp_path):
    """Test AllocEmpty with scalar dimension inputs."""
    # Create shape from scalars
    dim0 = pt.scalar("dim0", dtype="int64")
    dim1 = pt.scalar("dim1", dtype="int64")

    from pytensor.tensor.basic import AllocEmpty

    alloc_op = AllocEmpty(dtype="float32")

    x = alloc_op(dim0, dim1)

    dim0_val = np.array(3, dtype="int64")
    dim1_val = np.array(4, dtype="int64")

    # Note: AllocEmpty creates uninitialized memory, ONNX creates zeros
    # We can't compare values, but we can check shapes
    from pytensor.link.onnx import export_onnx

    f = pytensor.function([dim0, dim1], x)
    model_path = tmp_path / "test_alloc_empty.onnx"
    model = export_onnx(f, model_path)

    # Validate model structure
    onnx.checker.check_model(model)

    # Run with ONNX Runtime to check shape
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {
        onnx_inputs[0].name: dim0_val,
        onnx_inputs[1].name: dim1_val,
    }
    onnx_res = session.run(None, input_feed)

    # Check shape is correct
    assert onnx_res[0].shape == (3, 4)


def test_alloc_empty_from_shape(tmp_path):
    """Test AllocEmpty with dimensions extracted from another tensor's shape."""
    # Get dimensions from an existing tensor's shape
    x = pt.matrix("x", dtype="float32")
    dim0 = x.shape[0]
    dim1 = x.shape[1]

    from pytensor.tensor.basic import AllocEmpty

    alloc_op = AllocEmpty(dtype="float32")

    # Create a new tensor with the same shape
    y = alloc_op(dim0, dim1)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

    # Export and check
    from pytensor.link.onnx import export_onnx

    f = pytensor.function([x], y)
    model_path = tmp_path / "test_alloc_empty_from_shape.onnx"
    model = export_onnx(f, model_path)

    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()
    input_feed = {onnx_inputs[0].name: x_val}
    onnx_res = session.run(None, input_feed)

    # Should have same shape as input
    assert onnx_res[0].shape == x_val.shape


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_alloc_empty_dtypes(tmp_path, dtype):
    """Test AllocEmpty with different dtypes."""
    dim0 = pt.scalar("dim0", dtype="int64")
    dim1 = pt.scalar("dim1", dtype="int64")

    from pytensor.tensor.basic import AllocEmpty

    alloc_op = AllocEmpty(dtype=dtype)

    x = alloc_op(dim0, dim1)

    from pytensor.link.onnx import export_onnx

    f = pytensor.function([dim0, dim1], x)
    model_path = tmp_path / f"test_alloc_empty_{dtype}.onnx"
    model = export_onnx(f, model_path)

    onnx.checker.check_model(model)

    # Check output dtype
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dim0_val = np.array(2, dtype="int64")
    dim1_val = np.array(3, dtype="int64")

    onnx_inputs = session.get_inputs()
    input_feed = {
        onnx_inputs[0].name: dim0_val,
        onnx_inputs[1].name: dim1_val,
    }
    onnx_res = session.run(None, input_feed)

    expected_dtype = np.dtype(dtype)
    assert onnx_res[0].dtype == expected_dtype


def test_make_vector_from_scalars(tmp_path):
    """Test MakeVector creating a vector from scalar inputs."""
    from pytensor.tensor.basic import MakeVector

    # Create scalars
    a = pt.scalar("a", dtype="int64")
    b = pt.scalar("b", dtype="int64")
    c = pt.scalar("c", dtype="int64")

    # Make a vector from them
    make_vec_op = MakeVector(dtype="int64")
    vec = make_vec_op(a, b, c)

    a_val = np.array(1, dtype="int64")
    b_val = np.array(2, dtype="int64")
    c_val = np.array(3, dtype="int64")

    compare_onnx_and_py([a, b, c], vec, [a_val, b_val, c_val], tmp_path=tmp_path)


def test_make_vector_from_shape(tmp_path):
    """Test MakeVector with shape dimensions (common use case)."""
    from pytensor.tensor.basic import MakeVector

    # Get shape dimensions from a tensor
    x = pt.matrix("x", dtype="float32")
    dim0 = x.shape[0]

    # Create a shape vector like [dim0, -1] for reshape
    make_vec_op = MakeVector(dtype="int64")
    shape_vec = make_vec_op(dim0, np.int64(-1))

    # Use this in a computation
    y = x.reshape(shape_vec)

    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_make_vector_dtypes(tmp_path, dtype):
    """Test MakeVector with different dtypes."""
    from pytensor.tensor.basic import MakeVector

    a = pt.scalar("a", dtype=dtype)
    b = pt.scalar("b", dtype=dtype)

    make_vec_op = MakeVector(dtype=dtype)
    vec = make_vec_op(a, b)

    a_val = np.array(10, dtype=dtype)
    b_val = np.array(20, dtype=dtype)

    compare_onnx_and_py([a, b], vec, [a_val, b_val], tmp_path=tmp_path)
