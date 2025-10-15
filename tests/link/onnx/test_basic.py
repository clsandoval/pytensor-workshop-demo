"""Core ONNX export tests and comparison utilities.

For information on the ONNX test architecture and how to add tests,
see tests/link/onnx/README.md
"""

from functools import partial

import numpy as np
import pytest


# Skip entire module if ONNX not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt
from pytensor.compile.function import function


def compare_onnx_and_py(
    graph_inputs,
    graph_outputs,
    test_inputs,
    *,
    assert_fn=None,
    tmp_path=None,
):
    """Compare ONNX Runtime output with PyTensor output.

    Parameters
    ----------
    graph_inputs : list of Variable
        Symbolic input variables
    graph_outputs : Variable or list of Variable
        Symbolic output variables
    test_inputs : list
        Concrete test values for inputs
    assert_fn : callable, optional
        Custom assertion function (default: np.testing.assert_allclose)
    tmp_path : Path, optional
        Temporary directory for ONNX file (pytest fixture)

    Returns
    -------
    tuple
        (onnx_session, onnx_results)
    """
    from pytensor.link.onnx import export_onnx

    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-5)

    if tmp_path is None:
        import tempfile

        tmp_path = tempfile.mkdtemp()

    # Ensure graph_outputs is a list
    outputs_is_list = isinstance(graph_outputs, (list, tuple))
    if not outputs_is_list:
        graph_outputs = [graph_outputs]

    # Compile PyTensor function (reference implementation)
    pytensor_fn = function(graph_inputs, graph_outputs)
    py_res = pytensor_fn(*test_inputs)
    # PyTensor function returns a list when graph_outputs is a list,
    # so no need to wrap again

    # Export to ONNX
    onnx_path = f"{tmp_path}/test_model.onnx"
    model = export_onnx(pytensor_fn, onnx_path)

    # Validate ONNX model
    onnx.checker.check_model(model)

    # Run with ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create input feed dict
    onnx_inputs = session.get_inputs()
    input_feed = {}
    for onnx_inp, value in zip(onnx_inputs, test_inputs, strict=True):
        # Convert to numpy array with correct dtype (matching ONNX model)
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        # Match the ONNX model's expected dtype
        input_feed[onnx_inp.name] = value

    # Run inference
    onnx_res = session.run(None, input_feed)

    # Compare results
    assert len(onnx_res) == len(py_res), (
        f"Output count mismatch: {len(onnx_res)} vs {len(py_res)}"
    )

    for onnx_out, py_out in zip(onnx_res, py_res, strict=True):
        assert_fn(onnx_out, py_out)

    return session, onnx_res


def test_onnx_import():
    """Test that ONNX export can be imported."""
    from pytensor.link.onnx import export_onnx

    assert callable(export_onnx)


def test_dispatcher_registered():
    """Test that dispatch system is registered."""
    from pytensor.link.onnx.dispatch import onnx_funcify, onnx_typify

    assert callable(onnx_funcify)
    assert callable(onnx_typify)


def test_export_simple_add(tmp_path):
    """Test exporting a simple addition."""
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    f = pytensor.function([x, y], z)

    # Export
    model_path = tmp_path / "test_add.onnx"
    model = export_onnx(f, model_path)

    # Validate
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model)
    assert model_path.exists()

    # Test with ONNX Runtime
    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], [z], [x_val, y_val], tmp_path=tmp_path)


def test_export_multiple_ops(tmp_path):
    """Test exporting with multiple operations."""
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = (x + y) * 2 - y

    f = pytensor.function([x, y], z)

    # Export and validate
    model = export_onnx(f, tmp_path / "test_multi.onnx")
    onnx.checker.check_model(model)

    # Test execution
    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], [z], [x_val, y_val], tmp_path=tmp_path)


def test_unsupported_op_error():
    """Test that unsupported ops give clear error messages."""
    from pytensor.link.onnx import export_onnx
    from pytensor.tensor import nlinalg

    x = pt.matrix("x")
    # SVD is not supported in Phase 1
    u, s, vt = nlinalg.svd(x)

    f = pytensor.function([x], [u, s, vt])

    with pytest.raises(NotImplementedError, match="No ONNX conversion available"):
        export_onnx(f, "/tmp/test_svd.onnx")


def test_shared_variables_as_initializers(tmp_path):
    """Test that shared variables are converted to ONNX initializers."""
    from onnx import numpy_helper

    from pytensor import shared
    from pytensor.link.onnx import export_onnx

    # Create a simple linear model with shared weights
    W = shared(np.array([[1, 2], [3, 4], [5, 6]], dtype="float32"), name="W")
    b = shared(np.array([0.5, 1.5, 2.5], dtype="float32"), name="b")

    x = pt.vector("x", dtype="float32")
    y = pt.dot(W, x) + b

    f = pytensor.function([x], y)

    # Export to ONNX
    model_path = tmp_path / "test_shared.onnx"
    model = export_onnx(f, model_path)

    # Verify initializers exist in the model
    initializer_names = [init.name for init in model.graph.initializer]
    assert "W" in initializer_names
    assert "b" in initializer_names

    # Verify values are correct
    for init in model.graph.initializer:
        if init.name == "W":
            init_value = numpy_helper.to_array(init)
            np.testing.assert_allclose(init_value, W.get_value())
        elif init.name == "b":
            init_value = numpy_helper.to_array(init)
            np.testing.assert_allclose(init_value, b.get_value())

    # Test execution
    x_val = np.array([1, 2], dtype="float32")
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_deep_copy_operation(tmp_path):
    """Test DeepCopyOp maps to ONNX Identity."""
    from pytensor.compile.ops import DeepCopyOp

    x = pt.vector("x", dtype="float32")
    deep_copy_op = DeepCopyOp()
    y = deep_copy_op(x)

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_deep_copy_in_graph(tmp_path):
    """Test DeepCopyOp within a larger computation."""
    from pytensor.compile.ops import DeepCopyOp

    x = pt.vector("x", dtype="float32")

    # Copy, then do computation
    deep_copy_op = DeepCopyOp()
    x_copy = deep_copy_op(x)
    y = x_copy * 2 + 1

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)


def test_deep_copy_structure(tmp_path):
    """Test that DeepCopyOp generates ONNX Identity node."""
    from pytensor.compile.ops import DeepCopyOp
    from pytensor.link.onnx import export_onnx

    x = pt.vector("x", dtype="float32")
    deep_copy_op = DeepCopyOp()
    y = deep_copy_op(x)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_deep_copy.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    structure = validate_onnx_graph_structure(
        model,
        expected_node_types=["Identity"],
        expected_node_count=1,
    )

    assert structure["node_types"] == ["Identity"]


def validate_onnx_graph_structure(
    model,
    expected_node_types=None,
    expected_node_count=None,
    check_connections=True,
):
    """Validate ONNX graph structure beyond just output correctness.

    Parameters
    ----------
    model : onnx.ModelProto
        The ONNX model to validate
    expected_node_types : list of str, optional
        Expected node op_types in order (or subset)
    expected_node_count : int, optional
        Expected total number of nodes
    check_connections : bool
        Whether to validate all node connections

    Returns
    -------
    dict
        Graph structure information for inspection
    """
    graph = model.graph
    nodes = list(graph.node)

    # Check node count
    if expected_node_count is not None:
        assert len(nodes) == expected_node_count, (
            f"Expected {expected_node_count} nodes, got {len(nodes)}\n"
            f"Nodes: {[n.op_type for n in nodes]}"
        )

    # Check node types
    if expected_node_types is not None:
        actual_types = [n.op_type for n in nodes]
        # Check if expected types appear in order (subset match)
        idx = 0
        for expected_type in expected_node_types:
            found = False
            while idx < len(actual_types):
                if actual_types[idx] == expected_type:
                    found = True
                    idx += 1
                    break
                idx += 1
            assert found, (
                f"Expected node type '{expected_type}' not found in order\n"
                f"Expected: {expected_node_types}\n"
                f"Actual: {actual_types}"
            )

    # Check all connections are valid
    if check_connections:
        all_available = set()
        # Add inputs
        all_available.update(inp.name for inp in graph.input)
        # Add initializers
        all_available.update(init.name for init in graph.initializer)

        # Check each node
        for node in nodes:
            for inp in node.input:
                if inp:  # Skip empty strings (optional inputs)
                    assert inp in all_available, (
                        f"Node {node.name} ({node.op_type}) has undefined input: {inp}\n"
                        f"Available: {sorted(all_available)}"
                    )
            all_available.update(node.output)

    # Return structure info for inspection
    return {
        "node_count": len(nodes),
        "node_types": [n.op_type for n in nodes],
        "input_count": len(graph.input),
        "output_count": len(graph.output),
        "initializer_count": len(graph.initializer),
    }
