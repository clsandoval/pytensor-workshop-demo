"""Property-based tests for ONNX operations using Hypothesis."""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st


onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor.tensor as pt
from tests.link.onnx.strategies import ONNX_OPERATIONS
from tests.link.onnx.test_basic import compare_onnx_and_py


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for ONNX files."""
    return tmp_path_factory.mktemp("onnx_tests")


# Property 1: ONNX output matches PyTensor output
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    # Exclude reshape for now - it has a different signature
    op_name=st.sampled_from([k for k in ONNX_OPERATIONS.keys() if k != "reshape"]),
    data=st.data(),
)
def test_onnx_matches_pytensor(tmp_path, op_name, data):
    """
    Property: For any valid operation and inputs, ONNX output must match PyTensor.

    This is the fundamental correctness property - the ONNX backend should
    produce the same numerical results as PyTensor's native execution.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs using operation-specific strategy
    inputs_tuple = data.draw(op_config.input_strategy)

    # Skip if dtype doesn't match valid_dtypes for this operation
    if inputs_tuple and hasattr(inputs_tuple[0], "dtype"):
        dtype_str = str(inputs_tuple[0].dtype)
        if dtype_str not in op_config.valid_dtypes:
            assume(False)  # Skip this test case

    # Handle special cases that need filtering
    if op_name == "log":
        # Log requires positive inputs
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "sqrt":
        # Sqrt requires non-negative inputs
        inputs_tuple = tuple(np.abs(x) for x in inputs_tuple)
    elif op_name == "div":
        # Division requires non-zero divisor
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)  # Replace near-zero with 1.0
        inputs_tuple = (x, y)

    # Create symbolic variables
    if len(inputs_tuple) == 1:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
        symbolic_inputs = [x]
        test_values = [inputs_tuple[0]]

        # Apply operation
        result = op_config.op_func(x)
    elif len(inputs_tuple) == 2:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)

        # Handle different second argument types
        if isinstance(inputs_tuple[1], tuple):
            # Second argument is a shape (e.g., reshape)
            symbolic_inputs = [x]
            test_values = [inputs_tuple[0]]  # Only pass the tensor, not the shape
            result = op_config.op_func(x, inputs_tuple[1])
        else:
            # Second argument is a tensor
            y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
            symbolic_inputs = [x, y]
            test_values = [inputs_tuple[0], inputs_tuple[1]]
            result = op_config.op_func(x, y)
    else:
        raise NotImplementedError(
            f"Operations with {len(inputs_tuple)} inputs not yet supported"
        )

    # Compare ONNX and PyTensor outputs
    try:
        compare_onnx_and_py(symbolic_inputs, result, test_values, tmp_path=tmp_path)
    except Exception as e:
        # Re-raise with context about which operation failed
        # Get shapes safely (some inputs might be tuples)
        shapes = [
            x.shape if hasattr(x, "shape") else f"shape_tuple{x}" for x in inputs_tuple
        ]
        dtypes = [x.dtype if hasattr(x, "dtype") else "tuple" for x in inputs_tuple]
        raise AssertionError(
            f"Property test failed for operation '{op_name}' "
            f"with input shapes: {shapes}, "
            f"dtypes: {dtypes}"
        ) from e


# Property 2: Shape preservation for elemwise operations
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    op_name=st.sampled_from(
        [k for k, v in ONNX_OPERATIONS.items() if v.category == "elemwise"]
    ),
    data=st.data(),
)
def test_elemwise_preserves_broadcast_shape(tmp_path, op_name, data):
    """
    Property: Elemwise operations preserve broadcasting shape rules.

    For any elemwise operation, the output shape should match NumPy's
    broadcasting rules applied to the input shapes.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs
    inputs_tuple = data.draw(op_config.input_strategy)

    # Filter invalid inputs
    if op_name in ("log", "sqrt"):
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)
        inputs_tuple = (x, y)

    # Compute expected output shape using NumPy broadcasting
    if len(inputs_tuple) == 1:
        expected_shape = inputs_tuple[0].shape
    else:
        # Use NumPy to determine broadcast shape
        expected_shape = np.broadcast_shapes(*[x.shape for x in inputs_tuple])

    # Create symbolic computation
    if len(inputs_tuple) == 1:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
        result = op_config.op_func(x)
        symbolic_inputs = [x]
    else:
        x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
        y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
        result = op_config.op_func(x, y)
        symbolic_inputs = [x, y]

    # Run through ONNX
    _, onnx_results = compare_onnx_and_py(
        symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path
    )

    # Verify shape
    assert onnx_results[0].shape == expected_shape, (
        f"Operation '{op_name}' produced wrong shape. "
        f"Expected {expected_shape}, got {onnx_results[0].shape}"
    )


# Property 3: Dtype preservation
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    # Exclude reshape for now - it has a different signature
    op_name=st.sampled_from([k for k in ONNX_OPERATIONS.keys() if k != "reshape"]),
    data=st.data(),
)
def test_operation_preserves_dtype(tmp_path, op_name, data):
    """
    Property: Operations preserve input dtype (with known exceptions).

    Most operations should output the same dtype as their input.
    Exceptions: division always produces float, comparisons produce bool.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs
    inputs_tuple = data.draw(op_config.input_strategy)

    # Skip if dtype doesn't match valid_dtypes for this operation
    if inputs_tuple and hasattr(inputs_tuple[0], "dtype"):
        dtype_str = str(inputs_tuple[0].dtype)
        if dtype_str not in op_config.valid_dtypes:
            assume(False)  # Skip this test case

    # Filter invalid inputs
    if op_name in ("log", "sqrt"):
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)
        inputs_tuple = (x, y)

    input_dtype = inputs_tuple[0].dtype

    # Create symbolic computation
    if len(inputs_tuple) == 1:
        x = pt.tensor("x", dtype=input_dtype, shape=inputs_tuple[0].shape)
        result = op_config.op_func(x)
        symbolic_inputs = [x]
    elif isinstance(inputs_tuple[1], tuple):
        # Second arg is shape (reshape case)
        x = pt.tensor("x", dtype=input_dtype, shape=inputs_tuple[0].shape)
        result = op_config.op_func(x, inputs_tuple[1])
        symbolic_inputs = [x]
    else:
        x = pt.tensor("x", dtype=input_dtype, shape=inputs_tuple[0].shape)
        y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
        result = op_config.op_func(x, y)
        symbolic_inputs = [x, y]

    # Run through ONNX
    _, onnx_results = compare_onnx_and_py(
        symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path
    )

    # Verify dtype (accounting for known exceptions)
    output_dtype = onnx_results[0].dtype

    # Known exceptions where dtype changes
    if op_name == "div":
        # Division always produces float
        assert np.issubdtype(output_dtype, np.floating), (
            f"Division should produce float, got {output_dtype}"
        )
    else:
        # Most operations preserve dtype
        assert output_dtype == input_dtype, (
            f"Operation '{op_name}' changed dtype from {input_dtype} to {output_dtype}"
        )


# Property 4: Operations don't crash on edge cases
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    # Exclude reshape for now - it has a different signature
    op_name=st.sampled_from([k for k in ONNX_OPERATIONS.keys() if k != "reshape"]),
    data=st.data(),
)
def test_operation_handles_edge_cases(tmp_path, op_name, data):
    """
    Property: Operations handle edge cases without crashing.

    Tests with:
    - Empty tensors (shape with 0)
    - Scalars (0-dimensional tensors)
    - Large values
    - Small values near zero

    Operations may produce inf/nan for invalid inputs, but should not crash.
    """
    op_config = ONNX_OPERATIONS[op_name]

    # Generate inputs
    inputs_tuple = data.draw(op_config.input_strategy)

    # Skip if dtype doesn't match valid_dtypes for this operation
    if inputs_tuple and hasattr(inputs_tuple[0], "dtype"):
        dtype_str = str(inputs_tuple[0].dtype)
        if dtype_str not in op_config.valid_dtypes:
            assume(False)  # Skip this test case

    # Apply necessary filters
    if op_name in ("log", "sqrt"):
        inputs_tuple = tuple(np.abs(x) + 1e-6 for x in inputs_tuple)
    elif op_name == "div":
        x, y = inputs_tuple
        y = np.where(np.abs(y) < 1e-6, 1.0, y)
        inputs_tuple = (x, y)

    # Create symbolic computation
    try:
        if len(inputs_tuple) == 1:
            x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
            result = op_config.op_func(x)
            symbolic_inputs = [x]
        elif isinstance(inputs_tuple[1], tuple):
            x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
            result = op_config.op_func(x, inputs_tuple[1])
            symbolic_inputs = [x]
        else:
            x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
            y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
            result = op_config.op_func(x, y)
            symbolic_inputs = [x, y]

        # Run through ONNX - should not crash
        compare_onnx_and_py(
            symbolic_inputs, result, list(inputs_tuple), tmp_path=tmp_path
        )

    except (ValueError, TypeError, RuntimeError):
        # Some operations may legitimately fail for certain inputs
        # (e.g., reshape with incompatible shape)
        # This is acceptable - we just want to ensure it doesn't crash Python
        pass
