"""Core ONNX dispatch system for PyTensor.

This module provides the singledispatch-based conversion system for
converting PyTensor ops to ONNX nodes.
"""

from functools import singledispatch


try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install pytensor[onnx]"
    ) from e

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.raise_op import Assert, CheckAndRaise


# Target ONNX opset version
ONNX_OPSET_VERSION = 18


@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert PyTensor Op to ONNX representation.

    This is the main dispatch function. Register converters for specific
    Op types using @onnx_funcify.register(OpClass).

    Parameters
    ----------
    op : Op or FunctionGraph
        The operation to convert
    node : Apply, optional
        The Apply node containing the op (when op is an Op)
    **kwargs
        Additional conversion parameters:
        - var_names: Dict[Variable, str] - mapping of variables to names
        - get_var_name: Callable - function to get/create variable names

    Returns
    -------
    onnx.NodeProto or onnx.ModelProto
        ONNX representation of the operation

    Raises
    ------
    NotImplementedError
        If no converter is registered for this Op type
    """
    raise NotImplementedError(
        f"No ONNX conversion available for: {type(op).__name__}\n"
        f"Op: {op}\n"
        f"Node: {node}\n\n"
        f"This op is not yet supported for ONNX export.\n"
        f"Currently supported ops:\n"
        f"  - Elemwise: Add, Mul, Sub, Div, Neg, Exp, Log, Sqrt, Pow, Abs\n"
        f"  - Matrix: Dot\n"
        f"  - Activations: Softmax, Maximum (for ReLU)\n\n"
        f"To add support for this op, register a converter:\n"
        f"  @onnx_funcify.register({type(op).__name__})\n"
        f"  def onnx_funcify_{type(op).__name__}(op, node, var_names, get_var_name, **kwargs):\n"
        f"      # Return onnx.NodeProto\n"
    )


@onnx_funcify.register(Assert)
@onnx_funcify.register(CheckAndRaise)
def onnx_funcify_assert(op, node, var_names, get_var_name, **kwargs):
    """
    Assert and CheckAndRaise operations are skipped during ONNX export.

    These operations perform runtime validation and can raise exceptions,
    but ONNX has no exception handling mechanism. For ONNX export, we
    assume shapes are validated at compile time, so assertions are
    converted to pass-through (Identity) operations.

    Node structure:
    - inputs[0]: The value to return if assertion passes
    - inputs[1:]: Boolean conditions that must all be True

    We only pass through the value, ignoring the conditions.
    """
    # First input is the value, rest are conditions
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    return helper.make_node(
        "Identity",
        inputs=[input_name],
        outputs=[output_name],
        name=f"assert_passthrough_{output_name}",
    )


@singledispatch
def onnx_typify(data, dtype=None, **kwargs):
    """Convert Python/NumPy data to ONNX-compatible types.

    This is used for converting constants and shared variables to ONNX tensors.

    Parameters
    ----------
    data : Any
        Data to convert (typically numpy array or scalar)
    dtype : str, optional
        Target dtype for conversion

    Returns
    -------
    onnx.TensorProto or data
        ONNX tensor representation or original data
    """
    if dtype is None:
        return data
    else:
        return np.array(data, dtype=dtype)


@onnx_typify.register(np.ndarray)
def onnx_typify_ndarray(data, dtype=None, name="", **kwargs):
    """Convert numpy array to ONNX TensorProto."""
    if dtype is not None:
        data = data.astype(dtype)
    return numpy_helper.from_array(data, name=name)


def make_value_info(var: Variable, name: str) -> onnx.ValueInfoProto:
    """Create ONNX ValueInfoProto from PyTensor Variable.

    Parameters
    ----------
    var : Variable
        PyTensor variable
    name : str
        Name for the ONNX value

    Returns
    -------
    onnx.ValueInfoProto
        ONNX value info with type and shape
    """
    # Map PyTensor dtype to ONNX dtype
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "uint8": TensorProto.UINT8,
        "int8": TensorProto.INT8,
        "bool": TensorProto.BOOL,
    }

    dtype_str = str(var.type.dtype)
    onnx_dtype = dtype_map.get(dtype_str, TensorProto.FLOAT)

    # Get shape (use symbolic dimensions if needed)
    if hasattr(var.type, "shape"):
        shape = []
        for i, dim in enumerate(var.type.shape):
            if dim is None or (isinstance(dim, int) and dim < 0):
                # Dynamic dimension - use symbolic name
                shape.append(f"dim_{i}")
            else:
                shape.append(int(dim))
    else:
        shape = None

    # Create tensor type
    tensor_type = helper.make_tensor_type_proto(elem_type=onnx_dtype, shape=shape)

    return helper.make_value_info(name, tensor_type)


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph: FunctionGraph,
    node=None,
    opset_version: int = ONNX_OPSET_VERSION,
    model_name: str = "pytensor_model",
    user_inputs=None,
    **kwargs,
) -> onnx.ModelProto:
    """Convert a FunctionGraph to ONNX ModelProto.

    Parameters
    ----------
    fgraph : FunctionGraph
        The graph to convert
    opset_version : int
        ONNX opset version to target (default: 18)
    model_name : str
        Name for the ONNX model
    user_inputs : set, optional
        Set of Variables that are user inputs (not shared variables)

    Returns
    -------
    onnx.ModelProto
        Complete ONNX model
    """
    # Track converted nodes and initializers
    onnx_nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    # Default to treating all inputs as user inputs if not specified
    if user_inputs is None:
        user_inputs = set(fgraph.inputs)

    # Generate unique names for variables
    var_names: dict[Variable, str] = {}
    name_counter = 0

    def get_var_name(var: Variable) -> str:
        """Get or create unique name for a variable."""
        nonlocal name_counter
        if var not in var_names:
            if hasattr(var, "name") and var.name:
                base_name = var.name
                # Ensure uniqueness
                if base_name in var_names.values():
                    base_name = f"{base_name}_{name_counter}"
                    name_counter += 1
                var_names[var] = base_name
            else:
                var_names[var] = f"var_{name_counter}"
                name_counter += 1
        return var_names[var]

    # Pre-scan ops to collect flip requests (for filter_flip=True in Conv)
    # This must be done before creating initializers
    flip_initializers = {}
    for node in fgraph.toposort():
        # Import here to avoid circular dependency
        from pytensor.tensor.conv.abstract_conv import AbstractConv2d

        if isinstance(node.op, AbstractConv2d) and node.op.filter_flip:
            # This Conv needs its kernel flipped
            kernel_var = node.inputs[1]
            kernel_name = get_var_name(kernel_var)
            flip_initializers[kernel_name] = (2, 3)  # Flip spatial dimensions

    # Convert shared variables (inputs not in user_inputs) to initializers
    for inp in fgraph.inputs:
        if inp not in user_inputs and not isinstance(inp, Constant):
            # This is a shared variable - convert to initializer
            name = get_var_name(inp)
            if name not in [init.name for init in initializers]:
                # Get the actual value from the shared variable
                if hasattr(inp, "get_value"):
                    value = inp.get_value()
                elif hasattr(inp, "data"):
                    value = inp.data
                else:
                    # Fallback - try to get it from the variable itself
                    value = np.asarray(inp.data) if hasattr(inp, "data") else None

                if value is not None:
                    value_array = np.asarray(value)

                    # Check if this initializer needs to be flipped
                    if name in flip_initializers:
                        axes = flip_initializers[name]
                        # Flip along the specified axes
                        # For Conv2D kernel flipping: flip spatial dimensions (H and W)
                        value_array = np.flip(value_array, axis=axes).copy()

                    tensor = numpy_helper.from_array(value_array, name=name)
                    initializers.append(tensor)

    # Convert constants to initializers
    for node in fgraph.apply_nodes:
        for inp in node.inputs:
            if isinstance(inp, Constant):
                name = get_var_name(inp)
                if name not in [init.name for init in initializers]:
                    value_array = np.asarray(inp.data)

                    # Check if this initializer needs to be flipped
                    if name in flip_initializers:
                        axes = flip_initializers[name]
                        # Flip along the specified axes
                        # For Conv2D kernel flipping: flip spatial dimensions (H and W)
                        value_array = np.flip(value_array, axis=axes).copy()

                    tensor = numpy_helper.from_array(value_array, name=name)
                    initializers.append(tensor)

    # Convert ops in topological order
    for node in fgraph.toposort():
        # Get ONNX node for this Apply
        onnx_node = onnx_funcify(
            node.op,
            node=node,
            var_names=var_names,
            get_var_name=get_var_name,
            **kwargs,
        )

        if onnx_node is not None:
            # Handle both single nodes and lists of nodes
            if isinstance(onnx_node, list):
                onnx_nodes.extend(onnx_node)
            else:
                onnx_nodes.append(onnx_node)

    # Create inputs (only user inputs, not shared variables or constants)
    input_protos = []
    for inp in fgraph.inputs:
        if inp in user_inputs and not isinstance(inp, Constant):
            name = get_var_name(inp)
            input_protos.append(make_value_info(inp, name))

    # Create outputs
    output_protos = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        output_protos.append(make_value_info(out, name))

    # Create graph
    graph = helper.make_graph(
        nodes=onnx_nodes,
        name=f"{model_name}_graph",
        inputs=input_protos,
        outputs=output_protos,
        initializer=initializers,
    )

    # Create model with IR version 9 (compatible with ONNX Runtime 1.16+)
    model = helper.make_model(
        graph,
        producer_name="PyTensor",
        opset_imports=[helper.make_opsetid("", opset_version)],
        ir_version=9,
    )

    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        raise ValueError(f"Generated ONNX model is invalid: {e}") from e

    return model
