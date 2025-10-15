"""ONNX conversion for Join (Concat) operation."""

from onnx import helper

from pytensor.graph.basic import Constant
from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.basic import Join


@onnx_funcify.register(Join)
def onnx_funcify_Join(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor Join op to ONNX Concat node.

    PyTensor Join concatenates multiple tensors along a specified axis.
    ONNX Concat performs the same operation.

    Parameters
    ----------
    op : Join
        The Join operation instance
    node : Apply
        The apply node containing inputs and outputs
    var_names : dict
        Mapping of variables to ONNX names
    get_var_name : callable
        Function to get ONNX name for a variable

    Returns
    -------
    onnx.NodeProto
        ONNX Concat node

    Notes
    -----
    PyTensor Join takes axis as the first input (runtime value),
    but ONNX Concat requires axis as a compile-time attribute.

    In PyTensor graphs, the axis is typically a Constant, so we extract
    its value and pass it as an ONNX attribute.

    Join inputs: [axis (scalar constant), tensor1, tensor2, ...]
    Concat inputs: [tensor1, tensor2, ...]
    Concat attributes: axis=<int>

    Examples
    --------
    PyTensor:
    >>> x = pt.matrix("x")
    >>> y = pt.matrix("y")
    >>> z = pt.join(0, x, y)  # Concatenate along axis 0

    ONNX equivalent:
    >>> Concat(inputs=[x, y], axis=0)
    """
    # Extract inputs
    # node.inputs[0] is the axis (should be a Constant)
    # node.inputs[1:] are the tensors to concatenate

    axis_input = node.inputs[0]
    tensor_inputs = node.inputs[1:]

    # Extract axis value
    if not isinstance(axis_input, Constant):
        raise NotImplementedError(
            "ONNX Concat requires axis to be a compile-time constant. "
            f"Got: {axis_input}"
        )

    axis = int(axis_input.data)

    # Get ONNX names for tensor inputs
    input_names = [get_var_name(inp) for inp in tensor_inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX Concat node
    return helper.make_node(
        "Concat",
        inputs=input_names,
        outputs=output_names,
        axis=axis,
        name=f"Concat_{output_names[0]}",
    )
