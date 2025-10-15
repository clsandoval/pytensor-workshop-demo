"""ONNX conversion for pooling operations."""

from onnx import helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.pool import Pool


@onnx_funcify.register(Pool)
def onnx_funcify_Pool(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor Pool op to ONNX MaxPool node.

    Parameters
    ----------
    op : Pool
        The Pool operation instance
    node : Apply
        The apply node containing inputs and outputs
    var_names : dict
        Mapping of variables to ONNX names
    get_var_name : callable
        Function to get ONNX name for a variable

    Returns
    -------
    onnx.NodeProto
        ONNX MaxPool node

    Notes
    -----
    ONNX MaxPool operator:
    - Inputs: X (4D tensor in NCHW format)
    - Attributes:
      - kernel_shape (required): [pool_h, pool_w]
      - strides (optional): [stride_h, stride_w]
      - pads (optional): [pad_top, pad_left, pad_bottom, pad_right]
    - Outputs: Y (pooled tensor)

    PyTensor Pool op stores:
    - op.ws: window size (kernel_shape)
    - op.stride: stride for pooling
    - op.padding: (pad_h, pad_w) -> ONNX uses [pad_h, pad_w, pad_h, pad_w]
    - op.mode: 'max' or 'average'
    """
    if op.mode != "max":
        raise NotImplementedError(
            f"Only max pooling is supported for ONNX export, got: {op.mode}"
        )

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Extract pooling parameters
    kernel_shape = list(op.ws)
    strides = list(op.stride)

    # ONNX pads format: [pad_top, pad_left, pad_bottom, pad_right]
    # PyTensor padding: (pad_h, pad_w) - same padding on both sides
    pad_h, pad_w = op.padding
    pads = [pad_h, pad_w, pad_h, pad_w]

    # Build attributes
    attributes = {
        "kernel_shape": kernel_shape,
        "strides": strides,  # ONNX defaults to [1,1], so always include
    }

    # Add pads if non-zero
    if any(p > 0 for p in pads):
        attributes["pads"] = pads

    # Create ONNX MaxPool node
    return helper.make_node(
        "MaxPool",
        inputs=input_names,
        outputs=output_names,
        name=f"MaxPool_{output_names[0]}",
        **attributes,
    )
