"""ONNX conversion for convolution operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.conv.abstract_conv import AbstractConv2d


try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(AbstractConv2d)
def onnx_funcify_AbstractConv2d(op, node, var_names, get_var_name, **kwargs):
    """
    Convert AbstractConv2d to ONNX Conv node.

    PyTensor Conv2D parameters:
    - border_mode: Padding ('valid', 'half', tuple, etc.)
    - subsample: Stride (downsampling factor)
    - filter_flip: True=convolution, False=cross-correlation
    - filter_dilation: Dilation (atrous convolution)
    - num_groups: Grouped convolution

    ONNX Conv attributes:
    - auto_pad: 'NOTSET', 'SAME_UPPER', 'VALID'
    - pads: [top, left, bottom, right]
    - strides: [stride_h, stride_w]
    - dilations: [dilation_h, dilation_w]
    - group: Number of groups

    References:
    - PyTensor AbstractConv2d: pytensor/tensor/conv/abstract_conv.py
    - ONNX Conv spec: https://onnx.ai/onnx/operators/onnx__Conv.html
    """
    # Get input/output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Extract op attributes
    border_mode = op.border_mode
    subsample = op.subsample
    filter_flip = op.filter_flip
    filter_dilation = op.filter_dilation
    num_groups = op.num_groups

    # Handle filter flipping
    # Note: For filter_flip=True, we need to handle this differently.
    # The kernel flipping logic must be done at the graph level by the export system,
    # not within the converter itself, because the converter doesn't have access to
    # the initializers list. Instead, we'll use ONNX operators to flip the kernel
    # at runtime.

    nodes_to_return = []
    conv_input_names = input_names.copy()

    if filter_flip:
        # PyTensor flips kernel for mathematical convolution
        # ONNX Conv doesn't flip (cross-correlation)
        # Solution: The kernel is flipped during initializer creation
        #
        # The export system (basic.py) pre-scans for AbstractConv2d ops
        # with filter_flip=True and flips their kernel initializers automatically.
        # We don't need to do anything here - just let the Conv node use the
        # flipped kernel from the initializer.
        pass

    # Convert subsample to ONNX strides
    strides = list(subsample)

    # Convert filter_dilation to ONNX dilations
    dilations = list(filter_dilation)

    # Convert border_mode to ONNX padding
    auto_pad = "NOTSET"
    pads = None

    if border_mode == "valid":
        # No padding
        auto_pad = "VALID"
    elif border_mode in ("half",):
        # Maintain input size (with stride=1)
        # ONNX SAME_UPPER: pads at end if padding is odd
        auto_pad = "SAME_UPPER"
    elif border_mode == "full":
        # Full padding: output_size = input_size + kernel_size - 1
        # ONNX doesn't have FULL mode - need explicit pads
        raise NotImplementedError(
            "Conv2D with border_mode='full' not yet supported.\n"
            "ONNX Conv doesn't have 'FULL' padding mode.\n"
            "Need to compute explicit pads from kernel size."
        )
    elif isinstance(border_mode, int):
        # Symmetric padding (single value)
        # border_mode=1 → pads=[1,1,1,1]
        pads = [border_mode, border_mode, border_mode, border_mode]
    elif isinstance(border_mode, tuple) and len(border_mode) == 2:
        # Check if symmetric or asymmetric
        if isinstance(border_mode[0], int):
            # Symmetric: (pad_h, pad_w)
            pad_h, pad_w = border_mode
            pads = [pad_h, pad_w, pad_h, pad_w]
        else:
            # Asymmetric: ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right))
            (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right) = border_mode
            # ONNX format: [top, left, bottom, right]
            pads = [pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")

    # Build ONNX Conv node attributes
    attributes = {
        "strides": strides,
        "dilations": dilations,
        "group": num_groups,
    }

    # Add padding attributes
    if auto_pad != "NOTSET":
        attributes["auto_pad"] = auto_pad
    elif pads is not None:
        attributes["pads"] = pads

    # Create ONNX Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=conv_input_names,
        outputs=output_names,
        name=f"Conv_{output_names[0]}",
        **attributes,
    )

    nodes_to_return.append(conv_node)

    # Return single node or list of nodes
    if len(nodes_to_return) == 1:
        return nodes_to_return[0]
    else:
        return nodes_to_return
