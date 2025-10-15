"""ONNX conversion for resize operations."""

import numpy as np
from onnx import helper, numpy_helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.resize import Resize


@onnx_funcify.register(Resize)
def onnx_funcify_Resize(op, node, var_names, get_var_name, **kwargs):
    """
    Convert PyTensor Resize op to ONNX Resize node.

    ONNX Resize operator (opset 18):
    - Inputs:
      1. X: Input tensor
      2. roi: Region of interest (optional, we don't use)
      3. scales: Scale factors (what we use)
      4. sizes: Output sizes (alternative to scales, we don't use)
    - Attributes:
      - mode: "nearest" or "linear"
      - coordinate_transformation_mode: How to map coordinates
      - nearest_mode: Rounding mode for nearest neighbor

    Notes
    -----
    Nearest neighbor mode matches PyTensor's implementation exactly.
    Bilinear mode has known algorithmic differences between scipy.ndimage.zoom
    (used in PyTensor) and ONNX Resize, resulting in numerical differences.
    See tests/link/onnx/test_resize.py::test_resize_onnx_bilinear for details.

    Parameters
    ----------
    op : Resize
        The Resize operation instance
    node : Apply
        The apply node
    var_names : dict
        Variable name mapping
    get_var_name : callable
        Name generator

    Returns
    -------
    list of onnx.NodeProto
        ONNX nodes (Resize requires Constant nodes for scales)
    """
    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    input_name = input_names[0]
    output_name = output_names[0]

    # Map PyTensor mode to ONNX mode
    mode_mapping = {
        "nearest": "nearest",
        "linear": "linear",  # ONNX 'linear' = bilinear for 2D
    }

    onnx_mode = mode_mapping.get(op.mode)
    if onnx_mode is None:
        raise ValueError(f"Unsupported resize mode: {op.mode}")

    # ONNX Resize requires scales as a Constant input
    # scales format: [batch_scale, channel_scale, height_scale, width_scale]
    # We don't scale batch or channels, only spatial dimensions
    scale_h, scale_w = op.scale_factor
    scales = np.array([1.0, 1.0, scale_h, scale_w], dtype=np.float32)

    # Create Constant node for scales
    scales_name = f"scales_{output_name}"
    scales_tensor = numpy_helper.from_array(scales, name=scales_name)

    nodes = []

    # Constant node for scales
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[scales_name],
            value=scales_tensor,
            name=f"Const_{scales_name}",
        )
    )

    # ONNX Resize node
    # Inputs: X, roi (empty), scales
    # We create an empty roi tensor since we don't use it
    roi_name = f"roi_{output_name}"
    roi_tensor = numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name)

    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[roi_name],
            value=roi_tensor,
            name=f"Const_{roi_name}",
        )
    )

    # Build Resize node attributes
    # Use different coordinate transformation modes for different interpolation types
    if onnx_mode == "nearest":
        coord_transform_mode = "asymmetric"  # Matches floor-based nearest neighbor
        resize_attrs = {
            "mode": onnx_mode,
            "coordinate_transformation_mode": coord_transform_mode,
            "nearest_mode": "floor",
        }
    else:  # linear mode
        coord_transform_mode = "half_pixel"  # Better matches scipy.ndimage.zoom
        resize_attrs = {
            "mode": onnx_mode,
            "coordinate_transformation_mode": coord_transform_mode,
        }

    # Create Resize node
    nodes.append(
        helper.make_node(
            "Resize",
            inputs=[input_name, roi_name, scales_name],
            outputs=[output_name],
            name=f"Resize_{output_name}",
            **resize_attrs,
        )
    )

    return nodes
