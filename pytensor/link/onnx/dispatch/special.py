"""ONNX conversion for special functions and activations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.special import Softmax


try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Softmax)
def onnx_funcify_Softmax(op, node, var_names, get_var_name, **kwargs):
    """Convert Softmax to ONNX Softmax node.

    Note: PyTensor's softmax with axis=None flattens the input and applies
    softmax over all elements. ONNX doesn't have this behavior built-in,
    so we need to implement it with Flatten→Softmax→Reshape.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Get axis attribute
    axis = getattr(op, "axis", -1)

    # If axis=None, we need to flatten, apply softmax, then reshape
    if axis is None:
        # Create intermediate names
        flattened = f"softmax_flattened_{output_names[0]}"
        softmax_out = f"softmax_1d_{output_names[0]}"

        nodes = []

        # Step 1: Flatten to 1D
        nodes.append(
            helper.make_node(
                "Flatten",
                inputs=input_names,
                outputs=[flattened],
                axis=0,  # Flatten everything
                name=f"Flatten_{output_names[0]}",
            )
        )

        # Step 2: Apply softmax on flattened (1D) tensor
        nodes.append(
            helper.make_node(
                "Softmax",
                inputs=[flattened],
                outputs=[softmax_out],
                axis=-1,  # Last axis for 1D is axis 0
                name=f"Softmax1D_{output_names[0]}",
            )
        )

        # Step 3: Reshape back to original shape
        # We need to get the shape of the original input
        # This requires Shape + Reshape nodes
        shape_name = f"softmax_shape_{output_names[0]}"
        nodes.append(
            helper.make_node(
                "Shape",
                inputs=input_names,
                outputs=[shape_name],
                name=f"Shape_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[softmax_out, shape_name],
                outputs=output_names,
                name=f"Reshape_{output_names[0]}",
            )
        )

        return nodes
    else:
        # Regular softmax with specific axis
        onnx_node = helper.make_node(
            "Softmax",
            inputs=input_names,
            outputs=output_names,
            axis=axis,
            name=f"Softmax_{output_names[0]}",
        )
        return onnx_node
