"""ONNX conversion for shape operations."""

import numpy as np

from pytensor.compile.ops import DeepCopyOp
from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.basic import AllocEmpty, MakeVector
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.shape import Reshape, Shape_i


try:
    from onnx import helper, numpy_helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Shape_i)
def onnx_funcify_Shape_i(op, node, var_names, get_var_name, **kwargs):
    """Convert Shape_i to ONNX Shape + Gather nodes.

    Shape_i extracts a specific dimension from a tensor's shape.
    In ONNX: Shape(tensor) -> [d0, d1, ...] then Gather to get dimension i.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Get which dimension to extract
    dim_index = op.i

    # Create intermediate names
    shape_output = f"shape_{output_names[0]}"
    indices_const = f"indices_{output_names[0]}"

    nodes = []

    # Step 1: Get the shape of the input tensor
    nodes.append(
        helper.make_node(
            "Shape",
            inputs=input_names,
            outputs=[shape_output],
            name=f"Shape_{output_names[0]}",
        )
    )

    # Step 2: Create a Constant node for the index (ONNX opset 9+)
    indices_tensor = numpy_helper.from_array(
        np.array([dim_index], dtype=np.int64), name=""
    )
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[indices_const],
            value=indices_tensor,
            name=f"Const_{output_names[0]}",
        )
    )

    # Step 3: Gather the specific dimension
    gathered_output = f"gathered_{output_names[0]}"
    nodes.append(
        helper.make_node(
            "Gather",
            inputs=[shape_output, indices_const],
            outputs=[gathered_output],
            axis=0,
            name=f"Gather_{output_names[0]}",
        )
    )

    # Step 4: Squeeze to get a scalar (Gather outputs [1], we want scalar)
    axes_const = f"squeeze_axes_{output_names[0]}"
    axes_tensor = numpy_helper.from_array(np.array([0], dtype=np.int64), name="")
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[axes_const],
            value=axes_tensor,
            name=f"SqueezeAxesConst_{output_names[0]}",
        )
    )

    nodes.append(
        helper.make_node(
            "Squeeze",
            inputs=[gathered_output, axes_const],
            outputs=output_names,
            name=f"Squeeze_{output_names[0]}",
        )
    )

    return nodes


@onnx_funcify.register(Reshape)
def onnx_funcify_Reshape(op, node, var_names, get_var_name, **kwargs):
    """Convert Reshape to ONNX Reshape node."""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Reshape in ONNX takes input and shape as separate inputs
    # PyTensor's Reshape can have the shape as a second input
    onnx_node = helper.make_node(
        "Reshape",
        inputs=input_names,  # [data, shape]
        outputs=output_names,
        name=f"Reshape_{output_names[0]}",
    )

    return onnx_node


def decompose_dimshuffle_pattern(new_order, input_ndim):
    """Decompose DimShuffle into Squeeze, Transpose, Unsqueeze operations.

    Parameters
    ----------
    new_order : tuple
        DimShuffle pattern (e.g., (1, 'x', 0) or (2, 0))
    input_ndim : int
        Number of dimensions in input tensor

    Returns
    -------
    dict
        Dictionary with keys:
        - 'squeeze_axes': list of int - axes to remove (or None)
        - 'transpose_perm': list of int - permutation for transpose (or None)
        - 'unsqueeze_axes': list of int - axes to add (or None)

    Notes
    -----
    Follows PyTensor's DimShuffle.perform() decomposition:
    1. Squeeze: Remove dropped dimensions
    2. Transpose: Reorder kept dimensions
    3. Unsqueeze: Add new dimensions

    Examples
    --------
    >>> decompose_dimshuffle_pattern((1, "x", 0), input_ndim=2)
    {'squeeze_axes': None, 'transpose_perm': [1, 0], 'unsqueeze_axes': [1]}

    >>> decompose_dimshuffle_pattern((2, 0), input_ndim=3)  # (A,1,C) -> (C,A)
    {'squeeze_axes': [1], 'transpose_perm': [1, 0], 'unsqueeze_axes': None}
    """
    # Extract non-'x' dimensions (kept dimensions)
    non_x_dims = [d for d in new_order if d != "x"]

    # Find axes to add ('x' positions in new_order)
    axes_to_add = [i for i, d in enumerate(new_order) if d == "x"]

    # Find axes to drop (input dims not in non_x_dims)
    all_input_dims = set(range(input_ndim))
    kept_dims = set(non_x_dims)
    dropped_dims = sorted(all_input_dims - kept_dims)

    # Check if transpose is needed (non_x_dims not in sorted order)
    needs_transpose = non_x_dims != sorted(non_x_dims)

    # Build result
    result = {
        "squeeze_axes": dropped_dims if dropped_dims else None,
        "transpose_perm": non_x_dims if needs_transpose else None,
        "unsqueeze_axes": axes_to_add if axes_to_add else None,
    }

    # CRITICAL: Adjust transpose permutation after squeeze
    # After squeezing, dimension indices shift down
    if result["squeeze_axes"] and result["transpose_perm"]:
        # Create mapping from original dims to post-squeeze dims
        dim_mapping = {}
        new_idx = 0
        for old_idx in range(input_ndim):
            if old_idx not in result["squeeze_axes"]:
                dim_mapping[old_idx] = new_idx
                new_idx += 1

        # Remap transpose permutation
        result["transpose_perm"] = [
            dim_mapping[old_dim] for old_dim in result["transpose_perm"]
        ]

    return result


@onnx_funcify.register(DimShuffle)
def onnx_funcify_DimShuffle(op, node, var_names, get_var_name, **kwargs):
    """Convert DimShuffle to ONNX Unsqueeze/Squeeze/Transpose nodes.

    DimShuffle reorders and adds/removes dimensions:
    - 'x' in new_order: add dimension (Unsqueeze)
    - Missing dims: remove dimension (Squeeze)
    - Reordered dims: transpose (Transpose)

    Examples:
    - ('x', 0): adds dimension at position 0
    - (1,): removes dimension 0 from 2D input, keeps dimension 1
    - (1, 0): swaps dimensions of 2D input
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    new_order = op.new_order
    input_ndim = node.inputs[0].type.ndim

    # Extract which dimensions are kept and where new dims are added
    axes_to_add = [i for i, dim in enumerate(new_order) if dim == "x"]
    non_x_dims = [dim for dim in new_order if dim != "x"]

    # Case 1: Only adding dimensions (unsqueeze)
    if len(non_x_dims) == input_ndim and non_x_dims == list(range(input_ndim)):
        # No reordering, just adding dimensions
        if axes_to_add:
            axes_name = f"unsqueeze_axes_{output_names[0]}"
            axes_tensor = numpy_helper.from_array(
                np.array(axes_to_add, dtype=np.int64), name=""
            )

            nodes = []
            nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    value=axes_tensor,
                    name=f"ConstAxes_{output_names[0]}",
                )
            )

            nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[input_names[0], axes_name],
                    outputs=[output_names[0]],
                    name=f"Unsqueeze_{output_names[0]}",
                )
            )
            return nodes
        else:
            # Identity - no dims added or removed
            return helper.make_node(
                "Identity",
                inputs=input_names,
                outputs=output_names,
                name=f"Identity_{output_names[0]}",
            )

    # Case 2: Removing dimensions (squeeze)
    if len(non_x_dims) < input_ndim and not axes_to_add:
        # Find which axes to remove
        all_dims = set(range(input_ndim))
        kept_dims = set(non_x_dims)
        axes_to_remove = sorted(all_dims - kept_dims)

        # Check if dims are in order (no transpose needed)
        if non_x_dims == sorted(non_x_dims):
            axes_name = f"squeeze_axes_{output_names[0]}"
            axes_tensor = numpy_helper.from_array(
                np.array(axes_to_remove, dtype=np.int64), name=""
            )

            nodes = []
            nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    value=axes_tensor,
                    name=f"ConstAxes_{output_names[0]}",
                )
            )

            nodes.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[input_names[0], axes_name],
                    outputs=[output_names[0]],
                    name=f"Squeeze_{output_names[0]}",
                )
            )
            return nodes

    # Case 3: Transpose (reordering dimensions, no squeeze/unsqueeze)
    if (
        len(non_x_dims) == input_ndim
        and non_x_dims != list(range(input_ndim))
        and not axes_to_add
    ):
        return helper.make_node(
            "Transpose",
            inputs=input_names,
            outputs=output_names,
            perm=non_x_dims,
            name=f"Transpose_{output_names[0]}",
        )

    # Complex case: combination of operations
    # Decompose into Squeeze → Transpose → Unsqueeze sequence
    ops = decompose_dimshuffle_pattern(new_order, input_ndim)
    nodes = []
    current_var = input_names[0]

    # Step 1: Squeeze (if needed)
    if ops["squeeze_axes"]:
        squeeze_output = f"dimshuffle_squeeze_{output_names[0]}"
        axes_name = f"squeeze_axes_{output_names[0]}"
        axes_tensor = numpy_helper.from_array(
            np.array(ops["squeeze_axes"], dtype=np.int64), name=""
        )

        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"SqueezeAxesConst_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=[current_var, axes_name],
                outputs=[squeeze_output],
                name=f"Squeeze_{output_names[0]}",
            )
        )
        current_var = squeeze_output

    # Step 2: Transpose (if needed)
    if ops["transpose_perm"]:
        transpose_output = f"dimshuffle_transpose_{output_names[0]}"

        nodes.append(
            helper.make_node(
                "Transpose",
                inputs=[current_var],
                outputs=[transpose_output],
                perm=ops["transpose_perm"],
                name=f"Transpose_{output_names[0]}",
            )
        )
        current_var = transpose_output

    # Step 3: Unsqueeze (if needed)
    if ops["unsqueeze_axes"]:
        axes_name = f"unsqueeze_axes_{output_names[0]}"
        axes_tensor = numpy_helper.from_array(
            np.array(ops["unsqueeze_axes"], dtype=np.int64), name=""
        )

        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"UnsqueezeAxesConst_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[current_var, axes_name],
                outputs=output_names,
                name=f"Unsqueeze_{output_names[0]}",
            )
        )
    else:
        # If no unsqueeze, the last operation's output is the final output
        # Need to rename the last node's output
        if nodes:
            nodes[-1].output[0] = output_names[0]
        else:
            # Identity case (shouldn't happen, but handle it)
            nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=[current_var],
                    outputs=output_names,
                    name=f"Identity_{output_names[0]}",
                )
            )

    return nodes


@onnx_funcify.register(AllocEmpty)
def onnx_funcify_AllocEmpty(op, node, var_names, get_var_name, **kwargs):
    """Convert AllocEmpty to ONNX ConstantOfShape.

    AllocEmpty allocates uninitialized memory. For ONNX, we create
    a zero-filled tensor of the requested shape using ConstantOfShape.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # AllocEmpty takes shape dimensions as inputs
    # We need to pack them into a single shape tensor using Concat
    nodes = []

    # Check if inputs are scalars or vectors
    input_ndims = [inp.type.ndim for inp in node.inputs]

    if len(input_names) == 1 and input_ndims[0] == 1:
        # Single 1D input - already a shape vector, just cast to int64
        shape_name = f"shape_{output_names[0]}"
        nodes.append(
            helper.make_node(
                "Cast",
                inputs=[input_names[0]],
                outputs=[shape_name],
                to=7,  # TensorProto.INT64
                name=f"Cast_{output_names[0]}",
            )
        )
    elif len(input_names) == 1 and input_ndims[0] == 0:
        # Single scalar input - need to cast to int64 and make it 1D
        shape_name = f"shape_{output_names[0]}"

        # Cast to int64
        cast_name = f"cast_{output_names[0]}"
        nodes.append(
            helper.make_node(
                "Cast",
                inputs=[input_names[0]],
                outputs=[cast_name],
                to=7,  # TensorProto.INT64
                name=f"Cast_{output_names[0]}",
            )
        )

        # Unsqueeze to make it 1D [value]
        axes_name = f"axes_{output_names[0]}"
        axes_tensor = numpy_helper.from_array(np.array([0], dtype=np.int64), name="")
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"ConstAxes_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[cast_name, axes_name],
                outputs=[shape_name],
                name=f"Unsqueeze_{output_names[0]}",
            )
        )
    else:
        # Multiple dimensions - need to concat into shape vector
        shape_name = f"shape_{output_names[0]}"

        # Each input is a scalar dimension - cast to int64, then unsqueeze to 1D
        unsqueezed_names = []
        for i, dim_name in enumerate(input_names):
            # Cast to int64
            cast_name = f"cast_dim_{i}_{output_names[0]}"
            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[dim_name],
                    outputs=[cast_name],
                    to=7,  # TensorProto.INT64
                    name=f"Cast_{i}_{output_names[0]}",
                )
            )

            unsqueezed = f"unsqueezed_dim_{i}_{output_names[0]}"
            axes_name = f"axes_{i}_{output_names[0]}"

            # Create constant for axes [0]
            axes_tensor = numpy_helper.from_array(
                np.array([0], dtype=np.int64), name=""
            )
            nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    value=axes_tensor,
                    name=f"ConstAxes_{i}_{output_names[0]}",
                )
            )

            # Unsqueeze scalar to 1D
            nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[cast_name, axes_name],
                    outputs=[unsqueezed],
                    name=f"Unsqueeze_{i}_{output_names[0]}",
                )
            )
            unsqueezed_names.append(unsqueezed)

        # Concat all dimensions into shape vector
        nodes.append(
            helper.make_node(
                "Concat",
                inputs=unsqueezed_names,
                outputs=[shape_name],
                axis=0,
                name=f"ConcatShape_{output_names[0]}",
            )
        )

    # Create constant value (0) for ConstantOfShape attribute
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    dtype = dtype_map.get(op.dtype, np.float32)
    value_tensor = numpy_helper.from_array(np.array([0], dtype=dtype), name="")

    # Create tensor of given shape filled with the constant
    nodes.append(
        helper.make_node(
            "ConstantOfShape",
            inputs=[shape_name],
            outputs=output_names,
            value=value_tensor,
            name=f"ConstantOfShape_{output_names[0]}",
        )
    )

    return nodes


@onnx_funcify.register(MakeVector)
def onnx_funcify_MakeVector(op, node, var_names, get_var_name, **kwargs):
    """Convert MakeVector to ONNX Unsqueeze + Concat nodes.

    MakeVector takes multiple scalar inputs and creates a 1D vector from them.
    In ONNX: Unsqueeze each scalar to [value], then Concat along axis 0.

    Examples:
    - MakeVector(a, b, c) with a=1, b=2, c=3 -> [1, 2, 3]
    - Used commonly for creating shape vectors for reshape operations
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    nodes = []

    # If only one input, just unsqueeze it to make it 1D
    if len(input_names) == 1:
        axes_name = f"axes_{output_names[0]}"
        axes_tensor = numpy_helper.from_array(np.array([0], dtype=np.int64), name="")

        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"ConstAxes_{output_names[0]}",
            )
        )

        nodes.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[input_names[0], axes_name],
                outputs=output_names,
                name=f"Unsqueeze_{output_names[0]}",
            )
        )
        return nodes

    # Multiple inputs: unsqueeze each scalar to 1D, then concat
    unsqueezed_names = []
    for i, inp_name in enumerate(input_names):
        unsqueezed = f"unsqueezed_{i}_{output_names[0]}"
        axes_name = f"axes_{i}_{output_names[0]}"

        # Create constant for axes [0]
        axes_tensor = numpy_helper.from_array(np.array([0], dtype=np.int64), name="")
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=axes_tensor,
                name=f"ConstAxes_{i}_{output_names[0]}",
            )
        )

        # Unsqueeze scalar to 1D [value]
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[inp_name, axes_name],
                outputs=[unsqueezed],
                name=f"Unsqueeze_{i}_{output_names[0]}",
            )
        )
        unsqueezed_names.append(unsqueezed)

    # Concat all unsqueezed scalars into a single vector
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=unsqueezed_names,
            outputs=output_names,
            axis=0,
            name=f"Concat_{output_names[0]}",
        )
    )

    return nodes


@onnx_funcify.register(DeepCopyOp)
def onnx_funcify_DeepCopyOp(op, node, var_names, get_var_name, **kwargs):
    """Convert DeepCopyOp to ONNX Identity node.

    DeepCopyOp creates a copy of a tensor. For ONNX export, we don't need
    explicit copies since ONNX manages memory differently, so we use Identity.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    return helper.make_node(
        "Identity",
        inputs=input_names,
        outputs=output_names,
        name=f"Identity_{output_names[0]}",
    )
