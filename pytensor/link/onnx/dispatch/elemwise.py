"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.scalar import basic as scalar
from pytensor.scalar import math as scalar_math
from pytensor.tensor.elemwise import Elemwise


try:
    import numpy as np
    from onnx import helper, numpy_helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Sqr: "Mul",  # x^2 -> x * x (handled specially)
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
    scalar.ScalarMaximum: "Max",  # for ReLU pattern
    scalar.ScalarMinimum: "Min",
    scalar_math.Sigmoid: "Sigmoid",  # Logistic sigmoid activation (1 / (1 + exp(-x)))
}


def decompose_composite_elemwise(op, node, var_names, get_var_name, **kwargs):
    """Decompose a Composite scalar op into individual ONNX nodes.

    Composite ops contain a subgraph of scalar operations that have been
    fused together by PyTensor's optimizer. We need to unfold this subgraph
    and create individual ONNX nodes for each operation.
    """
    from pytensor.scalar.basic import ScalarConstant

    composite_op = op.scalar_op
    nodes = []

    # Map from composite's scalar variables to ONNX variable names
    scalar_to_onnx = {}

    # Map tensor inputs to their corresponding scalar inputs in the composite
    tensor_inputs = node.inputs
    scalar_inputs = composite_op.inputs
    for tensor_inp, scalar_inp in zip(tensor_inputs, scalar_inputs):
        scalar_to_onnx[scalar_inp] = get_var_name(tensor_inp)

    # Traverse the composite's graph in topological order
    from pytensor.graph.traversal import io_toposort

    # Get all nodes in the composite's graph
    composite_nodes = io_toposort(composite_op.inputs, composite_op.outputs)

    for scalar_node in composite_nodes:
        scalar_op_type = type(scalar_node.op)

        # Handle constants
        for inp in scalar_node.inputs:
            if isinstance(inp, ScalarConstant) and inp not in scalar_to_onnx:
                # Create ONNX constant
                const_name = f"const_{id(inp)}"
                const_value = np.array(inp.data, dtype=inp.type.dtype)
                const_tensor = numpy_helper.from_array(const_value, name="")

                nodes.append(
                    helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[const_name],
                        value=const_tensor,
                        name=const_name,
                    )
                )
                scalar_to_onnx[inp] = const_name

        # Get input names for this operation
        input_names = [scalar_to_onnx[inp] for inp in scalar_node.inputs]

        # Generate output name
        output_var = scalar_node.outputs[0]
        if output_var in composite_op.outputs:
            # This is a final output - use the tensor output name
            output_idx = composite_op.outputs.index(output_var)
            output_name = get_var_name(node.outputs[output_idx])
        else:
            # Intermediate result
            output_name = f"composite_tmp_{id(output_var)}"

        scalar_to_onnx[output_var] = output_name

        # Handle Cast operations specially (need dtype attribute)
        if scalar_op_type == scalar.Cast:
            # Map PyTensor dtype to ONNX TensorProto type
            dtype_map = {
                "float32": 1,  # TensorProto.FLOAT
                "float64": 11,  # TensorProto.DOUBLE
                "int32": 6,  # TensorProto.INT32
                "int64": 7,  # TensorProto.INT64
                "int8": 3,  # TensorProto.INT8
                "uint8": 2,  # TensorProto.UINT8
                "bool": 9,  # TensorProto.BOOL
            }

            target_dtype = scalar_node.outputs[0].type.dtype
            onnx_dtype = dtype_map.get(target_dtype)

            if onnx_dtype is None:
                raise NotImplementedError(
                    f"Cast to dtype {target_dtype} not supported for ONNX export"
                )

            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=input_names,
                    outputs=[output_name],
                    to=onnx_dtype,
                    name=f"Cast_{output_name}",
                )
            )
            continue

        # Handle Sqr operations specially (x^2 -> x * x)
        if scalar_op_type == scalar.Sqr:
            # Sqr has only one input, but Mul needs two (both the same)
            assert len(input_names) == 1, "Sqr should have exactly one input"
            nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[input_names[0], input_names[0]],  # x * x
                    outputs=[output_name],
                    name=f"Sqr_{output_name}",
                )
            )
            continue

        # Handle SiLU operations specially (decompose into Sigmoid + Mul)
        if scalar_op_type == scalar_math.SiLU:
            # SiLU(x) = x * sigmoid(x)
            assert len(input_names) == 1, "SiLU should have exactly one input"

            # Create intermediate sigmoid output name
            sigmoid_out = f"composite_sigmoid_{id(output_var)}"

            # 1. Sigmoid(x)
            nodes.append(
                helper.make_node(
                    "Sigmoid",
                    inputs=input_names,
                    outputs=[sigmoid_out],
                    name=f"Sigmoid_{output_name}",
                )
            )

            # 2. Mul(x, sigmoid(x))
            nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[input_names[0], sigmoid_out],
                    outputs=[output_name],
                    name=f"SiLU_{output_name}",
                )
            )
            continue

        # Convert the scalar operation to ONNX
        if scalar_op_type not in SCALAR_OP_TO_ONNX:
            raise NotImplementedError(
                f"Scalar op in Composite not supported: {scalar_op_type.__name__}"
            )

        onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

        # Create ONNX node
        nodes.append(
            helper.make_node(
                onnx_op_type,
                inputs=input_names,
                outputs=[output_name],
                name=f"{onnx_op_type}_{output_name}",
            )
        )

    return nodes


@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node.

    Elemwise ops perform element-wise operations on tensors.
    They map directly to ONNX ops like Add, Mul, etc.
    """
    scalar_op_type = type(op.scalar_op)

    # Handle Composite scalar ops by decomposing them
    if scalar_op_type == scalar.Composite:
        return decompose_composite_elemwise(op, node, var_names, get_var_name, **kwargs)

    # Handle SiLU operations specially (decompose into Sigmoid + Mul)
    # SiLU(x) = x * sigmoid(x), requires multi-node decomposition since ONNX has no native SiLU
    if scalar_op_type == scalar_math.SiLU:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_names = [get_var_name(out) for out in node.outputs]

        # Create intermediate variable name for sigmoid output
        sigmoid_out = f"sigmoid_{output_names[0]}"

        # Create two nodes:
        # 1. Sigmoid(x)
        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=input_names,
            outputs=[sigmoid_out],
            name=f"Sigmoid_{output_names[0]}",
        )

        # 2. Mul(x, sigmoid(x))
        mul_node = helper.make_node(
            "Mul",
            inputs=[input_names[0], sigmoid_out],
            outputs=output_names,
            name=f"SiLU_{output_names[0]}",
        )

        # Return list of nodes for multi-node decomposition
        return [sigmoid_node, mul_node]

    # Handle Cast operations specially
    if scalar_op_type == scalar.Cast:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_names = [get_var_name(out) for out in node.outputs]

        # Map PyTensor dtype to ONNX TensorProto type
        dtype_map = {
            "float32": 1,  # TensorProto.FLOAT
            "float64": 11,  # TensorProto.DOUBLE
            "int32": 6,  # TensorProto.INT32
            "int64": 7,  # TensorProto.INT64
            "int8": 3,  # TensorProto.INT8
            "uint8": 2,  # TensorProto.UINT8
            "bool": 9,  # TensorProto.BOOL
        }

        target_dtype = op.scalar_op.o_type.dtype
        onnx_dtype = dtype_map.get(target_dtype)

        if onnx_dtype is None:
            raise NotImplementedError(
                f"Cast to dtype {target_dtype} not supported for ONNX export"
            )

        return helper.make_node(
            "Cast",
            inputs=input_names,
            outputs=output_names,
            to=onnx_dtype,
            name=f"Cast_{output_names[0]}",
        )

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(
            f"Elemwise scalar op not supported for ONNX export: {scalar_op_type.__name__}\n"
            f"Supported scalar ops: {', '.join(op.__name__ for op in SCALAR_OP_TO_ONNX.keys())}"
        )

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX node
    onnx_node = helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )

    return onnx_node
