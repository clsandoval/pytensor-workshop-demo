"""ONNX conversion for elementwise operations.

This module handles conversion of PyTensor elementwise operations to ONNX nodes.
Elementwise operations perform the same operation on each element of the input
tensor(s), supporting NumPy-style broadcasting.

Comparison Operations
---------------------
Comparison operations (EQ, NE, GT, LT, GE, LE) map directly to ONNX operators:
- EQ → Equal: Element-wise equality, output dtype is bool
- Output is always bool regardless of input dtype
- Supports: float32, float64, int32, int64, bool

Usage in YOLO
-------------
EQ is used for shape validation in dynamic upsampling:
- Compare shape dimensions: Eq(Shape_i{0}.0, Shape_i{0}.0)
- Check for -1 (dynamic dimension): Eq(dim, -1)
- Results used in Switch for conditional logic
"""

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
    # Binary arithmetic operations
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.IntDiv: "Div",  # Floor division (handled specially with Floor node)
    # Unary arithmetic operations
    scalar.Neg: "Neg",
    scalar.Abs: "Abs",
    scalar.Sqr: "Mul",  # x^2 -> x * x (handled specially in line 156)
    scalar.Pow: "Pow",
    # Math functions
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    # Comparison operations
    scalar.EQ: "Equal",  # Element-wise equality comparison
    # Min/Max operations
    scalar.ScalarMaximum: "Max",
    scalar.ScalarMinimum: "Min",
    # Activation functions
    scalar_math.Sigmoid: "Sigmoid",  # Logistic sigmoid: 1 / (1 + exp(-x))
    # Control flow
    scalar.Switch: "Where",  # Conditional selection: if cond then x else y
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

        # Handle EQ operations specially (ensure inputs have matching dtypes)
        # ONNX Equal operator requires both inputs to have identical dtypes
        if scalar_op_type == scalar.EQ:
            # EQ(x, y) where x and y might have different dtypes
            assert len(input_names) == 2, "EQ should have exactly two inputs"

            # Get input dtypes from the scalar node
            x_dtype = scalar_node.inputs[0].type.dtype
            y_dtype = scalar_node.inputs[1].type.dtype

            # If dtypes match, no casting needed
            if x_dtype == y_dtype:
                nodes.append(
                    helper.make_node(
                        "Equal",
                        inputs=input_names,
                        outputs=[output_name],
                        name=f"Equal_{output_name}",
                    )
                )
            else:
                # Cast both to int64 (the most general integer type)
                # Choose int64 as it can represent all integer values
                x_casted = f"composite_x_casted_{id(output_var)}"
                y_casted = f"composite_y_casted_{id(output_var)}"

                # 1. Cast x to int64 (TensorProto.INT64 = 7)
                nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[input_names[0]],
                        outputs=[x_casted],
                        to=7,  # TensorProto.INT64
                        name=f"Cast_x_{output_name}",
                    )
                )

                # 2. Cast y to int64 (TensorProto.INT64 = 7)
                nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[input_names[1]],
                        outputs=[y_casted],
                        to=7,  # TensorProto.INT64
                        name=f"Cast_y_{output_name}",
                    )
                )

                # 3. Equal(x_casted, y_casted)
                nodes.append(
                    helper.make_node(
                        "Equal",
                        inputs=[x_casted, y_casted],
                        outputs=[output_name],
                        name=f"Equal_{output_name}",
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

        # Handle IntDiv operations specially (decompose into Cast + Div + Floor + Cast)
        if scalar_op_type == scalar.IntDiv:
            # IntDiv(x, y) = cast_int(floor(div(cast_float(x), cast_float(y))))
            # ONNX Floor only works with float types, so we cast to float first
            assert len(input_names) == 2, "IntDiv should have exactly two inputs"

            # Determine output dtype from the scalar node
            output_dtype = scalar_node.outputs[0].type.dtype
            dtype_map = {
                "int32": 6,  # TensorProto.INT32
                "int64": 7,  # TensorProto.INT64
                "int8": 3,  # TensorProto.INT8
                "uint8": 2,  # TensorProto.UINT8
            }
            output_onnx_dtype = dtype_map.get(output_dtype, 7)  # Default to INT64

            # Create intermediate variable names
            x_float = f"composite_x_float_{id(output_var)}"
            y_float = f"composite_y_float_{id(output_var)}"
            div_out = f"composite_div_{id(output_var)}"
            floor_out = f"composite_floor_{id(output_var)}"

            # 1. Cast x to float (TensorProto.FLOAT = 1)
            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[input_names[0]],
                    outputs=[x_float],
                    to=1,  # TensorProto.FLOAT
                    name=f"Cast_x_{output_name}",
                )
            )

            # 2. Cast y to float (TensorProto.FLOAT = 1)
            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[input_names[1]],
                    outputs=[y_float],
                    to=1,  # TensorProto.FLOAT
                    name=f"Cast_y_{output_name}",
                )
            )

            # 3. Div(x_float, y_float)
            nodes.append(
                helper.make_node(
                    "Div",
                    inputs=[x_float, y_float],
                    outputs=[div_out],
                    name=f"Div_{output_name}",
                )
            )

            # 4. Floor(div_out)
            nodes.append(
                helper.make_node(
                    "Floor",
                    inputs=[div_out],
                    outputs=[floor_out],
                    name=f"Floor_{output_name}",
                )
            )

            # 5. Cast back to original integer dtype
            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[floor_out],
                    outputs=[output_name],
                    to=output_onnx_dtype,
                    name=f"IntDiv_{output_name}",
                )
            )
            continue

        # Convert the scalar operation to ONNX
        if scalar_op_type not in SCALAR_OP_TO_ONNX:
            raise NotImplementedError(
                f"Scalar op in Composite not supported: {scalar_op_type.__name__}"
            )

        onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

        # Define which operations are chainable (commutative/associative binary ops)
        chainable_ops = {"Add", "Mul", "And", "Or", "Max", "Min"}

        # ONNX binary operations only accept 2 inputs
        # If we have more and the op is chainable, chain them: Add(Add(a, b), c)
        if len(input_names) > 2 and onnx_op_type in chainable_ops:
            # Start with first two inputs
            intermediate = f"composite_chain_0_{id(output_var)}"
            nodes.append(
                helper.make_node(
                    onnx_op_type,
                    inputs=[input_names[0], input_names[1]],
                    outputs=[intermediate],
                    name=f"{onnx_op_type}_0_{output_name}",
                )
            )

            # Chain remaining inputs
            for i, next_input in enumerate(input_names[2:], start=1):
                prev_intermediate = intermediate
                # Last operation outputs to the final output name
                if i == len(input_names) - 2:
                    intermediate = output_name
                else:
                    intermediate = f"composite_chain_{i}_{id(output_var)}"

                nodes.append(
                    helper.make_node(
                        onnx_op_type,
                        inputs=[prev_intermediate, next_input],
                        outputs=[intermediate],
                        name=f"{onnx_op_type}_{i}_{output_name}",
                    )
                )
        else:
            # Other operations - pass inputs as-is
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

    # Handle EQ operations specially (ensure inputs have matching dtypes)
    # ONNX Equal operator requires both inputs to have identical dtypes
    if scalar_op_type == scalar.EQ:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_names = [get_var_name(out) for out in node.outputs]

        # Get input dtypes
        x_dtype = node.inputs[0].type.dtype
        y_dtype = node.inputs[1].type.dtype

        # If dtypes match, no casting needed
        if x_dtype == y_dtype:
            return helper.make_node(
                "Equal",
                inputs=input_names,
                outputs=output_names,
                name=f"Equal_{output_names[0]}",
            )

        # Cast both to int64 (the most general integer type)
        x_casted = f"x_casted_{output_names[0]}"
        y_casted = f"y_casted_{output_names[0]}"

        # Create three nodes:
        # 1. Cast x to int64 (TensorProto.INT64 = 7)
        cast_x_node = helper.make_node(
            "Cast",
            inputs=[input_names[0]],
            outputs=[x_casted],
            to=7,  # TensorProto.INT64
            name=f"Cast_x_{output_names[0]}",
        )

        # 2. Cast y to int64 (TensorProto.INT64 = 7)
        cast_y_node = helper.make_node(
            "Cast",
            inputs=[input_names[1]],
            outputs=[y_casted],
            to=7,  # TensorProto.INT64
            name=f"Cast_y_{output_names[0]}",
        )

        # 3. Equal(x_casted, y_casted)
        equal_node = helper.make_node(
            "Equal",
            inputs=[x_casted, y_casted],
            outputs=output_names,
            name=f"Equal_{output_names[0]}",
        )

        # Return list of nodes for multi-node decomposition
        return [cast_x_node, cast_y_node, equal_node]

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

    # Handle IntDiv operations specially (decompose into Cast + Div + Floor + Cast)
    # IntDiv(x, y) = cast_int(floor(div(cast_float(x), cast_float(y))))
    # ONNX Floor only works with float types, so we cast to float first
    if scalar_op_type == scalar.IntDiv:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_names = [get_var_name(out) for out in node.outputs]

        # Determine output dtype from the op
        output_dtype = op.scalar_op.output_types([inp.type for inp in node.inputs])[
            0
        ].dtype
        dtype_map = {
            "int32": 6,  # TensorProto.INT32
            "int64": 7,  # TensorProto.INT64
            "int8": 3,  # TensorProto.INT8
            "uint8": 2,  # TensorProto.UINT8
        }
        output_onnx_dtype = dtype_map.get(output_dtype, 7)  # Default to INT64

        # Create intermediate variable names
        x_float = f"x_float_{output_names[0]}"
        y_float = f"y_float_{output_names[0]}"
        div_out = f"div_{output_names[0]}"
        floor_out = f"floor_{output_names[0]}"

        # Create five nodes:
        # 1. Cast x to float (TensorProto.FLOAT = 1)
        cast_x_node = helper.make_node(
            "Cast",
            inputs=[input_names[0]],
            outputs=[x_float],
            to=1,  # TensorProto.FLOAT
            name=f"Cast_x_{output_names[0]}",
        )

        # 2. Cast y to float (TensorProto.FLOAT = 1)
        cast_y_node = helper.make_node(
            "Cast",
            inputs=[input_names[1]],
            outputs=[y_float],
            to=1,  # TensorProto.FLOAT
            name=f"Cast_y_{output_names[0]}",
        )

        # 3. Div(x_float, y_float)
        div_node = helper.make_node(
            "Div",
            inputs=[x_float, y_float],
            outputs=[div_out],
            name=f"Div_{output_names[0]}",
        )

        # 4. Floor(div_out)
        floor_node = helper.make_node(
            "Floor",
            inputs=[div_out],
            outputs=[floor_out],
            name=f"Floor_{output_names[0]}",
        )

        # 5. Cast back to original integer dtype
        cast_node = helper.make_node(
            "Cast",
            inputs=[floor_out],
            outputs=output_names,
            to=output_onnx_dtype,
            name=f"IntDiv_{output_names[0]}",
        )

        # Return list of nodes for multi-node decomposition
        return [cast_x_node, cast_y_node, div_node, floor_node, cast_node]

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

    # Define which operations are chainable (commutative/associative binary ops)
    chainable_ops = {"Add", "Mul", "And", "Or", "Max", "Min"}

    # ONNX binary operations (Add, Mul, etc.) only accept 2 inputs
    # PyTensor can have n-ary operations like Add(a, b, c)
    # We need to chain them: Add(Add(a, b), c)
    if len(input_names) > 2 and onnx_op_type in chainable_ops:
        nodes = []
        # Start with first two inputs
        intermediate = f"intermediate_0_{output_names[0]}"
        nodes.append(
            helper.make_node(
                onnx_op_type,
                inputs=[input_names[0], input_names[1]],
                outputs=[intermediate],
                name=f"{onnx_op_type}_0_{output_names[0]}",
            )
        )

        # Chain remaining inputs
        for i, next_input in enumerate(input_names[2:], start=1):
            prev_intermediate = intermediate
            # Last operation outputs to the final output name
            if i == len(input_names) - 2:
                intermediate = output_names[0]
            else:
                intermediate = f"intermediate_{i}_{output_names[0]}"

            nodes.append(
                helper.make_node(
                    onnx_op_type,
                    inputs=[prev_intermediate, next_input],
                    outputs=[intermediate],
                    name=f"{onnx_op_type}_{i}_{output_names[0]}",
                )
            )

        return nodes
    else:
        # Other operations - direct mapping
        onnx_node = helper.make_node(
            onnx_op_type,
            inputs=input_names,
            outputs=output_names,
            name=f"{onnx_op_type}_{output_names[0]}",
        )

        return onnx_node
