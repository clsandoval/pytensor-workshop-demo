"""ONNX conversion for linear algebra operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.blas import Dot22, Gemv
from pytensor.tensor.math import Dot


try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(Dot)
def onnx_funcify_Dot(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot to ONNX MatMul node.

    PyTensor's Dot operation maps to ONNX MatMul for matrix multiplication.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    onnx_node = helper.make_node(
        "MatMul",
        inputs=input_names,
        outputs=output_names,
        name=f"MatMul_{output_names[0]}",
    )

    return onnx_node


@onnx_funcify.register(Dot22)
def onnx_funcify_Dot22(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot22 (optimized 2x2 dot) to ONNX MatMul node."""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    onnx_node = helper.make_node(
        "MatMul",
        inputs=input_names,
        outputs=output_names,
        name=f"MatMul_{output_names[0]}",
    )

    return onnx_node


@onnx_funcify.register(Gemv)
def onnx_funcify_Gemv(op, node, var_names, get_var_name, **kwargs):
    """Convert Gemv (General Matrix-Vector multiplication) to ONNX operations.

    Gemv computes: y = alpha * A @ x + beta * y
    We need to decompose this into ONNX operations.
    """
    # Gemv nodes have inputs: [y, alpha, A, x, beta]
    y_in, alpha, A, x, beta = node.inputs
    output_names = [get_var_name(out) for out in node.outputs]

    # Get names for inputs
    y_name = get_var_name(y_in)
    alpha_name = get_var_name(alpha)
    A_name = get_var_name(A)
    x_name = get_var_name(x)
    beta_name = get_var_name(beta)

    # Create intermediate variable names
    matmul_result = f"gemv_matmul_{output_names[0]}"
    scaled_matmul = f"gemv_scaled_matmul_{output_names[0]}"
    scaled_y = f"gemv_scaled_y_{output_names[0]}"

    nodes = []

    # Step 1: A @ x
    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=[A_name, x_name],
            outputs=[matmul_result],
            name=matmul_result,
        )
    )

    # Step 2: alpha * (A @ x)
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=[alpha_name, matmul_result],
            outputs=[scaled_matmul],
            name=scaled_matmul,
        )
    )

    # Step 3: beta * y
    nodes.append(
        helper.make_node(
            "Mul", inputs=[beta_name, y_name], outputs=[scaled_y], name=scaled_y
        )
    )

    # Step 4: alpha * (A @ x) + beta * y
    nodes.append(
        helper.make_node(
            "Add",
            inputs=[scaled_matmul, scaled_y],
            outputs=output_names,
            name=f"gemv_add_{output_names[0]}",
        )
    )

    # Return all nodes (we'll need to handle multiple nodes in the FunctionGraph converter)
    # For now, return the first node and let the caller handle the rest
    return nodes
