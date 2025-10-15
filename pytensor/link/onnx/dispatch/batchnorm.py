"""ONNX conversion for batch normalization operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.batchnorm import BatchNormalization


try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


@onnx_funcify.register(BatchNormalization)
def onnx_funcify_BatchNormalization(op, node, var_names, get_var_name, **kwargs):
    """
    Convert BatchNormalization op to ONNX BatchNormalization node.

    ONNX BatchNormalization parameters:
    - X: input tensor
    - scale: gamma (scale parameter)
    - B: beta (bias parameter)
    - input_mean: mean
    - input_var: variance
    - epsilon: numerical stability constant

    Parameters
    ----------
    op : BatchNormalization
        The PyTensor BatchNormalization op
    node : Apply
        The Apply node
    var_names : dict
        Mapping from variables to ONNX names
    get_var_name : callable
        Function to get/create variable names
    **kwargs
        Additional parameters

    Returns
    -------
    onnx.NodeProto
        ONNX BatchNormalization node
    """
    # Get input names
    # node.inputs = [x, gamma, beta, mean, variance]
    x_name = get_var_name(node.inputs[0])
    gamma_name = get_var_name(node.inputs[1])
    beta_name = get_var_name(node.inputs[2])
    mean_name = get_var_name(node.inputs[3])
    variance_name = get_var_name(node.inputs[4])

    # Get output name
    output_name = get_var_name(node.outputs[0])

    # Create ONNX BatchNormalization node
    # ONNX BatchNormalization signature:
    # BatchNormalization(
    #     X, scale, B, input_mean, input_var,
    #     epsilon=1e-5, momentum=0.9, training_mode=0
    # )
    #
    # Inputs:
    #   - X: input tensor (NCHW format for 4D)
    #   - scale: gamma (per-channel scale)
    #   - B: beta (per-channel bias)
    #   - input_mean: running mean (per-channel)
    #   - input_var: running variance (per-channel)
    #
    # Attributes:
    #   - epsilon: numerical stability constant
    #   - training_mode: 0 for inference (we only support inference)
    #
    # Output:
    #   - Y: normalized output (same shape as X)

    onnx_node = helper.make_node(
        "BatchNormalization",
        inputs=[x_name, gamma_name, beta_name, mean_name, variance_name],
        outputs=[output_name],
        epsilon=op.epsilon,
        # training_mode=0 means inference mode (use provided mean/var)
        # This is the default in ONNX opset >= 9
        name=f"BatchNormalization_{output_name}",
    )

    return onnx_node
