"""ONNX export API for PyTensor."""

from pathlib import Path


try:
    import onnx
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install pytensor[onnx]"
    ) from e

from pytensor.compile.function.types import Function
from pytensor.link.onnx.dispatch.basic import onnx_funcify


def export_onnx(
    pytensor_function: Function,
    output_path: str | Path,
    *,
    opset_version: int = 18,
    model_name: str = "pytensor_model",
    **kwargs,
) -> onnx.ModelProto:
    """Export a PyTensor function to ONNX format.

    Parameters
    ----------
    pytensor_function : Function
        Compiled PyTensor function to export
    output_path : str or Path
        Path where the .onnx file will be saved
    opset_version : int, optional
        ONNX opset version to target (default: 18)
    model_name : str, optional
        Name for the ONNX model (default: "pytensor_model")
    **kwargs
        Additional parameters passed to onnx_funcify

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> from pytensor.link.onnx import export_onnx
    >>>
    >>> # Create function
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> z = x + y * 2
    >>> f = pytensor.function([x, y], z)
    >>>
    >>> # Export to ONNX
    >>> model = export_onnx(f, "model.onnx")
    >>>
    >>> # Load in ONNX Runtime
    >>> import onnxruntime as ort
    >>> session = ort.InferenceSession("model.onnx")
    >>> result = session.run(None, {"x": [1, 2, 3], "y": [4, 5, 6]})

    Troubleshooting
    ---------------
    **ImportError: No module named 'onnx'**
    Install ONNX: `pip install pytensor[onnx]`

    **NotImplementedError: No ONNX conversion available for: <OpName>**
    The operation is not yet supported. Check the list of supported ops in the
    error message or PyTensor documentation.

    **ValueError: Generated ONNX model is invalid**
    The generated ONNX graph failed validation. This is likely a bug in the
    ONNX backend. Please report it with a minimal reproducible example.

    **Shape mismatch in ONNX Runtime**
    Ensure input shapes match what the model expects. ONNX models have specific
    shape requirements that may differ from PyTensor's dynamic shapes.
    """
    # Get the FunctionGraph from the compiled function
    fgraph = pytensor_function.maker.fgraph

    # Identify which inputs are user inputs vs shared variables
    # User inputs have inp.value = None, shared variables have inp.value set
    user_inputs = set()
    for inp in pytensor_function.maker.inputs:
        if inp.value is None:
            # This is an explicit input
            user_inputs.add(inp.variable)

    # Convert to ONNX
    model = onnx_funcify(
        fgraph,
        opset_version=opset_version,
        model_name=model_name,
        user_inputs=user_inputs,
        **kwargs,
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return model
