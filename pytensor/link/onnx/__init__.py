"""ONNX export functionality for PyTensor.

This module provides functionality to export PyTensor functions to ONNX format
for deployment in environments like WebAssembly, mobile, or edge devices.

Example
-------
>>> import pytensor
>>> import pytensor.tensor as pt
>>> from pytensor.link.onnx import export_onnx
>>>
>>> # Create and compile function
>>> x = pt.vector("x")
>>> y = pt.vector("y")
>>> z = x + y * 2
>>> f = pytensor.function([x, y], z)
>>>
>>> # Export to ONNX
>>> export_onnx(f, "model.onnx")
"""

from pytensor.link.onnx.export import export_onnx


__all__ = ["export_onnx"]
