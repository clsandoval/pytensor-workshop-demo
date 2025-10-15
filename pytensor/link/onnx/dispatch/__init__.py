"""ONNX dispatch system initialization.

Imports all dispatch modules to trigger @onnx_funcify.register() decorators.
"""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Import dispatch modules to register converters
import pytensor.link.onnx.dispatch.batchnorm
import pytensor.link.onnx.dispatch.conv
import pytensor.link.onnx.dispatch.elemwise
import pytensor.link.onnx.dispatch.join
import pytensor.link.onnx.dispatch.nlinalg
import pytensor.link.onnx.dispatch.pool
import pytensor.link.onnx.dispatch.resize
import pytensor.link.onnx.dispatch.shape
import pytensor.link.onnx.dispatch.special

__all__ = ["onnx_funcify", "onnx_typify"]
# isort: on
