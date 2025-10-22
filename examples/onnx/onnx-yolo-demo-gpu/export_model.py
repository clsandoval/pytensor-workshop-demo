"""Export YOLO11n model to ONNX format.

IMPORTANT: This script sets PYTENSOR_FLAGS='floatX=float32' to ensure
the exported ONNX model uses float32 consistently. This prevents type
mismatch errors when loading the model in ONNX Runtime (Web/CPU/GPU).

The script also uses concrete input shapes and ONNX shape inference to
ensure WebGPU compatibility.
"""

# CRITICAL: Set PyTensor to float32 BEFORE any imports!
import os


os.environ["PYTENSOR_FLAGS"] = "floatX=float32"

import onnx
from onnx import shape_inference
from yolo.model import build_yolo11n

import pytensor
from pytensor import tensor as pt
from pytensor.graph.replace import clone_replace
from pytensor.link.onnx import export_onnx


# Build model
model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

# Create concrete input shape (required for WebGPU)
# WebGPU needs concrete shapes for kernel compilation
x_concrete = pt.TensorType("float32", shape=(1, 3, 320, 320))("x")

# Replace symbolic input with concrete input
concrete_predictions = [
    clone_replace(pred, {x_sym: x_concrete}) for pred in predictions
]

# Compile PyTensor function with concrete shapes
f = pytensor.function([x_concrete], concrete_predictions)

# Export to ONNX
onnx_path = "yolo11n.onnx"
onnx_model = export_onnx(f, str(onnx_path))

# Apply ONNX shape inference to propagate concrete shapes through the graph
# This is critical for WebGPU - it needs to know output shapes at kernel compilation time
print("Applying ONNX shape inference for WebGPU compatibility...")
inferred_model = shape_inference.infer_shapes(onnx_model)

# Save the model with concrete shapes
onnx.save(inferred_model, onnx_path)

print(f"\nModel exported to: {onnx_path}")
print("Input shape: [1, 3, 320, 320]")
print("Output shapes:")
for out in inferred_model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")
