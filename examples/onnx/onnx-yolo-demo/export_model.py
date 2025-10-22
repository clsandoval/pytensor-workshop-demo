"""Export YOLO11n model to ONNX format."""

from yolo.model import build_yolo11n

import pytensor
from pytensor.link.onnx import export_onnx


# Build model
model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

# Compile PyTensor function
f = pytensor.function([x_sym], predictions)

# Export to ONNX
onnx_path = "yolo11n.onnx"
onnx_model = export_onnx(f, str(onnx_path))
