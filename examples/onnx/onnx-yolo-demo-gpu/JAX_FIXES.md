# JAX Backend Test Fixes

## Issues Fixed

### 1. JAX JIT Compilation Error
**Problem**: Tests were failing with `TypeError: Shapes must be 1D sequences of concrete values`

**Root Cause**: PyTensor's graph optimizer introduces dynamic shape computations that are incompatible with JAX JIT compilation. Operations like `local_reshape_chain`, `local_subtensor_of_batch_dims`, and other "shape_unsafe" rewrites create JAX tracer objects instead of concrete integer values.

**Solution**: Added `optimizer_excluding=shape_unsafe` to PYTENSOR_FLAGS in:
- `tests/conftest.py` (line 8)
- `scripts/run_tests.sh`
- `scripts/run_tests.py`

### 2. AttributeError in Gradient Test
**Problem**: Test was calling non-existent method `model.backbone.get_params()`

**Root Cause**: The model uses `params` attribute, not a `get_params()` method.

**Solution**: Changed line 107 in `tests/test_jax_backend.py` from:
```python
model.backbone.get_params() + model.head.get_params()
```
to:
```python
model.backbone.params + model.head.params
```

## Running Tests

### On GPU VM (Linux):
```bash
cd /path/to/pytensor/examples/onnx/onnx-yolo-demo-gpu
bash scripts/run_tests.sh
```

### On Windows or any Python environment:
```bash
cd /path/to/pytensor/examples/onnx/onnx-yolo-demo-gpu
python scripts/run_tests.py
```

### Manual test run:
```bash
export PYTENSOR_FLAGS="floatX=float32,optimizer_excluding=shape_unsafe"
export JAX_PLATFORMS="cuda"
python -m pytest tests/test_jax_backend.py -xvs
```

## Important Notes

1. **Environment Variables Must Be Set Before Import**: The PYTENSOR_FLAGS must be set before importing any PyTensor modules. This is why it's at the top of `conftest.py`.

2. **JAX Platform**: Set `JAX_PLATFORMS="cuda"` to force JAX to use GPU.

3. **Float32 Required**: JAX and ONNX require float32, not float64.

## Troubleshooting

If tests still fail:

1. **Verify environment**:
```python
import pytensor
print(pytensor.config.floatX)  # Should be 'float32'
print(pytensor.config.optimizer_excluding)  # Should be 'shape_unsafe'
```

2. **Check JAX GPU availability**:
```python
import jax
print(jax.devices())  # Should show GPU device
```

3. **Clear PyTensor cache**:
```bash
rm -rf ~/.pytensor
```

## Training Script

The same fix applies to training. Run training with:
```bash
PYTENSOR_FLAGS="floatX=float32,optimizer_excluding=shape_unsafe" python train.py
```

Or use the provided script:
```bash
bash scripts/train.sh
```