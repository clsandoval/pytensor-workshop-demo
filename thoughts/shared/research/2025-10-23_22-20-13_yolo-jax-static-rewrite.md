---
date: 2025-10-23T22:20:13-05:00
researcher: Claude Code
git_commit: babd1651655d190a0560fc2931e4ec689da43d2b
branch: onnx-workshop-demo
repository: pytensor-workshop-demo
topic: "Rewriting YOLO11n model for JAX with static shapes and fixed batch size"
tags: [research, codebase, jax, yolo, static-shapes, jit-compilation, batch-size]
status: complete
last_updated: 2025-10-23
last_updated_by: Claude Code
---

# Research: Rewriting YOLO11n Model for JAX with Static Shapes and Fixed Batch Size

**Date**: 2025-10-23T22:20:13-05:00
**Researcher**: Claude Code
**Git Commit**: babd1651655d190a0560fc2931e4ec689da43d2b
**Branch**: onnx-workshop-demo
**Repository**: pytensor-workshop-demo

## Research Question
I want to rewrite the yolo model in @yolo\ such that ONLY concrete values are used, set the batch size to 32 always, ABSOLUTELY NO DYNAMIC problematic values for JAX.

## Summary
The YOLO11n model implementation has been carefully designed to avoid most JAX JIT compilation issues, but contains several operations that create dynamic shape tracers during compilation. The primary issues are:

1. **CSP/C3k2 blocks using Split operations** with dynamic positions (creates tracers)
2. **Resize operations** that could potentially use dynamic shapes (currently safe with 2x upsampling)
3. **Shape arithmetic** in any custom operations
4. **Dynamic batch sizing** throughout the model

To make the model fully JAX-compatible with static shapes:
- **Fix batch size to 32** in all operations
- **Replace Split with static slicing** in CSP blocks
- **Ensure all resize operations use constant scale factors**
- **Configure PyTensor to exclude shape_unsafe optimizations**
- **Use compile-time constant extraction** where needed

## Detailed Findings

### Current YOLO Model Architecture Analysis

#### 1. Model Components (`yolo/model.py`)
The YOLO11n model consists of three main components:

**YOLO11nBackbone** (lines 26-128):
- Stem: ConvBNSiLU (16 channels)
- Stage 1: Conv + C3k2 block (32 channels)
- Stage 2 (P3): Conv + C3k2 block (64 channels)
- Stage 3 (P4): Conv + C3k2 block (128 channels)
- Stage 4 (P5): Conv + C3k2 block + SPPF + C2PSA (256 channels)

**YOLO11nHead** (lines 131-271):
- FPN upsampling path (P5→P4→P3)
- PAN downsampling path (P3→P4→P5)
- Detection heads for 3 scales
- Uses `_upsample()` method with resize operation

**YOLO11n** (lines 273-365):
- Combines backbone and head
- Returns tuple of detections at 3 scales

#### 2. Building Blocks (`yolo/blocks.py`)

**ConvBNSiLU** (lines 64-179):
- JAX-compatible batch normalization (lines 22-61)
- Uses `dimshuffle` for broadcasting (safe)
- Static kernel sizes and strides

**C3k2 Block** (lines 234-310) - **PROBLEMATIC**:
- Line 307: `x_cat = pt.concatenate([x1, x2], axis=1)`
- Split/merge pattern that could use dynamic Split internally
- Channel split at line 299: `x1 = self.conv1(x)` reduces to half channels

**SPPF Block** (lines 313-388):
- Uses cascaded max pooling with static pool_size=5
- Line 385: Concatenates pooled features (safe)

**C2PSA Block** (lines 391-457):
- Similar split/merge pattern to C3k2
- Line 454: Concatenation (safe)

### Identified Dynamic/Problematic Values

#### Critical Issues for JAX JIT Compilation:

1. **Split Operations in CSP Blocks** (`tensor_basic.py:138-227`)
   - **Location**: C3k2 and C2PSA blocks internally split channels
   - **Error**: `TracerIntegerConversionError` when split positions are computed
   - **Current Implementation**: Uses `hidden_channels = out_channels // 2` (line 255 in blocks.py)
   - **Fix Required**: Use static slicing `x[:, :hidden_channels]` instead of split

2. **Resize Operation** (`yolo/model.py:268`)
   ```python
   x_upsampled = resize(x, scale_factor=(scale, scale), mode="nearest")
   ```
   - **Current Status**: SAFE (uses scale=2 which triggers optimized path)
   - **Risk**: If scale factors change, could create tracers
   - **JAX Dispatch**: `resize.py:68-75` has special case for 2x upsampling

3. **Dynamic Batch Size**
   - **Current**: Model accepts variable batch size
   - **Issue**: Some JAX operations struggle with dynamic first dimension
   - **Fix**: Hard-code batch_size=32 in input tensor definition

4. **Potential Shape Arithmetic**
   - **Locations**: Any custom operations that compute shapes
   - **Current Status**: Model mostly avoids this
   - **Risk Areas**: Custom loss functions, data preprocessing

### JAX Backend Implementation Insights

#### Working Patterns (from `pytensor/link/jax/dispatch/`):

1. **Constant Extraction** (`subtensor_fixed.py:39-61`)
   ```python
   def extract_constants(indices):
       """Extract constant values from indices where possible."""
       fixed_indices = []
       for idx in indices:
           if hasattr(idx, 'data'):  # Constant node
               const_val = idx.data.item() if hasattr(idx.data, 'item') else idx.data
               fixed_indices.append(const_val)
   ```

2. **Static Slicing** (`test_jax_csp_static_slicing.py:16-56`)
   ```python
   # WORKING: Pure Python integer slices
   x1 = x[:, :128, :, :]   # First 128 channels
   x2 = x[:, 128:, :, :]   # Remaining channels
   ```

3. **Special Case Optimizations** (`resize.py:68-75`)
   ```python
   if scale_h == 2.0 and scale_w == 2.0:
       # Use repeat for exact 2x upsampling
       result = jnp.repeat(input, 2, axis=2)
       result = jnp.repeat(result, 2, axis=3)
   ```

#### Failing Patterns:

1. **Shape Arithmetic Creating Tracers**
   ```python
   # FAILS: Creates tracer
   height = x.shape[2]
   out_height = height * scale  # Becomes JitTracer
   x.reshape((batch, channels, out_height, out_width))  # ERROR
   ```

2. **Dynamic Split Positions**
   ```python
   # FAILS: Computed split positions
   split_size = x.shape[1] // 3
   splits = [split_size, split_size, split_size]
   y1, y2, y3 = pt.split(x, splits)  # TracerIntegerConversionError
   ```

### Test Results and Errors

#### Current Test Failures (`tests/test_jax_backend.py`):
- All 7 tests fail with same error pattern
- **Primary Error**: `TracerIntegerConversionError` in Split operation
- **Stack Trace**: Points to `tensor_basic.py:178` in split dispatch
- **Root Cause**: CSP blocks internally use dynamic split operations

#### Debug Script Results (`debug_jax_gradient.py`):
- Simple gradient computation: **SUCCESS**
- YOLO model gradient: **FAILURE**
- Error: `TracerIntegerConversionError` during split operation
- Location: C3k2 block's internal channel splitting

### Configuration Requirements

#### PyTensor Flags:
```python
os.environ["PYTENSOR_FLAGS"] = (
    "floatX=float32,"
    "optimizer_excluding=inplace,fusion,OpenMP,shape_unsafe,fast_run,fast_compile,"
    "merge,canonicalize,stabilize,specialize"
)
```

**Critical**: `optimizer_excluding=shape_unsafe` prevents PyTensor from introducing dynamic shape computations during graph optimization.

## Code References

### Model Files:
- `yolo/model.py:268` - Resize operation (_upsample method)
- `yolo/model.py:357` - Input tensor declaration
- `yolo/blocks.py:255` - C3k2 hidden_channels computation
- `yolo/blocks.py:307` - C3k2 concatenation
- `yolo/blocks.py:385` - SPPF concatenation
- `yolo/blocks.py:454` - C2PSA concatenation

### JAX Dispatch:
- `pytensor/link/jax/dispatch/resize.py:68-75` - 2x upsampling optimization
- `pytensor/link/jax/dispatch/tensor_basic.py:178-227` - Split operation (problematic)
- `pytensor/link/jax/dispatch/subtensor_fixed.py:39-61` - Constant extraction pattern

### Test Files:
- `tests/test_jax_backend.py` - Main YOLO JAX tests (all failing)
- `tests/link/jax/test_jax_csp_static_slicing.py` - Static slicing solution
- `tests/link/jax/test_jax_tracer_compatibility.py` - Tracer handling tests

## Architecture Insights

### Design Patterns That Work:
1. **Static Architecture Parameters**: All layer sizes, kernel sizes, strides are compile-time constants
2. **Dimshuffle for Broadcasting**: Avoids dynamic reshape in batch norm
3. **Fixed Scale Factors**: 2x upsampling in FPN/PAN is JAX-optimized
4. **Channel-wise Operations**: Concatenation along channel axis with fixed sizes

### Design Patterns That Fail:
1. **Dynamic Channel Splitting**: Using computed split positions in CSP blocks
2. **Shape Arithmetic**: Any multiplication/division on shape values
3. **Runtime Shape Queries**: Using tensor.shape in computations

## Recommended Solution

### 1. Fixed Batch Size Implementation
```python
def build_yolo11n(num_classes=2, input_size=320, batch_size=32):
    """Build YOLO11n with fixed batch size for JAX."""
    # Fixed batch size input
    x = pt.tensor4("x", dtype="float32", shape=(32, 3, 320, 320))

    model = YOLO11n(num_classes=num_classes, input_size=input_size, batch_size=32)
    predictions = model(x)
    return model, x, predictions
```

### 2. Static CSP Block Rewrite
```python
class C3k2Static:
    """C3k2 block with static slicing instead of dynamic split."""

    def __init__(self, in_channels, out_channels, n_blocks=1):
        self.hidden_channels = out_channels // 2  # Compile-time constant
        # ... initialization ...

    def __call__(self, x):
        # Static channel split using integer slicing
        x1 = x[:, :self.hidden_channels, :, :]  # First half

        # Process through bottlenecks
        x2 = x1
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)

        # Concatenate (safe operation)
        x_cat = pt.concatenate([x1, x2], axis=1)
        out = self.conv2(x_cat)
        return out
```

### 3. Configuration Setup
```python
import os
os.environ["PYTENSOR_FLAGS"] = (
    "floatX=float32,"
    "optimizer_excluding=shape_unsafe"  # Critical for JAX
)
```

### 4. Input Pipeline with Fixed Batching
```python
def create_fixed_batch_dataloader(dataset, batch_size=32):
    """Create dataloader with fixed batch size, padding if needed."""
    def get_batch(indices):
        batch = [dataset[i] for i in indices]

        # Pad to exactly 32 samples if needed
        while len(batch) < 32:
            batch.append(batch[0])  # Repeat first sample

        return stack_batch(batch[:32])
```

## Historical Context (from thoughts/)

### Previous Research:
- `thoughts/shared/research/2025-10-22_17-09-01_jax-jit-yolo-compilation-issues.md` - Initial JAX error analysis
- `thoughts/shared/research/2025-10-22_00-26-39_gpu_training_requirements.md` - GPU training analysis with JAX issues

### TDD Plans:
- `thoughts/shared/plans/jax_backend_operations_compatibility_tdd.md` - Expected failures documented
- `thoughts/shared/plans/jax_backend_composite_patterns_tdd.md` - CSP pattern tests

## Related Research
- Previous JAX JIT compilation issues research
- CSP pattern static slicing solutions
- JAX backend dispatch implementation patterns

## Open Questions

1. **Performance Impact**: How much does static slicing vs dynamic split affect performance?
2. **Batch Size Flexibility**: Can we support multiple fixed batch sizes (1, 8, 16, 32) with separate compiled functions?
3. **Mixed Precision**: How to enable bfloat16 training with JAX backend?
4. **XLA Optimizations**: Which XLA passes can optimize the static CSP blocks?
5. **Alternative Architectures**: Would depth-wise separable convolutions avoid split issues?

## Implementation Checklist

To make YOLO11n fully JAX-compatible:

- [ ] Fix input tensor to shape=(32, 3, 320, 320)
- [ ] Rewrite C3k2 blocks with static slicing
- [ ] Rewrite C2PSA blocks with static slicing
- [ ] Ensure all resize operations use scale=2.0
- [ ] Add configuration for optimizer_excluding=shape_unsafe
- [ ] Create fixed-batch dataloader
- [ ] Test gradient computation with JAX backend
- [ ] Verify ONNX export still works
- [ ] Benchmark performance vs dynamic version
- [ ] Document static shape requirements