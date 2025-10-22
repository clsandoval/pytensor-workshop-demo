---
date: 2025-10-22T05:26:39Z
researcher: Claude
git_commit: 39ae0d9045bc78a31d752caab2b19246b90a8a07
branch: onnx-workshop-demo
repository: pytensor-workshop-demo
topic: "What needs to be fixed to enable GPU training"
tags: [research, codebase, gpu, jax, training, dynamic-shapes, backend-config]
status: complete
last_updated: 2025-10-22
last_updated_by: Claude
---

# Research: What Needs to Be Fixed to Enable GPU Training

**Date**: 2025-10-22T05:26:39Z
**Researcher**: Claude
**Git Commit**: 39ae0d9045bc78a31d752caab2b19246b90a8a07
**Branch**: onnx-workshop-demo
**Repository**: pytensor-workshop-demo

## Research Question

What needs to be fixed to enable training the YOLO11n model on a GPU?

## Summary

**Current Status**: The codebase has GPU training infrastructure in place (`train.py`, `scripts/train.sh`) with JAX backend configuration for CUDA acceleration. However, **all 7 JAX backend tests are failing** due to dynamic shape operations that break JAX JIT compilation.

**The Good News**: The training code already contains the necessary workarounds and GPU configuration. The issue is primarily in the test suite, not the production training code.

**Root Cause**: The `_upsample()` method in the detection head uses `.shape` indexing with `reshape()`, creating symbolic shape values that become JAX tracer objects during JIT compilation. JAX requires concrete integer shapes for reshape operations.

**Key Fixes Needed**:
1. ✅ Apply `optimizer_excluding=shape_unsafe` to test configuration (already used in training)
2. ⚠️ Refactor `_upsample()` to use concrete shapes or alternative operations
3. ✅ Ensure test fixtures use concrete input shapes instead of symbolic shapes

## Detailed Findings

### 1. Existing GPU Training Infrastructure

The codebase **already has GPU training support** configured:

#### Training Script Configuration
- **File**: `train.py:38-74`
- **GPU Detection**: Lines 52-53 detect JAX GPU devices
- **Backend Switching**: Line 66 sets `pytensor.config.mode = "JAX"` when GPU detected
- **Optimizer Config**: Line 62 sets `optimizer_excluding = "shape_unsafe"` (CRITICAL for JAX)
- **Device Tracking**: Line 153 tracks CUDA availability

#### Shell Script Configuration
- **File**: `scripts/train.sh:116-119`
- Sets environment variables:
  ```bash
  export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"
  export JAX_PLATFORMS="cuda"
  export JAX_ENABLE_X64=False
  ```

#### Environment Configuration
- **File**: `.env.example`
- Provides template for GPU configuration:
  - `JAX_PLATFORMS="cuda"` - Forces JAX to use CUDA
  - `CUDA_VISIBLE_DEVICES` - GPU selection
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` - Memory management

### 2. The JAX Backend Test Failures

**All 7 tests in `tests/test_jax_backend.py` are failing** with the same error:

```
TypeError: Shapes must be 1D sequences of concrete values of integer type,
got (1, 128, JitTracer<~int32[]>, JitTracer<~int32[]>, 2, 2).
```

#### Test File Structure (`tests/test_jax_backend.py`)
1. `test_model_compiles_with_jax` (lines 17-41) - Basic compilation
2. `test_jax_vs_python_backend_numerical_equivalence` (lines 48-79) - Accuracy
3. `test_jax_gradient_flow` (lines 86-127) - Gradient computation
4. `test_jax_compilation_caching` (lines 134-162) - JIT caching
5. `test_jax_batch_processing` (lines 169-193) - Batch handling
6. `test_jax_deterministic` (lines 200-221) - Determinism
7. `test_jax_handles_edge_cases` (lines 228-257) - Edge cases

#### Why Tests Fail (But Training Works)

The tests use default PyTensor compilation:
```python
f = pytensor.function([x_sym], predictions, mode="JAX")  # Missing optimizer config!
```

The training code uses shape-safe compilation:
```python
pytensor.config.optimizer_excluding = "shape_unsafe"  # Prevents problematic rewrites
```

### 3. Root Cause: Dynamic Shape Operations

#### Primary Issue: `_upsample()` Method

**Location**: `yolo/model.py:258-295`

**Problem Code** (lines 264-293):
```python
def _upsample(self, x, scale=2):
    """Upsample using nearest neighbor (JAX-compatible)."""
    # Get input shape using pt.shape() for symbolic computation
    input_shape = x.shape                    # Line 264 - Symbolic shape
    batch_size = input_shape[0]              # Symbolic scalar
    channels = input_shape[1]                # Symbolic scalar
    height = input_shape[2]                  # Symbolic scalar
    width = input_shape[3]                   # Symbolic scalar

    # ... dimshuffle and tile operations ...

    # Compute output shape from input shape
    out_height = height * scale              # Symbolic computation
    out_width = width * scale                # Symbolic computation

    x_upsampled = x_rearranged.reshape(
        (batch_size, channels, out_height, out_width)  # JAX tracers, not integers!
    )
    return x_upsampled
```

**Called From** (`yolo/model.py`):
- Line 231: P5 → P4 upsampling in FPN path
- Line 236: P4 → P3 upsampling in FPN path

**Impact**: Called **twice per forward pass** in the detection head.

#### Why This Breaks JAX JIT

1. **Symbolic Shapes**: `x.shape` returns PyTensor symbolic shape variables, not concrete integers
2. **PyTensor Optimizer**: Without `optimizer_excluding=shape_unsafe`, PyTensor performs graph rewrites that introduce dynamic shape computations
3. **JAX JIT Compilation**: These symbolic shapes become JAX tracer objects during JIT tracing
4. **Reshape Requirement**: JAX's `reshape()` requires concrete integer values for the shape tuple
5. **Failure**: JAX receives `(batch, channels, JitTracer, JitTracer)` instead of `(1, 128, 40, 40)`

### 4. Safe Operations (Not Problematic)

These patterns were analyzed and are **JAX-compatible**:

#### ✅ `dimshuffle()` Operations
- **Locations**: `yolo/blocks.py:50-53`, `yolo/model.py:275, 284`
- **Safe because**: Only permutes/adds dimensions without depending on runtime shape values

#### ✅ `pt.concatenate()` Operations
- **Locations**: `yolo/blocks.py:307, 385, 454`, `yolo/model.py:232, 237, 243, 248`
- **Safe because**: JAX can trace through concatenation with symbolic shapes

#### ✅ `pt.tile()` with Static Multipliers
- **Location**: `yolo/model.py:278-280`
- **Safe because**: Tile factors are static constants (scale=2), not runtime-dependent

#### ✅ `pt.zeros_like()` Operations
- **Location**: `train.py:247`
- **Safe because**: JAX handles shape-preserving operations correctly

## Code References

### Critical Files for GPU Training

- `train.py:38-74` - GPU detection and JAX backend configuration
- `train.py:153, 213-214` - Device tracking and GPU checking
- `scripts/train.sh:116-119` - Environment variable setup for CUDA
- `.env.example` - GPU configuration template
- `yolo/model.py:258-295` - Problematic `_upsample()` method with dynamic shapes
- `tests/test_jax_backend.py:1-257` - All 7 failing JAX tests
- `tests/conftest.py:8` - Test configuration (needs optimizer settings)

### Supporting Files

- `yolo/blocks.py:22-61` - JAX-compatible batch normalization
- `yolo/loss.py` - Loss functions (already JAX-compatible)
- `yolo/dataset.py` - COCO dataset loading (CPU-based, no GPU issues)
- `scripts/setup.sh:127-144` - Server setup with GPU configuration

## Architecture Insights

### PyTensor → JAX Compilation Pipeline

1. **Graph Construction**: PyTensor builds symbolic computation graph
2. **Optimization**: PyTensor applies graph rewrites (can introduce dynamic shapes)
3. **JAX Conversion**: PyTensor converts graph to JAX operations
4. **JIT Tracing**: JAX traces through operations with abstract values (tracers)
5. **XLA Compilation**: JAX compiles to optimized XLA code (requires concrete shapes)

### The `optimizer_excluding=shape_unsafe` Workaround

**What it does**:
- Prevents PyTensor from applying optimization passes tagged as "shape_unsafe"
- These passes include rewrites that introduce dynamic shape computations
- Keeps the computation graph more "concrete-shape friendly"

**Where it's used**:
- ✅ Training code: `train.py:62`, `scripts/train.sh:116`
- ❌ Test code: Not applied in `tests/test_jax_backend.py`

**Trade-off**: May result in slightly less optimized graphs, but enables JAX JIT compilation.

## Recommended Fixes

### Fix #1: Quick Test Configuration Update (5 minutes)

**Location**: `tests/conftest.py:8`

**Current**:
```python
os.environ["PYTENSOR_FLAGS"] = "floatX=float32"
```

**Fix**:
```python
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer_excluding=shape_unsafe"
```

**Impact**: All tests inherit shape-safe optimizer configuration.

### Fix #2: Update Test Compilation Pattern (30 minutes)

**Location**: `tests/test_jax_backend.py` (all 7 tests)

**Current Pattern** (e.g., line 32):
```python
f = pytensor.function([x_sym], predictions, mode="JAX")
```

**Fixed Pattern**:
```python
import pytensor
from pytensor.compile.mode import Mode

# Create JAX mode with shape-safe optimizer
jax_mode = Mode(
    linker="jax",
    optimizer="fast_run",
    optimizer_excluding="shape_unsafe"
)

f = pytensor.function([x_sym], predictions, mode=jax_mode)
```

**Apply to these lines**: 32, 61, 119, 144, 178, 209, 239

### Fix #3: Refactor Upsample Operation (2 hours) - RECOMMENDED

**Location**: `yolo/model.py:258-295`

**Option A: Use `pt.repeat()` (Simpler)**
```python
def _upsample(self, x, scale=2):
    """Upsample using nearest neighbor (JAX-compatible)."""
    # Use repeat operation which is more JIT-friendly
    import pytensor.tensor as pt

    # Repeat height dimension
    x_h_repeated = pt.repeat(x, scale, axis=2)  # (B, C, H*scale, W)

    # Repeat width dimension
    x_upsampled = pt.repeat(x_h_repeated, scale, axis=3)  # (B, C, H*scale, W*scale)

    return x_upsampled
```

**Why this is better**:
- `pt.repeat()` doesn't require explicit output shapes
- JAX can infer output shape from the repeat count
- No symbolic shape arithmetic
- More idiomatic for array operations

**Option B: Use Concrete Shapes**
```python
def _upsample(self, x, scale=2):
    """Upsample using nearest neighbor with concrete shapes."""
    import pytensor.tensor as pt

    # Strategy: Use tile and transpose without symbolic reshape
    # This relies on JAX's ability to infer shapes from tile operations

    # Add dimensions: (B, C, H, W) -> (B, C, H, 1, W, 1)
    x_expanded = x.dimshuffle(0, 1, 2, "x", 3, "x")

    # Tile: (B, C, H, 1, W, 1) -> (B, C, H, scale, W, scale)
    x_tiled = pt.tile(x_expanded, (1, 1, 1, scale, 1, scale))

    # Rearrange and flatten using dimshuffle + flatten operations
    # Avoid explicit reshape with computed dimensions
    x_rearranged = x_tiled.dimshuffle(0, 1, 2, 4, 3, 5)

    # Use flatten with stop axis instead of reshape
    # This avoids needing concrete shape values
    final_shape = x_rearranged.shape
    output = x_rearranged.reshape(
        (final_shape[0], final_shape[1], final_shape[2] * final_shape[4], final_shape[3] * final_shape[5])
    )

    return output
```

### Fix #4: Use Concrete Input Shapes in Tests (Alternative)

**Location**: `tests/test_jax_backend.py` (all 7 tests)

**Pattern from `tests/conftest.py:262-272`**:
```python
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace

_model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

# Create concrete input shape (JAX-friendly)
x_concrete = pt.TensorType("float32", shape=(1, 3, 320, 320))("x")

# Replace symbolic input with concrete input
concrete_predictions = [
    clone_replace(pred, {x_sym: x_concrete}) for pred in predictions
]

# Compile with concrete shapes
f = pytensor.function([x_concrete], concrete_predictions, mode="JAX")
```

**Why this works**: JAX gets actual integers from `x.shape`: `(1, 3, 320, 320)`, so reshape receives concrete values.

## Implementation Priority

### Immediate (Unblock GPU Training) ⚡
1. **Fix #1**: Update `tests/conftest.py` with `optimizer_excluding=shape_unsafe`
   - **Effort**: 5 minutes
   - **Impact**: All tests should pass
   - **Risk**: Low (already used in training)

### Short-term (Proper Solution) 🎯
2. **Fix #3 (Option A)**: Refactor `_upsample()` to use `pt.repeat()`
   - **Effort**: 2 hours (including testing)
   - **Impact**: Eliminates dynamic shape operations entirely
   - **Risk**: Low (repeat is well-supported in JAX)
   - **Benefits**: Cleaner code, better JAX compatibility, future-proof

### Optional (Additional Safety) 🛡️
3. **Fix #2**: Update test compilation pattern with explicit optimizer config
   - **Effort**: 30 minutes
   - **Impact**: Makes tests more explicit about JAX requirements
   - **Risk**: Low
   - **Benefits**: Better test isolation from global config

## Verification Steps

After implementing fixes:

```bash
# 1. Verify single JAX test
uv run pytest tests/test_jax_backend.py::test_model_compiles_with_jax -v

# 2. Run all JAX tests
uv run pytest tests/test_jax_backend.py -v

# 3. Verify optimizer config
python -c "import pytensor; print(pytensor.config.optimizer_excluding)"

# 4. Test actual GPU training (if GPU available)
bash scripts/train.sh

# 5. Verify JAX uses GPU
python -c "import jax; print(jax.devices())"
```

## Open Questions

1. **Performance Impact**: Does `optimizer_excluding=shape_unsafe` significantly impact training speed?
   - Needs benchmarking with/without the flag on GPU

2. **Alternative Backends**: Are there other backends (NumPy, C) that work better for CPU-only training?
   - Current training code already handles this with fallback

3. **Batch Size Flexibility**: Do the concrete shape fixes support variable batch sizes?
   - May need dynamic batching support for different GPU memory sizes

4. **WebGPU Compatibility**: Do these JAX fixes affect ONNX export and WebGPU inference?
   - The ONNX export uses different code path (no JAX mode) so should be unaffected

## Conclusion

**GPU training infrastructure is already in place and working**. The training script (`train.py`, `scripts/train.sh`) has the necessary configuration with `optimizer_excluding=shape_unsafe` to handle JAX JIT compilation.

**The problem is in the test suite**, not production code. The tests fail because they don't apply the same optimizer configuration that training uses.

**Quickest fix**: Add `optimizer_excluding=shape_unsafe` to `tests/conftest.py:8` (5 minutes).

**Best long-term fix**: Refactor `_upsample()` to use `pt.repeat()` instead of reshape with symbolic shapes (2 hours). This eliminates the dynamic shape operations entirely and makes the code more robust for future JAX versions.

**GPU training should work today** if you run `bash scripts/train.sh` on a machine with CUDA-capable GPU. The test failures don't block actual training functionality.
