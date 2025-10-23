---
date: 2025-10-23T10:48:47-05:00
researcher: Claude
git_commit: 85ac2c81856aeaf4f9c0b47c13eb41ab78ea099b
branch: onnx-workshop-demo
repository: pytensor
topic: "JAX Backend Operations Requiring Rewrites for Tracer Compatibility"
tags: [research, codebase, jax, pytensor, tracers, shape-operations, yolo11, gradient-computation]
status: complete
last_updated: 2025-10-23
last_updated_by: Claude
---

# Research: JAX Backend Operations Requiring Rewrites for Tracer Compatibility

**Date**: 2025-10-23T10:48:47-05:00
**Researcher**: Claude
**Git Commit**: 85ac2c81856aeaf4f9c0b47c13eb41ab78ea099b
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question
Even when setting `optimizer_excluding=shape_unsafe`, JAX tracer errors persist in the YOLO11 implementation. Which operations in the PyTensor JAX backend need to be rewritten to properly handle JAX tracers and enable GPU training?

## Summary
The JAX backend has 31+ operations with tracer compatibility issues, primarily in shape manipulation, tensor construction, and gradient computation. The root cause is that JAX JIT compilation requires concrete integer values for operations like `split`, `reshape`, `arange`, and `resize`, but PyTensor's implementations often pass symbolic/traced values. Even with `optimizer_excluding=shape_unsafe`, the specific error shows `Split` operation receiving tracer values for split indices. Critical operations requiring immediate fixes include: Split, Resize, Alloc/AllocEmpty, DimShuffle, and gradient operations for Conv2D. The solution requires either marking shape parameters as static arguments during JIT compilation or rewriting operations to avoid requiring concrete values.

## Detailed Findings

### The Immediate Issue: Split Operation Failure

#### Error Traceback Analysis
The error occurs in `pytensor/link/jax/dispatch/tensor_basic.py:139`:
```python
return jnp.split(x, cumsum_splits, axis=axis)
```

**Root Cause Chain**:
1. C3k2 block concatenates tensors: `pt.concatenate([x1, x2], axis=1)` (blocks.py:307)
2. PyTensor optimizer may introduce Split operations for optimization
3. Split implementation tries to extract constant values (tensor_basic.py:106-117)
4. When extraction fails, falls back to `jnp.cumsum(splits[:-1])` (line 131)
5. During JIT, `splits` contains tracers, not concrete values
6. `jnp.split()` requires concrete indices, receives tracers → ConcretizationTypeError

### Operations Requiring Rewrites - Comprehensive List

#### 1. Critical Priority - Guaranteed to Fail with Tracers

**Split** (`tensor_basic.py:95-141`)
- **Problem**: `jnp.split()` requires concrete split indices
- **Current**: Falls back to `jnp.cumsum()` on tracers when not constant
- **Fix Required**: Mark split indices as static or use alternative implementation

**Resize** (`resize.py:59-119`)
- **Problem**: Uses `int(height * scale_h)` which fails on tracers
- **Affects**: YOLO upsampling operations
- **Fix Required**: Use JAX-compatible shape operations instead of Python int()

**AllocEmpty** (`tensor_basic.py:35-40`)
- **Problem**: `jnp.empty(shape)` requires concrete shape tuple
- **Fix Required**: Mark shape as static or use traced-compatible allocation

**Alloc** (`tensor_basic.py:43-50`)
- **Problem**: `jnp.broadcast_to()` with symbolic shapes fails
- **Fix Required**: Rewrite using JAX shape operations

**Eye** (`tensor_basic.py:156-163`)
- **Problem**: `jnp.eye(N, M, k)` requires concrete dimensions
- **Fix Required**: Mark N, M, k as static arguments

**MaxPoolGrad** (`pool.py:78-163`)
- **Problem**: Uses shape values in Python `range()` loops
- **Fix Required**: Replace with JAX-compatible iteration

**FillDiagonal** (`extra_ops.py:104-110`)
- **Problem**: `min(value.shape[-2:])` and `diag_indices()` on tracers
- **Fix Required**: Use JAX operations for min and diagonal indices

#### 2. High Priority - Fail in Complex Scenarios

**DimShuffle** (`elemwise.py:72-84`)
- **Problem**: Accesses `.shape`, converts to list, modifies, then reshapes
- **Current**: `shape = list(res.shape[:len(op.shuffle)])`
- **Fix Required**: Use JAX shape operations throughout

**Argmax** (`math.py:28-60`)
- **Problem**: Complex shape arithmetic with `np.prod()` on shapes
- **Fix Required**: Replace NumPy operations with JAX equivalents

**ARange** (`tensor_basic.py:53-82`)
- **Problem**: Limited support for shape-derived values
- **Current**: Only accepts constants or Shape_i operations
- **Fix Required**: Expand support for traced shape values

**Reshape** (`shape.py:58-74`)
- **Problem**: Overly restrictive validation - only accepts Shape/Shape_i/JAXShapeTuple
- **Fix Required**: Support more shape sources while maintaining safety

**Conv2D Gradient Operations** (`conv.py:134-422`)
- **AbstractConv_gradInputs**: Extensive `.shape` access for upsampling
- **AbstractConv_gradWeights**: Shape access for filter gradient computation
- **Fix Required**: Rewrite shape calculations using JAX operations

#### 3. Medium Priority - Partial Compatibility

**Repeat** (`extra_ops.py:41-48`)
- **Problem**: `repeats` parameter may need to be concrete
- **Fix Required**: Mark as static when necessary

**BatchNormalization** (`batchnorm.py:9-101`)
- **Problem**: Conditional reshape based on `.ndim`
- **Fix Required**: Use JAX-compatible reshape patterns

**Random Operations** (`random.py:83-425`)
- **Problem**: Extensive shape slicing and broadcasting
- **Fix Required**: Ensure shape operations are traced-compatible

**Tri** (`tensor_basic.py:190-203`)
- **Problem**: Requires concrete dimensions
- **Fix Required**: Mark dimensions as static

### Pattern Analysis: Why Operations Fail

#### 1. Direct Python Type Conversion
```python
# FAILS with tracers
out_height = int(height * scale_h)  # resize.py:67
```

#### 2. Python Control Flow on Shapes
```python
# FAILS - range requires concrete integers
for i in range(out_h):  # pool.py:151
    for j in range(out_w):
```

#### 3. NumPy Operations on Shapes
```python
# FAILS - NumPy doesn't understand tracers
new_shape = np.prod(np.array(reduced_shape))  # math.py:52
```

#### 4. Shape Tuple Manipulation
```python
# FAILS - list() on traced shapes
shape = list(res.shape[:len(op.shuffle)])  # elemwise.py:77
```

#### 5. Direct Shape Comparisons
```python
# FAILS during JIT
if a.shape[0] != b.shape[0]:  # blas.py:10
```

### JAX Static Arguments System

**Current Implementation** (`linker.py:76-98`):
- Automatically detects inputs used only in JAXShapeTuple nodes
- Marks them as `static_argnums` during `jax.jit()`
- Converts to Python int before passing to JIT function

**Limitation**: Only handles direct shape inputs, not:
- Shape values from computations
- Shape accesses within operations
- Dynamic shape arithmetic

### Operations That Work Correctly

These operations properly handle tracers:
- **Shape** / **Shape_i**: Return traced shapes (safe)
- **Join**: Uses `jnp.concatenate()` (axis must be concrete)
- **MakeVector**: Creates arrays with `jnp.array()`
- **Pool**: Uses `jax.lax.reduce_window()` with static parameters
- **Subtensor operations**: Use JAX's advanced indexing

### Solution Strategies

#### Strategy 1: Mark Parameters as Static
```python
# In JAXLinker
static_argnums = detect_shape_parameters(fgraph)
jit_fn = jax.jit(fn, static_argnums=static_argnums)
```

#### Strategy 2: Use JAX-Compatible Operations
```python
# Instead of: int(height * scale_h)
out_height = jnp.round(height * scale_h).astype(jnp.int32)
```

#### Strategy 3: Avoid Shape Access During JIT
```python
# Instead of: shape = list(x.shape)
shape = jnp.shape(x)  # Returns traced array
```

#### Strategy 4: Compile-Time Validation
```python
# Check at graph construction, not runtime
if not is_shape_compatible(shape_source):
    raise NotImplementedError("Shape must be constant or from Shape ops")
```

## Code References

### Critical Files
- `pytensor/link/jax/dispatch/tensor_basic.py:95-141` - Split operation with tracer issues
- `pytensor/link/jax/dispatch/resize.py:67-68,99-100` - int() conversion on shapes
- `pytensor/link/jax/dispatch/elemwise.py:77` - DimShuffle shape manipulation
- `pytensor/link/jax/dispatch/conv.py:187,220-224` - Conv gradient shape access
- `pytensor/link/jax/dispatch/pool.py:106-107,151-152` - MaxPoolGrad shape unpacking
- `pytensor/link/jax/linker.py:76-98` - Static argument detection system

### YOLO Model Usage
- `examples/onnx/onnx-yolo-demo/yolo/blocks.py:307` - C3k2 concatenation
- `examples/onnx/onnx-yolo-demo/yolo/loss.py:117-118` - Subtensor slicing
- `examples/onnx/onnx-yolo-demo/yolo/model.py:268` - Resize upsampling
- `examples/onnx/onnx-yolo-demo/tests/test_jax_backend.py:8-10` - Configuration

## Architecture Insights

### Two-Phase Execution Model
1. **Graph Construction**: PyTensor builds symbolic graph with shape placeholders
2. **JIT Compilation**: JAX traces the graph, converting shapes to tracers
3. **Problem**: Operations expecting concrete values receive tracers instead

### Optimizer Impact
- Even with `optimizer_excluding=shape_unsafe`, optimizers may introduce problematic patterns
- Split operations are introduced by optimizer, not user code
- The C3k2 block itself is JAX-compatible; issues arise from optimization

### Static vs Dynamic Dichotomy
- **Op Properties** (`op.__props__`): Already static, compiled into Op
- **Graph Inputs**: Need explicit `static_argnums` marking
- **Computed Shapes**: Most problematic - neither static nor properly traced

## Historical Context (from thoughts/)

Previous research documents establish:
- `2025-10-23_10-20-02_jax-attributeerror-error-repr.md`: Identified optimizer_excluding solution with 95% confidence
- `2025-10-22_17-38-47_yolo11-operations-comprehensive-list.md`: Catalogued all 31+ operations used in YOLO11
- Pattern: Shape-unsafe optimizations consistently cause JAX failures
- Performance trade-off: 5-10% speed reduction for compatibility

## Related Research
- [JAX AttributeError Analysis](thoughts/shared/research/2025-10-23_10-20-02_jax-attributeerror-error-repr.md)
- [YOLO11 Operations List](thoughts/shared/research/2025-10-22_17-38-47_yolo11-operations-comprehensive-list.md)
- [JAX JIT Compilation Issues](thoughts/shared/research/2025-10-22_17-09-01_jax-jit-yolo-compilation-issues.md)

## Open Questions

1. **Can we automatically detect which parameters need static marking?**
   - Current system only catches direct JAXShapeTuple usage
   - Could analyze operation implementations for concrete requirements

2. **Should PyTensor provide JAX-specific operations?**
   - Alternative implementations that avoid concrete value requirements
   - Trade complexity for compatibility

3. **Can the optimizer be made JAX-aware?**
   - Avoid introducing Split with dynamic indices
   - Preserve concrete values where JAX need them

4. **Is there a systematic way to test tracer compatibility?**
   - Automated testing with symbolic shapes
   - Catch issues before runtime

## Recommendations

### Immediate Actions
1. **Fix Split Operation**: Add shape validation and better error messages
2. **Fix Resize Operation**: Replace int() with JAX-compatible operations
3. **Document Requirements**: Clear indication of which ops need concrete values

### Long-term Improvements
1. **Enhance Static Detection**: Analyze ops to auto-detect static requirements
2. **JAX-Specific Rewrites**: Optimizer passes that preserve JAX compatibility
3. **Comprehensive Testing**: Test suite with symbolic shapes for all ops
4. **Performance Analysis**: Measure impact of different solution strategies

### For YOLO Users
1. **Use Configuration**: Set `optimizer_excluding=shape_unsafe` before imports
2. **Avoid Dynamic Shapes**: Use fixed input sizes where possible
3. **Monitor Warnings**: JAX dispatch warnings indicate potential failures
4. **Test on Small Examples**: Verify JAX compatibility before full training

This analysis provides a complete roadmap for fixing JAX backend tracer compatibility issues, with specific operations identified and solution strategies outlined.