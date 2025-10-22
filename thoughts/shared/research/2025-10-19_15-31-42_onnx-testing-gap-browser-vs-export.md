---
date: 2025-10-19T15:31:42-0500
researcher: clsandoval
git_commit: 226f34c37775b44b18b783723a7357f56fb5e116
branch: onnx-workshop-demo
repository: pytensor
topic: "ONNX Testing Gap: Browser Runtime vs Export Tests"
tags: [research, onnx, testing, yolo, browser, webgpu, composite-operations]
status: complete
last_updated: 2025-10-19
last_updated_by: clsandoval
---

# Research: ONNX Testing Gap: Browser Runtime vs Export Tests

**Date**: 2025-10-19T15:31:42-0500
**Researcher**: clsandoval
**Git Commit**: 226f34c37775b44b18b783723a7357f56fb5e116
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question

When running the YOLO11n model in the browser (webgpu_vs_wasm_benchmark.html), there's an error in `Add_composite_tmp_2443371660016`, but the ONNX export tests pass. What is the testing gap?

## Summary

**The Critical Gap**: PyTensor's ONNX export tests validate that models export successfully and produce correct numerical results when run with **ONNX Runtime's CPU backend**, but they **never validate** that exported models actually work in the **browser environment** (WebGPU/WASM) where they're intended to run.

**The Error**: The `Add_composite_tmp_*` variable name in the error message indicates an issue with **Composite operation decomposition** - specifically, PyTensor fuses multiple operations into a single Composite op during optimization, then decomposes them back into individual ONNX nodes during export. The intermediate variable naming suggests this decomposition creates temporary variables that may not be handled correctly by the browser runtime.

**The Root Cause**: Three distinct gaps exist:
1. **Runtime Environment Gap**: Tests use CPU ExecutionProvider only, never WebGPU/WASM
2. **Integration Testing Gap**: Tests validate individual operations but not complete neural network models
3. **Variable Naming Gap**: Composite decomposition creates intermediate variables with generated names that may violate browser runtime constraints

## Detailed Findings

### 1. What the Tests DO Validate

**ONNX Export Tests** (`tests/link/onnx/test_*.py` - 157 tests):

✓ **Numerical Correctness**: ONNX Runtime (CPU) produces same results as PyTensor
- Method: `compare_onnx_and_py()` helper in `test_basic.py:22`
- Tolerance: rtol=1e-4, atol=1e-5
- Execution: Sequential with `CPUExecutionProvider` only

✓ **ONNX Spec Compliance**: Models pass `onnx.checker.check_model()`
- Validates: Graph structure, node types, attributes, type inference
- Location: Called in every test via `test_basic.py:348`

✓ **Operation Coverage**: Comprehensive test suite for:
- Elemwise ops (45 tests): Add, Mul, Sub, Div, Sigmoid, SiLU, Switch, Cast, etc.
- Shape ops (35 tests): Reshape, DimShuffle, Squeeze, Unsqueeze, etc.
- Conv ops (19 tests): All parameter combinations (stride, padding, dilation, groups)
- Other ops: BatchNorm (7), Pool (7), Join (10), LinAlg (7), Resize (6), etc.

✓ **YOLO-Specific Patterns**: Explicitly tested
- Floor division for padding: `test_floor_div_yolo_padding_pattern`
- SPPF pooling: `test_maxpool2d_yolo_sppf_pattern`
- FPN upsampling: `test_resize_onnx_yolo_fpn_pattern`
- C3k2 blocks, SiLU activation, attention mechanisms

**YOLO Demo Tests** (`examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py` - 8 tests):

✓ **Export Success**: Model exports without errors (`test_model_exports_to_onnx`)
✓ **CPU Runtime Equivalence**: ONNX Runtime matches PyTensor (`test_onnx_runtime_vs_pytensor_equivalence`)
✓ **Output Shapes**: Correct dimensions for P3, P4, P5 scales
✓ **Batch Sizes**: Dynamic batching works (batch=1, 2, 4)
✓ **Determinism**: Same input produces identical output
✓ **Metadata**: Inputs/outputs have correct shapes
✓ **Opset Version**: Uses opset >= 11
✓ **No Training Ops**: No Dropout, RandomUniform, etc.

### 2. What the Tests DO NOT Validate

✗ **Browser/WebGPU Execution**: Zero tests run in browser environment
- No Selenium/Puppeteer-based browser tests
- No WebGPU backend validation
- No ONNX.js runtime tests
- No JavaScript execution tests

✗ **WASM Execution**: Zero tests use WASM ExecutionProvider
- Tests only use `CPUExecutionProvider`
- No validation of browser WASM runtime

✗ **Cross-Runtime Validation**: Only CPU backend tested
- Not tested: GPU, TensorRT, CoreML, Web backends
- Risk: Operators may have backend-specific bugs

✗ **End-to-End Model Deployment**: Isolated operation tests only
- No complete neural network validation beyond YOLO export test
- No inference pipeline testing (preprocessing → model → postprocessing)
- No real-world deployment scenarios

✗ **Variable Naming Constraints**: No validation of intermediate variable names
- Composite decomposition creates names like `composite_tmp_{id(var)}`
- No checks for browser runtime naming requirements
- No validation against WebGPU/WASM identifier constraints

✗ **Performance/Latency**: No benchmarking
- No FPS measurements
- No memory usage validation
- No optimization verification

### 3. The Composite Operation Decomposition Issue

**What are Composite Operations?**

From `pytensor/link/onnx/dispatch/elemwise.py:70-397`:

Composite operations are **fused operations** created by PyTensor's optimizer:
```python
# Original PyTensor code:
z = (x * 2 + y) * 3

# After optimization:
z = Elemwise(Composite([Mul, Add, Mul]))(x, y)  # Fused into single op

# During ONNX export, decomposed back:
tmp1 = Mul(x, 2)
tmp2 = Add(tmp1, y)        # <- Creates intermediate "composite_tmp_*" variable
z = Mul(tmp2, 3)
```

**The Decomposition Process** (`decompose_composite_elemwise` function):

1. **Traverses composite subgraph** in topological order (`io_toposort`)
2. **Creates ONNX constants** for embedded scalar constants
3. **Generates intermediate variable names**:
   ```python
   output_name = f"composite_tmp_{id(output_var)}"  # Line 123
   ```
4. **Maps scalar operations to ONNX ops** via `SCALAR_OP_TO_ONNX` dictionary
5. **Returns list of ONNX nodes** with intermediate connections

**Where Composites Appear in YOLO11n**:

Critical location: `yolo/model.py:249-286` - `_upsample()` method
```python
def _upsample(self, x, scale=2):
    # Get input shape using pt.shape() for symbolic computation
    input_shape = x.shape
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Compute output shape from input shape
    out_height = height * scale  # <- Likely becomes Composite
    out_width = width * scale    # <- Likely becomes Composite

    x_upsampled = x_rearranged.reshape(
        (batch_size, channels, out_height, out_width)  # <- Dynamic shapes
    )
```

**Why This Matters**:

The `_upsample()` method uses:
- **Symbolic shape access**: `x.shape` returns symbolic variables (line 255)
- **Arithmetic on shapes**: `height * scale` creates Elemwise ops on shape values (line 279)
- **Dynamic reshape**: `reshape()` with computed dimensions (line 282-284)

PyTensor's optimizer likely fuses these shape arithmetic operations into Composite ops, which then get decomposed during ONNX export with intermediate variable names like `Add_composite_tmp_2443371660016`.

### 4. Browser Runtime Differences from CPU Runtime

**ONNX Runtime CPU** (what tests use):
- Fully featured C++ implementation
- Permissive variable naming
- Comprehensive operator support
- Forgiving error handling

**ONNX Runtime Web (WebGPU/WASM)** (what browser uses):
- JavaScript/TypeScript implementation
- Potential naming constraints (JavaScript identifier rules)
- Limited operator support (subset of CPU ops)
- Stricter error handling

**Potential Issues with `Add_composite_tmp_*` Names**:

1. **Identifier Length**: Very long generated names may hit browser limits
2. **Naming Collisions**: `id(output_var)` may not be unique across graph
3. **Special Characters**: Underscore-heavy names may cause issues
4. **Name Resolution**: Browser may have different variable scoping rules

### 5. YOLO-Specific Test Coverage Analysis

**Tested YOLO Patterns** (`tests/link/onnx/test_*.py`):

| Pattern | Test Location | Status |
|---------|---------------|--------|
| Floor div padding | `test_elemwise.py:152` | ✓ Tested |
| SiLU activation | `test_elemwise.py:230` | ✓ Tested |
| SPPF pooling | `test_pool.py:68` | ✓ Tested |
| FPN upsampling | `test_resize.py:73` | ✓ Tested |
| Sigmoid attention | `test_elemwise.py:240` | ✓ Tested |
| C3k2 blocks | `test_batchnorm.py:78` | ✓ Tested |
| Conv2D filter flip | `test_regressions.py:45` | ✓ Tested |
| Switch dimension calc | `test_elemwise.py:201` | ✓ Tested |

**But NOT Tested**:
- ✗ Complete YOLO11n model in browser
- ✗ Composite ops from `_upsample()` in WebGPU
- ✗ Dynamic shape handling in WASM
- ✗ End-to-end inference in web runtime

### 6. The Testing Architecture

**Current Test Flow**:
```
PyTensor Function → ONNX Export → ONNX Runtime (CPU) → NumPy Comparison
                                      ↓
                              onnx.checker.check_model()
```

**What's Missing**:
```
PyTensor Function → ONNX Export → [Browser Runtime NOT TESTED]
                                      ↓
                              - WebGPU Execution
                              - WASM Execution
                              - JavaScript Integration
                              - Real-world Deployment
```

**Test Infrastructure** (`tests/link/onnx/`):

**Helper Functions**:
- `compare_onnx_and_py()` - Numerical validation helper (`test_basic.py:22`)
- `validate_onnx_graph_structure()` - Graph structure validator
- Hypothesis strategies - Property-based test generation (`strategies/`)

**Hypothesis Profiles** (`conftest.py`):
- **dev**: 10 examples, 500ms deadline (fast feedback)
- **ci**: 100 examples (CI pipelines)
- **thorough**: 1000 examples (release validation)
- **onnx**: 50 examples, 5s deadline (ONNX-specific)

**But No Browser Testing Infrastructure**:
- No Selenium/Puppeteer setup
- No JavaScript test harness
- No WebGPU emulation
- No WASM validation

### 7. Code References

**ONNX Export Implementation**:
- `pytensor/link/onnx/dispatch/elemwise.py:70-397` - Composite decomposition
- `pytensor/link/onnx/dispatch/elemwise.py:400-656` - Elemwise converter
- `pytensor/link/onnx/dispatch/basic.py:184-352` - Model export
- `pytensor/link/onnx/export.py:18` - Export API

**Test Files**:
- `tests/link/onnx/test_elemwise.py:21-247` - Elemwise operation tests
- `tests/link/onnx/test_basic.py:22-102` - `compare_onnx_and_py` helper
- `tests/link/onnx/test_properties.py:30` - Property-based tests
- `examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py:12-269` - YOLO export tests

**YOLO Model**:
- `examples/onnx/onnx-yolo-demo/yolo/model.py:249-286` - `_upsample()` method
- `examples/onnx/onnx-yolo-demo/yolo/model.py:352-382` - `build_yolo11n()`
- `examples/onnx/onnx-yolo-demo/site/webgpu_vs_wasm_benchmark.html:462-464` - Browser loading

**Analysis Scripts**:
- `examples/onnx/onnx-yolo-demo/export_model.py` - Simple export test
- `examples/onnx/onnx-yolo-demo/find_missing_ops.py` - Missing op detector
- `examples/onnx/onnx-yolo-demo/find_all_missing_ops.py` - Comprehensive analysis

## Architecture Insights

### Singledispatch-Based Export System

PyTensor uses Python's `functools.singledispatch` for extensible ONNX conversion:

```python
@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert PyTensor Op to ONNX."""
    raise NotImplementedError(...)

@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise ops."""
    if isinstance(op.scalar_op, scalar.Composite):
        return decompose_composite_elemwise(...)  # Returns list of nodes
    # ... handle regular operations
```

**Key Design Decisions**:

1. **Multi-Node Decomposition**: Handlers can return single node OR list of nodes
2. **Variable Name Generation**: `get_var_name()` closure maintains uniqueness
3. **Intermediate Variables**: Pattern `f"{operation}_{output_name}"` for temps
4. **Operation Chaining**: N-ary ops converted to binary chains (Add, Mul, etc.)
5. **Dtype Handling**: Automatic Cast insertion for type mismatches

### The Composite Naming Pattern

From `elemwise.py:123`:
```python
# For intermediate results in composite decomposition:
output_name = f"composite_tmp_{id(output_var)}"

# For final outputs:
output_idx = composite_op.outputs.index(output_var)
output_name = get_var_name(node.outputs[output_idx])
```

**Problem**: `id(output_var)` uses Python's object ID, which:
- Can be very large numbers (memory addresses)
- May not be deterministic across runs
- Creates long variable names
- May collide in complex graphs

## Key Findings: The Three-Level Testing Gap

### Gap Level 1: Runtime Environment (Critical)

**What's Tested**: ONNX Runtime CPU backend only
**What's Not Tested**: WebGPU, WASM, browser JavaScript runtime
**Impact**: ★★★★★ CRITICAL

**Evidence**:
```python
# ALL tests use this:
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

# NONE use this:
# - WebGPU provider
# - WASM provider
# - Browser execution
```

**Why This Matters**: The demo HTML loads the model in browser with WebGPU/WASM:
```javascript
// webgpu_vs_wasm_benchmark.html:462-464
webgpuSession = await ort.InferenceSession.create('yolo11n.onnx', {
    executionProviders: ['webgpu']  // NOT TESTED IN PYTENSOR
});
```

### Gap Level 2: Integration Testing (High)

**What's Tested**: Individual operations in isolation
**What's Not Tested**: Complete neural network models
**Impact**: ★★★★☆ HIGH

**Evidence**:
- 157 operation-level tests (`tests/link/onnx/test_*.py`)
- 8 YOLO export tests (`test_onnx_export.py`)
- But only 1 test validates the FULL model: `test_onnx_runtime_vs_pytensor_equivalence`
- That test still uses CPU backend only

**Why This Matters**:
- Operations work individually
- May fail when combined in complex graphs
- Composite decomposition only appears in real models
- State management issues only visible in integration

### Gap Level 3: Variable Naming (Medium)

**What's Tested**: ONNX spec compliance via `onnx.checker.check_model()`
**What's Not Tested**: Browser runtime naming constraints
**Impact**: ★★★☆☆ MEDIUM

**Evidence**:
```python
# Composite decomposition creates names like:
output_name = f"composite_tmp_{id(output_var)}"
# Example: "composite_tmp_2443371660016"

# Browser error mentions:
# "Add_composite_tmp_2443371660016"
```

**Why This Matters**:
- JavaScript has identifier length limits
- Variable scoping may differ between runtimes
- Name collisions possible with generated IDs
- Browser debuggers may truncate long names

## Recommendations

### Immediate Actions (Close Gap Level 1)

**1. Add Browser Runtime Tests**

Create new test file: `tests/link/onnx/test_browser_runtime.py`

```python
@pytest.mark.browser
@pytest.mark.skipif(not HAS_SELENIUM, reason="Selenium not installed")
def test_yolo11n_webgpu_inference():
    """Test YOLO11n runs in browser with WebGPU."""
    # Export model
    # Start local server
    # Launch browser with Selenium
    # Load model in WebGPU
    # Run inference
    # Compare results with CPU runtime
```

**2. Add ONNX Runtime Web Provider Tests**

If onnxruntime-web Python bindings exist:
```python
session = ort.InferenceSession(
    path,
    providers=["WebGpuExecutionProvider"]  # Test WebGPU provider
)
```

**3. Add End-to-End Integration Tests**

Create `test_yolo_e2e_integration.py`:
```python
def test_yolo11n_full_inference_pipeline():
    """Test complete YOLO11n: preprocess → model → postprocess."""
    # Load image
    # Preprocess (resize, normalize)
    # Run model (CPU, WebGPU, WASM)
    # Parse detections
    # Validate bounding boxes
```

### Medium-Term Actions (Close Gap Level 2)

**4. Improve Composite Variable Naming**

Change from:
```python
output_name = f"composite_tmp_{id(output_var)}"
```

To:
```python
# Use counter instead of id() for deterministic, short names
output_name = f"composite_tmp_{next(counter)}"
```

**5. Add Graph Validation for Browser Compatibility**

New validator: `validate_browser_compatibility(onnx_model)`:
- Check variable name lengths
- Validate identifier characters
- Ensure no reserved JavaScript keywords
- Verify operator support in onnxruntime-web

**6. Add Property-Based Integration Tests**

Extend Hypothesis tests to generate complete networks:
```python
@given(neural_network_architecture())
def test_random_network_exports_to_browser(network):
    """Property: Any network exports and runs in browser."""
    # Generate random network
    # Export to ONNX
    # Validate in browser runtime
```

### Long-Term Actions (Close Gap Level 3)

**7. Create Browser Testing Infrastructure**

Components needed:
- Selenium/Puppeteer test harness
- Local web server for model serving
- JavaScript assertion library
- WebGPU emulation for CI
- Cross-browser testing (Chrome, Firefox, Safari)

**8. Add Performance Validation**

Benchmark suite comparing:
- CPU vs WebGPU inference time
- WASM vs WebGPU performance
- Memory usage across runtimes
- FPS for video inference

**9. Document Browser Deployment Best Practices**

Create guide: `docs/onnx_browser_deployment.md`
- Known limitations of onnxruntime-web
- Recommended model architectures
- Performance optimization tips
- Debugging browser inference issues

## Related Research

- ONNX operation dispatch system: See `pytensor/link/onnx/dispatch/`
- PyTensor optimizer and fusion: See composite operation creation
- ONNX Runtime architecture: CPU vs Web providers
- WebGPU limitations: Operator support matrix

## Open Questions

1. **Why does the browser error occur specifically with Add operations?**
   - Need to inspect actual ONNX graph to see Add node connections
   - May be Add node with composite_tmp inputs/outputs
   - Could be issue with Add's broadcast semantics in WebGPU

2. **Is the error in model loading or inference?**
   - HTML shows error during session creation (line 462-464)
   - Suggests graph validation failure, not runtime execution error
   - Need browser console output to confirm

3. **Are there other operations with similar issues?**
   - Composite decomposition affects all Elemwise ops
   - Other operations: Mul, Sub, Div may have same naming issues
   - Need comprehensive browser testing to identify all failures

4. **Does the error occur with both WebGPU and WASM providers?**
   - HTML supports both providers (line 336-339)
   - Error message doesn't specify which provider failed
   - Need to test both separately

5. **Is this a PyTensor issue or onnxruntime-web issue?**
   - PyTensor generates ONNX that passes `onnx.checker.check_model()`
   - ONNX is spec-compliant
   - May be browser runtime parsing bug
   - Need to file issue with appropriate project

## Summary Table: Testing Coverage vs Reality

| Test Aspect | Current Coverage | Required Coverage | Gap |
|-------------|-----------------|-------------------|-----|
| **CPU Runtime** | ✓✓✓✓✓ Excellent | CPU not deployment target | N/A |
| **WebGPU Runtime** | ✗✗✗✗✗ None | ★★★★★ Critical | **CRITICAL GAP** |
| **WASM Runtime** | ✗✗✗✗✗ None | ★★★★★ Critical | **CRITICAL GAP** |
| **Operation Tests** | ✓✓✓✓✓ Excellent (157 tests) | Individual ops covered | Adequate |
| **Integration Tests** | ✗✗✗✓✓ Poor (1 e2e test) | ★★★★☆ High | **HIGH GAP** |
| **Browser Deployment** | ✗✗✗✗✗ None | ★★★★★ Critical | **CRITICAL GAP** |
| **Variable Naming** | ✓✓✓✗✗ Partial (spec only) | ★★★☆☆ Medium | **MEDIUM GAP** |
| **Performance** | ✗✗✗✗✗ None | ★★☆☆☆ Low | LOW GAP |

## Conclusion

**The testing gap is clear and critical**: PyTensor thoroughly validates ONNX export correctness using CPU runtime, but **never validates** that exported models actually work in the browser environment (WebGPU/WASM) where they're intended to run.

**The specific error** (`Add_composite_tmp_2443371660016`) indicates an issue with Composite operation decomposition creating intermediate variables with generated names that may violate browser runtime constraints or cause parsing failures.

**The solution path**:
1. **Immediate**: Add browser runtime tests using Selenium + onnxruntime-web
2. **Short-term**: Improve composite variable naming with deterministic, short names
3. **Long-term**: Build comprehensive browser testing infrastructure

**Impact**: Without browser testing, PyTensor cannot reliably support ONNX deployment to web applications, despite extensive CPU-based validation. This is a **critical gap** for production use of PyTensor-exported models in browser environments.
