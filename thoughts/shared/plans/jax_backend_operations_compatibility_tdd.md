# JAX Backend Operations Compatibility TDD Implementation Plan

## Overview

This plan addresses the critical JAX backend tracer compatibility issues preventing GPU training of YOLO models. We'll implement comprehensive tests for all 31+ operations that fail with JAX tracers, then fix each operation systematically using TDD principles.

## Current State Analysis

The PyTensor JAX backend has fundamental compatibility issues with JAX's JIT compilation when operations receive traced (abstract) values instead of concrete integers for shape parameters, indices, and control flow. This prevents gradient computation and GPU training.

### Current Testing Landscape:
- Testing framework: pytest
- Available test utilities: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()`
- Existing test patterns to follow: Split tests in `test_tensor_basic.py:95-189`, Resize tests in `test_resize.py`
- Test fixtures/mocks available: `jax_mode`, `py_mode` configurations

## Desired End State

All PyTensor operations used in YOLO models should either:
1. Correctly handle JAX tracers during JIT compilation, OR
2. Provide clear, actionable error messages at graph construction time (not runtime)

After implementation, the gradient flow test in `test_jax_backend.py:95-135` should pass, enabling GPU training.

### Key Discoveries:
- Split operation fails at `pytensor/link/jax/dispatch/tensor_basic.py:139` with traced indices
- Resize fails at `pytensor/link/jax/dispatch/resize.py:67-68` with int() on tracers
- 31+ operations need fixes or better error handling
- JAXLinker can mark parameters as static at `pytensor/link/jax/linker.py:76-113`

## What We're NOT Testing/Implementing

- Operations not used in YOLO models (can be addressed later)
- NumPy backend compatibility (already works)
- Performance optimizations (focus on correctness first)
- Operations that already work correctly (Shape, Join, MakeVector, Pool)

## TDD Approach

Write comprehensive tests that define correct behavior, verify they fail with informative messages, then implement fixes by making tests pass. Each operation gets tests for forward pass, gradient computation, and error handling.

### Test Design Philosophy:
- Tests should fail with clear, diagnostic messages that guide implementation
- Each test should validate one specific aspect of tracer compatibility
- Gradient tests ensure end-to-end training capability
- Error tests ensure graceful failure for unsupported patterns

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive, informative tests that define the feature completely. These tests should fail in expected, diagnostic ways.

### Test Categories:

#### 1. Critical Operation Compatibility Tests
**Test File**: `tests/link/jax/test_jax_tracer_compatibility.py`
**Purpose**: Verify critical operations handle tracers correctly during JIT compilation

**Test Cases to Write:**

##### Test: `test_split_with_dynamic_splits`
**Purpose**: Verify Split handles traced split positions correctly
**Test Data**: Matrix with dynamic dimension, computed split positions
**Expected Behavior**: Should either work with tracers or raise clear error at graph construction
**Assertions**: Output shapes match expected splits, gradients flow

```python
def test_split_with_dynamic_splits():
    """
    Test that Split operation handles dynamic split positions.

    This test verifies:
    - Split works when positions come from shape computations
    - Gradients can flow through the operation
    - Clear error if not supported
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - dynamic splits from shape
    x = pt.matrix("x", shape=(None, 60))  # Dynamic first dim, fixed second
    split_size = x.shape[1] // 3  # Should be 20
    splits = pt.stack([split_size, split_size, split_size])

    # Act
    y1, y2, y3 = pt.split(x, splits, n_splits=3, axis=1)

    # Compute gradients
    loss = y1.sum() + y2.sum() * 2 + y3.sum() * 3
    grad_x = pt.grad(loss, x)

    # Test data
    x_val = np.random.randn(4, 60).astype("float32")

    # Assert - should work or raise NotImplementedError
    try:
        compare_jax_and_py([x], [y1, y2, y3, grad_x], [x_val])
    except NotImplementedError as e:
        assert "constant split positions" in str(e)
```

**Expected Failure Mode**: Currently raises `ConcretizationTypeError` at runtime
- Error type: `jax.errors.ConcretizationTypeError`
- Expected message: Should be `NotImplementedError` at graph construction instead

##### Test: `test_resize_with_dynamic_dimensions`
**Purpose**: Verify Resize handles traced dimensions correctly
**Test Data**: Images with dynamic batch size
**Expected Behavior**: Upsampling should work with traced shapes
**Assertions**: Output shape is correct, gradients flow

```python
def test_resize_with_dynamic_dimensions():
    """
    Test that Resize operation handles dynamic input dimensions.

    This test verifies:
    - Resize works when input has dynamic dimensions
    - Scale factors apply correctly to traced shapes
    - Gradients flow through the operation
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.nnet.abstract_conv import bilinear_upsampling
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - dynamic batch size
    x = pt.tensor4("x", shape=(None, 3, 32, 32))

    # Act - upsample by 2x
    y = bilinear_upsampling(x, 2)

    # Compute gradient
    loss = y.sum()
    grad_x = pt.grad(loss, x)

    # Test data
    x_val = np.random.randn(2, 3, 32, 32).astype("float32")

    # Assert
    compare_jax_and_py([x], [y, grad_x], [x_val])

    # Verify output shape
    assert y.eval({x: x_val}).shape == (2, 3, 64, 64)
```

**Expected Failure Mode**: `TracerIntegerConversionError` at `int(height * scale_h)`
- Error type: `jax.errors.TracerIntegerConversionError`
- Expected message: Error when converting tracer to int

##### Test: `test_alloc_with_dynamic_shape`
**Purpose**: Verify Alloc operations handle traced shapes
**Test Data**: Shape derived from input dimensions
**Expected Behavior**: Should allocate with dynamic shape or error clearly
**Assertions**: Output has correct shape, values broadcast correctly

```python
def test_alloc_with_dynamic_shape():
    """
    Test that Alloc handles dynamic shapes from traced dimensions.

    This test verifies:
    - Alloc can broadcast to shapes with traced dimensions
    - AllocEmpty can create arrays with traced shapes
    - Clear errors if not supported
    """
    import pytensor.tensor as pt
    import pytensor.tensor.basic as ptb
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.matrix("x", shape=(None, None))
    value = pt.scalar("value")

    # Act - Alloc with dynamic shape
    shape = pt.stack([x.shape[0], x.shape[1] * 2])
    y = ptb.alloc(value, shape[0], shape[1])

    # AllocEmpty with dynamic shape
    z = ptb.AllocEmpty("float32")(x.shape[0], 10)

    # Test data
    x_val = np.random.randn(3, 4).astype("float32")
    value_val = np.array(5.0, dtype="float32")

    # Assert
    compare_jax_and_py([x, value], [y, z], [x_val, value_val])
```

**Expected Failure Mode**: `ConcretizationTypeError` at `jnp.broadcast_to()`
- Error type: `jax.errors.ConcretizationTypeError`
- Expected message: Abstract tracer value where concrete expected

#### 2. Gradient Flow Integration Tests
**Test File**: `tests/link/jax/test_jax_gradient_flow.py`
**Purpose**: Verify end-to-end gradient flow through operation chains

##### Test: `test_yolo_c3k2_gradient_flow`
**Purpose**: Test gradient flow through C3k2 block (concatenation pattern)
**Test Data**: Feature maps at different scales
**Expected Behavior**: Gradients flow through split/concat operations
**Assertions**: All parameter gradients are non-zero and finite

```python
def test_yolo_c3k2_gradient_flow():
    """
    Test gradient flow through YOLO C3k2 block pattern.

    This test verifies:
    - Gradients flow through concatenation operations
    - Split operations (if introduced by optimizer) work
    - No gradient vanishing or explosion
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - simulate C3k2 pattern
    x = pt.tensor4("x", shape=(None, 256, 32, 32))

    # Split channels (this is what optimizer might introduce)
    x1 = x[:, :128]
    x2 = x[:, 128:]

    # Process separately (simplified)
    y1 = x1 * 2.0 + 1.0
    y2 = x2 * 3.0 - 1.0

    # Concatenate back
    y = pt.concatenate([y1, y2], axis=1)

    # Compute loss and gradient
    loss = y.sum()
    grad_x = pt.grad(loss, x)

    # Test data
    x_val = np.random.randn(2, 256, 32, 32).astype("float32")

    # Assert
    fn, (loss_val, grad_val) = compare_jax_and_py(
        [x], [loss, grad_x], [x_val]
    )

    # Verify gradient properties
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert np.all(np.isfinite(grad_val)), "Gradient contains NaN or Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is all zeros"
```

##### Test: `test_full_yolo_forward_backward`
**Purpose**: Test complete YOLO model gradient computation
**Test Data**: Full resolution input images
**Expected Behavior**: Gradients flow through entire model
**Assertions**: All layer gradients computed successfully

```python
def test_full_yolo_forward_backward():
    """
    Test gradient flow through complete YOLO model.

    This test verifies:
    - Full model compiles with JAX
    - Gradients flow through all layers
    - No operations block gradient computation
    """
    import pytensor.tensor as pt
    import numpy as np
    from yolo.model import YOLO11n
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)

    # Act - forward pass
    det_p3, det_p4, det_p5 = model(x)

    # Compute loss (simplified)
    loss = det_p3.sum() + det_p4.sum() + det_p5.sum()

    # Get all model parameters
    params = [
        p for p in model.backbone.params + model.head.params
        if hasattr(p, "name")
    ]

    # Compute gradients for all parameters
    grads = pt.grad(loss, params)

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x], [loss] + grads, [x_val]
    )

    loss_val = outputs[0]
    grad_vals = outputs[1:]

    # Verify all gradients computed
    assert len(grad_vals) == len(params)
    for i, (param, grad_val) in enumerate(zip(params, grad_vals)):
        assert grad_val is not None, f"Gradient for {param.name} is None"
        assert np.all(np.isfinite(grad_val)), f"Gradient for {param.name} has NaN/Inf"
```

#### 3. Static Parameter Detection Tests
**Test File**: `tests/link/jax/test_jax_static_detection.py`
**Purpose**: Verify operations correctly identify parameters that need to be static

##### Test: `test_static_parameter_detection`
**Purpose**: Verify JAXLinker detects and marks static parameters
**Test Data**: Graphs with shape parameters
**Expected Behavior**: Shape inputs marked as static_argnums
**Assertions**: Correct parameters marked static

```python
def test_static_parameter_detection():
    """
    Test that JAXLinker correctly identifies static parameters.

    This test verifies:
    - Shape inputs are detected as needing static treatment
    - Static parameters are properly marked in jax.jit
    - Dynamic parameters are not marked static
    """
    import pytensor
    import pytensor.tensor as pt
    from pytensor.link.jax.dispatch.shape import JAXShapeTuple
    from pytensor.link.jax import JAXLinker

    # Arrange - graph with shape parameter
    shape = pt.vector("shape", dtype="int64")
    x = pt.tensor("x", shape=(None,))

    # Use shape in reshape (requires concrete value)
    y = x.reshape(shape)

    # Act - compile with JAX
    fn = pytensor.function([x, shape], y, mode="JAX")

    # Look at the compiled function's static_argnums
    jax_linker = fn.maker.linker
    if hasattr(jax_linker, 'static_argnums'):
        static_argnums = jax_linker.static_argnums
    else:
        # Extract from jitted function
        import jax
        jitted_fn = fn.vm.jit_fn  # Access the jitted function
        static_argnums = jitted_fn.keywords.get('static_argnums', [])

    # Assert - shape parameter should be marked static
    assert 1 in static_argnums, "Shape parameter not marked as static"
```

#### 4. Error Handling Tests
**Test File**: `tests/link/jax/test_jax_error_messages.py`
**Purpose**: Verify operations provide clear, actionable error messages

##### Test: `test_clear_error_messages`
**Purpose**: Verify operations raise clear errors for unsupported patterns
**Test Data**: Various unsupported input patterns
**Expected Behavior**: NotImplementedError with helpful message at graph construction
**Assertions**: Error message guides user to solution

```python
@pytest.mark.parametrize("op_name,create_op,error_pattern", [
    ("ARange",
     lambda: pt.arange(pt.scalar() * 2),
     "JAX requires the arguments of.*arange.*to be constants"),
    ("Reshape",
     lambda: pt.vector().reshape((pt.scalar() * 2, 3)),
     "JAX requires concrete values for the.*shape.*parameter"),
    ("Split",
     lambda: pt.split(pt.matrix(), [pt.scalar()] * 3, n_splits=3),
     "Split node does not have constant split positions"),
])
def test_clear_error_messages(op_name, create_op, error_pattern):
    """
    Test that operations provide clear error messages for unsupported patterns.

    This test verifies:
    - Errors are raised at graph construction, not runtime
    - Error messages clearly explain the limitation
    - Messages suggest alternatives when possible
    """
    import re
    import pytest
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    with pytest.raises(NotImplementedError) as exc_info:
        # Act
        out = create_op()
        # Try to compile
        compare_jax_and_py([], [out], [])

    # Assert
    assert re.search(error_pattern, str(exc_info.value)), \
        f"Error message doesn't match expected pattern for {op_name}"
```

### Test Implementation Steps:

1. **Create test file structure**:
   ```bash
   tests/link/jax/
   ├── test_jax_tracer_compatibility.py
   ├── test_jax_gradient_flow.py
   ├── test_jax_static_detection.py
   └── test_jax_error_messages.py
   ```

2. **Import necessary testing utilities**:
   ```python
   import numpy as np
   import pytest
   import pytensor
   import pytensor.tensor as pt
   import pytensor.tensor.basic as ptb
   from pytensor import grad
   from pytensor.configdefaults import config
   from tests.link.jax.test_basic import compare_jax_and_py

   jax = pytest.importorskip("jax")
   ```

3. **Create test fixtures for common data**:
   ```python
   @pytest.fixture
   def sample_tensor():
       """Fixture providing sample tensor data."""
       return np.random.randn(2, 3, 32, 32).astype("float32")

   @pytest.fixture
   def dynamic_shape_graph():
       """Fixture providing graph with dynamic shapes."""
       x = pt.tensor4("x", shape=(None, 3, None, None))
       return x
   ```

4. **Implement each test case** with clear arrange-act-assert structure

5. **Add comprehensive docstrings** explaining what each test validates

### Success Criteria:

#### Automated Verification:
- [ ] All test files created with proper structure
- [ ] Tests use existing test utilities correctly
- [ ] Test code follows project conventions: `pytest tests/link/jax/test_jax_*.py`
- [ ] Tests are discoverable: `pytest --collect-only tests/link/jax/test_jax_*`

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Assertion messages are diagnostic
- [ ] Test code is readable and maintainable

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and verify they fail in the expected, diagnostic ways. This ensures our tests are actually testing something and will catch regressions.

### Verification Steps:

1. **Run the compatibility test suite**:
   ```bash
   pytest tests/link/jax/test_jax_tracer_compatibility.py -v
   ```

2. **Run the gradient flow tests**:
   ```bash
   pytest tests/link/jax/test_jax_gradient_flow.py -v
   ```

3. **Run the error handling tests**:
   ```bash
   pytest tests/link/jax/test_jax_error_messages.py -v
   ```

4. **For each test, verify**:
   - Test fails (not passes or errors unexpectedly)
   - Failure message is informative
   - Failure points to the right location in dispatch code
   - Error type matches expectations

### Expected Failures:

For each test, document what we expect:

- **test_split_with_dynamic_splits**:
  - Expected: `jax.errors.ConcretizationTypeError` at `jnp.split()`
  - Points to: `pytensor/link/jax/dispatch/tensor_basic.py:139`

- **test_resize_with_dynamic_dimensions**:
  - Expected: `jax.errors.TracerIntegerConversionError` at `int(height * scale_h)`
  - Points to: `pytensor/link/jax/dispatch/resize.py:67`

- **test_alloc_with_dynamic_shape**:
  - Expected: `jax.errors.ConcretizationTypeError` at `jnp.broadcast_to()`
  - Points to: `pytensor/link/jax/dispatch/tensor_basic.py:46`

- **test_yolo_c3k2_gradient_flow**:
  - Expected: Failure during gradient computation
  - Shows: Cannot compute gradients through Split operation

- **test_full_yolo_forward_backward**:
  - Expected: Compilation failure or gradient computation error
  - Shows: Specific operation blocking gradient flow

### Success Criteria:

#### Automated Verification:
- [ ] All tests run and are discovered: `pytest --collect-only tests/link/jax/test_jax_*`
- [ ] All tests fail (none pass): `pytest tests/link/jax/test_jax_* --tb=short`
- [ ] No unexpected import or syntax errors: `pytest tests/link/jax/test_jax_* --tb=line`

#### Manual Verification:
- [ ] Each test fails with expected error type
- [ ] Failure messages clearly indicate what's missing
- [ ] Failure messages would help during implementation
- [ ] Stack traces point to relevant dispatch code locations
- [ ] No cryptic or misleading error messages

### Adjustment Phase:

If tests don't fail properly:
- [ ] Fix tests that pass unexpectedly (too lenient)
- [ ] Fix tests with confusing error messages
- [ ] Fix tests that error instead of fail (missing imports, etc.)
- [ ] Improve assertion messages for clarity

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement the feature by making tests pass, one at a time. Work like debugging - let the test failures guide implementation.

### Implementation Strategy:

**Order of Implementation:**
1. Start with Split operation (most critical, blocks optimizer)
2. Then Resize (needed for upsampling)
3. Then Alloc/AllocEmpty (fundamental operations)
4. Continue with other operations based on YOLO usage frequency

### Implementation Steps:

#### Implementation 1: Fix Split Operation

**Target Test**: `test_split_with_dynamic_splits`
**Current Failure**: `ConcretizationTypeError` at runtime

**Changes Required:**

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Changes**: Improve constant detection and error handling

```python
@jax_funcify.register(Split)
def jax_funcify_Split(op: Split, node, **kwargs):
    _, axis, splits = node.inputs

    # Check if axis is constant
    try:
        constant_axis = get_scalar_constant_value(axis)
    except NotScalarConstantError:
        constant_axis = None
        warnings.warn(
            "Split node does not have constant axis. "
            "JAX implementation will likely fail during JIT compilation.",
            UserWarning
        )

    # Check if splits are constant
    try:
        constant_splits = np.array([
            get_scalar_constant_value(splits[i])
            for i in range(get_vector_length(splits))
        ])
    except (ValueError, NotScalarConstantError):
        constant_splits = None
        # Raise error at graph construction time
        raise NotImplementedError(
            "JAX Split requires constant split positions. "
            "The split sizes in your graph come from dynamic computations "
            "and cannot be JIT-compiled by JAX. Consider using constant "
            "split sizes or reshaping operations instead."
        )

    def split(x, axis, splits):
        # Use pre-extracted constants
        if constant_axis is not None:
            axis = constant_axis

        if constant_splits is not None:
            splits = constant_splits
            cumsum_splits = np.cumsum(splits[:-1])
        else:
            # Should never reach here due to NotImplementedError above
            raise RuntimeError("Dynamic splits should have been caught earlier")

        # Validate at runtime
        if len(splits) != op.len_splits:
            raise ValueError(f"Length of splits {len(splits)} != n_splits {op.len_splits}")

        if (splits < 0).any():
            raise ValueError("Split sizes cannot be negative")

        return jnp.split(x, cumsum_splits, axis=axis)

    return split
```

**Debugging Approach:**
1. Run the test: `pytest tests/link/jax/test_jax_tracer_compatibility.py::test_split_with_dynamic_splits -v`
2. Verify NotImplementedError raised at graph construction
3. Check error message is clear and actionable
4. Implement alternative using `jax.lax.dynamic_slice` if needed

**Success Criteria:**

##### Automated Verification:
- [ ] Target test passes: `pytest tests/link/jax/test_jax_tracer_compatibility.py::test_split_with_dynamic_splits -v`
- [ ] No regressions in existing Split tests: `pytest tests/link/jax/test_tensor_basic.py::TestJaxSplit -v`
- [ ] Linting passes: `make lint`

##### Manual Verification:
- [ ] Error raised at graph construction, not runtime
- [ ] Error message is clear and suggests alternatives
- [ ] Code follows project conventions

#### Implementation 2: Fix Resize Operation

**Target Test**: `test_resize_with_dynamic_dimensions`
**Current Failure**: `TracerIntegerConversionError` at `int(height * scale_h)`

**Changes Required:**

**File**: `pytensor/link/jax/dispatch/resize.py`
**Changes**: Replace Python int() with JAX-compatible operations

```python
def resize_nearest(input):
    batch, channels, height, width = input.shape

    # Use JAX operations instead of Python int()
    out_height = jnp.round(height * scale_h).astype(jnp.int32)
    out_width = jnp.round(width * scale_w).astype(jnp.int32)

    # Create coordinate grids using JAX operations
    out_h_coords = jnp.floor(
        jnp.arange(out_height, dtype=jnp.float32) / scale_h
    ).astype(jnp.int32)
    out_w_coords = jnp.floor(
        jnp.arange(out_width, dtype=jnp.float32) / scale_w
    ).astype(jnp.int32)

    # Clip coordinates to valid range
    out_h_coords = jnp.clip(out_h_coords, 0, height - 1)
    out_w_coords = jnp.clip(out_w_coords, 0, width - 1)

    # Use advanced indexing
    return input[:, :, out_h_coords[:, None], out_w_coords[None, :]]
```

**Similar changes for resize_linear function**

#### Implementation 3: Fix Alloc Operations

**Target Test**: `test_alloc_with_dynamic_shape`
**Current Failure**: `ConcretizationTypeError` at `jnp.broadcast_to()`

**Changes Required:**

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Changes**: Add shape validation and use JAX-compatible broadcast

```python
@jax_funcify.register(Alloc)
def jax_funcify_Alloc(op, node, **kwargs):
    _, *shape_args = node.inputs

    # Validate shape arguments
    for i, shape_arg in enumerate(shape_args):
        if not _is_jax_shape_compatible(shape_arg):
            raise NotImplementedError(
                f"JAX Alloc requires concrete or Shape-derived dimensions. "
                f"Dimension {i} comes from {shape_arg.owner.op if shape_arg.owner else 'input'} "
                f"which is not supported. Use constant dimensions or Shape ops."
            )

    def alloc(x, *shape):
        # Use JAX operations that handle tracers
        x_expanded = jnp.ones(shape, dtype=x.dtype) * x
        Alloc._check_runtime_broadcast(node, jnp.asarray(x), x_expanded.shape)
        return x_expanded

    return alloc

def _is_jax_shape_compatible(shape_var):
    """Check if shape variable is JAX-compatible."""
    if isinstance(shape_var, Constant):
        return True
    if shape_var.owner and isinstance(shape_var.owner.op, (Shape, Shape_i, JAXShapeTuple)):
        return True
    return False
```

### Complete Feature Implementation:

Once individual operations are fixed:

**Integration Testing:**
- Run full YOLO gradient test: `pytest tests/onnx/onnx-yolo-demo/tests/test_jax_backend.py::test_jax_gradient_flow -v`
- Check all operation tests: `pytest tests/link/jax/test_jax_tracer_compatibility.py -v`
- Verify no regressions: `pytest tests/link/jax/ -v`

**Success Criteria:**

##### Automated Verification:
- [ ] All new tests pass: `pytest tests/link/jax/test_jax_* -v`
- [ ] YOLO gradient test passes: `pytest examples/onnx/onnx-yolo-demo/tests/test_jax_backend.py::test_jax_gradient_flow -v`
- [ ] No regressions in existing tests: `pytest tests/link/jax/ -v`
- [ ] Code coverage acceptable: `pytest --cov=pytensor.link.jax.dispatch tests/link/jax/`
- [ ] Linting passes: `make lint`

##### Manual Verification:
- [ ] All operations handle tracers or fail gracefully
- [ ] Error messages are clear and actionable
- [ ] Code is maintainable and documented
- [ ] Performance is acceptable

---

## Phase 4: Refactoring & Cleanup

### Overview
Now that tests are green, refactor to improve code quality while keeping tests passing.

### Refactoring Targets:

1. **Code Duplication**:
   - Extract common shape validation logic
   - Create shared utilities for constant extraction
   - Consolidate error messages

2. **Code Clarity**:
   - Improve variable names in dispatch functions
   - Add comments explaining tracer handling
   - Simplify complex conditional logic

3. **Performance**:
   - Cache constant extractions where possible
   - Optimize JAX operations for common cases
   - Remove unnecessary validations

4. **Test Quality**:
   - Extract common test patterns to fixtures
   - Reduce test duplication with parametrization
   - Improve test documentation

### Refactoring Steps:

1. **Extract common utilities**:

   Create `pytensor/link/jax/dispatch/utils.py`:
   ```python
   def validate_concrete_value(var, param_name, op_name):
       """Validate that a variable has a concrete value for JAX."""
       try:
           return get_scalar_constant_value(var)
       except NotScalarConstantError:
           raise NotImplementedError(
               f"JAX {op_name} requires concrete value for {param_name}. "
               f"Got dynamic value from {var.owner.op if var.owner else 'input'}."
           )

   def is_jax_shape_compatible(shape_var):
       """Check if shape variable is JAX-compatible."""
       if isinstance(shape_var, Constant):
           return True
       if shape_var.owner and isinstance(shape_var.owner.op, (Shape, Shape_i, JAXShapeTuple)):
           return True
       return False
   ```

2. **Refactor dispatch operations** to use utilities

3. **Improve test organization** with shared fixtures

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass: `pytest tests/link/jax/ -v`
- [ ] Code coverage maintained: `pytest --cov=pytensor.link.jax.dispatch tests/link/jax/`
- [ ] Linting passes: `make lint`
- [ ] No performance regressions in benchmarks

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] No unnecessary complexity added
- [ ] Functions have single responsibilities
- [ ] Comments explain complex tracer handling

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] All 31+ operations have tracer compatibility tests
- [ ] Critical operations have gradient tests
- [ ] Error paths have clear message tests
- [ ] Integration tests verify end-to-end functionality

### Test Organization:
- Test files: `tests/link/jax/test_jax_*.py`
- Fixtures: Defined in each test file or shared in `conftest.py`
- Test utilities: Import from `test_basic.py`
- Test data: Generated with numpy.random.default_rng

### Running Tests:

```bash
# Run all JAX compatibility tests
pytest tests/link/jax/test_jax_tracer_compatibility.py -v

# Run specific operation test
pytest tests/link/jax/test_jax_tracer_compatibility.py::test_split_with_dynamic_splits -v

# Run with coverage
pytest tests/link/jax/ --cov=pytensor.link.jax.dispatch --cov-report=term-missing

# Run gradient flow tests
pytest tests/link/jax/test_jax_gradient_flow.py -v

# Run YOLO integration test
pytest examples/onnx/onnx-yolo-demo/tests/test_jax_backend.py::test_jax_gradient_flow -v
```

## Performance Considerations

- JAX-compatible operations may be slightly slower than specialized implementations
- Static parameter detection adds compilation overhead but enables JIT
- Trade-off: 5-10% performance reduction for full GPU acceleration capability

### Performance Testing:
- [ ] Benchmark before and after changes
- [ ] Profile JAX compilation time
- [ ] Measure gradient computation speed
- [ ] Compare GPU vs CPU performance

## Migration Notes

For existing code using affected operations:
1. Operations that can't handle tracers will raise clear errors
2. Errors occur at graph construction, not runtime
3. Migration path: use constant values or Shape operations
4. Alternative: disable specific optimizations with `optimizer_excluding`

## References

- Original ticket: `thoughts/shared/research/2025-10-23_10-48-47_jax-backend-operations-rewrite-requirements.md`
- Related research: `thoughts/shared/research/2025-10-23_10-20-02_jax-attributeerror-error-repr.md`
- Current test file: `examples/onnx/onnx-yolo-demo/tests/test_jax_backend.py`
- JAX dispatch implementations: `pytensor/link/jax/dispatch/*.py`
- Test utilities: `tests/link/jax/test_basic.py:36-96`