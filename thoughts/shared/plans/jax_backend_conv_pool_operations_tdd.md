# JAX Backend Conv2D and Pool2D Operations TDD Implementation Plan

## Overview

This plan addresses the critical missing Conv2D and Pool2D operations for JAX backend compatibility. Conv2D is used 22+ times in YOLO11, making it the most critical missing operation. Pool2D is essential for the SPPF block. We'll implement comprehensive tests first, verify they fail properly, then implement JAX-compatible dispatch for these operations.

## Current State Analysis

Conv2D and Pool2D operations are not currently tested for JAX tracer compatibility despite being fundamental to YOLO11. Conv2D appears in every ConvBNSiLU block, while Pool2D is critical for multi-scale feature extraction in SPPF.

### Current Testing Landscape:
- Testing framework: pytest
- Available test utilities: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()`
- Existing test patterns: Conv tests in `test_conv.py`, Pool tests in `test_pool.py`
- Test fixtures/mocks available: `jax_mode`, `py_mode` configurations

## Desired End State

Conv2D and Pool2D operations should work seamlessly with JAX tracers during JIT compilation, enabling gradient flow through all convolution and pooling layers in YOLO11. This includes proper handling of dynamic batch sizes and gradient computation.

### Key Discoveries:
- Conv2D implementation at `pytensor.tensor.conv.abstract_conv`
- Pool2D implementation at `pytensor.tensor.pool`
- Existing JAX dispatch may not handle dynamic batch dimensions
- SPPF uses cascaded 5x5 max pooling with stride=1

## What We're NOT Testing/Implementing

- 3D convolutions (Conv3D) - not used in YOLO11
- Average pooling variations not used in YOLO
- Dilated convolutions (not used in this model)
- Depthwise separable convolutions

## TDD Approach

Write comprehensive tests for Conv2D and Pool2D with dynamic shapes, verify they fail with clear messages indicating tracer issues, then implement JAX-compatible dispatch to make tests pass.

### Test Design Philosophy:
- Each test validates specific aspect of tracer compatibility
- Tests should reveal exactly what fails with tracers
- Gradient tests ensure end-to-end training capability
- Clear failure messages guide implementation

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define correct Conv2D and Pool2D behavior with JAX tracers. These tests should fail in expected, diagnostic ways.

### Test Categories:

#### 1. Conv2D Basic Operation Tests
**Test File**: `tests/link/jax/test_jax_conv2d_tracer.py`
**Purpose**: Verify Conv2D handles dynamic batch sizes and computes gradients correctly

**Test Cases to Write:**

##### Test: `test_conv2d_with_dynamic_batch`
**Purpose**: Verify Conv2D works with dynamic batch dimension
**Test Data**: Input with dynamic batch, fixed spatial dimensions
**Expected Behavior**: Conv2D should handle traced batch dimension
**Assertions**: Output shape correct, gradients flow

```python
def test_conv2d_with_dynamic_batch():
    """
    Test Conv2D with dynamic batch sizes during JIT compilation.

    This test verifies:
    - Dynamic batch dimension handling
    - Weight gradient computation
    - Proper shape inference with tracers
    - Gradient flow through convolution layers
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.conv import conv2d
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - dynamic batch size
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 32, 32))
    filters = pt.tensor4("filters", dtype="float32")

    # Act - basic convolution
    out = conv2d(x, filters, border_mode="valid")

    # Compute gradients
    loss = out.sum()
    grad_x = grad(loss, x)
    grad_filters = grad(loss, filters)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 32, 32)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    # Assert
    compare_jax_and_py(
        [x, filters],
        [out, grad_x, grad_filters],
        [x_val, filters_val]
    )

    # Verify output shape
    out_val = out.eval({x: x_val, filters: filters_val})
    assert out_val.shape == (2, 16, 30, 30), f"Unexpected shape: {out_val.shape}"
```

**Expected Failure Mode**: Potential tracer issues with shape computation
- Error type: May pass if Conv2D already handles tracers
- Expected message: Should work or raise clear NotImplementedError

##### Test: `test_conv2d_with_padding`
**Purpose**: Test Conv2D with "same" and numeric padding
**Test Data**: Various padding configurations
**Expected Behavior**: Padding should work with dynamic dimensions
**Assertions**: Output maintains spatial dimensions with "same" padding

```python
@pytest.mark.parametrize("padding,expected_shape", [
    ("same", (2, 16, 32, 32)),
    ((1, 1), (2, 16, 32, 32)),
    ((2, 3), (2, 16, 34, 36)),
])
def test_conv2d_with_padding(padding, expected_shape):
    """
    Test Conv2D with various padding modes.

    This test verifies:
    - "same" padding maintains dimensions
    - Numeric padding works correctly
    - Gradients flow through padded convolutions
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.conv import conv2d
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32")
    filters = pt.tensor4("filters", dtype="float32")

    # Act
    out = conv2d(x, filters, border_mode=padding)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 32, 32)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    # Assert
    fn, (out_val,) = compare_jax_and_py(
        [x, filters],
        [out],
        [x_val, filters_val]
    )

    assert out_val.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {out_val.shape}"
```

##### Test: `test_conv2d_with_stride`
**Purpose**: Test strided convolutions
**Test Data**: Conv2D with stride=2 (common in downsampling)
**Expected Behavior**: Stride should work with tracers
**Assertions**: Output dimensions reduced correctly

```python
def test_conv2d_with_stride():
    """
    Test Conv2D with stride > 1 (downsampling).

    This test verifies:
    - Strided convolution with dynamic batch
    - Correct output shape computation
    - Gradient flow through strided conv
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.conv import conv2d
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 32, 32))
    filters = pt.tensor4("filters", dtype="float32")

    # Act - stride=2 convolution
    out = conv2d(x, filters, subsample=(2, 2), border_mode="valid")

    # Compute gradient
    loss = out.sum()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 32, 32)).astype("float32")
    filters_val = rng.normal(size=(16, 3, 3, 3)).astype("float32")

    # Assert
    fn, (out_val, grad_val) = compare_jax_and_py(
        [x, filters],
        [out, grad_x],
        [x_val, filters_val]
    )

    # Verify downsampled shape
    assert out_val.shape == (2, 16, 15, 15), \
        f"Expected stride=2 output shape (2, 16, 15, 15), got {out_val.shape}"

    # Verify gradient shape matches input
    assert grad_val.shape == x_val.shape
```

#### 2. Pool2D Operation Tests
**Test File**: `tests/link/jax/test_jax_pool2d_tracer.py`
**Purpose**: Verify Pool2D operations handle dynamic dimensions

**Test Cases to Write:**

##### Test: `test_pool2d_with_dynamic_dimensions`
**Purpose**: Test max pooling with dynamic batch size
**Test Data**: 5x5 pooling as used in SPPF
**Expected Behavior**: Pooling should work with traced dimensions
**Assertions**: Output shape correct, gradients flow

```python
def test_pool2d_with_dynamic_dimensions():
    """
    Test max pooling (5x5, stride=1) with dynamic dimensions.

    This test verifies:
    - Cascaded pooling as in SPPF
    - Gradient flow through pooling
    - Padding handling with tracers
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.pool import pool_2d
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - dynamic batch
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # Act - SPPF-style cascaded pooling
    pool1 = pool_2d(x, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), mode="max", padding=(2, 2))

    # Concatenate (simplified - just sum for gradient test)
    out = pool1 + pool2 + pool3

    # Compute gradient
    loss = out.sum()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 256, 32, 32)).astype("float32")

    # Assert
    compare_jax_and_py(
        [x],
        [pool1, pool2, pool3, out, grad_x],
        [x_val]
    )

    # Verify shapes preserved with padding
    for pool_out in [pool1, pool2, pool3]:
        pool_val = pool_out.eval({x: x_val})
        assert pool_val.shape == (2, 256, 32, 32), \
            f"Pool output shape {pool_val.shape} != input shape"
```

**Expected Failure Mode**: Should work if Pool2D dispatch handles tracers
- Error type: Likely passes as Pool2D was mentioned as working
- Expected message: None

##### Test: `test_pool2d_gradient_flow`
**Purpose**: Verify gradients flow correctly through max pooling
**Test Data**: Small feature maps to test gradient routing
**Expected Behavior**: Gradients should route to max elements
**Assertions**: Non-max positions get zero gradient

```python
def test_pool2d_gradient_flow():
    """
    Test gradient flow through max pooling.

    This test verifies:
    - Gradients route only to max elements
    - Correct gradient shape
    - No gradient explosion/vanishing
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.pool import pool_2d
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - small input for clear gradient routing
    x = pt.tensor4("x", dtype="float32")

    # Act - 2x2 max pooling
    out = pool_2d(x, ws=(2, 2), stride=(2, 2), mode="max")

    # Gradient computation
    loss = out.sum()
    grad_x = grad(loss, x)

    # Test data - arranged so we know which elements are max
    x_val = np.array([[[[1, 2, 5, 6],
                        [3, 4, 7, 8],
                        [9, 10, 13, 14],
                        [11, 12, 15, 16]]]]).astype("float32")

    # Assert
    fn, (out_val, grad_val) = compare_jax_and_py(
        [x],
        [out, grad_x],
        [x_val]
    )

    # Expected: gradients only at positions [1,1], [1,3], [3,1], [3,3]
    # which are 4, 8, 12, 16 - the max values in each 2x2 block
    expected_grad = np.array([[[[0, 0, 0, 0],
                                [0, 1, 0, 1],
                                [0, 0, 0, 0],
                                [0, 1, 0, 1]]]]).astype("float32")

    np.testing.assert_array_equal(grad_val, expected_grad,
                                  "Gradient routing through max pool incorrect")
```

#### 3. ConvBNSiLU Block Integration Test
**Test File**: `tests/link/jax/test_jax_convbnsilu_block.py`
**Purpose**: Test complete ConvBNSiLU block as used 22+ times in YOLO11

##### Test: `test_convbnsilu_block_complete`
**Purpose**: End-to-end test of Conv2D + BN + SiLU pattern
**Test Data**: Typical YOLO feature dimensions
**Expected Behavior**: Complete block should work with JAX
**Assertions**: Gradients flow through entire block

```python
def test_convbnsilu_block_complete():
    """
    Test complete ConvBNSiLU block (Conv2D + BN + SiLU).

    This is the atomic unit used 22+ times in YOLO11.
    Verifies end-to-end gradient flow through the complete block.
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.conv import conv2d
    from pytensor import grad, shared
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - build ConvBNSiLU block
    x = pt.tensor4("x", dtype="float32", shape=(None, 128, 32, 32))

    # Conv2D layer
    filters = pt.tensor4("filters", dtype="float32")
    conv_out = conv2d(x, filters, border_mode="same")

    # Batch Norm (simplified for testing)
    bn_gamma = pt.vector("bn_gamma", dtype="float32")
    bn_beta = pt.vector("bn_beta", dtype="float32")

    # Normalize per channel
    mean = conv_out.mean(axis=(0, 2, 3), keepdims=True)
    var = conv_out.var(axis=(0, 2, 3), keepdims=True)
    bn_out = (conv_out - mean) / pt.sqrt(var + 1e-5)
    bn_out = bn_out * bn_gamma.dimshuffle('x', 0, 'x', 'x') + \
              bn_beta.dimshuffle('x', 0, 'x', 'x')

    # SiLU activation (x * sigmoid(x))
    silu_out = bn_out * pt.sigmoid(bn_out)

    # Compute loss and gradients
    loss = silu_out.sum()
    grads = grad(loss, [x, filters, bn_gamma, bn_beta])

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 128, 32, 32)).astype("float32")
    filters_val = rng.normal(size=(256, 128, 3, 3)).astype("float32") * 0.1
    bn_gamma_val = np.ones(256).astype("float32")
    bn_beta_val = np.zeros(256).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, filters, bn_gamma, bn_beta],
        [silu_out, loss] + grads,
        [x_val, filters_val, bn_gamma_val, bn_beta_val]
    )

    silu_val, loss_val = outputs[:2]
    grad_vals = outputs[2:]

    # Verify gradients exist and are finite
    for i, grad_val in enumerate(grad_vals):
        assert grad_val is not None, f"Gradient {i} is None"
        assert np.all(np.isfinite(grad_val)), f"Gradient {i} has NaN/Inf"
        assert np.abs(grad_val).sum() > 0, f"Gradient {i} is all zeros"
```

**Expected Failure Mode**: May fail on BN or SiLU operations
- Error type: Depends on which sub-operation fails
- Expected message: Tracer issues in normalization or activation

### Test Implementation Steps:

1. **Create test file structure**:
   ```bash
   tests/link/jax/
   ├── test_jax_conv2d_tracer.py
   ├── test_jax_pool2d_tracer.py
   └── test_jax_convbnsilu_block.py
   ```

2. **Import necessary testing utilities**:
   ```python
   import numpy as np
   import pytest
   import pytensor
   import pytensor.tensor as pt
   from pytensor.tensor.conv import conv2d
   from pytensor.tensor.pool import pool_2d
   from pytensor import grad, shared
   from pytensor.configdefaults import config
   from tests.link.jax.test_basic import compare_jax_and_py

   jax = pytest.importorskip("jax")

   # Set tolerances based on precision
   floatX = config.floatX
   RTOL = 1e-6 if floatX.endswith("64") else 1e-5
   ATOL = 1e-6 if floatX.endswith("64") else 1e-5
   ```

3. **Create test fixtures for common data**:
   ```python
   @pytest.fixture
   def conv_test_data():
       """Fixture providing convolution test data."""
       rng = np.random.default_rng(42)
       return {
           "x": rng.normal(size=(2, 3, 32, 32)).astype("float32"),
           "filters_3x3": rng.normal(size=(16, 3, 3, 3)).astype("float32"),
           "filters_1x1": rng.normal(size=(16, 3, 1, 1)).astype("float32"),
       }

   @pytest.fixture
   def pool_test_data():
       """Fixture providing pooling test data."""
       rng = np.random.default_rng(42)
       return rng.normal(size=(2, 256, 32, 32)).astype("float32")
   ```

4. **Implement each test case** with clear arrange-act-assert structure

5. **Add comprehensive docstrings** explaining what each test validates

### Success Criteria:

#### Automated Verification:
- [ ] All test files created with proper imports
- [ ] Tests use compare_jax_and_py utility correctly
- [ ] Test code follows project conventions
- [ ] Tests are discoverable: `pytest --collect-only tests/link/jax/test_jax_conv*`

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Assertion messages are diagnostic
- [ ] Test code is readable and maintainable

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and verify they fail in the expected, diagnostic ways.

### Verification Steps:

1. **Run the Conv2D test suite**:
   ```bash
   pytest tests/link/jax/test_jax_conv2d_tracer.py -v
   ```

2. **Run the Pool2D test suite**:
   ```bash
   pytest tests/link/jax/test_jax_pool2d_tracer.py -v
   ```

3. **Run the ConvBNSiLU integration test**:
   ```bash
   pytest tests/link/jax/test_jax_convbnsilu_block.py -v
   ```

4. **For each test, document**:
   - Does it pass or fail?
   - If fails, what's the error?
   - Is the error informative?
   - Does it point to the right code?

### Expected Failures:

Document actual vs expected for each test:

- **test_conv2d_with_dynamic_batch**:
  - Expected: May pass if Conv2D already handles tracers
  - Actual: [To be determined]

- **test_pool2d_with_dynamic_dimensions**:
  - Expected: Likely passes (Pool2D mentioned as working)
  - Actual: [To be determined]

- **test_convbnsilu_block_complete**:
  - Expected: May fail on BN or SiLU operations
  - Actual: [To be determined]

### Success Criteria:

#### Automated Verification:
- [ ] All tests run without import errors
- [ ] Tests that should fail do fail
- [ ] No unexpected passes for operations that shouldn't work

#### Manual Verification:
- [ ] Failure messages indicate the specific issue
- [ ] Stack traces point to dispatch code
- [ ] Errors would guide implementation

### Adjustment Phase:

If tests don't fail properly:
- [ ] Add more challenging test cases
- [ ] Test edge cases that might fail
- [ ] Ensure we're testing actual tracer compatibility

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement Conv2D and Pool2D dispatch to make tests pass.

### Implementation Strategy:

**Order of Implementation:**
1. Fix Conv2D basic operation
2. Fix Conv2D with padding/stride
3. Verify Pool2D works (may already work)
4. Fix ConvBNSiLU integration issues

### Implementation Steps:

#### Implementation 1: Conv2D JAX Dispatch

**Target Test**: `test_conv2d_with_dynamic_batch`
**Current Failure**: [To be determined after running tests]

**File**: `pytensor/link/jax/dispatch/conv.py`
**Changes**: Ensure Conv2D handles dynamic batch dimensions

```python
import jax.numpy as jnp
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.conv import Conv2D

@jax_funcify.register(Conv2D)
def jax_funcify_Conv2D(op, node, **kwargs):
    """JAX implementation of Conv2D that handles dynamic shapes."""

    border_mode = op.border_mode
    subsample = op.subsample
    filter_flip = op.filter_flip

    def conv2d_fn(x, filters):
        # Handle different padding modes
        if border_mode == "valid":
            padding = "VALID"
        elif border_mode == "same" or border_mode == "half":
            padding = "SAME"
        elif isinstance(border_mode, tuple):
            # Numeric padding
            padding = border_mode
        else:
            padding = "VALID"

        # JAX conv expects filter shape: [out, in, h, w]
        # PyTensor uses: [out, in, h, w] - same!

        # Use JAX's conv_general_dilated
        from jax import lax

        # Flip filters if needed (correlation vs convolution)
        if filter_flip:
            filters = filters[:, :, ::-1, ::-1]

        # Perform convolution
        y = lax.conv_general_dilated(
            x,                    # Input: [N, C, H, W]
            filters,              # Kernel: [O, I, Kh, Kw]
            window_strides=subsample,
            padding=padding,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )

        return y

    return conv2d_fn
```

**Debugging Approach:**
1. Run test: `pytest tests/link/jax/test_jax_conv2d_tracer.py::test_conv2d_with_dynamic_batch -v`
2. If fails, check error message
3. Adjust implementation based on error
4. Re-run until test passes

#### Implementation 2: Verify Pool2D Works

**Target Test**: `test_pool2d_with_dynamic_dimensions`
**Current Status**: Expected to work already

**File**: `pytensor/link/jax/dispatch/pool.py`
**Action**: Verify existing implementation handles tracers

```python
# Check existing pool implementation
# If it doesn't handle dynamic batch, update similar to Conv2D
```

#### Implementation 3: Fix BN and SiLU for ConvBNSiLU

**Target Test**: `test_convbnsilu_block_complete`
**Potential Issues**: Dimshuffle and sigmoid operations

**Files to check/fix**:
- `pytensor/link/jax/dispatch/elemwise.py` - for sigmoid
- `pytensor/link/jax/dispatch/tensor_basic.py` - for dimshuffle

### Complete Feature Implementation:

Once individual operations work:

**Integration Testing:**
```bash
# Run all new tests
pytest tests/link/jax/test_jax_conv2d_tracer.py -v
pytest tests/link/jax/test_jax_pool2d_tracer.py -v
pytest tests/link/jax/test_jax_convbnsilu_block.py -v

# Check for regressions
pytest tests/link/jax/test_conv.py -v
pytest tests/link/jax/test_pool.py -v
```

### Success Criteria:

##### Automated Verification:
- [ ] All Conv2D tests pass
- [ ] All Pool2D tests pass
- [ ] ConvBNSiLU integration test passes
- [ ] No regressions in existing tests
- [ ] Linting passes: `make lint`

##### Manual Verification:
- [ ] Implementation handles all padding modes
- [ ] Stride and dilation work correctly
- [ ] Gradients compute properly
- [ ] Code is well-documented

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor to improve code quality while keeping tests green.

### Refactoring Targets:

1. **Extract Common Patterns**:
   - Padding mode conversion logic
   - Dimension reordering utilities
   - Filter flipping logic

2. **Improve Documentation**:
   - Add docstrings explaining JAX conventions
   - Document dimension order expectations
   - Add examples of usage

3. **Optimize Performance**:
   - Cache padding computations
   - Optimize for common cases (3x3, 1x1 convolutions)

### Refactoring Steps:

1. **Create utilities module** if needed:
   ```python
   # pytensor/link/jax/dispatch/conv_utils.py
   def convert_padding_mode(border_mode):
       """Convert PyTensor padding to JAX format."""
       if border_mode == "valid":
           return "VALID"
       elif border_mode in ("same", "half"):
           return "SAME"
       elif isinstance(border_mode, tuple):
           return border_mode
       else:
           return "VALID"
   ```

2. **Add comprehensive docstrings**

3. **Run tests after each change** to ensure nothing breaks

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] No performance regressions
- [ ] Code coverage maintained

#### Manual Verification:
- [ ] Code is cleaner and more maintainable
- [ ] Documentation is comprehensive
- [ ] No code duplication

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] Conv2D with all padding modes tested
- [ ] Conv2D with strides tested
- [ ] Pool2D with SPPF pattern tested
- [ ] Complete ConvBNSiLU block tested
- [ ] Gradient flow verified for all operations

### Test Organization:
- Test files: `tests/link/jax/test_jax_conv*.py`, `test_jax_pool*.py`
- Fixtures: Defined in each test file
- Test utilities: Import from `test_basic.py`
- Tolerances: Defined based on float precision

### Running Tests:

```bash
# Run all Conv2D tests
pytest tests/link/jax/test_jax_conv2d_tracer.py -v

# Run specific test
pytest tests/link/jax/test_jax_conv2d_tracer.py::test_conv2d_with_dynamic_batch -v

# Run with coverage
pytest tests/link/jax/test_jax_conv* --cov=pytensor.link.jax.dispatch.conv

# Run integration test
pytest tests/link/jax/test_jax_convbnsilu_block.py -v
```

## Performance Considerations

Conv2D and Pool2D are performance-critical operations:
- JAX's XLA compilation should provide excellent performance
- Dynamic batch handling may add slight overhead
- GPU execution will be significantly faster than CPU

### Performance Testing:
- [ ] Benchmark Conv2D against PyTorch/TensorFlow
- [ ] Profile different kernel sizes (1x1, 3x3, 5x5)
- [ ] Measure JIT compilation time
- [ ] Test memory usage with large batches

## Migration Notes

For existing code:
- Conv2D should work transparently with dynamic batches
- No changes needed to model code
- Existing tests should continue to pass

## References

- Original research: `thoughts/shared/research/2025-10-23_20-11-25_jax-backend-missing-operations-test-coverage.md`
- YOLO11 implementation: `blocks.py:163-169` (Conv2D usage)
- SPPF implementation: `blocks.py:362-382` (Pool2D usage)
- Test utilities: `tests/link/jax/test_basic.py:36-96`
- Existing Conv tests: `tests/link/jax/test_conv.py`
- Existing Pool tests: `tests/link/jax/test_pool.py`