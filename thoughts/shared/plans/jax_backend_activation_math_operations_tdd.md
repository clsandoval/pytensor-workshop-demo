# JAX Backend Activation and Math Operations TDD Implementation Plan

## Overview

This plan addresses critical activation functions (sigmoid for SiLU) and mathematical operations (sqrt, maximum, minimum, log, mean) required for YOLO11. These operations are essential for activation functions, batch normalization, loss computation, and IoU calculations. We'll implement comprehensive tests first, verify failures, then implement JAX-compatible dispatch.

## Current State Analysis

Activation and math operations are fundamental to neural networks but may not properly handle JAX tracers during JIT compilation. Sigmoid is required for every SiLU activation (used in all 22+ ConvBNSiLU blocks), while math operations are critical for batch normalization and loss calculations.

### Current Testing Landscape:
- Testing framework: pytest
- Available test utilities: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()`
- Existing test patterns: Elemwise tests in `test_elemwise.py`
- Test fixtures/mocks available: `jax_mode`, `py_mode` configurations

## Desired End State

All activation and mathematical operations should seamlessly handle JAX tracers, enabling gradient computation through SiLU activations, batch normalization, and loss functions. Operations should work with dynamic shapes and maintain numerical stability.

### Key Discoveries:
- Sigmoid used in SiLU: `blocks.py:177` - pattern is `x * sigmoid(x)`
- Sqrt used in batch norm: `blocks.py:56` - for variance normalization
- Maximum/minimum for IoU: `loss.py:44-49` - intersection calculations
- Log for BCE loss: `loss.py:122,125` - binary cross-entropy
- Mean for loss aggregation: `loss.py:141,148,233,238`

## What We're NOT Testing/Implementing

- Activation functions not used in YOLO11 (tanh, relu, etc.)
- Math operations not in the critical path
- Complex mathematical functions (FFT, special functions)
- Statistical operations beyond mean

## TDD Approach

Write comprehensive tests for each activation and math operation with dynamic shapes, verify they fail with tracer issues, then implement proper JAX dispatch to make tests pass.

### Test Design Philosophy:
- Test both scalar and tensor operations
- Verify gradient flow through all operations
- Test numerical stability edge cases
- Ensure broadcasting works correctly

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define correct behavior for activation and math operations with JAX tracers.

### Test Categories:

#### 1. Sigmoid Activation Tests
**Test File**: `tests/link/jax/test_jax_sigmoid_activation.py`
**Purpose**: Verify sigmoid and SiLU pattern work with tracers

**Test Cases to Write:**

##### Test: `test_sigmoid_activation`
**Purpose**: Test sigmoid with dynamic shapes
**Test Data**: Various tensor shapes including dynamic batch
**Expected Behavior**: Sigmoid should handle traced dimensions
**Assertions**: Output in range [0,1], gradients flow correctly

```python
def test_sigmoid_activation():
    """
    Test sigmoid activation with dynamic shapes.

    This test verifies:
    - SiLU pattern (x * sigmoid(x))
    - Gradient flow through activation
    - Numerical stability with JAX
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - dynamic batch size
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))

    # Act - compute sigmoid and SiLU
    sigmoid_out = pt.sigmoid(x)
    silu_out = x * sigmoid_out  # SiLU activation

    # Compute gradients
    loss = silu_out.sum()
    grad_x = grad(loss, x)

    # Test data with various ranges
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 64, 32, 32)).astype("float32")

    # Assert
    fn, (sig_val, silu_val, grad_val) = compare_jax_and_py(
        [x],
        [sigmoid_out, silu_out, grad_x],
        [x_val]
    )

    # Verify sigmoid properties
    assert np.all(sig_val >= 0) and np.all(sig_val <= 1), \
        "Sigmoid output not in [0, 1]"

    # Verify SiLU gradient exists
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is all zeros"
```

**Expected Failure Mode**: Should work if elemwise ops handle tracers
- Error type: None expected
- Expected message: Should pass

##### Test: `test_sigmoid_numerical_stability`
**Purpose**: Test sigmoid with extreme values
**Test Data**: Very large/small inputs
**Expected Behavior**: Should not produce NaN/Inf
**Assertions**: Stable outputs for extreme inputs

```python
def test_sigmoid_numerical_stability():
    """
    Test sigmoid numerical stability with extreme values.

    This test verifies:
    - No NaN/Inf for large positive inputs
    - No NaN/Inf for large negative inputs
    - Correct asymptotic behavior
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - extreme values
    x = pt.vector("x", dtype="float32")

    # Act
    y = pt.sigmoid(x)
    grad_y = grad(y.sum(), x)

    # Test data with extreme values
    x_val = np.array([-100, -50, -10, 0, 10, 50, 100], dtype="float32")

    # Assert
    fn, (y_val, grad_val) = compare_jax_and_py(
        [x],
        [y, grad_y],
        [x_val]
    )

    # Verify numerical stability
    assert np.all(np.isfinite(y_val)), f"Sigmoid produced NaN/Inf: {y_val}"
    assert np.all(np.isfinite(grad_val)), f"Gradient has NaN/Inf: {grad_val}"

    # Verify asymptotic behavior
    assert y_val[0] < 1e-6, f"sigmoid(-100) = {y_val[0]}, expected ~0"
    assert y_val[-1] > 1 - 1e-6, f"sigmoid(100) = {y_val[-1]}, expected ~1"
```

##### Test: `test_silu_pattern_complete`
**Purpose**: Test complete SiLU pattern as used in YOLO
**Test Data**: Realistic feature map dimensions
**Expected Behavior**: Pattern should work end-to-end
**Assertions**: Correct forward and backward pass

```python
def test_silu_pattern_complete():
    """
    Test complete SiLU (Swish) activation pattern.

    This test verifies:
    - Multiplication of tensor with its sigmoid
    - Gradient computation through SiLU
    - Pattern matches expected SiLU behavior
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - typical ConvBNSiLU dimensions
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # Act - SiLU pattern
    silu = x * pt.sigmoid(x)

    # Gradient
    loss = silu.sum()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.randn(2, 256, 32, 32).astype("float32") * 0.5

    # Assert
    fn, (silu_val, grad_val) = compare_jax_and_py(
        [x],
        [silu, grad_x],
        [x_val]
    )

    # Manually compute expected SiLU for verification
    def silu_numpy(x):
        return x / (1 + np.exp(-x))

    expected_silu = silu_numpy(x_val)
    np.testing.assert_allclose(silu_val, expected_silu, rtol=1e-5,
                               err_msg="SiLU output doesn't match expected")

    # Verify gradient shape
    assert grad_val.shape == x_val.shape, "Gradient shape mismatch"
```

#### 2. Mathematical Operations Tests
**Test File**: `tests/link/jax/test_jax_math_operations.py`
**Purpose**: Test math operations used in BN and loss computation

**Test Cases to Write:**

##### Test: `test_math_operations_with_tracers`
**Purpose**: Test all critical math operations
**Test Data**: Dynamic shapes for each operation
**Expected Behavior**: All ops should handle tracers
**Assertions**: Correct outputs and gradients

```python
@pytest.mark.parametrize("op_name,op_func,test_input_fn", [
    ("sqrt", pt.sqrt, lambda: np.array([1, 4, 9, 16], dtype="float32")),
    ("maximum", lambda x: pt.maximum(x, 0.5), lambda: np.array([0, 0.3, 0.7, 1], dtype="float32")),
    ("minimum", lambda x: pt.minimum(x, 0.5), lambda: np.array([0, 0.3, 0.7, 1], dtype="float32")),
    ("log", pt.log, lambda: np.array([0.1, 1, 2, 10], dtype="float32")),
])
def test_math_operations_with_tracers(op_name, op_func, test_input_fn):
    """
    Test mathematical operations with JAX tracers.

    This test verifies:
    - Operations work with dynamic shapes
    - Gradients flow correctly
    - Numerical stability
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.vector("x", dtype="float32")

    # Act
    y = op_func(x)

    # Gradient (skip for operations that don't have gradients everywhere)
    if op_name not in ["maximum", "minimum"]:
        grad_x = grad(y.sum(), x)
    else:
        # For max/min, test gradient where differentiable
        grad_x = None

    # Test data
    x_val = test_input_fn()

    # Assert
    if grad_x is not None:
        fn, (y_val, grad_val) = compare_jax_and_py(
            [x],
            [y, grad_x],
            [x_val]
        )

        # Verify gradient exists
        assert np.all(np.isfinite(grad_val)), \
            f"{op_name} gradient has NaN/Inf"
    else:
        fn, (y_val,) = compare_jax_and_py(
            [x],
            [y],
            [x_val]
        )

    # Verify output is finite
    assert np.all(np.isfinite(y_val)), \
        f"{op_name} output has NaN/Inf: {y_val}"
```

**Expected Failure Mode**: Should mostly work
- Error type: None for basic math ops
- Expected message: Should pass

##### Test: `test_sqrt_for_batch_norm`
**Purpose**: Test sqrt as used in batch normalization
**Test Data**: Variance-like inputs (positive values)
**Expected Behavior**: Stable for small positive values
**Assertions**: No NaN for variance + epsilon pattern

```python
def test_sqrt_for_batch_norm():
    """
    Test sqrt operation as used in batch normalization.

    This test verifies:
    - Sqrt of variance + epsilon pattern
    - Numerical stability for small variances
    - Gradient flow through normalization
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - batch norm pattern
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))
    epsilon = 1e-5

    # Compute variance (simplified)
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

    # Sqrt for normalization
    std = pt.sqrt(var + epsilon)
    normalized = (x - mean) / std

    # Gradient
    loss = normalized.sum()
    grad_x = grad(loss, x)

    # Test data - include small variance case
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 64, 32, 32)).astype("float32") * 0.01

    # Assert
    fn, (std_val, norm_val, grad_val) = compare_jax_and_py(
        [x],
        [std, normalized, grad_x],
        [x_val]
    )

    # Verify no NaN/Inf
    assert np.all(np.isfinite(std_val)), "Std has NaN/Inf"
    assert np.all(np.isfinite(norm_val)), "Normalized output has NaN/Inf"
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"

    # Verify std is positive
    assert np.all(std_val > 0), "Std should be positive"
```

##### Test: `test_maximum_minimum_for_iou`
**Purpose**: Test max/min operations for IoU computation
**Test Data**: Box coordinate tensors
**Expected Behavior**: Correct intersection computation
**Assertions**: Proper clipping and intersection areas

```python
def test_maximum_minimum_for_iou():
    """
    Test maximum/minimum operations for IoU computation.

    This test verifies:
    - Intersection area computation
    - Proper handling of non-overlapping boxes
    - Gradient flow for differentiable IoU
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - box coordinates
    boxes1 = pt.matrix("boxes1", dtype="float32")  # [N, 4]
    boxes2 = pt.matrix("boxes2", dtype="float32")  # [M, 4]

    # Compute intersections (simplified for 1 box each)
    x1_max = pt.maximum(boxes1[0, 0], boxes2[0, 0])
    y1_max = pt.maximum(boxes1[0, 1], boxes2[0, 1])
    x2_min = pt.minimum(boxes1[0, 2], boxes2[0, 2])
    y2_min = pt.minimum(boxes1[0, 3], boxes2[0, 3])

    # Intersection area
    inter_w = pt.maximum(x2_min - x1_max, 0)
    inter_h = pt.maximum(y2_min - y1_max, 0)
    inter_area = inter_w * inter_h

    # Test data - overlapping and non-overlapping boxes
    boxes1_val = np.array([[0, 0, 2, 2]], dtype="float32")  # Box at origin
    boxes2_val = np.array([[1, 1, 3, 3]], dtype="float32")  # Overlapping box

    # Assert
    fn, (inter_val,) = compare_jax_and_py(
        [boxes1, boxes2],
        [inter_area],
        [boxes1_val, boxes2_val]
    )

    # Expected intersection area is 1 (1x1 overlap)
    expected_inter = 1.0
    np.testing.assert_almost_equal(inter_val, expected_inter,
                                   err_msg="Intersection area incorrect")
```

#### 3. Mean Reduction Tests
**Test File**: `tests/link/jax/test_jax_mean_reduction.py`
**Purpose**: Test mean reduction for loss aggregation

##### Test: `test_mean_reduction_with_axes`
**Purpose**: Test mean over specific axes with dynamic shapes
**Test Data**: Multi-dimensional tensors
**Expected Behavior**: Correct reduction over dynamic dimensions
**Assertions**: Output shape and values correct

```python
def test_mean_reduction_with_axes():
    """
    Test mean reduction over specific axes with dynamic shapes.

    This test verifies:
    - Reduction over dynamic batch dimension
    - Keepdims parameter handling
    - Multiple axis reduction
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - dynamic batch
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))

    # Act - various mean reductions
    mean_batch = pt.mean(x, axis=0)  # Average over batch
    mean_spatial = pt.mean(x, axis=(2, 3), keepdims=True)  # Spatial average
    mean_all = pt.mean(x)  # Global average

    # Gradients
    loss = mean_all
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(3, 64, 32, 32)).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x],
        [mean_batch, mean_spatial, mean_all, grad_x],
        [x_val]
    )

    mean_batch_val, mean_spatial_val, mean_all_val, grad_val = outputs

    # Verify shapes
    assert mean_batch_val.shape == (64, 32, 32), \
        f"Batch mean shape {mean_batch_val.shape} != (64, 32, 32)"
    assert mean_spatial_val.shape == (3, 64, 1, 1), \
        f"Spatial mean shape {mean_spatial_val.shape} != (3, 64, 1, 1)"
    assert mean_all_val.shape == (), \
        f"Global mean should be scalar, got shape {mean_all_val.shape}"

    # Verify gradient
    expected_grad = np.ones_like(x_val) / x_val.size
    np.testing.assert_allclose(grad_val, expected_grad, rtol=1e-6,
                               err_msg="Mean gradient incorrect")
```

**Expected Failure Mode**: Should work for basic reductions
- Error type: None expected
- Expected message: Should pass

##### Test: `test_mean_for_loss_aggregation`
**Purpose**: Test mean in loss computation context
**Test Data**: Loss tensors with various shapes
**Expected Behavior**: Proper averaging for different loss types
**Assertions**: Correct loss values and gradients

```python
def test_mean_for_loss_aggregation():
    """
    Test mean operation for loss aggregation patterns.

    This test verifies:
    - Mean over batch dimension
    - Mean over spatial dimensions
    - Weighted mean computation
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - typical loss tensors
    per_pixel_loss = pt.tensor4("per_pixel_loss", dtype="float32",
                                 shape=(None, 1, 32, 32))
    weights = pt.tensor4("weights", dtype="float32",
                         shape=(None, 1, 32, 32))

    # Act - different aggregation patterns
    # Pattern 1: Simple mean over all dimensions
    loss1 = pt.mean(per_pixel_loss)

    # Pattern 2: Mean over spatial, then batch
    loss2 = pt.mean(pt.mean(per_pixel_loss, axis=(2, 3)))

    # Pattern 3: Weighted mean
    weighted_loss = per_pixel_loss * weights
    loss3 = pt.sum(weighted_loss) / pt.sum(weights)

    # Test data
    rng = np.random.default_rng(42)
    loss_val = rng.uniform(0, 1, size=(2, 1, 32, 32)).astype("float32")
    weights_val = rng.uniform(0.5, 1.5, size=(2, 1, 32, 32)).astype("float32")

    # Assert
    fn, (l1, l2, l3) = compare_jax_and_py(
        [per_pixel_loss, weights],
        [loss1, loss2, loss3],
        [loss_val, weights_val]
    )

    # All aggregations should produce scalars
    assert l1.shape == (), "Loss1 should be scalar"
    assert l2.shape == (), "Loss2 should be scalar"
    assert l3.shape == (), "Loss3 should be scalar"

    # Pattern 1 and 2 should be equal
    np.testing.assert_almost_equal(l1, l2, decimal=6,
                                   err_msg="Different mean patterns gave different results")
```

#### 4. Log Operation for BCE Loss
**Test File**: `tests/link/jax/test_jax_log_bce.py`
**Purpose**: Test log operation for binary cross-entropy

##### Test: `test_log_for_bce_loss`
**Purpose**: Test log in BCE loss computation
**Test Data**: Probability values [0,1]
**Expected Behavior**: Stable for near-zero values
**Assertions**: No NaN/Inf for clipped probabilities

```python
def test_log_for_bce_loss():
    """
    Test log operation for binary cross-entropy loss.

    This test verifies:
    - Numerical stability with small probabilities
    - Proper clipping to avoid log(0)
    - Gradient flow through BCE
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - BCE loss pattern
    pred = pt.matrix("pred", dtype="float32")  # Predictions [N, C]
    target = pt.matrix("target", dtype="float32")  # Targets [N, C]

    # Clip predictions for numerical stability
    eps = 1e-7
    pred_clipped = pt.clip(pred, eps, 1 - eps)

    # BCE loss computation
    bce = -target * pt.log(pred_clipped) - (1 - target) * pt.log(1 - pred_clipped)
    loss = pt.mean(bce)

    # Gradient
    grad_pred = grad(loss, pred)

    # Test data - include edge cases
    pred_val = np.array([[0.01, 0.5, 0.99],
                         [0.1, 0.9, 0.001]], dtype="float32")
    target_val = np.array([[0, 1, 1],
                          [0, 1, 0]], dtype="float32")

    # Assert
    fn, (loss_val, grad_val) = compare_jax_and_py(
        [pred, target],
        [loss, grad_pred],
        [pred_val, target_val]
    )

    # Verify no NaN/Inf
    assert np.isfinite(loss_val), f"BCE loss is not finite: {loss_val}"
    assert np.all(np.isfinite(grad_val)), "BCE gradient has NaN/Inf"

    # Verify loss is positive
    assert loss_val > 0, f"BCE loss should be positive, got {loss_val}"
```

### Test Implementation Steps:

1. **Create test file structure**:
   ```bash
   tests/link/jax/
   ├── test_jax_sigmoid_activation.py
   ├── test_jax_math_operations.py
   ├── test_jax_mean_reduction.py
   └── test_jax_log_bce.py
   ```

2. **Import necessary utilities**:
   ```python
   import numpy as np
   import pytest
   import pytensor.tensor as pt
   from pytensor import grad
   from pytensor.configdefaults import config
   from tests.link.jax.test_basic import compare_jax_and_py

   jax = pytest.importorskip("jax")
   ```

3. **Create shared fixtures**:
   ```python
   @pytest.fixture
   def activation_test_data():
       """Test data for activation functions."""
       rng = np.random.default_rng(42)
       return {
           "small": rng.randn(10).astype("float32"),
           "medium": rng.randn(2, 64, 32, 32).astype("float32"),
           "extreme": np.array([-100, -10, 0, 10, 100], dtype="float32"),
       }
   ```

4. **Implement each test** with clear documentation

### Success Criteria:

#### Automated Verification:
- [ ] All test files created and importable
- [ ] Tests use compare_jax_and_py correctly
- [ ] Parametrized tests work properly
- [ ] Tests discoverable with pytest

#### Manual Verification:
- [ ] Test coverage includes all operations from research
- [ ] Edge cases properly tested
- [ ] Gradient tests included where applicable
- [ ] Clear documentation for each test

---

## Phase 2: Test Failure Verification

### Overview
Run tests and document which operations need fixes.

### Verification Steps:

1. **Run activation tests**:
   ```bash
   pytest tests/link/jax/test_jax_sigmoid_activation.py -v
   ```

2. **Run math operation tests**:
   ```bash
   pytest tests/link/jax/test_jax_math_operations.py -v
   ```

3. **Run reduction tests**:
   ```bash
   pytest tests/link/jax/test_jax_mean_reduction.py -v
   ```

4. **Run BCE/log tests**:
   ```bash
   pytest tests/link/jax/test_jax_log_bce.py -v
   ```

### Expected Failures:

Most of these operations should work as they're basic elemwise ops:

- **test_sigmoid_activation**: Expected to pass
- **test_math_operations_with_tracers**: Expected to mostly pass
- **test_mean_reduction_with_axes**: Expected to pass
- **test_log_for_bce_loss**: Expected to pass

### Success Criteria:

#### Automated Verification:
- [ ] All tests run without import errors
- [ ] Document which tests pass/fail
- [ ] Error messages are informative

#### Manual Verification:
- [ ] Failures point to specific issues
- [ ] Stack traces identify dispatch code
- [ ] Can identify what needs fixing

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Fix any operations that don't handle tracers properly.

### Implementation Strategy:

Most of these operations likely already work. Focus on:
1. Verifying existing implementations
2. Fixing any edge cases
3. Improving numerical stability

### Implementation Steps:

#### Implementation 1: Verify Sigmoid/SiLU

**File**: `pytensor/link/jax/dispatch/elemwise.py`
**Expected**: Should already work

```python
# Sigmoid should be handled by elemwise dispatch
# Verify it works with dynamic shapes
```

#### Implementation 2: Verify Math Operations

**File**: `pytensor/link/jax/dispatch/elemwise.py`
**Expected**: Basic math ops should work

```python
# sqrt, log, maximum, minimum should be handled
# May need to ensure proper broadcasting
```

#### Implementation 3: Verify Mean Reduction

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Expected**: Reductions likely work

```python
# Mean with axes should be handled
# Verify keepdims parameter works
```

### Complete Feature Implementation:

Run all tests to verify operations work:

```bash
# Run all activation/math tests
pytest tests/link/jax/test_jax_sigmoid_activation.py -v
pytest tests/link/jax/test_jax_math_operations.py -v
pytest tests/link/jax/test_jax_mean_reduction.py -v
pytest tests/link/jax/test_jax_log_bce.py -v
```

### Success Criteria:

##### Automated Verification:
- [ ] All activation tests pass
- [ ] All math operation tests pass
- [ ] All reduction tests pass
- [ ] No regressions in existing tests

##### Manual Verification:
- [ ] Numerical stability verified
- [ ] Gradient computation works
- [ ] Performance acceptable

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve code quality and add optimizations if needed.

### Refactoring Targets:

1. **Numerical Stability**:
   - Add epsilon parameters where needed
   - Improve clipping strategies
   - Document stability considerations

2. **Performance Optimizations**:
   - Use JAX-specific optimized operations
   - Avoid unnecessary copies

3. **Documentation**:
   - Add examples for common patterns
   - Document numerical considerations

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] No performance regressions
- [ ] Code coverage maintained

#### Manual Verification:
- [ ] Code is well-documented
- [ ] Numerical stability improved
- [ ] No unnecessary complexity

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] Sigmoid and SiLU pattern tested
- [ ] All critical math ops tested
- [ ] Mean reduction patterns tested
- [ ] BCE loss pattern tested
- [ ] Numerical stability verified

### Test Organization:
- Test files: Organized by operation category
- Fixtures: Shared test data generators
- Utilities: Use compare_jax_and_py
- Edge cases: Extreme values, empty tensors

### Running Tests:

```bash
# Run all activation/math tests
pytest tests/link/jax/test_jax_sigmoid*.py test_jax_math*.py -v

# Run with coverage
pytest tests/link/jax/test_jax_*.py --cov=pytensor.link.jax.dispatch

# Run specific operation test
pytest tests/link/jax/test_jax_math_operations.py::test_sqrt_for_batch_norm -v
```

## Performance Considerations

These operations are called frequently:
- Sigmoid in every ConvBNSiLU block
- Sqrt in batch normalization
- Mean in loss computation
- Should leverage JAX's vectorization

### Performance Testing:
- [ ] Benchmark against NumPy
- [ ] Profile with different batch sizes
- [ ] Measure JIT compilation overhead

## Migration Notes

These operations should be transparent:
- No model code changes needed
- Existing tests should pass
- Better numerical stability possible

## References

- Original research: `thoughts/shared/research/2025-10-23_20-11-25_jax-backend-missing-operations-test-coverage.md`
- SiLU usage: `blocks.py:177`
- Math ops in BN: `blocks.py:56`
- IoU computation: `loss.py:44-49`
- BCE loss: `loss.py:122,125`
- Mean aggregation: `loss.py:141,148,233,238`
- Test utilities: `tests/link/jax/test_basic.py:36-96`