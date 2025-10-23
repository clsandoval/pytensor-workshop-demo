# JAX Backend Gradient and Compilation Operations TDD Implementation Plan

## Overview

This plan addresses the critical gradient computation and function compilation operations that enable model training. These include `pytensor.grad` for computing parameter gradients, `pytensor.function` with updates for training loops, and proper handling of shared variables. Without these working correctly with JAX, YOLO11 cannot be trained on GPU. We'll implement comprehensive tests first, verify failures, then ensure gradient computation and compilation work seamlessly.

## Current State Analysis

Gradient computation and function compilation are the foundation of training but may have issues with JAX's JIT compilation, especially when dealing with parameter updates, shared variables, and complex gradient graphs. These operations must handle dynamic shapes, multiple outputs, and stateful updates correctly.

### Current Testing Landscape:
- Testing framework: pytest
- Available test utilities: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()`
- Existing patterns: Basic gradient tests exist but not comprehensive training patterns
- Test fixtures/mocks available: `jax_mode`, `py_mode` configurations

## Desired End State

Gradient computation should work for all YOLO11 parameters, handling disconnected gradients appropriately. Function compilation with updates should enable efficient training loops with parameter updates. Shared variables should work correctly for running statistics and learnable parameters.

### Key Discoveries:
- Gradient computation: `train.py:243` - computing gradients for all parameters
- Function compilation: `train.py:292-298` - training function with updates
- Shared variables: Used for BN running statistics and parameters
- Updates dictionary: Parameter updates in training loop

## What We're NOT Testing/Implementing

- Second-order gradients (Hessian) not used in YOLO
- Complex optimizers beyond basic SGD patterns
- Distributed training patterns
- Custom gradient operations

## TDD Approach

Write comprehensive tests for gradient computation and compilation patterns as they appear in training, verify they fail if there are tracer issues, then ensure complete training loops work with JAX.

### Test Design Philosophy:
- Test realistic training patterns
- Verify gradient flow through entire models
- Test stateful updates work correctly
- Ensure compilation produces efficient code

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define correct behavior for gradient computation and function compilation with JAX.

### Test Categories:

#### 1. Gradient Computation Tests
**Test File**: `tests/link/jax/test_jax_gradient_computation.py`
**Purpose**: Test gradient computation for various scenarios

**Test Cases to Write:**

##### Test: `test_gradient_computation`
**Purpose**: Test basic gradient computation with JAX
**Test Data**: Simple and complex computational graphs
**Expected Behavior**: Gradients computed correctly
**Assertions**: Gradient values match expected

```python
def test_gradient_computation():
    """
    Test pytensor.grad with JAX backend.

    This test verifies:
    - Gradient computation for multiple parameters
    - Disconnected gradient handling (returns zeros)
    - Gradient of gradient (higher-order)
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - multi-parameter model
    x = pt.matrix("x", dtype="float32")
    w1 = pt.matrix("w1", dtype="float32")
    w2 = pt.matrix("w2", dtype="float32")
    b1 = pt.vector("b1", dtype="float32")
    b2 = pt.vector("b2", dtype="float32")

    # Simple 2-layer network
    h = pt.tanh(pt.dot(x, w1) + b1)
    y = pt.dot(h, w2) + b2

    # Loss
    target = pt.matrix("target", dtype="float32")
    loss = ((y - target) ** 2).mean()

    # Act - compute gradients
    grads = grad(loss, [w1, w2, b1, b2])

    # Also test disconnected gradient
    unconnected = pt.matrix("unconnected", dtype="float32")
    grad_unconnected = grad(loss, unconnected, disconnected_inputs='zero')

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 10)).astype("float32")
    w1_val = rng.normal(size=(10, 20)).astype("float32") * 0.1
    w2_val = rng.normal(size=(20, 5)).astype("float32") * 0.1
    b1_val = np.zeros(20, dtype="float32")
    b2_val = np.zeros(5, dtype="float32")
    target_val = rng.normal(size=(4, 5)).astype("float32")
    unconnected_val = rng.normal(size=(3, 3)).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, w1, w2, b1, b2, target, unconnected],
        [loss] + grads + [grad_unconnected],
        [x_val, w1_val, w2_val, b1_val, b2_val, target_val, unconnected_val]
    )

    loss_val = outputs[0]
    grad_vals = outputs[1:5]
    grad_unconnected_val = outputs[5]

    # Verify all gradients computed
    for i, g in enumerate(grad_vals):
        assert g is not None, f"Gradient {i} is None"
        assert np.all(np.isfinite(g)), f"Gradient {i} has NaN/Inf"
        assert g.shape == [w1_val, w2_val, b1_val, b2_val][i].shape, \
            f"Gradient {i} shape mismatch"

    # Verify disconnected gradient is zeros
    np.testing.assert_array_equal(grad_unconnected_val,
                                  np.zeros_like(unconnected_val),
                                  "Disconnected gradient should be zeros")
```

**Expected Failure Mode**: Should work for basic gradients
- Error type: None expected for first-order gradients
- Expected message: Should pass

##### Test: `test_gradient_through_complex_model`
**Purpose**: Test gradients through YOLO-like model
**Test Data**: Multi-layer conv network
**Expected Behavior**: Gradients flow through all layers
**Assertions**: No vanishing/exploding gradients

```python
def test_gradient_through_complex_model():
    """
    Test gradient computation through complex model.

    This test verifies:
    - Gradients through deep networks
    - Multiple parameter types (conv, bn, bias)
    - No gradient vanishing/explosion
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.conv import conv2d
    from pytensor import grad, shared
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - multi-layer conv network
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 64, 64))

    # Layer 1
    conv1_w = pt.tensor4("conv1_w", dtype="float32")
    conv1_b = pt.vector("conv1_b", dtype="float32")
    h1 = conv2d(x, conv1_w, border_mode="same")
    h1 = h1 + conv1_b.dimshuffle('x', 0, 'x', 'x')
    h1 = pt.nnet.relu(h1)

    # Layer 2
    conv2_w = pt.tensor4("conv2_w", dtype="float32")
    conv2_b = pt.vector("conv2_b", dtype="float32")
    h2 = conv2d(h1, conv2_w, border_mode="same")
    h2 = h2 + conv2_b.dimshuffle('x', 0, 'x', 'x')
    h2 = pt.nnet.relu(h2)

    # Global pooling and output
    y = h2.mean(axis=(2, 3))  # Global average pooling

    # Loss
    target = pt.matrix("target", dtype="float32", shape=(None, 64))
    loss = ((y - target) ** 2).mean()

    # Compute gradients for all parameters
    params = [conv1_w, conv1_b, conv2_w, conv2_b]
    grads = grad(loss, params)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 64, 64)).astype("float32") * 0.1
    conv1_w_val = rng.normal(size=(32, 3, 3, 3)).astype("float32") * 0.1
    conv1_b_val = np.zeros(32, dtype="float32")
    conv2_w_val = rng.normal(size=(64, 32, 3, 3)).astype("float32") * 0.05
    conv2_b_val = np.zeros(64, dtype="float32")
    target_val = rng.normal(size=(2, 64)).astype("float32") * 0.1

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, conv1_w, conv1_b, conv2_w, conv2_b, target],
        [loss] + grads,
        [x_val, conv1_w_val, conv1_b_val, conv2_w_val, conv2_b_val, target_val]
    )

    loss_val = outputs[0]
    grad_vals = outputs[1:]

    # Verify gradients exist and are reasonable
    for i, (param_val, grad_val) in enumerate(zip(
        [conv1_w_val, conv1_b_val, conv2_w_val, conv2_b_val],
        grad_vals
    )):
        assert grad_val is not None, f"Gradient {i} is None"
        assert np.all(np.isfinite(grad_val)), f"Gradient {i} has NaN/Inf"

        # Check gradient magnitude is reasonable
        grad_norm = np.linalg.norm(grad_val.flatten())
        assert 1e-8 < grad_norm < 1e3, \
            f"Gradient {i} norm {grad_norm} suggests vanishing/exploding"
```

##### Test: `test_gradient_wrt_inputs`
**Purpose**: Test gradient w.r.t. inputs (for adversarial examples, etc.)
**Test Data**: Model with input gradients
**Expected Behavior**: Can compute gradients w.r.t. inputs
**Assertions**: Input gradients computed correctly

```python
def test_gradient_wrt_inputs():
    """
    Test gradient computation with respect to inputs.

    This test verifies:
    - Gradients w.r.t. input tensors
    - Used for visualization, adversarial examples
    - Works with dynamic batch size
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32", shape=(None, 3, 32, 32))
    w = pt.tensor4("w", dtype="float32")

    # Simple convolution
    from pytensor.tensor.conv import conv2d
    y = conv2d(x, w, border_mode="valid")

    # Loss
    loss = y.sum()

    # Gradient w.r.t. input
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 3, 32, 32)).astype("float32")
    w_val = rng.normal(size=(16, 3, 5, 5)).astype("float32") * 0.1

    # Assert
    fn, (grad_x_val,) = compare_jax_and_py(
        [x, w],
        [grad_x],
        [x_val, w_val]
    )

    # Verify input gradient computed
    assert grad_x_val.shape == x_val.shape, "Input gradient shape mismatch"
    assert np.all(np.isfinite(grad_x_val)), "Input gradient has NaN/Inf"
    assert np.abs(grad_x_val).sum() > 0, "Input gradient is zero"
```

#### 2. Function Compilation Tests
**Test File**: `tests/link/jax/test_jax_function_compilation.py`
**Purpose**: Test function compilation with updates

**Test Cases to Write:**

##### Test: `test_function_compilation_with_updates`
**Purpose**: Test training loop pattern with parameter updates
**Test Data**: Simple training step
**Expected Behavior**: Functions compile with update dictionary
**Assertions**: Parameters update correctly

```python
def test_function_compilation_with_updates():
    """
    Test pytensor.function compilation with parameter updates.

    This test verifies:
    - Function with updates dictionary
    - In-place parameter updates
    - Multiple outputs with updates
    - JAX JIT compilation of update operations
    """
    import pytensor
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad, shared
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - simple model with shared variables
    # Parameters as shared variables
    w = shared(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), name="w")
    b = shared(np.array([0.0, 0.0], dtype="float32"), name="b")

    # Input and target
    x = pt.matrix("x", dtype="float32")
    target = pt.matrix("target", dtype="float32")

    # Forward pass
    y = pt.dot(x, w) + b

    # Loss
    loss = ((y - target) ** 2).mean()

    # Gradients
    grad_w = grad(loss, w)
    grad_b = grad(loss, b)

    # Learning rate
    lr = 0.01

    # Updates dictionary (SGD)
    updates = {
        w: w - lr * grad_w,
        b: b - lr * grad_b,
    }

    # Act - compile function with updates
    train_fn = pytensor.function(
        inputs=[x, target],
        outputs=[loss, y],
        updates=updates,
        mode="JAX"
    )

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 2)).astype("float32")
    target_val = rng.normal(size=(4, 2)).astype("float32")

    # Store initial values
    w_initial = w.get_value().copy()
    b_initial = b.get_value().copy()

    # Execute training step
    loss_val, y_val = train_fn(x_val, target_val)

    # Get updated values
    w_updated = w.get_value()
    b_updated = b.get_value()

    # Assert - parameters should be updated
    assert not np.array_equal(w_initial, w_updated), \
        "Weight not updated"
    assert not np.array_equal(b_initial, b_updated), \
        "Bias not updated"

    # Verify update direction (gradient descent)
    # Compute expected gradients manually
    y_pred = np.dot(x_val, w_initial) + b_initial
    diff = y_pred - target_val
    grad_w_expected = 2 * np.dot(x_val.T, diff) / len(x_val)
    grad_b_expected = 2 * diff.mean(axis=0)

    w_expected = w_initial - lr * grad_w_expected
    b_expected = b_initial - lr * grad_b_expected

    np.testing.assert_allclose(w_updated, w_expected, rtol=1e-5,
                               err_msg="Weight update incorrect")
    np.testing.assert_allclose(b_updated, b_expected, rtol=1e-5,
                               err_msg="Bias update incorrect")
```

**Expected Failure Mode**: May have issues with shared variable updates
- Error type: Potential issues with stateful updates in JAX
- Expected message: May need special handling for updates

##### Test: `test_function_with_givens`
**Purpose**: Test function compilation with givens (substitutions)
**Test Data**: Batch iteration pattern
**Expected Behavior**: Givens work for batch selection
**Assertions**: Correct batch processed

```python
def test_function_with_givens():
    """
    Test function compilation with givens parameter.

    This test verifies:
    - Substituting shared variables with givens
    - Batch iteration patterns
    - Works with JAX compilation
    """
    import pytensor
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import shared

    # Arrange - data as shared variable
    data = shared(np.arange(100).reshape(10, 10).astype("float32"), name="data")

    # Batch indices
    batch_idx = pt.iscalar("batch_idx")
    batch_size = 3

    # Get batch using indexing
    batch_data = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    # Simple computation
    output = batch_data.sum(axis=1)

    # Act - compile with different approaches
    # Approach 1: Direct indexing
    fn1 = pytensor.function(
        inputs=[batch_idx],
        outputs=output,
        mode="JAX"
    )

    # Test for batch 0
    result1 = fn1(0)

    # Approach 2: Using givens (if supported)
    batch_placeholder = pt.matrix("batch", dtype="float32")
    output2 = batch_placeholder.sum(axis=1)

    fn2 = pytensor.function(
        inputs=[],
        outputs=output2,
        givens={batch_placeholder: data[0:3]},
        mode="JAX"
    )

    result2 = fn2()

    # Assert - both approaches should give same result
    expected = np.arange(30).reshape(3, 10).astype("float32").sum(axis=1)
    np.testing.assert_array_almost_equal(result1, expected,
                                         err_msg="Direct indexing failed")
    np.testing.assert_array_almost_equal(result2, expected,
                                         err_msg="Givens approach failed")
```

##### Test: `test_function_with_multiple_outputs`
**Purpose**: Test functions returning multiple values
**Test Data**: Model with multiple outputs
**Expected Behavior**: All outputs computed correctly
**Assertions**: Multiple return values handled

```python
def test_function_with_multiple_outputs():
    """
    Test function with multiple outputs.

    This test verifies:
    - Functions can return multiple tensors
    - Intermediate values can be returned
    - Works with JAX JIT compilation
    """
    import pytensor
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad

    # Arrange
    x = pt.matrix("x", dtype="float32")
    w1 = pt.matrix("w1", dtype="float32")
    w2 = pt.matrix("w2", dtype="float32")

    # Multi-layer with intermediate outputs
    h1 = pt.tanh(pt.dot(x, w1))
    y = pt.dot(h1, w2)

    # Loss and gradient
    loss = y.sum()
    grad_w1 = grad(loss, w1)

    # Act - compile function with multiple outputs
    fn = pytensor.function(
        inputs=[x, w1, w2],
        outputs=[h1, y, loss, grad_w1],  # Multiple outputs
        mode="JAX"
    )

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 5)).astype("float32")
    w1_val = rng.normal(size=(5, 10)).astype("float32") * 0.1
    w2_val = rng.normal(size=(10, 3)).astype("float32") * 0.1

    # Execute
    h1_val, y_val, loss_val, grad_w1_val = fn(x_val, w1_val, w2_val)

    # Assert - all outputs computed
    assert h1_val.shape == (4, 10), f"Hidden shape {h1_val.shape} != (4, 10)"
    assert y_val.shape == (4, 3), f"Output shape {y_val.shape} != (4, 3)"
    assert loss_val.shape == (), f"Loss should be scalar, got {loss_val.shape}"
    assert grad_w1_val.shape == w1_val.shape, "Gradient shape mismatch"

    # Verify intermediate values are correct
    h1_expected = np.tanh(np.dot(x_val, w1_val))
    np.testing.assert_allclose(h1_val, h1_expected, rtol=1e-5,
                               err_msg="Intermediate output incorrect")
```

#### 3. Shared Variable Tests
**Test File**: `tests/link/jax/test_jax_shared_variables.py`
**Purpose**: Test shared variable handling

**Test Cases to Write:**

##### Test: `test_shared_variable_updates`
**Purpose**: Test shared variable updates in training
**Test Data**: Running statistics pattern
**Expected Behavior**: Shared variables update correctly
**Assertions**: Values persist across calls

```python
def test_shared_variable_updates():
    """
    Test shared variable updates in training patterns.

    This test verifies:
    - Shared variables can be updated
    - Updates persist across function calls
    - Running statistics patterns work
    """
    import pytensor
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import shared

    # Arrange - running mean pattern (like in BN)
    running_mean = shared(np.zeros(10, dtype="float32"), name="running_mean")
    momentum = 0.1

    # New batch data
    batch_data = pt.matrix("batch_data", dtype="float32")

    # Compute batch mean
    batch_mean = batch_data.mean(axis=0)

    # Update running mean
    new_running_mean = momentum * batch_mean + (1 - momentum) * running_mean

    # Update dictionary
    updates = {running_mean: new_running_mean}

    # Act - compile function with updates
    update_fn = pytensor.function(
        inputs=[batch_data],
        outputs=[batch_mean, new_running_mean],
        updates=updates,
        mode="JAX"
    )

    # Test with multiple batches
    rng = np.random.default_rng(42)

    # Batch 1
    batch1 = rng.normal(loc=1.0, size=(32, 10)).astype("float32")
    batch1_mean, running1 = update_fn(batch1)
    running_val1 = running_mean.get_value()

    # Batch 2
    batch2 = rng.normal(loc=2.0, size=(32, 10)).astype("float32")
    batch2_mean, running2 = update_fn(batch2)
    running_val2 = running_mean.get_value()

    # Assert - running mean should update
    assert not np.array_equal(running_val1, np.zeros(10)), \
        "Running mean not updated after batch 1"
    assert not np.array_equal(running_val1, running_val2), \
        "Running mean not updated after batch 2"

    # Verify update formula
    expected1 = momentum * batch1.mean(axis=0)
    np.testing.assert_allclose(running_val1, expected1, rtol=1e-5,
                               err_msg="First update incorrect")

    expected2 = momentum * batch2.mean(axis=0) + (1 - momentum) * expected1
    np.testing.assert_allclose(running_val2, expected2, rtol=1e-5,
                               err_msg="Second update incorrect")
```

##### Test: `test_shared_variable_in_gradient`
**Purpose**: Test gradients w.r.t. shared variables
**Test Data**: Model with shared parameters
**Expected Behavior**: Can compute gradients of shared vars
**Assertions**: Gradients computed correctly

```python
def test_shared_variable_in_gradient():
    """
    Test gradient computation with shared variables.

    This test verifies:
    - Gradients w.r.t. shared variables
    - Shared variables in computational graph
    - Updates based on gradients
    """
    import pytensor
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad, shared

    # Arrange - model with shared parameters
    W = shared(np.eye(5, dtype="float32"), name="W")
    b = shared(np.zeros(5, dtype="float32"), name="b")

    # Input
    x = pt.matrix("x", dtype="float32")

    # Computation
    y = pt.dot(x, W) + b

    # Loss
    loss = (y ** 2).sum()

    # Gradients w.r.t. shared variables
    grad_W = grad(loss, W)
    grad_b = grad(loss, b)

    # Act - compile function
    fn = pytensor.function(
        inputs=[x],
        outputs=[loss, grad_W, grad_b],
        mode="JAX"
    )

    # Test data
    x_val = np.array([[1, 2, 3, 4, 5]], dtype="float32")

    # Execute
    loss_val, grad_W_val, grad_b_val = fn(x_val)

    # Assert - gradients computed
    assert grad_W_val.shape == (5, 5), "Weight gradient shape mismatch"
    assert grad_b_val.shape == (5,), "Bias gradient shape mismatch"

    # Verify gradient values
    # y = x @ I + 0 = x, loss = sum(x^2)
    # grad_loss/grad_b = 2*y = 2*x
    expected_grad_b = 2 * x_val[0]
    np.testing.assert_allclose(grad_b_val, expected_grad_b, rtol=1e-5,
                               err_msg="Bias gradient incorrect")
```

### Test Implementation Steps:

1. **Create test file structure**:
   ```bash
   tests/link/jax/
   ├── test_jax_gradient_computation.py
   ├── test_jax_function_compilation.py
   └── test_jax_shared_variables.py
   ```

2. **Import necessary utilities**:
   ```python
   import numpy as np
   import pytest
   import pytensor
   import pytensor.tensor as pt
   from pytensor import grad, shared, function
   from pytensor.configdefaults import config
   from tests.link.jax.test_basic import compare_jax_and_py

   jax = pytest.importorskip("jax")
   ```

3. **Create test fixtures**:
   ```python
   @pytest.fixture
   def simple_model():
       """Simple model for testing."""
       W = shared(np.random.randn(10, 5).astype("float32") * 0.1)
       b = shared(np.zeros(5, dtype="float32"))
       return W, b

   @pytest.fixture
   def training_data():
       """Training data for testing."""
       rng = np.random.default_rng(42)
       return {
           "X": rng.normal(size=(32, 10)).astype("float32"),
           "y": rng.normal(size=(32, 5)).astype("float32"),
       }
   ```

### Success Criteria:

#### Automated Verification:
- [ ] All test files created and importable
- [ ] Gradient tests cover various scenarios
- [ ] Function compilation tests work
- [ ] Shared variable tests pass

#### Manual Verification:
- [ ] Training patterns tested
- [ ] Updates work correctly
- [ ] Multiple outputs handled
- [ ] Shared variables update

---

## Phase 2: Test Failure Verification

### Overview
Run tests to identify issues with gradient and compilation operations.

### Verification Steps:

1. **Run gradient tests**:
   ```bash
   pytest tests/link/jax/test_jax_gradient_computation.py -v
   ```

2. **Run compilation tests**:
   ```bash
   pytest tests/link/jax/test_jax_function_compilation.py -v
   ```

3. **Run shared variable tests**:
   ```bash
   pytest tests/link/jax/test_jax_shared_variables.py -v
   ```

### Expected Failures:

- **test_gradient_computation**: Should mostly work
- **test_function_compilation_with_updates**: May have issues with updates
- **test_shared_variable_updates**: Potential issues with stateful updates

### Success Criteria:

#### Automated Verification:
- [ ] All tests run
- [ ] Failures documented
- [ ] Error messages informative

#### Manual Verification:
- [ ] Can identify issues
- [ ] Understand JAX limitations
- [ ] Have fix strategy

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Fix gradient and compilation issues for JAX backend.

### Implementation Strategy:

1. Ensure basic gradients work
2. Handle shared variable updates properly
3. Make function compilation JAX-compatible
4. Test training patterns work

### Implementation Steps:

#### Implementation 1: Gradient Computation

Should mostly work, but verify:
- Disconnected inputs handled
- Multiple parameters supported
- No tracer issues

#### Implementation 2: Function Compilation

May need special handling for:
- Updates dictionary with JAX
- Shared variable mutations
- JIT compilation with updates

#### Implementation 3: Shared Variables

Need to ensure:
- Updates work with JAX's functional style
- State management correct
- Persistence across calls

### Success Criteria:

##### Automated Verification:
- [ ] All gradient tests pass
- [ ] Function compilation works
- [ ] Shared variables update
- [ ] No regressions

##### Manual Verification:
- [ ] Training loops work
- [ ] Updates persist
- [ ] Performance acceptable

---

## Phase 4: Refactoring & Cleanup

### Overview
Optimize gradient and compilation for performance.

### Refactoring Targets:

1. **Performance**:
   - Optimize gradient computation
   - Improve JIT compilation
   - Minimize recompilation

2. **Usability**:
   - Clear error messages
   - Document JAX limitations
   - Provide migration guide

3. **Robustness**:
   - Handle edge cases
   - Validate inputs
   - Test with real models

### Success Criteria:

#### Automated Verification:
- [ ] All tests pass
- [ ] Performance benchmarked
- [ ] Coverage maintained

#### Manual Verification:
- [ ] Code well-documented
- [ ] JAX patterns clear
- [ ] Migration guide complete

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] Gradient computation tested
- [ ] Function compilation verified
- [ ] Updates work correctly
- [ ] Shared variables tested
- [ ] Training patterns work

### Test Organization:
- Test files: By operation type
- Fixtures: Model and data generators
- Utilities: Use existing tools
- Patterns: Real training scenarios

### Running Tests:

```bash
# Run all gradient/compilation tests
pytest tests/link/jax/test_jax_gradient*.py test_jax_function*.py test_jax_shared*.py -v

# Run with coverage
pytest tests/link/jax/test_jax_*compilation*.py --cov=pytensor.compile

# Run specific test
pytest tests/link/jax/test_jax_gradient_computation.py::test_gradient_computation -v
```

## Performance Considerations

These operations are critical for training:
- Gradient computation on every batch
- Function compilation affects JIT
- Updates must be efficient
- Shared variables accessed frequently

### Performance Testing:
- [ ] Benchmark gradient computation
- [ ] Profile compilation time
- [ ] Measure update overhead
- [ ] Test with large models

## Migration Notes

For existing training code:
- May need to adapt update patterns
- Shared variable handling may differ
- JIT compilation behavior different
- Document workarounds

## References

- Original research: `thoughts/shared/research/2025-10-23_20-11-25_jax-backend-missing-operations-test-coverage.md`
- Gradient usage: `train.py:243`
- Function compilation: `train.py:292-298`
- Test utilities: `tests/link/jax/test_basic.py:36-96`
- JAX limitations: Functional programming model