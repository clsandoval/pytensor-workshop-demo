# JAX Backend Tensor Manipulation Operations TDD Implementation Plan

## Overview

This plan addresses critical tensor manipulation operations including dimshuffle (for broadcasting), concatenate (for feature fusion), tensor creation (zeros_like, constant, as_tensor_variable), and type casting. These operations are fundamental for batch normalization, CSP blocks, parameter initialization, and ONNX compatibility. We'll implement comprehensive tests first, verify failures, then implement JAX-compatible dispatch.

## Current State Analysis

Tensor manipulation operations are essential building blocks but may not properly handle JAX tracers. Dimshuffle is critical for batch normalization broadcasting, concatenate is used in every CSP block for feature fusion, tensor creation is needed for initialization, and casting ensures float32 for ONNX export.

### Current Testing Landscape:
- Testing framework: pytest
- Available test utilities: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()`
- Existing test patterns: Tensor basic tests in `test_tensor_basic.py`
- Test fixtures/mocks available: `jax_mode`, `py_mode` configurations

## Desired End State

All tensor manipulation operations should seamlessly handle JAX tracers, enabling proper broadcasting in batch normalization, feature concatenation in CSP blocks, gradient initialization, and type conversions for ONNX compatibility.

### Key Discoveries:
- Dimshuffle used in BN: `blocks.py:50-53` - broadcasting parameters
- Dimshuffle in loss: `loss.py:114,209` - dimension manipulation
- Concatenate in blocks: Feature fusion in C3k2, C2PSA, SPPF
- Tensor creation: `model.py:357`, `train.py:247,268-270`
- Cast operation: `train.py:276` - float32 enforcement

## What We're NOT Testing/Implementing

- Complex indexing operations not used in YOLO
- Tensor manipulation ops not in critical path
- Advanced broadcasting beyond dimshuffle
- Type conversions beyond float32/float64

## TDD Approach

Write comprehensive tests for tensor manipulation operations with dynamic shapes, verify they fail with tracer issues if present, then implement proper JAX dispatch to make tests pass.

### Test Design Philosophy:
- Test dynamic shape handling thoroughly
- Verify broadcasting semantics are preserved
- Test gradient flow through all operations
- Ensure type safety and conversions work

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define correct behavior for tensor manipulation operations with JAX tracers.

### Test Categories:

#### 1. Dimshuffle Broadcasting Tests
**Test File**: `tests/link/jax/test_jax_dimshuffle.py`
**Purpose**: Verify dimshuffle handles dynamic dimensions and broadcasting

**Test Cases to Write:**

##### Test: `test_dimshuffle_broadcasting`
**Purpose**: Test dimension shuffling and broadcasting with JAX tracers
**Test Data**: Various broadcasting patterns used in BN
**Expected Behavior**: Proper dimension manipulation with tracers
**Assertions**: Output shape correct, gradients flow

```python
def test_dimshuffle_broadcasting():
    """
    Test dimension shuffling and broadcasting with JAX tracers.

    This test verifies:
    - Adding/removing dimensions using 'x'
    - Dimension reordering
    - Broadcasting compatibility in JAX
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - test various dimshuffle patterns
    x = pt.vector("x", dtype="float32")  # Shape: (256,)
    y = pt.tensor4("y", dtype="float32", shape=(None, 256, 32, 32))

    # Act - common dimshuffle patterns
    # Pattern 1: Add dimensions for broadcasting (BN parameters)
    x_broadcast = x.dimshuffle('x', 0, 'x', 'x')  # (1, 256, 1, 1)

    # Pattern 2: Apply broadcasted operation (BN pattern)
    result = y * x_broadcast

    # Pattern 3: Reorder dimensions
    y_transposed = y.dimshuffle(0, 2, 3, 1)  # NCHW -> NHWC

    # Compute gradients
    loss = result.sum()
    grad_x = grad(loss, x)
    grad_y = grad(loss, y)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(256,)).astype("float32")
    y_val = rng.normal(size=(2, 256, 32, 32)).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, y],
        [x_broadcast, result, y_transposed, grad_x, grad_y],
        [x_val, y_val]
    )

    x_bc_val, result_val, y_trans_val, grad_x_val, grad_y_val = outputs

    # Verify shapes
    assert x_bc_val.shape == (1, 256, 1, 1), \
        f"Broadcast shape {x_bc_val.shape} != (1, 256, 1, 1)"
    assert result_val.shape == (2, 256, 32, 32), \
        f"Result shape {result_val.shape} != (2, 256, 32, 32)"
    assert y_trans_val.shape == (2, 32, 32, 256), \
        f"Transposed shape {y_trans_val.shape} != (2, 32, 32, 256)"

    # Verify gradients exist
    assert grad_x_val.shape == x_val.shape
    assert grad_y_val.shape == y_val.shape
```

**Expected Failure Mode**: Dimshuffle might have issues with 'x' broadcasting
- Error type: Potential tracer issues with dimension manipulation
- Expected message: May fail on 'x' dimension addition

##### Test: `test_dimshuffle_batch_norm_pattern`
**Purpose**: Test exact BN parameter broadcasting pattern
**Test Data**: BN gamma/beta broadcasting to feature maps
**Expected Behavior**: Parameters broadcast correctly
**Assertions**: BN pattern works end-to-end

```python
def test_dimshuffle_batch_norm_pattern():
    """
    Test dimshuffle pattern used in batch normalization.

    This test verifies:
    - BN parameter broadcasting (gamma, beta)
    - Gradient flow through broadcasted parameters
    - Dynamic batch size handling
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - BN pattern
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 32, 32))
    gamma = pt.vector("gamma", dtype="float32")  # Per-channel scale
    beta = pt.vector("beta", dtype="float32")   # Per-channel shift

    # Act - BN parameter broadcasting
    # Compute stats (simplified)
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)

    # Normalize
    x_norm = (x - mean) / pt.sqrt(var + 1e-5)

    # Apply affine transform with dimshuffle
    gamma_bc = gamma.dimshuffle('x', 0, 'x', 'x')
    beta_bc = beta.dimshuffle('x', 0, 'x', 'x')
    y = x_norm * gamma_bc + beta_bc

    # Compute gradients
    loss = y.sum()
    grads = grad(loss, [x, gamma, beta])

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(3, 64, 32, 32)).astype("float32")
    gamma_val = np.ones(64, dtype="float32")
    beta_val = np.zeros(64, dtype="float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, gamma, beta],
        [y] + grads,
        [x_val, gamma_val, beta_val]
    )

    y_val = outputs[0]
    grad_vals = outputs[1:]

    # Verify output shape preserved
    assert y_val.shape == x_val.shape, \
        f"BN output shape {y_val.shape} != input shape {x_val.shape}"

    # Verify all gradients computed
    for i, g in enumerate(grad_vals):
        assert g is not None, f"Gradient {i} is None"
        assert np.all(np.isfinite(g)), f"Gradient {i} has NaN/Inf"
```

##### Test: `test_dimshuffle_loss_broadcasting`
**Purpose**: Test dimshuffle in loss computation
**Test Data**: Loss weight broadcasting patterns
**Expected Behavior**: Proper broadcasting for element-wise loss
**Assertions**: Loss computed correctly with broadcasting

```python
def test_dimshuffle_loss_broadcasting():
    """
    Test dimshuffle for loss weight broadcasting.

    This test verifies:
    - Broadcasting loss weights to predictions
    - Gradient flow through broadcasted weights
    - Common loss broadcasting patterns
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - loss broadcasting pattern
    pred = pt.tensor3("pred", dtype="float32", shape=(None, 5, 100))  # [B, C, N]
    target = pt.tensor3("target", dtype="float32", shape=(None, 5, 100))
    class_weights = pt.vector("weights", dtype="float32")  # [C]

    # Act - broadcast class weights
    weights_bc = class_weights.dimshuffle('x', 0, 'x')  # [1, C, 1]

    # Compute weighted loss
    loss_per_elem = (pred - target) ** 2
    weighted_loss = loss_per_elem * weights_bc
    loss = weighted_loss.mean()

    # Gradient
    grad_pred = grad(loss, pred)

    # Test data
    rng = np.random.default_rng(42)
    pred_val = rng.normal(size=(2, 5, 100)).astype("float32")
    target_val = rng.normal(size=(2, 5, 100)).astype("float32")
    weights_val = np.array([1.0, 2.0, 0.5, 1.5, 1.0], dtype="float32")

    # Assert
    fn, (loss_val, grad_val) = compare_jax_and_py(
        [pred, target, class_weights],
        [loss, grad_pred],
        [pred_val, target_val, weights_val]
    )

    # Verify loss is scalar
    assert loss_val.shape == (), f"Loss should be scalar, got {loss_val.shape}"

    # Verify gradient shape matches input
    assert grad_val.shape == pred_val.shape
```

#### 2. Concatenate Operation Tests
**Test File**: `tests/link/jax/test_jax_concatenate.py`
**Purpose**: Verify concatenate works with dynamic dimensions

**Test Cases to Write:**

##### Test: `test_concatenate_dynamic_batch`
**Purpose**: Test concatenation with dynamic batch size
**Test Data**: Multiple tensors to concatenate
**Expected Behavior**: Concatenation works with traced dimensions
**Assertions**: Output shape correct, gradients distributed

```python
def test_concatenate_dynamic_batch():
    """
    Test concatenation with dynamic batch dimension.

    This test verifies:
    - Concatenation along channel axis
    - Dynamic batch size handling
    - Gradient distribution to inputs
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - CSP block pattern
    x1 = pt.tensor4("x1", dtype="float32", shape=(None, 128, 32, 32))
    x2 = pt.tensor4("x2", dtype="float32", shape=(None, 128, 32, 32))
    x3 = pt.tensor4("x3", dtype="float32", shape=(None, 64, 32, 32))

    # Act - concatenate along channel axis
    y = pt.concatenate([x1, x2, x3], axis=1)  # Output: [B, 320, 32, 32]

    # Compute gradients
    loss = y.sum()
    grads = grad(loss, [x1, x2, x3])

    # Test data
    rng = np.random.default_rng(42)
    x1_val = rng.normal(size=(2, 128, 32, 32)).astype("float32")
    x2_val = rng.normal(size=(2, 128, 32, 32)).astype("float32")
    x3_val = rng.normal(size=(2, 64, 32, 32)).astype("float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x1, x2, x3],
        [y] + grads,
        [x1_val, x2_val, x3_val]
    )

    y_val = outputs[0]
    grad_vals = outputs[1:]

    # Verify concatenated shape
    assert y_val.shape == (2, 320, 32, 32), \
        f"Concatenated shape {y_val.shape} != (2, 320, 32, 32)"

    # Verify gradients distributed correctly
    assert grad_vals[0].shape == x1_val.shape
    assert grad_vals[1].shape == x2_val.shape
    assert grad_vals[2].shape == x3_val.shape

    # All gradients should be ones (from sum loss)
    for g in grad_vals:
        np.testing.assert_array_almost_equal(g, np.ones_like(g))
```

**Expected Failure Mode**: Should work as concatenate is basic
- Error type: None expected
- Expected message: Should pass

##### Test: `test_concatenate_sppf_pattern`
**Purpose**: Test SPPF-style 4-way concatenation
**Test Data**: Multiple pooled features
**Expected Behavior**: Multi-input concatenation works
**Assertions**: Correct output shape and gradients

```python
def test_concatenate_sppf_pattern():
    """
    Test SPPF-style 4-way concatenation.

    This test verifies:
    - Concatenating 4 feature maps
    - Same spatial size concatenation
    - Gradient flow to all inputs
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - SPPF outputs
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 16, 16))

    # Simulate SPPF pooling outputs (all same spatial size)
    pool1 = x * 1.1  # Simulated pooled features
    pool2 = x * 1.2
    pool3 = x * 1.3

    # Act - 4-way concatenation
    y = pt.concatenate([x, pool1, pool2, pool3], axis=1)

    # Gradient
    loss = y.mean()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 256, 16, 16)).astype("float32")

    # Assert
    fn, (y_val, grad_val) = compare_jax_and_py(
        [x],
        [y, grad_x],
        [x_val]
    )

    # Verify 4x channel concatenation
    assert y_val.shape == (2, 1024, 16, 16), \
        f"SPPF concat shape {y_val.shape} != (2, 1024, 16, 16)"

    # Verify gradient flows back
    assert grad_val.shape == x_val.shape
    assert np.all(np.isfinite(grad_val))
```

#### 3. Tensor Creation Operations Tests
**Test File**: `tests/link/jax/test_jax_tensor_creation.py`
**Purpose**: Test tensor creation with dynamic shapes

**Test Cases to Write:**

##### Test: `test_tensor_creation_operations`
**Purpose**: Test various tensor creation methods
**Test Data**: Different creation patterns
**Expected Behavior**: All creation methods work with JAX
**Assertions**: Correct shapes and values

```python
@pytest.mark.parametrize("create_fn,test_name,expected_check", [
    (lambda x: pt.zeros_like(x), "zeros_like",
     lambda result, x: np.all(result == 0)),
    (lambda x: pt.ones_like(x), "ones_like",
     lambda result, x: np.all(result == 1)),
    (lambda: pt.constant(5.0), "scalar_constant",
     lambda result, x: result == 5.0),
    (lambda: pt.constant([[1, 2], [3, 4]], dtype="float32"), "matrix_constant",
     lambda result, x: np.array_equal(result, [[1, 2], [3, 4]])),
])
def test_tensor_creation_operations(create_fn, test_name, expected_check):
    """
    Test tensor creation operations with JAX backend.

    This test verifies:
    - Various tensor creation methods
    - Dynamic shape handling where applicable
    - Proper dtype preservation
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.matrix("x", dtype="float32") if test_name.endswith("_like") else None

    # Act
    if x is not None:
        y = create_fn(x)
        inputs = [x]
        input_vals = [np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")]
    else:
        y = create_fn()
        inputs = []
        input_vals = []

    # Assert
    fn, (y_val,) = compare_jax_and_py(
        inputs,
        [y],
        input_vals
    )

    # Verify expected values
    x_val = input_vals[0] if input_vals else None
    assert expected_check(y_val, x_val), \
        f"{test_name} produced unexpected values: {y_val}"

    # Verify shape for *_like operations
    if test_name.endswith("_like"):
        assert y_val.shape == x_val.shape, \
            f"{test_name} shape {y_val.shape} != input shape {x_val.shape}"
```

**Expected Failure Mode**: Should mostly work
- Error type: None for basic creation
- Expected message: Should pass

##### Test: `test_as_tensor_variable`
**Purpose**: Test numpy array conversion
**Test Data**: Various numpy arrays
**Expected Behavior**: Proper conversion to tensor variables
**Assertions**: Values and dtypes preserved

```python
def test_as_tensor_variable():
    """
    Test conversion of numpy arrays to tensor variables.

    This test verifies:
    - Numpy to tensor conversion
    - Dtype preservation
    - Shape preservation
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - various numpy arrays
    np_scalar = np.array(5.0, dtype="float32")
    np_vector = np.array([1, 2, 3, 4], dtype="float32")
    np_matrix = np.array([[1, 2], [3, 4]], dtype="float32")

    # Act - convert to tensor variables
    scalar = pt.as_tensor_variable(np_scalar)
    vector = pt.as_tensor_variable(np_vector)
    matrix = pt.as_tensor_variable(np_matrix)

    # Combine in computation
    result = scalar * vector.sum() + matrix.sum()

    # Assert
    fn, (result_val,) = compare_jax_and_py(
        [],
        [result],
        []
    )

    # Expected: 5.0 * 10 + 10 = 60
    expected = 60.0
    np.testing.assert_almost_equal(result_val, expected,
                                   err_msg=f"Result {result_val} != {expected}")
```

##### Test: `test_gradient_initialization_pattern`
**Purpose**: Test gradient initialization patterns
**Test Data**: Parameter-like tensors
**Expected Behavior**: Proper initialization for training
**Assertions**: Gradients initialized correctly

```python
def test_gradient_initialization_pattern():
    """
    Test gradient initialization patterns used in training.

    This test verifies:
    - Zero gradient initialization
    - Gradient accumulation patterns
    - Dynamic shape handling
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad, shared
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - parameters and gradients
    param = pt.matrix("param", dtype="float32")
    x = pt.matrix("x", dtype="float32")

    # Compute loss
    loss = (x @ param).sum()

    # Initialize gradient to zero
    grad_init = pt.zeros_like(param)

    # Compute actual gradient
    grad_param = grad(loss, param)

    # Accumulate (training pattern)
    grad_accum = grad_init + grad_param

    # Test data
    rng = np.random.default_rng(42)
    param_val = rng.normal(size=(4, 3)).astype("float32")
    x_val = rng.normal(size=(2, 4)).astype("float32")

    # Assert
    fn, (grad_init_val, grad_val, grad_accum_val) = compare_jax_and_py(
        [param, x],
        [grad_init, grad_param, grad_accum],
        [param_val, x_val]
    )

    # Verify zero initialization
    np.testing.assert_array_equal(grad_init_val, np.zeros_like(param_val))

    # Verify gradient computed
    assert np.all(np.isfinite(grad_val))

    # Verify accumulation equals gradient (since init was zero)
    np.testing.assert_array_almost_equal(grad_accum_val, grad_val)
```

#### 4. Type Casting Tests
**Test File**: `tests/link/jax/test_jax_cast.py`
**Purpose**: Test type casting operations

**Test Cases to Write:**

##### Test: `test_cast_operations`
**Purpose**: Test casting between dtypes
**Test Data**: Various dtype conversions
**Expected Behavior**: Proper type conversion with JAX
**Assertions**: Dtypes converted correctly

```python
def test_cast_operations():
    """
    Test type casting with JAX backend.

    This test verifies:
    - Float32 casting for ONNX
    - Casting with dynamic shapes
    - Gradient preservation through cast
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - float64 input
    x = pt.matrix("x", dtype="float64")

    # Act - cast to float32 (ONNX pattern)
    x_f32 = pt.cast(x, "float32")

    # Compute something
    y = x_f32 * 2.0

    # Cast back for gradient test
    y_f64 = pt.cast(y, "float64")

    # Gradient (through cast operations)
    grad_x = grad(y_f64.sum(), x)

    # Test data
    x_val = np.array([[1.5, 2.5], [3.5, 4.5]], dtype="float64")

    # Assert
    fn, (x_f32_val, y_val, grad_val) = compare_jax_and_py(
        [x],
        [x_f32, y, grad_x],
        [x_val]
    )

    # Verify dtypes
    assert x_f32_val.dtype == np.float32, \
        f"Cast to float32 failed: {x_f32_val.dtype}"
    assert y_val.dtype == np.float32, \
        f"Operation preserved float32: {y_val.dtype}"

    # Verify gradient flows through cast
    expected_grad = np.ones_like(x_val) * 2.0
    np.testing.assert_array_almost_equal(grad_val, expected_grad)
```

**Expected Failure Mode**: Cast should work
- Error type: None expected
- Expected message: Should pass

##### Test: `test_cast_for_onnx_export`
**Purpose**: Test ONNX export casting pattern
**Test Data**: Model outputs
**Expected Behavior**: Proper float32 enforcement
**Assertions**: All outputs are float32

```python
def test_cast_for_onnx_export():
    """
    Test casting pattern for ONNX export.

    This test verifies:
    - Enforcing float32 for all outputs
    - Handling mixed precision inputs
    - Preserving values during cast
    """
    import pytensor.tensor as pt
    import numpy as np
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - mixed precision model outputs
    out1 = pt.matrix("out1", dtype="float64")
    out2 = pt.tensor3("out2", dtype="float32")
    out3 = pt.vector("out3", dtype="float64")

    # Act - cast all to float32 for ONNX
    onnx_out1 = pt.cast(out1, "float32")
    onnx_out2 = out2  # Already float32
    onnx_out3 = pt.cast(out3, "float32")

    # Test data
    rng = np.random.default_rng(42)
    out1_val = rng.normal(size=(2, 3)).astype("float64")
    out2_val = rng.normal(size=(2, 3, 4)).astype("float32")
    out3_val = rng.normal(size=(5,)).astype("float64")

    # Assert
    fn, outputs = compare_jax_and_py(
        [out1, out2, out3],
        [onnx_out1, onnx_out2, onnx_out3],
        [out1_val, out2_val, out3_val]
    )

    # Verify all outputs are float32
    for i, out in enumerate(outputs):
        assert out.dtype == np.float32, \
            f"Output {i} not float32: {out.dtype}"

    # Verify values preserved (within float32 precision)
    np.testing.assert_allclose(outputs[0], out1_val.astype("float32"),
                               rtol=1e-6)
    np.testing.assert_array_equal(outputs[1], out2_val)
    np.testing.assert_allclose(outputs[2], out3_val.astype("float32"),
                               rtol=1e-6)
```

### Test Implementation Steps:

1. **Create test file structure**:
   ```bash
   tests/link/jax/
   ├── test_jax_dimshuffle.py
   ├── test_jax_concatenate.py
   ├── test_jax_tensor_creation.py
   └── test_jax_cast.py
   ```

2. **Import necessary utilities**:
   ```python
   import numpy as np
   import pytest
   import pytensor.tensor as pt
   from pytensor import grad, shared
   from pytensor.configdefaults import config
   from tests.link.jax.test_basic import compare_jax_and_py

   jax = pytest.importorskip("jax")
   ```

3. **Create test fixtures**:
   ```python
   @pytest.fixture
   def tensor_shapes():
       """Common tensor shapes for testing."""
       return {
           "batch": (2, 3, 32, 32),
           "features": (2, 256, 16, 16),
           "vector": (256,),
           "matrix": (4, 5),
       }
   ```

### Success Criteria:

#### Automated Verification:
- [ ] All test files created and importable
- [ ] Tests use compare_jax_and_py correctly
- [ ] Parametrized tests work properly
- [ ] Tests are discoverable

#### Manual Verification:
- [ ] Dimshuffle patterns tested thoroughly
- [ ] Concatenation with dynamic batch tested
- [ ] All tensor creation methods covered
- [ ] Cast operations verified

---

## Phase 2: Test Failure Verification

### Overview
Run tests and document which operations need fixes.

### Verification Steps:

1. **Run dimshuffle tests**:
   ```bash
   pytest tests/link/jax/test_jax_dimshuffle.py -v
   ```

2. **Run concatenate tests**:
   ```bash
   pytest tests/link/jax/test_jax_concatenate.py -v
   ```

3. **Run tensor creation tests**:
   ```bash
   pytest tests/link/jax/test_jax_tensor_creation.py -v
   ```

4. **Run cast tests**:
   ```bash
   pytest tests/link/jax/test_jax_cast.py -v
   ```

### Expected Failures:

- **test_dimshuffle_broadcasting**: May fail on 'x' dimension handling
- **test_concatenate_dynamic_batch**: Expected to pass
- **test_tensor_creation_operations**: Expected to mostly pass
- **test_cast_operations**: Expected to pass

### Success Criteria:

#### Automated Verification:
- [ ] All tests run without import errors
- [ ] Document which tests pass/fail
- [ ] Error messages are informative

#### Manual Verification:
- [ ] Failures point to specific issues
- [ ] Can identify what needs fixing
- [ ] Stack traces helpful

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Fix operations that don't handle tracers properly.

### Implementation Strategy:

Focus on operations that fail:
1. Fix dimshuffle if needed
2. Verify concatenate works
3. Ensure tensor creation handles dynamic shapes
4. Verify cast operations

### Implementation Steps:

#### Implementation 1: Fix Dimshuffle

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Changes**: Ensure dimshuffle handles 'x' broadcasting

```python
@jax_funcify.register(DimShuffle)
def jax_funcify_DimShuffle(op, node, **kwargs):
    """JAX implementation of dimension shuffle."""

    def dimshuffle(x):
        # Handle 'x' for new axes and reordering
        res = x

        # Add new axes for 'x' entries
        for i, dim in enumerate(op.new_order):
            if dim == 'x':
                res = jnp.expand_dims(res, axis=i)

        # Reorder dimensions
        perm = [i for i, d in enumerate(op.new_order) if d != 'x']
        if perm:
            res = jnp.transpose(res, perm)

        return res

    return dimshuffle
```

#### Implementation 2: Verify Concatenate

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Expected**: Should already work

```python
# Concatenate should be handled by existing dispatch
# Verify it works with dynamic batch dimensions
```

#### Implementation 3: Tensor Creation

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Expected**: Most creation ops should work

```python
# zeros_like, ones_like should be handled
# Constant creation should work
# as_tensor_variable conversion should work
```

#### Implementation 4: Cast Operations

**File**: `pytensor/link/jax/dispatch/tensor_basic.py`
**Expected**: Cast should work with JAX

```python
# Cast should use jnp.astype
# Verify gradient flows through cast
```

### Success Criteria:

##### Automated Verification:
- [ ] All dimshuffle tests pass
- [ ] All concatenate tests pass
- [ ] All tensor creation tests pass
- [ ] All cast tests pass

##### Manual Verification:
- [ ] Operations handle dynamic shapes
- [ ] Gradients flow correctly
- [ ] No performance issues

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve code quality and optimize implementations.

### Refactoring Targets:

1. **Code Organization**:
   - Group related operations
   - Extract common patterns
   - Improve documentation

2. **Performance**:
   - Optimize broadcasting operations
   - Cache dimension permutations
   - Avoid unnecessary copies

3. **Error Handling**:
   - Add informative error messages
   - Validate inputs where needed
   - Document limitations

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] No performance regressions
- [ ] Code coverage maintained

#### Manual Verification:
- [ ] Code is well-organized
- [ ] Documentation improved
- [ ] No unnecessary complexity

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] All dimshuffle patterns tested
- [ ] Concatenation with dynamic batch tested
- [ ] Tensor creation methods covered
- [ ] Type casting verified
- [ ] Gradient flow tested

### Test Organization:
- Test files: One per operation category
- Fixtures: Shared shape configurations
- Utilities: Use compare_jax_and_py
- Patterns: Test real usage patterns

### Running Tests:

```bash
# Run all tensor manipulation tests
pytest tests/link/jax/test_jax_dimshuffle.py test_jax_concatenate.py test_jax_tensor_creation.py test_jax_cast.py -v

# Run with coverage
pytest tests/link/jax/test_jax_*.py --cov=pytensor.link.jax.dispatch

# Run specific test
pytest tests/link/jax/test_jax_dimshuffle.py::test_dimshuffle_broadcasting -v
```

## Performance Considerations

These operations are frequently used:
- Dimshuffle in every BN layer
- Concatenate in CSP blocks
- Tensor creation for initialization
- Cast for ONNX export

### Performance Testing:
- [ ] Benchmark dimshuffle patterns
- [ ] Profile concatenation
- [ ] Measure creation overhead
- [ ] Test cast performance

## Migration Notes

These operations should be transparent:
- No model code changes needed
- Existing tests should pass
- Better JAX integration

## References

- Original research: `thoughts/shared/research/2025-10-23_20-11-25_jax-backend-missing-operations-test-coverage.md`
- Dimshuffle in BN: `blocks.py:50-53`
- Concatenate in blocks: CSP patterns
- Tensor creation: `model.py:357`, `train.py:247`
- Cast for ONNX: `train.py:276`
- Test utilities: `tests/link/jax/test_basic.py:36-96`