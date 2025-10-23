# JAX Backend Composite Patterns TDD Implementation Plan

## Overview

This plan addresses complex composite patterns that combine multiple operations: complete Batch Normalization (custom implementation for JAX), SiLU activation pattern (x * sigmoid(x)), and CSP (Cross Stage Partial) split-process-merge patterns. These patterns are fundamental to YOLO11 architecture, with ConvBNSiLU used 22+ times and CSP patterns in C3k2/C2PSA blocks. We'll implement comprehensive tests first, verify failures, then ensure these patterns work seamlessly with JAX.

## Current State Analysis

Composite patterns combine multiple operations that may individually work but fail when combined due to tracer propagation issues. Batch normalization involves complex statistics computation, parameter broadcasting, and mode switching. SiLU requires element-wise multiplication with sigmoid. CSP patterns involve channel splitting, parallel processing, and concatenation.

### Current Testing Landscape:
- Testing framework: pytest
- Available test utilities: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()`
- Existing patterns: Individual op tests exist but not composite patterns
- Test fixtures/mocks available: `jax_mode`, `py_mode` configurations

## Desired End State

All composite patterns should work end-to-end with JAX tracers, enabling complete YOLO11 model training. Batch normalization should handle training/inference modes, running statistics, and affine transformations. SiLU should compute efficiently with proper gradient flow. CSP patterns should handle split/merge operations seamlessly.

### Key Discoveries:
- BN pattern in ConvBNSiLU: Custom implementation without standard BN ops
- SiLU in every ConvBNSiLU: `blocks.py:177`
- CSP in C3k2: Split-process-concatenate pattern
- C2PSA: Attention-based CSP variant
- SPPF: Special pooling-based CSP pattern

## What We're NOT Testing/Implementing

- Batch normalization variants not used in YOLO (GroupNorm, LayerNorm)
- Activation patterns beyond SiLU
- CSP patterns not in YOLO11
- Complex attention mechanisms beyond C2PSA

## TDD Approach

Write comprehensive tests for complete patterns as they appear in YOLO11, verify they fail if component operations have issues, then ensure the full patterns work with JAX.

### Test Design Philosophy:
- Test complete patterns as atomic units
- Verify both forward and backward passes
- Test training vs inference modes where applicable
- Ensure numerical stability and correct statistics

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests for composite patterns that define their complete behavior with JAX tracers.

### Test Categories:

#### 1. Complete Batch Normalization Pattern Tests
**Test File**: `tests/link/jax/test_jax_batch_norm_complete.py`
**Purpose**: Test complete BN implementation as used in YOLO11

**Test Cases to Write:**

##### Test: `test_batch_norm_complete_pattern`
**Purpose**: Test complete BN pattern from ConvBNSiLU
**Test Data**: Typical CNN feature maps
**Expected Behavior**: Full BN with statistics and affine transform
**Assertions**: Correct normalization, gradient flow

```python
def test_batch_norm_complete_pattern():
    """
    Test complete batch normalization pattern from ConvBNSiLU.

    This test verifies:
    - Custom BN implementation for JAX
    - Running mean/var computation
    - Training vs inference modes
    - Gradient flow through BN
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad, shared
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - BN with learnable parameters
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # BN parameters
    gamma = pt.vector("gamma", dtype="float32")  # Scale
    beta = pt.vector("beta", dtype="float32")   # Shift

    # Running statistics (shared variables)
    running_mean = shared(np.zeros(256, dtype="float32"), name="running_mean")
    running_var = shared(np.ones(256, dtype="float32"), name="running_var")

    # Training mode flag
    training = True
    momentum = 0.1
    eps = 1e-5

    # Act - Batch normalization implementation
    if training:
        # Compute batch statistics
        batch_mean = x.mean(axis=(0, 2, 3))
        batch_var = x.var(axis=(0, 2, 3))

        # Update running statistics (exponential moving average)
        new_running_mean = momentum * batch_mean + (1 - momentum) * running_mean
        new_running_var = momentum * batch_var + (1 - momentum) * running_var

        # Normalize using batch statistics
        x_normalized = (x - batch_mean.dimshuffle('x', 0, 'x', 'x')) / \
                       pt.sqrt(batch_var.dimshuffle('x', 0, 'x', 'x') + eps)
    else:
        # Use running statistics for inference
        x_normalized = (x - running_mean.dimshuffle('x', 0, 'x', 'x')) / \
                       pt.sqrt(running_var.dimshuffle('x', 0, 'x', 'x') + eps)

    # Apply affine transformation
    y = x_normalized * gamma.dimshuffle('x', 0, 'x', 'x') + \
        beta.dimshuffle('x', 0, 'x', 'x')

    # Compute gradients
    loss = y.sum()
    grads = grad(loss, [x, gamma, beta])

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 256, 32, 32)).astype("float32")
    gamma_val = np.ones(256, dtype="float32")
    beta_val = np.zeros(256, dtype="float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, gamma, beta],
        [y, batch_mean if training else running_mean,
         batch_var if training else running_var] + grads,
        [x_val, gamma_val, beta_val]
    )

    y_val = outputs[0]
    mean_val = outputs[1]
    var_val = outputs[2]
    grad_vals = outputs[3:]

    # Verify normalization
    assert y_val.shape == x_val.shape, "BN output shape mismatch"

    # Check statistics computation
    if training:
        expected_mean = x_val.mean(axis=(0, 2, 3))
        expected_var = x_val.var(axis=(0, 2, 3))
        np.testing.assert_allclose(mean_val, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(var_val, expected_var, rtol=1e-5)

    # Verify all gradients computed
    for i, g in enumerate(grad_vals):
        assert g is not None, f"Gradient {i} is None"
        assert np.all(np.isfinite(g)), f"Gradient {i} has NaN/Inf"
```

**Expected Failure Mode**: Complex pattern may have issues
- Error type: Potential issues with statistics computation or broadcasting
- Expected message: May fail on dimshuffle or mean/var computation

##### Test: `test_batch_norm_inference_mode`
**Purpose**: Test BN in inference mode with frozen statistics
**Test Data**: Fixed running statistics
**Expected Behavior**: Uses running stats, no updates
**Assertions**: Deterministic output, no stat updates

```python
def test_batch_norm_inference_mode():
    """
    Test batch normalization in inference mode.

    This test verifies:
    - Using running statistics for normalization
    - No updates to running stats
    - Deterministic behavior
    - Gradient computation still works
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad, shared
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32", shape=(None, 64, 16, 16))
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")

    # Pre-computed statistics
    running_mean_val = np.random.randn(64).astype("float32") * 0.1
    running_var_val = np.random.uniform(0.5, 1.5, 64).astype("float32")

    running_mean = pt.constant(running_mean_val)
    running_var = pt.constant(running_var_val)

    eps = 1e-5

    # Act - Inference mode BN
    x_normalized = (x - running_mean.dimshuffle('x', 0, 'x', 'x')) / \
                   pt.sqrt(running_var.dimshuffle('x', 0, 'x', 'x') + eps)

    y = x_normalized * gamma.dimshuffle('x', 0, 'x', 'x') + \
        beta.dimshuffle('x', 0, 'x', 'x')

    # Gradient (can still compute in inference)
    loss = y.mean()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 64, 16, 16)).astype("float32")
    gamma_val = np.ones(64, dtype="float32")
    beta_val = np.zeros(64, dtype="float32")

    # Assert - run twice to verify determinism
    fn, (y_val1, grad_val1) = compare_jax_and_py(
        [x, gamma, beta],
        [y, grad_x],
        [x_val, gamma_val, beta_val]
    )

    fn2, (y_val2, grad_val2) = compare_jax_and_py(
        [x, gamma, beta],
        [y, grad_x],
        [x_val, gamma_val, beta_val]
    )

    # Verify deterministic output
    np.testing.assert_array_equal(y_val1, y_val2,
                                  "Inference BN not deterministic")

    # Verify gradient exists
    assert np.all(np.isfinite(grad_val1)), "Gradient has NaN/Inf"
```

##### Test: `test_batch_norm_gradient_flow`
**Purpose**: Test gradient flow through BN layers
**Test Data**: Multi-layer BN setup
**Expected Behavior**: Gradients flow through normalization
**Assertions**: No gradient vanishing/explosion

```python
def test_batch_norm_gradient_flow():
    """
    Test gradient flow through batch normalization.

    This test verifies:
    - Gradients flow to input, gamma, and beta
    - No gradient vanishing in deep networks
    - Proper gradient scaling
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - simplified BN for gradient testing
    x = pt.tensor4("x", dtype="float32", shape=(None, 128, 8, 8))
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")

    # Compute BN
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - mean) / pt.sqrt(var + 1e-5)

    # Apply affine
    y = x_norm * gamma.dimshuffle('x', 0, 'x', 'x') + \
        beta.dimshuffle('x', 0, 'x', 'x')

    # Loss with specific target
    target = pt.tensor4("target", dtype="float32", shape=(None, 128, 8, 8))
    loss = ((y - target) ** 2).mean()

    # Compute all gradients
    grad_x, grad_gamma, grad_beta = grad(loss, [x, gamma, beta])

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(4, 128, 8, 8)).astype("float32")
    gamma_val = np.ones(128, dtype="float32")
    beta_val = np.zeros(128, dtype="float32")
    target_val = rng.normal(size=(4, 128, 8, 8)).astype("float32") * 0.1

    # Assert
    fn, (loss_val, gx, gg, gb) = compare_jax_and_py(
        [x, gamma, beta, target],
        [loss, grad_x, grad_gamma, grad_beta],
        [x_val, gamma_val, beta_val, target_val]
    )

    # Verify gradients are reasonable
    assert np.all(np.isfinite(gx)), "Input gradient has NaN/Inf"
    assert np.all(np.isfinite(gg)), "Gamma gradient has NaN/Inf"
    assert np.all(np.isfinite(gb)), "Beta gradient has NaN/Inf"

    # Check gradient magnitudes are reasonable
    gx_norm = np.linalg.norm(gx)
    assert 0.001 < gx_norm < 1000, f"Input gradient norm {gx_norm} out of range"
```

#### 2. SiLU Activation Pattern Tests
**Test File**: `tests/link/jax/test_jax_silu_pattern.py`
**Purpose**: Test complete SiLU pattern as used in ConvBNSiLU

**Test Cases to Write:**

##### Test: `test_silu_activation_pattern`
**Purpose**: Test complete SiLU (x * sigmoid(x)) pattern
**Test Data**: Various input ranges
**Expected Behavior**: Correct SiLU computation and gradients
**Assertions**: Matches reference implementation

```python
def test_silu_activation_pattern():
    """
    Test complete SiLU (Swish) activation pattern.

    This test verifies:
    - x * sigmoid(x) computation
    - Gradient computation
    - Numerical stability
    - Performance with large tensors
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32", shape=(None, 512, 16, 16))

    # Act - SiLU pattern
    sigmoid_x = pt.sigmoid(x)
    silu = x * sigmoid_x

    # Gradient
    loss = silu.mean()
    grad_x = grad(loss, x)

    # Test data with various ranges
    rng = np.random.default_rng(42)
    x_val = np.concatenate([
        rng.normal(0, 1, size=(1, 512, 16, 16)),    # Normal range
        rng.normal(0, 10, size=(1, 512, 16, 16)),   # Large values
        rng.normal(0, 0.1, size=(1, 512, 16, 16)),  # Small values
    ], axis=0).astype("float32")

    # Assert
    fn, (silu_val, grad_val) = compare_jax_and_py(
        [x],
        [silu, grad_x],
        [x_val]
    )

    # Verify against reference implementation
    def silu_ref(x):
        return x / (1 + np.exp(-x))

    expected = silu_ref(x_val)
    np.testing.assert_allclose(silu_val, expected, rtol=1e-5,
                               err_msg="SiLU output doesn't match reference")

    # Verify gradient computation
    def silu_grad_ref(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 + x * (1 - sig))

    expected_grad = silu_grad_ref(x_val) / x_val.size  # Divided by size for mean
    np.testing.assert_allclose(grad_val, expected_grad, rtol=1e-4,
                               err_msg="SiLU gradient doesn't match reference")
```

**Expected Failure Mode**: Should work if sigmoid and multiplication work
- Error type: None expected
- Expected message: Should pass

##### Test: `test_silu_in_convbnsilu_block`
**Purpose**: Test SiLU integrated in complete ConvBNSiLU
**Test Data**: Complete block computation
**Expected Behavior**: SiLU works after BN
**Assertions**: Full block gradient flow

```python
def test_silu_in_convbnsilu_block():
    """
    Test SiLU activation in complete ConvBNSiLU block.

    This test verifies:
    - SiLU after batch normalization
    - Gradient flow through complete block
    - No numerical issues in combination
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.conv import conv2d
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - complete ConvBNSiLU
    x = pt.tensor4("x", dtype="float32", shape=(None, 128, 32, 32))
    filters = pt.tensor4("filters", dtype="float32")
    gamma = pt.vector("gamma", dtype="float32")
    beta = pt.vector("beta", dtype="float32")

    # Conv2D
    conv_out = conv2d(x, filters, border_mode="same")

    # Batch Norm (simplified)
    mean = conv_out.mean(axis=(0, 2, 3), keepdims=True)
    var = conv_out.var(axis=(0, 2, 3), keepdims=True)
    bn_out = (conv_out - mean) / pt.sqrt(var + 1e-5)
    bn_out = bn_out * gamma.dimshuffle('x', 0, 'x', 'x') + \
              beta.dimshuffle('x', 0, 'x', 'x')

    # SiLU activation
    output = bn_out * pt.sigmoid(bn_out)

    # Gradient through entire block
    loss = output.sum()
    grads = grad(loss, [x, filters, gamma, beta])

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 128, 32, 32)).astype("float32") * 0.1
    filters_val = rng.normal(size=(256, 128, 3, 3)).astype("float32") * 0.02
    gamma_val = np.ones(256, dtype="float32")
    beta_val = np.zeros(256, dtype="float32")

    # Assert
    fn, outputs = compare_jax_and_py(
        [x, filters, gamma, beta],
        [output] + grads,
        [x_val, filters_val, gamma_val, beta_val]
    )

    output_val = outputs[0]
    grad_vals = outputs[1:]

    # Verify output shape
    assert output_val.shape == (2, 256, 32, 32), \
        f"ConvBNSiLU output shape {output_val.shape} != expected"

    # Verify all gradients flow
    for i, g in enumerate(grad_vals):
        assert g is not None, f"Gradient {i} is None"
        assert np.all(np.isfinite(g)), f"Gradient {i} has NaN/Inf"
        assert np.abs(g).sum() > 0, f"Gradient {i} is zero"
```

#### 3. CSP Pattern Tests
**Test File**: `tests/link/jax/test_jax_csp_patterns.py`
**Purpose**: Test Cross Stage Partial patterns

**Test Cases to Write:**

##### Test: `test_csp_split_merge_pattern`
**Purpose**: Test basic CSP split-process-merge
**Test Data**: Feature maps for splitting
**Expected Behavior**: Proper channel split and merge
**Assertions**: Gradient flows through both paths

```python
def test_csp_split_merge_pattern():
    """
    Test Cross Stage Partial split-process-merge pattern.

    This test verifies:
    - Channel splitting (slicing)
    - Parallel processing paths
    - Concatenation of processed features
    - Gradient distribution through split paths
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - CSP pattern
    x = pt.tensor4("x", dtype="float32", shape=(None, 256, 32, 32))

    # Split channels
    split_idx = 128
    x1 = x[:, :split_idx, :, :]     # First half
    x2 = x[:, split_idx:, :, :]     # Second half

    # Process paths differently (simplified)
    # Path 1: Direct connection
    y1 = x1

    # Path 2: Processing (simplified as scaling)
    y2 = x2 * 2.0 + 1.0

    # Merge
    output = pt.concatenate([y1, y2], axis=1)

    # Gradient
    loss = output.sum()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 256, 32, 32)).astype("float32")

    # Assert
    fn, (output_val, grad_val) = compare_jax_and_py(
        [x],
        [output, grad_x],
        [x_val]
    )

    # Verify shape preserved
    assert output_val.shape == x_val.shape, \
        f"CSP output shape {output_val.shape} != input shape"

    # Verify gradient flows through both paths
    grad_path1 = grad_val[:, :split_idx, :, :]
    grad_path2 = grad_val[:, split_idx:, :, :]

    # Path 1 gradient should be 1 (direct connection)
    np.testing.assert_array_almost_equal(grad_path1, np.ones_like(grad_path1))

    # Path 2 gradient should be 2 (from scaling)
    np.testing.assert_array_almost_equal(grad_path2, np.ones_like(grad_path2) * 2)
```

**Expected Failure Mode**: Should work if slicing and concat work
- Error type: None expected
- Expected message: Should pass

##### Test: `test_c3k2_block_pattern`
**Purpose**: Test C3k2 block pattern from YOLO
**Test Data**: Typical C3k2 dimensions
**Expected Behavior**: Complete block works
**Assertions**: Proper bottleneck processing

```python
def test_c3k2_block_pattern():
    """
    Test C3k2 block pattern with bottlenecks.

    This test verifies:
    - Initial channel adjustment
    - Bottleneck processing
    - CSP split/merge
    - Final projection
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange - C3k2 structure
    x = pt.tensor4("x", dtype="float32", shape=(None, 512, 16, 16))

    # Parameters for convolutions (simplified)
    w1 = pt.tensor4("w1", dtype="float32")  # Input projection
    w2 = pt.tensor4("w2", dtype="float32")  # Bottleneck conv1
    w3 = pt.tensor4("w3", dtype="float32")  # Bottleneck conv2
    w4 = pt.tensor4("w4", dtype="float32")  # Output projection

    # Channel dimensions
    hidden_channels = 256

    # Initial projection (simplified as matrix multiply)
    def simple_conv(x, w):
        # Simplified convolution as matrix multiply
        batch, in_ch, h, w = x.shape
        out_ch = w.shape[0]
        x_flat = x.reshape((batch, in_ch, -1))
        w_flat = w.reshape((out_ch, in_ch, -1)).mean(axis=2)
        out_flat = pt.dot(w_flat, x_flat)
        return out_flat.reshape((batch, out_ch, h, w))

    # Act - C3k2 processing
    # Project to hidden channels
    h1 = simple_conv(x, w1)

    # Split
    h1_split = h1[:, :hidden_channels//2, :, :]
    h2_split = h1[:, hidden_channels//2:, :, :]

    # Bottleneck on second split
    h2_bn1 = simple_conv(h2_split, w2)
    h2_bn2 = simple_conv(h2_bn1, w3)

    # Merge
    merged = pt.concatenate([h1_split, h2_bn2], axis=1)

    # Final projection
    output = simple_conv(merged, w4)

    # Gradient
    loss = output.mean()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 512, 16, 16)).astype("float32") * 0.1
    w1_val = rng.normal(size=(256, 512, 3, 3)).astype("float32") * 0.02
    w2_val = rng.normal(size=(64, 128, 3, 3)).astype("float32") * 0.02
    w3_val = rng.normal(size=(128, 64, 3, 3)).astype("float32") * 0.02
    w4_val = rng.normal(size=(512, 256, 1, 1)).astype("float32") * 0.02

    # Assert
    fn, (output_val, grad_val) = compare_jax_and_py(
        [x, w1, w2, w3, w4],
        [output, grad_x],
        [x_val, w1_val, w2_val, w3_val, w4_val]
    )

    # Verify shape preserved
    assert output_val.shape == x_val.shape, \
        f"C3k2 should preserve shape: {output_val.shape} != {x_val.shape}"

    # Verify gradient flows
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is zero"
```

##### Test: `test_sppf_pattern`
**Purpose**: Test SPPF multi-scale pooling pattern
**Test Data**: Features for multi-scale pooling
**Expected Behavior**: Cascaded pooling and concatenation
**Assertions**: Multi-scale features extracted

```python
def test_sppf_pattern():
    """
    Test SPPF (Spatial Pyramid Pooling Fast) pattern.

    This test verifies:
    - Initial 1x1 convolution
    - Cascaded max pooling (3x)
    - 4-way concatenation
    - Final 1x1 convolution
    """
    import pytensor.tensor as pt
    import numpy as np
    from pytensor.tensor.pool import pool_2d
    from pytensor import grad
    from tests.link.jax.test_basic import compare_jax_and_py

    # Arrange
    x = pt.tensor4("x", dtype="float32", shape=(None, 512, 16, 16))

    # SPPF structure (simplified - no conv for clarity)
    # Initial feature
    h0 = x

    # Cascaded pooling (5x5, stride 1, padding 2)
    pool1 = pool_2d(h0, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool2 = pool_2d(pool1, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")
    pool3 = pool_2d(pool2, ws=(5, 5), stride=(1, 1), padding=(2, 2), mode="max")

    # 4-way concatenation
    output = pt.concatenate([h0, pool1, pool2, pool3], axis=1)

    # Gradient
    loss = output.sum()
    grad_x = grad(loss, x)

    # Test data
    rng = np.random.default_rng(42)
    x_val = rng.normal(size=(2, 512, 16, 16)).astype("float32")

    # Assert
    fn, (output_val, grad_val) = compare_jax_and_py(
        [x],
        [output, grad_x],
        [x_val]
    )

    # Verify 4x channel expansion
    assert output_val.shape == (2, 2048, 16, 16), \
        f"SPPF output shape {output_val.shape} != (2, 2048, 16, 16)"

    # Verify gradient flows
    assert np.all(np.isfinite(grad_val)), "Gradient has NaN/Inf"

    # Gradient should be larger than 1 due to multiple paths
    mean_grad = np.abs(grad_val).mean()
    assert mean_grad > 1, f"Mean gradient {mean_grad} suggests blocked paths"
```

### Test Implementation Steps:

1. **Create test file structure**:
   ```bash
   tests/link/jax/
   ├── test_jax_batch_norm_complete.py
   ├── test_jax_silu_pattern.py
   └── test_jax_csp_patterns.py
   ```

2. **Import necessary utilities**:
   ```python
   import numpy as np
   import pytest
   import pytensor.tensor as pt
   from pytensor.tensor.conv import conv2d
   from pytensor.tensor.pool import pool_2d
   from pytensor import grad, shared
   from pytensor.configdefaults import config
   from tests.link.jax.test_basic import compare_jax_and_py

   jax = pytest.importorskip("jax")
   ```

3. **Create test fixtures**:
   ```python
   @pytest.fixture
   def bn_test_data():
       """Test data for batch normalization."""
       rng = np.random.default_rng(42)
       return {
           "x": rng.normal(size=(4, 256, 32, 32)).astype("float32"),
           "gamma": np.ones(256, dtype="float32"),
           "beta": np.zeros(256, dtype="float32"),
       }

   @pytest.fixture
   def csp_test_data():
       """Test data for CSP patterns."""
       rng = np.random.default_rng(42)
       return {
           "features": rng.normal(size=(2, 512, 16, 16)).astype("float32"),
       }
   ```

### Success Criteria:

#### Automated Verification:
- [ ] All test files created and importable
- [ ] Tests cover complete patterns
- [ ] Forward and backward passes tested
- [ ] Tests are discoverable

#### Manual Verification:
- [ ] BN training vs inference tested
- [ ] SiLU pattern verified
- [ ] CSP split/merge tested
- [ ] SPPF multi-scale tested

---

## Phase 2: Test Failure Verification

### Overview
Run tests to identify which composite patterns have issues.

### Verification Steps:

1. **Run batch norm tests**:
   ```bash
   pytest tests/link/jax/test_jax_batch_norm_complete.py -v
   ```

2. **Run SiLU pattern tests**:
   ```bash
   pytest tests/link/jax/test_jax_silu_pattern.py -v
   ```

3. **Run CSP pattern tests**:
   ```bash
   pytest tests/link/jax/test_jax_csp_patterns.py -v
   ```

### Expected Failures:

- **test_batch_norm_complete_pattern**: May fail on statistics or broadcasting
- **test_silu_activation_pattern**: Expected to pass
- **test_csp_split_merge_pattern**: Expected to pass
- **test_sppf_pattern**: Depends on pooling implementation

### Success Criteria:

#### Automated Verification:
- [ ] All tests run
- [ ] Failures documented
- [ ] Error messages clear

#### Manual Verification:
- [ ] Can identify failing components
- [ ] Understand failure causes
- [ ] Have path to fix

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Fix composite patterns by ensuring all component operations work together.

### Implementation Strategy:

1. Fix individual operations if needed
2. Ensure operations compose properly
3. Test end-to-end patterns
4. Optimize for performance

### Implementation Steps:

#### Implementation 1: Fix Batch Normalization

Ensure all BN components work:
- Mean/variance computation with dynamic batch
- Dimshuffle for broadcasting
- Running statistics updates
- Gradient flow

#### Implementation 2: Verify SiLU

Should work if sigmoid and multiplication work:
- Verify sigmoid handles all ranges
- Ensure element-wise multiplication works
- Check gradient computation

#### Implementation 3: Verify CSP Patterns

Should work if slicing and concatenation work:
- Channel slicing with dynamic batch
- Concatenation along channel axis
- Gradient distribution

### Success Criteria:

##### Automated Verification:
- [ ] All BN tests pass
- [ ] All SiLU tests pass
- [ ] All CSP tests pass
- [ ] No regressions

##### Manual Verification:
- [ ] Patterns match YOLO implementation
- [ ] Numerically stable
- [ ] Performance acceptable

---

## Phase 4: Refactoring & Cleanup

### Overview
Optimize composite patterns for performance and maintainability.

### Refactoring Targets:

1. **Performance Optimization**:
   - Fuse operations where possible
   - Minimize memory allocations
   - Use JAX-specific optimizations

2. **Code Organization**:
   - Create reusable pattern implementations
   - Document pattern usage
   - Add pattern validation

3. **Numerical Stability**:
   - Add stability checks
   - Document precision requirements
   - Test edge cases

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass
- [ ] Performance improved
- [ ] Code coverage maintained

#### Manual Verification:
- [ ] Patterns well-documented
- [ ] Code reusable
- [ ] Optimizations documented

---

## Testing Strategy Summary

### Test Coverage Goals:
- [ ] Complete BN pattern tested
- [ ] SiLU activation verified
- [ ] CSP patterns covered
- [ ] SPPF pattern tested
- [ ] All patterns work end-to-end

### Test Organization:
- Test files: One per pattern type
- Fixtures: Pattern-specific test data
- Utilities: Use compare_jax_and_py
- Integration: Test complete blocks

### Running Tests:

```bash
# Run all composite pattern tests
pytest tests/link/jax/test_jax_batch_norm_complete.py test_jax_silu_pattern.py test_jax_csp_patterns.py -v

# Run with coverage
pytest tests/link/jax/test_jax_*pattern*.py --cov=pytensor.link.jax

# Run specific pattern test
pytest tests/link/jax/test_jax_batch_norm_complete.py::test_batch_norm_complete_pattern -v
```

## Performance Considerations

Composite patterns are performance-critical:
- ConvBNSiLU used 22+ times
- CSP patterns in every block
- Batch norm in every conv layer
- Should leverage JAX's fusion capabilities

### Performance Testing:
- [ ] Benchmark complete blocks
- [ ] Profile pattern execution
- [ ] Measure memory usage
- [ ] Test JIT compilation time

## Migration Notes

These patterns are building blocks:
- No changes to model architecture
- Existing models should work
- Better JAX integration
- Potential for optimization

## References

- Original research: `thoughts/shared/research/2025-10-23_20-11-25_jax-backend-missing-operations-test-coverage.md`
- ConvBNSiLU: `blocks.py:163-177`
- C3k2 block: Used throughout model
- SPPF: `blocks.py:362-382`
- Test utilities: `tests/link/jax/test_basic.py:36-96`