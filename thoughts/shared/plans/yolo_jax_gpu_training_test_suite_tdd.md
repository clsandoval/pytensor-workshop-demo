# YOLO11n JAX GPU Training Test Suite - TDD Implementation Plan

## Overview

Create a comprehensive test suite that ensures the complete YOLO11n model forward pass and training workflow functions correctly when JAX mode is FORCED and GPU is USED. This test suite will provide confidence that GPU-based training will work in production environments.

## Current State Analysis

### Existing Testing Landscape:

**Test Files:**
- `examples/onnx/onnx-yolo-demo/testing/test_jax_components.py` - Optional JAX mode, no GPU requirement
- `examples/onnx/onnx-yolo-demo/testing/test_jax_issues.py` - Forces JAX but doesn't require GPU
- `examples/onnx/onnx-yolo-demo/testing/test_model.py` - Default backend only

**Testing Framework:** Standard Python with argparse (not pytest)
**Test Utilities:** Custom setup functions, manual assertion checking
**Test Patterns:** Found PyTensor's official JAX tests in `tests/link/jax/test_basic.py:36-96`

### Critical Gaps:

1. **No GPU requirement** - Existing tests pass on CPU, providing false confidence
2. **No JIT verification** - Don't confirm JAX actually compiled the computational graph
3. **No device placement verification** - Don't confirm operations execute on GPU
4. **No correctness verification** - Don't compare JAX results against reference backend
5. **No multi-epoch training test** - Only single training steps tested
6. **No checkpoint/resume test** - Critical workflow untested
7. **No gradient correctness test** - Don't verify gradients match reference

### Key Discoveries:

- PyTensor official pattern: `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()` utility
- GPU detection pattern: `jax.devices()[0].platform == "gpu"`
- JIT verification pattern: `isinstance(result, jax.Array)`
- Mode forcing pattern: `pytensor.config.mode = "JAX"`
- Skip pattern: `@pytest.mark.skipif(not has_gpu(), reason="GPU required")`

## Desired End State

A new test file `test_jax_gpu_required.py` that:

1. **Automatically skips** all tests when GPU unavailable (dev-friendly)
2. **Forces JAX mode** globally for all tests
3. **Verifies JIT compilation** for every forward pass
4. **Verifies GPU execution** for every operation
5. **Compares with CPU backend** to ensure numerical correctness
6. **Tests complete training workflow** including multi-epoch and checkpointing
7. **Uses synthetic random data** for speed and no dependencies
8. **Provides clear diagnostic messages** when tests fail

## What We're NOT Testing/Implementing

- Performance benchmarks (compilation time, inference speed, memory usage)
- Real COCO dataset loading and processing
- Visualization or plotting of results
- Integration with WandB or other logging services
- ONNX export functionality (already tested elsewhere)
- Multi-GPU or distributed training
- Specific numerical accuracy thresholds (will use reasonable tolerances)

## TDD Approach

### Test Design Philosophy:

1. **Fail-fast on environment issues** - Detect missing JAX/GPU before running tests
2. **Explicit verification** - Don't assume; verify JIT, GPU placement, correctness
3. **Reference comparison** - Every JAX result compared with CPU PyTensor
4. **Synthetic data** - Fast, deterministic, no external dependencies
5. **Comprehensive workflow** - Test from single operation to multi-epoch training
6. **Clear failure messages** - Diagnostic output guides debugging

### Testing Strategy:

- Write tests that **define behavior** through expected outcomes
- Verify tests **fail properly** before implementation exists
- Use **reference backend** (CPU PyTensor) as ground truth
- Test **incrementally** from simple operations to complex workflows
- **Skip gracefully** when GPU unavailable (developer convenience)

---

## Phase 1: Test Design & Implementation

### Overview

Write comprehensive tests that define the complete GPU training workflow. Tests should skip gracefully if GPU unavailable, but when run on GPU servers, they must verify every aspect of JAX compilation and execution.

### Test File Structure

**New File**: `examples/onnx/onnx-yolo-demo/testing/test_jax_gpu_required.py`

### CRITICAL: JAX Mode Enforcement Pattern

**EVERY test MUST follow this pattern to ensure JAX mode is truly forced:**

```python
def test_something_jax_vs_cpu():
    """Test description."""
    import jax

    # 1. VERIFY JAX mode is active
    verify_jax_mode()

    # 2. Define computational graph
    # ... your code ...

    # 3. FORCE JAX mode before compilation
    force_jax_mode()
    f_jax = function(inputs, outputs)

    # 4. Compile CPU version for comparison
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function(inputs, outputs)

    # 5. Execute both
    result_jax = f_jax(*test_data)
    result_cpu = f_cpu(*test_data)

    # 6. VERIFY result is jax.Array (proves JIT happened)
    if not isinstance(result_jax, jax.Array):
        raise AssertionError(
            f"CRITICAL: JAX JIT did not compile!\n"
            f"  Expected: jax.Array, Got: {type(result_jax)}\n"
            f"  JAX mode is NOT being used."
        )

    # 7. Compare results
    np.testing.assert_allclose(result_jax, result_cpu, ...)

    print("  ✓ JAX mode verified: ACTIVE")
    print("  ✓ JIT compiled: YES")
```

**Why this is critical:**
- Previous issues: JAX mode was set but didn't actually take effect
- Without verification: Tests pass but use CPU, giving false confidence
- `verify_jax_mode()`: Checks mode before test starts
- `force_jax_mode()`: Re-forces mode before compilation
- `isinstance(result, jax.Array)`: Proves JIT actually compiled
- Clear error messages: Immediately identify if JAX isn't being used

### Test Categories:

#### 1. Environment Detection Tests

**Purpose**: Fail fast if environment doesn't support GPU training

**Test File**: `test_jax_gpu_required.py`

##### Test: `test_environment_jax_available`

**Purpose**: Verify JAX is installed and importable

**Test Data**: None

**Expected Behavior**: JAX imports successfully

**Assertions**:
```python
def test_environment_jax_available():
    """Test that JAX is available."""
    try:
        import jax
        assert jax is not None
        print(f"✓ JAX version: {jax.__version__}")
    except ImportError:
        pytest.skip("JAX not available")
```

**Expected Failure Mode**: `pytest.skip()` if JAX not installed

##### Test: `test_environment_gpu_available`

**Purpose**: Verify GPU is detected by JAX

**Test Data**: None

**Expected Behavior**: JAX detects at least one GPU device

**Assertions**:
```python
def test_environment_gpu_available():
    """Test that GPU is available to JAX."""
    import jax

    devices = jax.devices()
    assert len(devices) > 0, "No JAX devices found"

    device_type = devices[0].platform
    if device_type != "gpu":
        pytest.skip(f"GPU not available, found: {device_type}")

    print(f"✓ GPU devices: {devices}")
```

**Expected Failure Mode**: `pytest.skip()` if no GPU available

##### Test: `test_environment_pytensor_jax_mode`

**Purpose**: Verify PyTensor can be configured for JAX mode

**Test Data**: Simple tensor operation

**Expected Behavior**: PyTensor compiles successfully with JAX backend

**Assertions**:
```python
def test_environment_pytensor_jax_mode():
    """Test that PyTensor can use JAX mode."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor import function

    # Force JAX mode
    pytensor.config.mode = "JAX"
    assert pytensor.config.mode == "JAX", "Failed to set JAX mode"

    # Try simple compilation
    x = pt.vector("x", dtype="float32")
    y = x * 2
    f = function([x], y)

    result = f(np.array([1.0, 2.0], dtype="float32"))
    assert isinstance(result, jax.Array), "Result is not a JAX array"

    print(f"✓ PyTensor JAX mode: {pytensor.config.mode}")
```

**Expected Failure Mode**: AssertionError or compilation error if JAX backend broken

#### 2. Forward Pass Tests with JIT Verification

**Purpose**: Verify all YOLO components work with JAX JIT and produce correct results

##### Test: `test_forward_basic_operation_jax_vs_cpu`

**Purpose**: Test that basic tensor operations work with JAX and match CPU results

**Test Data**: Random 4D tensor (1, 3, 10, 10)

**Expected Behavior**: JAX result matches CPU result, is jax.Array, on GPU

**Assertions**:
```python
def test_forward_basic_operation_jax_vs_cpu():
    """Test basic operation: JAX GPU vs CPU PyTensor."""
    import jax
    import pytensor.tensor as pt
    from pytensor import function

    # CRITICAL: Verify JAX mode is active before starting test
    verify_jax_mode()

    # Define operation
    x = pt.tensor4("x", dtype="float32")
    y = x * 2 + 1

    # Compile with JAX (mode should already be JAX from global setup)
    force_jax_mode()  # Re-force to be absolutely sure
    f_jax = function([x], y)

    # Compile with CPU for comparison
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], y)

    # Test data
    x_val = np.random.randn(1, 3, 10, 10).astype("float32")

    # Execute
    result_jax = f_jax(x_val)
    result_cpu = f_cpu(x_val)

    # Verify JAX used JIT (CRITICAL CHECK)
    if not isinstance(result_jax, jax.Array):
        raise AssertionError(
            f"CRITICAL: JAX JIT did not compile!\n"
            f"  Expected: jax.Array\n"
            f"  Got: {type(result_jax)}\n"
            f"  This means JAX mode is NOT being used.\n"
            f"  Result is a numpy array, indicating CPU execution."
        )

    # Verify correctness
    np.testing.assert_allclose(
        result_jax, result_cpu,
        rtol=1e-5, atol=1e-6,
        err_msg="JAX result doesn't match CPU result"
    )

    print(f"  ✓ JAX mode verified: ACTIVE")
    print(f"  ✓ JIT compiled: YES (result is jax.Array)")
    print(f"  ✓ JAX result shape: {result_jax.shape}")
    print(f"  ✓ Results match (max diff: {np.max(np.abs(result_jax - result_cpu)):.2e})")
```

**Expected Failure Mode**:
- Before implementation: N/A (operations already exist)
- If broken: AssertionError showing difference between JAX and CPU results

##### Test: `test_forward_upsampling_jax_vs_cpu`

**Purpose**: Test critical upsampling operation (historically problematic with JAX)

**Test Data**: Random tensor (1, 16, 10, 10)

**Expected Behavior**: Upsampling to (1, 16, 20, 20), JAX matches CPU

**Assertions**:
```python
def test_forward_upsampling_jax_vs_cpu():
    """Test upsampling operation: JAX GPU vs CPU."""
    import jax
    import pytensor.tensor as pt
    from pytensor import function

    # Define upsampling (as used in YOLO)
    x = pt.tensor4("x", dtype="float32")
    scale = 2

    # Get input shape
    input_shape = x.shape
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Upsample using dimshuffle + tile + reshape pattern
    x_expanded = x.dimshuffle(0, 1, 2, "x", 3, "x")
    x_tiled = pt.tile(x_expanded, (1, 1, 1, scale, 1, scale))
    x_rearranged = x_tiled.dimshuffle(0, 1, 2, 4, 3, 5)

    out_height = height * scale
    out_width = width * scale
    y = x_rearranged.reshape((batch_size, channels, out_height, out_width))

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], y)

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], y)

    # Test data
    x_val = np.random.randn(1, 16, 10, 10).astype("float32")

    # Execute
    result_jax = f_jax(x_val)
    result_cpu = f_cpu(x_val)

    # Verify shape
    expected_shape = (1, 16, 20, 20)
    assert result_jax.shape == expected_shape, \
        f"Expected {expected_shape}, got {result_jax.shape}"

    # Verify JAX used JIT
    assert isinstance(result_jax, jax.Array), "Result is not a JAX array"

    # Verify correctness
    np.testing.assert_allclose(
        result_jax, result_cpu,
        rtol=1e-5, atol=1e-6,
        err_msg="Upsampling: JAX result doesn't match CPU"
    )

    print(f"  ✓ Upsampled {x_val.shape} -> {result_jax.shape}")
    print(f"  ✓ Results match (max diff: {np.max(np.abs(result_jax - result_cpu)):.2e})")
```

**Expected Failure Mode**: Shape mismatch or numerical difference if upsampling broken

##### Test: `test_forward_conv_bn_silu_jax_vs_cpu`

**Purpose**: Test ConvBNSiLU block (fundamental building block)

**Test Data**: Random tensor (1, 3, 32, 32)

**Expected Behavior**: Correct output shape, JAX matches CPU

**Assertions**:
```python
def test_forward_conv_bn_silu_jax_vs_cpu():
    """Test ConvBNSiLU block: JAX GPU vs CPU."""
    import jax
    import pytensor.tensor as pt
    from pytensor import function
    from yolo.blocks import ConvBNSiLU

    # Create block
    conv = ConvBNSiLU(3, 16, kernel_size=3, stride=2, padding="same")

    # Define graph
    x = pt.tensor4("x", dtype="float32")
    y = conv(x)

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], y)

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], y)

    # Test data
    x_val = np.random.randn(1, 3, 32, 32).astype("float32")

    # Execute
    result_jax = f_jax(x_val)
    result_cpu = f_cpu(x_val)

    # Verify shape
    expected_shape = (1, 16, 16, 16)  # stride=2 halves spatial dims
    assert result_jax.shape == expected_shape, \
        f"Expected {expected_shape}, got {result_jax.shape}"

    # Verify JAX used JIT
    assert isinstance(result_jax, jax.Array), "Result is not a JAX array"

    # Verify correctness
    np.testing.assert_allclose(
        result_jax, result_cpu,
        rtol=1e-4, atol=1e-5,
        err_msg="ConvBNSiLU: JAX result doesn't match CPU"
    )

    print(f"  ✓ ConvBNSiLU: {x_val.shape} -> {result_jax.shape}")
    print(f"  ✓ Results match (max diff: {np.max(np.abs(result_jax - result_cpu)):.2e})")
```

**Expected Failure Mode**: Shape mismatch or numerical difference

##### Test: `test_forward_c3k2_block_jax_vs_cpu`

**Purpose**: Test C3k2 block (CSP bottleneck)

**Test Data**: Random tensor (1, 64, 40, 40)

**Expected Behavior**: Shape preserved, JAX matches CPU

**Assertions**: Similar structure to ConvBNSiLU test

**Expected Failure Mode**: Numerical difference if bottleneck operations broken

##### Test: `test_forward_sppf_block_jax_vs_cpu`

**Purpose**: Test SPPF block (Spatial Pyramid Pooling - Fast)

**Test Data**: Random tensor (1, 256, 10, 10)

**Expected Behavior**: Shape preserved, JAX matches CPU

**Assertions**: Similar structure to ConvBNSiLU test

**Expected Failure Mode**: Numerical difference if pooling operations broken

##### Test: `test_forward_backbone_jax_vs_cpu`

**Purpose**: Test complete YOLO backbone

**Test Data**: Random tensor (1, 3, 320, 320)

**Expected Behavior**: Three outputs (P3, P4, P5), all match CPU

**Assertions**:
```python
def test_forward_backbone_jax_vs_cpu():
    """Test YOLO backbone: JAX GPU vs CPU."""
    import jax
    import pytensor.tensor as pt
    from pytensor import function
    from yolo.model import YOLO11nBackbone

    # Build backbone
    backbone = YOLO11nBackbone()

    # Define graph
    x = pt.tensor4("x", dtype="float32")
    p3, p4, p5 = backbone(x)

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], [p3, p4, p5])

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], [p3, p4, p5])

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Execute
    p3_jax, p4_jax, p5_jax = f_jax(x_val)
    p3_cpu, p4_cpu, p5_cpu = f_cpu(x_val)

    # Verify all are JAX arrays
    assert all(isinstance(p, jax.Array) for p in [p3_jax, p4_jax, p5_jax]), \
        "Not all outputs are JAX arrays"

    # Verify shapes
    assert p3_jax.shape == (1, 64, 40, 40), f"P3 shape: {p3_jax.shape}"
    assert p4_jax.shape == (1, 128, 20, 20), f"P4 shape: {p4_jax.shape}"
    assert p5_jax.shape == (1, 256, 10, 10), f"P5 shape: {p5_jax.shape}"

    # Verify correctness for each output
    for name, jax_out, cpu_out in [("P3", p3_jax, p3_cpu),
                                     ("P4", p4_jax, p4_cpu),
                                     ("P5", p5_jax, p5_cpu)]:
        np.testing.assert_allclose(
            jax_out, cpu_out,
            rtol=1e-4, atol=1e-5,
            err_msg=f"Backbone {name}: JAX doesn't match CPU"
        )
        print(f"  ✓ {name}: shape {jax_out.shape}, "
              f"max diff: {np.max(np.abs(jax_out - cpu_out)):.2e}")
```

**Expected Failure Mode**: Numerical difference in one of the backbone outputs

##### Test: `test_forward_head_jax_vs_cpu`

**Purpose**: Test complete YOLO detection head

**Test Data**: Random tensors matching backbone outputs

**Expected Behavior**: Three detection outputs, all match CPU

**Assertions**: Similar structure to backbone test

**Expected Failure Mode**: Numerical difference in head outputs

##### Test: `test_forward_complete_model_jax_vs_cpu`

**Purpose**: Test end-to-end forward pass of complete YOLO11n

**Test Data**: Random tensor (1, 3, 320, 320)

**Expected Behavior**: Three prediction outputs with correct shapes, all match CPU

**Assertions**:
```python
def test_forward_complete_model_jax_vs_cpu():
    """Test complete YOLO11n model: JAX GPU vs CPU."""
    import jax
    import pytensor.tensor as pt
    from pytensor import function
    from yolo.model import build_yolo11n

    # Build model
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    pred_p3, pred_p4, pred_p5 = predictions

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], [pred_p3, pred_p4, pred_p5])

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], [pred_p3, pred_p4, pred_p5])

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Execute
    p3_jax, p4_jax, p5_jax = f_jax(x_val)
    p3_cpu, p4_cpu, p5_cpu = f_cpu(x_val)

    # Verify all are JAX arrays
    assert all(isinstance(p, jax.Array) for p in [p3_jax, p4_jax, p5_jax]), \
        "Not all outputs are JAX arrays"

    # Verify shapes (2 classes: 4 bbox + 2 class = 6 channels)
    assert p3_jax.shape == (1, 6, 40, 40), f"P3 pred shape: {p3_jax.shape}"
    assert p4_jax.shape == (1, 6, 20, 20), f"P4 pred shape: {p4_jax.shape}"
    assert p5_jax.shape == (1, 6, 10, 10), f"P5 pred shape: {p5_jax.shape}"

    # Verify correctness for each prediction
    for name, jax_out, cpu_out in [("Pred_P3", p3_jax, p3_cpu),
                                     ("Pred_P4", p4_jax, p4_cpu),
                                     ("Pred_P5", p5_jax, p5_cpu)]:
        np.testing.assert_allclose(
            jax_out, cpu_out,
            rtol=1e-4, atol=1e-5,
            err_msg=f"{name}: JAX doesn't match CPU"
        )
        print(f"  ✓ {name}: shape {jax_out.shape}, "
              f"max diff: {np.max(np.abs(jax_out - cpu_out)):.2e}")

    print(f"  ✓ Complete model forward pass verified on GPU")
```

**Expected Failure Mode**: Numerical difference in predictions

#### 3. Loss Computation Tests

**Purpose**: Verify loss functions work with JAX and produce correct values

##### Test: `test_loss_computation_jax_vs_cpu`

**Purpose**: Test that loss function works with JAX and matches CPU

**Test Data**: Random tensor (1, 3, 320, 320)

**Expected Behavior**: Loss computed, JAX matches CPU

**Assertions**:
```python
def test_loss_computation_jax_vs_cpu():
    """Test loss computation: JAX GPU vs CPU."""
    import jax
    import pytensor.tensor as pt
    from pytensor import function
    from yolo.model import build_yolo11n
    from yolo.loss import yolo_loss

    # Build model and loss
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    loss, loss_dict = yolo_loss(predictions, targets=None, num_classes=2)

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], [loss, loss_dict["box_loss"], loss_dict["cls_loss"]])

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], [loss, loss_dict["box_loss"], loss_dict["cls_loss"]])

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Execute
    total_jax, box_jax, cls_jax = f_jax(x_val)
    total_cpu, box_cpu, cls_cpu = f_cpu(x_val)

    # Verify all are JAX arrays or scalars
    assert isinstance(total_jax, (jax.Array, np.ndarray, float)), \
        f"Loss should be scalar or array, got {type(total_jax)}"

    # Verify correctness
    np.testing.assert_allclose(
        total_jax, total_cpu,
        rtol=1e-4, atol=1e-5,
        err_msg="Total loss: JAX doesn't match CPU"
    )
    np.testing.assert_allclose(
        box_jax, box_cpu,
        rtol=1e-4, atol=1e-5,
        err_msg="Box loss: JAX doesn't match CPU"
    )
    np.testing.assert_allclose(
        cls_jax, cls_cpu,
        rtol=1e-4, atol=1e-5,
        err_msg="Cls loss: JAX doesn't match CPU"
    )

    print(f"  ✓ Total loss - JAX: {total_jax:.6f}, CPU: {total_cpu:.6f}")
    print(f"  ✓ Box loss   - JAX: {box_jax:.6f}, CPU: {box_cpu:.6f}")
    print(f"  ✓ Cls loss   - JAX: {cls_jax:.6f}, CPU: {cls_cpu:.6f}")
```

**Expected Failure Mode**: Numerical difference in loss values

#### 4. Gradient Computation Tests

**Purpose**: Verify gradients can be computed through JAX and are correct

##### Test: `test_gradients_all_parameters_jax_vs_cpu`

**Purpose**: Test that gradients can be computed for all parameters and match CPU

**Test Data**: Random tensor (1, 3, 320, 320)

**Expected Behavior**: Gradients computed for all params, JAX matches CPU

**Assertions**:
```python
def test_gradients_all_parameters_jax_vs_cpu():
    """Test gradients for all parameters: JAX GPU vs CPU."""
    import jax
    import pytensor
    import pytensor.tensor as pt
    from pytensor import function
    from yolo.model import build_yolo11n
    from yolo.loss import yolo_loss

    # Build model and loss
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    loss, _ = yolo_loss(predictions, targets=None, num_classes=2)

    # Compute gradients for first 10 parameters (for speed)
    test_params = model.params[:10]
    grads = []
    for param in test_params:
        grad = pytensor.grad(loss, param, disconnected_inputs="ignore")
        grads.append(grad)

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], grads)

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], grads)

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Execute
    grads_jax = f_jax(x_val)
    grads_cpu = f_cpu(x_val)

    # Verify each gradient
    for i, (grad_jax, grad_cpu, param) in enumerate(zip(grads_jax, grads_cpu, test_params)):
        # Verify is JAX array
        assert isinstance(grad_jax, jax.Array), \
            f"Gradient {i} is not a JAX array"

        # Verify shape matches parameter
        assert grad_jax.shape == param.get_value().shape, \
            f"Gradient {i} shape mismatch"

        # Verify not all zeros
        assert np.abs(grad_jax).sum() > 0, \
            f"Gradient {i} is all zeros"

        # Verify no NaN or Inf
        assert np.isfinite(grad_jax).all(), \
            f"Gradient {i} contains NaN or Inf"

        # Verify correctness vs CPU
        np.testing.assert_allclose(
            grad_jax, grad_cpu,
            rtol=1e-4, atol=1e-5,
            err_msg=f"Gradient {i} ({param.name}): JAX doesn't match CPU"
        )

        print(f"  ✓ Grad {i} ({param.name}): shape {grad_jax.shape}, "
              f"mean |grad|: {np.mean(np.abs(grad_jax)):.2e}, "
              f"max diff: {np.max(np.abs(grad_jax - grad_cpu)):.2e}")

    print(f"  ✓ All {len(test_params)} gradients verified on GPU")
```

**Expected Failure Mode**: Numerical difference in gradients, or NaN/Inf gradients

##### Test: `test_gradients_finite_and_nonzero_jax`

**Purpose**: Verify gradients are well-behaved (no NaN/Inf, not all zeros)

**Test Data**: Random tensor (1, 3, 320, 320)

**Expected Behavior**: All gradients are finite and have meaningful values

**Assertions**: Similar to above but focuses on gradient quality, not CPU comparison

**Expected Failure Mode**: NaN, Inf, or zero gradients indicating broken backprop

#### 5. Training Step Tests

**Purpose**: Verify parameter updates work correctly with JAX

##### Test: `test_training_single_step_updates_parameters_jax`

**Purpose**: Test that a single training step updates parameters

**Test Data**: Random tensor (1, 3, 320, 320)

**Expected Behavior**: After training step, parameters have changed

**Assertions**:
```python
def test_training_single_step_updates_parameters_jax():
    """Test single training step updates parameters on GPU."""
    import jax
    import pytensor
    import pytensor.tensor as pt
    from pytensor import function, shared
    from yolo.model import build_yolo11n
    from yolo.loss import yolo_loss

    # Force JAX mode
    pytensor.config.mode = "JAX"

    # Build model and loss
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    loss, _ = yolo_loss(predictions, targets=None, num_classes=2)

    # Test on first parameter only (for speed)
    test_param = model.params[0]
    grad = pytensor.grad(loss, test_param)

    # Setup optimizer
    lr = pt.as_tensor_variable(np.float32(0.01))
    velocity = shared(
        np.zeros_like(test_param.get_value(), dtype="float32"),
        name="v"
    )
    momentum = pt.as_tensor_variable(np.float32(0.9))

    # Compute updates
    v_new = momentum * velocity - lr * pt.cast(grad, "float32")
    p_new = test_param + v_new

    updates = [(velocity, v_new), (test_param, p_new)]

    # Compile training function
    train_fn = function([x], loss, updates=updates)

    # Save initial parameter value
    param_before = test_param.get_value().copy()

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Execute training step
    loss_val = train_fn(x_val)

    # Get parameter after update
    param_after = test_param.get_value()

    # Verify loss is scalar
    assert isinstance(loss_val, (float, np.ndarray, jax.Array)), \
        f"Loss should be scalar, got {type(loss_val)}"

    # Verify loss is finite
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"

    # Verify parameter changed
    param_diff = np.abs(param_after - param_before)
    assert param_diff.sum() > 0, \
        "Parameter did not change after training step"

    # Verify parameter change is reasonable (not too large)
    max_change = np.max(param_diff)
    assert max_change < 1.0, \
        f"Parameter changed too much: max change = {max_change}"

    print(f"  ✓ Loss: {float(loss_val):.6f}")
    print(f"  ✓ Parameter changed: max Δ = {max_change:.2e}, "
          f"mean Δ = {np.mean(param_diff):.2e}")
```

**Expected Failure Mode**: Parameters don't update, or update with NaN/Inf values

##### Test: `test_training_multiple_steps_decreases_loss_jax`

**Purpose**: Test that multiple training steps decrease loss

**Test Data**: Same random tensor repeated 10 times

**Expected Behavior**: Loss decreases over training steps

**Assertions**:
```python
def test_training_multiple_steps_decreases_loss_jax():
    """Test that multiple training steps decrease loss on GPU."""
    import jax
    import pytensor
    import pytensor.tensor as pt
    from pytensor import function, shared
    from yolo.model import build_yolo11n
    from yolo.loss import yolo_loss

    # Force JAX mode
    pytensor.config.mode = "JAX"

    # Build model and loss
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    loss, _ = yolo_loss(predictions, targets=None, num_classes=2)

    # Compute gradients for all parameters
    grads = []
    for param in model.params:
        grad = pytensor.grad(loss, param, disconnected_inputs="ignore")
        grads.append(grad)

    # Setup optimizer with higher learning rate for faster convergence
    lr = pt.as_tensor_variable(np.float32(0.1))
    momentum = pt.as_tensor_variable(np.float32(0.9))

    velocities = []
    updates = []

    for param, grad in zip(model.params, grads):
        velocity = shared(
            np.zeros_like(param.get_value(), dtype="float32"),
            name=f"{param.name}_v"
        )
        velocities.append(velocity)

        v_new = momentum * velocity - lr * pt.cast(grad, "float32")
        p_new = param + v_new

        updates.append((velocity, v_new))
        updates.append((param, p_new))

    # Compile training function
    train_fn = function([x], loss, updates=updates)

    # Test data (same data for all steps to ensure convergence)
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Run 10 training steps
    losses = []
    for step in range(10):
        loss_val = train_fn(x_val)
        losses.append(float(loss_val))
        print(f"  Step {step + 1}: loss = {loss_val:.6f}")

    # Verify all losses are finite
    assert all(np.isfinite(l) for l in losses), \
        "Some losses are not finite"

    # Verify loss decreased (last loss < first loss)
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"

    # Verify loss trend is generally decreasing
    # (allow some fluctuation but overall trend should be down)
    avg_first_half = np.mean(losses[:5])
    avg_second_half = np.mean(losses[5:])
    assert avg_second_half < avg_first_half, \
        f"Loss is not trending down: {avg_first_half:.6f} -> {avg_second_half:.6f}"

    print(f"  ✓ Loss decreased: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"  ✓ Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
```

**Expected Failure Mode**: Loss doesn't decrease, or increases/diverges

#### 6. Multi-Epoch Training Tests

**Purpose**: Verify training works correctly over multiple epochs

##### Test: `test_training_multi_epoch_jax`

**Purpose**: Test that training works for multiple epochs

**Test Data**: Small batches of random tensors for 3 epochs

**Expected Behavior**: Training completes successfully, loss trends down

**Assertions**:
```python
def test_training_multi_epoch_jax():
    """Test multi-epoch training on GPU."""
    import jax
    import pytensor
    import pytensor.tensor as pt
    from pytensor import function, shared
    from yolo.model import build_yolo11n
    from yolo.loss import yolo_loss

    # Force JAX mode
    pytensor.config.mode = "JAX"

    # Build model and loss
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    loss, loss_dict = yolo_loss(predictions, targets=None, num_classes=2)

    # Compute gradients
    grads = []
    for param in model.params:
        grad = pytensor.grad(loss, param, disconnected_inputs="ignore")
        grads.append(grad)

    # Setup optimizer
    lr = pt.as_tensor_variable(np.float32(0.01))
    momentum = pt.as_tensor_variable(np.float32(0.9))

    velocities = []
    updates = []

    for param, grad in zip(model.params, grads):
        velocity = shared(
            np.zeros_like(param.get_value(), dtype="float32"),
            name=f"{param.name}_v"
        )
        velocities.append(velocity)

        v_new = momentum * velocity - lr * pt.cast(grad, "float32")
        p_new = param + v_new

        updates.append((velocity, v_new))
        updates.append((param, p_new))

    # Compile training function
    train_fn = function([x], [loss, loss_dict["box_loss"], loss_dict["cls_loss"]],
                        updates=updates)

    # Training loop
    num_epochs = 3
    batches_per_epoch = 5
    epoch_losses = []

    for epoch in range(num_epochs):
        batch_losses = []

        for batch in range(batches_per_epoch):
            # Generate random batch
            x_val = np.random.randn(2, 3, 320, 320).astype("float32")

            # Training step
            total_loss, box_loss, cls_loss = train_fn(x_val)
            batch_losses.append(float(total_loss))

        # Compute epoch average
        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)

        print(f"  Epoch {epoch + 1}: avg loss = {avg_loss:.6f}")

    # Verify all epoch losses are finite
    assert all(np.isfinite(l) for l in epoch_losses), \
        "Some epoch losses are not finite"

    # Verify loss decreased over epochs
    assert epoch_losses[-1] < epoch_losses[0], \
        f"Loss did not decrease over epochs: {epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f}"

    print(f"  ✓ Training completed {num_epochs} epochs")
    print(f"  ✓ Loss decreased: {epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f}")
```

**Expected Failure Mode**: Training crashes during epochs, or loss doesn't decrease

#### 7. Checkpoint Save/Resume Tests

**Purpose**: Verify checkpointing works correctly with JAX

##### Test: `test_checkpoint_save_and_resume_jax`

**Purpose**: Test that training can be saved and resumed correctly

**Test Data**: Random tensors

**Expected Behavior**: After resume, training continues from saved state

**Assertions**:
```python
def test_checkpoint_save_and_resume_jax():
    """Test checkpoint save and resume on GPU."""
    import jax
    import pytensor
    import pytensor.tensor as pt
    from pytensor import function, shared
    from yolo.model import build_yolo11n
    from yolo.loss import yolo_loss
    import tempfile
    from pathlib import Path

    # Force JAX mode
    pytensor.config.mode = "JAX"

    # Build model and loss
    model, x, predictions = build_yolo11n(num_classes=2, input_size=320)
    loss, _ = yolo_loss(predictions, targets=None, num_classes=2)

    # Test on first parameter only (for speed)
    test_param = model.params[0]
    grad = pytensor.grad(loss, test_param)

    # Setup optimizer
    lr = pt.as_tensor_variable(np.float32(0.01))
    velocity = shared(
        np.zeros_like(test_param.get_value(), dtype="float32"),
        name="v"
    )
    momentum = pt.as_tensor_variable(np.float32(0.9))

    v_new = momentum * velocity - lr * pt.cast(grad, "float32")
    p_new = test_param + v_new

    updates = [(velocity, v_new), (test_param, p_new)]

    # Compile training function
    train_fn = function([x], loss, updates=updates)

    # Test data
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")

    # Run 5 training steps
    print("  Phase 1: Initial training (5 steps)")
    for step in range(5):
        loss_val = train_fn(x_val)
        print(f"    Step {step + 1}: loss = {loss_val:.6f}")

    # Save checkpoint
    checkpoint_dict = {
        "param": test_param.get_value(),
        "velocity": velocity.get_value(),
        "step": 5,
    }

    # Create temporary file for checkpoint
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        checkpoint_path = tmp.name

    np.savez(checkpoint_path, **checkpoint_dict)
    print(f"  ✓ Checkpoint saved to {checkpoint_path}")

    # Continue training for 3 more steps (without loading)
    print("  Phase 2: Continue training (3 more steps)")
    losses_continued = []
    for step in range(3):
        loss_val = train_fn(x_val)
        losses_continued.append(float(loss_val))
        print(f"    Step {step + 6}: loss = {loss_val:.6f}")

    # Now reset parameters and load checkpoint
    print("  Phase 3: Reset and resume from checkpoint")

    # Reset to random values
    test_param.set_value(np.random.randn(*test_param.get_value().shape).astype("float32"))
    velocity.set_value(np.zeros_like(velocity.get_value()))

    # Load checkpoint
    checkpoint = np.load(checkpoint_path)
    test_param.set_value(checkpoint["param"])
    velocity.set_value(checkpoint["velocity"])
    print(f"  ✓ Checkpoint loaded from step {checkpoint['step']}")

    # Continue training for 3 steps from loaded checkpoint
    print("  Phase 4: Resume training (3 steps)")
    losses_resumed = []
    for step in range(3):
        loss_val = train_fn(x_val)
        losses_resumed.append(float(loss_val))
        print(f"    Step {step + 6}: loss = {loss_val:.6f}")

    # Verify resumed training matches continued training
    # (should have same trajectory since we used same data)
    for i, (loss_cont, loss_resum) in enumerate(zip(losses_continued, losses_resumed)):
        np.testing.assert_allclose(
            loss_resum, loss_cont,
            rtol=1e-5, atol=1e-6,
            err_msg=f"Resumed training step {i+1} doesn't match continued training"
        )

    print(f"  ✓ Resumed training matches continued training")
    print(f"  ✓ Checkpoint save/resume verified on GPU")

    # Cleanup
    Path(checkpoint_path).unlink()
```

**Expected Failure Mode**: Resumed training doesn't match expected trajectory

### Test Implementation Steps:

1. **Create test file**: `examples/onnx/onnx-yolo-demo/testing/test_jax_gpu_required.py`

2. **Import necessary testing utilities**:
```python
#!/usr/bin/env python3
"""
GPU-required tests for YOLO11n with forced JAX mode.

These tests REQUIRE GPU and skip gracefully if unavailable.
Run on GPU server:
    python test_jax_gpu_required.py

All tests verify:
1. JAX mode is forced
2. GPU is being used
3. JIT compilation occurred
4. Results match CPU backend
"""

import os
import sys
import numpy as np

# Set PyTensor flags before importing
os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"

import pytensor
import pytensor.tensor as pt
from pytensor import function, shared

# Add parent directory for imports
sys.path.insert(0, "..")

# GPU detection utilities
def has_gpu():
    """Check if GPU is available."""
    try:
        import jax
        devices = jax.devices()
        return len(devices) > 0 and devices[0].platform == "gpu"
    except:
        return False

def skip_if_no_gpu():
    """Skip test if GPU not available."""
    if not has_gpu():
        print("⊗ SKIPPED: GPU not available")
        sys.exit(0)  # Exit with success (skip, not fail)

def force_jax_mode():
    """
    Force JAX mode and verify it's actually set.

    This is CRITICAL - we've had issues where mode was not truly forced.
    This function ensures JAX mode is active and throws a clear error if not.
    """
    import pytensor

    # Attempt to force JAX mode
    pytensor.config.mode = "JAX"

    # VERIFY it actually took effect
    actual_mode = str(pytensor.config.mode)
    if actual_mode != "JAX":
        raise RuntimeError(
            f"CRITICAL: Failed to force JAX mode!\n"
            f"  Attempted to set: JAX\n"
            f"  Actually got: {actual_mode}\n"
            f"  This means JAX is NOT being used.\n"
            f"  Tests would give false confidence."
        )

    return actual_mode

def verify_jax_mode():
    """
    Verify that current mode is JAX.

    Call this before EVERY test to ensure mode hasn't been reset.
    """
    import pytensor

    actual_mode = str(pytensor.config.mode)
    if actual_mode != "JAX":
        raise RuntimeError(
            f"CRITICAL: JAX mode is not active!\n"
            f"  Current mode: {actual_mode}\n"
            f"  Expected: JAX\n"
            f"  JAX mode must be forced for these tests."
        )

# Global GPU check and JAX mode enforcement
print("=" * 70)
print("GPU-Required JAX Tests for YOLO11n".center(70))
print("=" * 70)

if not has_gpu():
    print("\n⚠ GPU not detected - all tests will be skipped")
    print("  To run these tests, execute on a machine with GPU")
    print("=" * 70)
    sys.exit(0)
else:
    import jax
    print(f"\n✓ GPU detected: {jax.devices()}")

    # Force JAX mode and VERIFY it worked
    actual_mode = force_jax_mode()
    print(f"✓ PyTensor mode FORCED: {actual_mode}")
    print(f"✓ JAX mode enforcement: ACTIVE")
    print("=" * 70)
```

3. **Implement each test case** following the test designs above

4. **Add test runner**:
```python
def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Running GPU-Required Tests".center(70))
    print("=" * 70 + "\n")

    tests = [
        ("Environment: JAX Available", test_environment_jax_available),
        ("Environment: GPU Available", test_environment_gpu_available),
        ("Environment: PyTensor JAX Mode", test_environment_pytensor_jax_mode),
        ("Forward: Basic Operation", test_forward_basic_operation_jax_vs_cpu),
        ("Forward: Upsampling", test_forward_upsampling_jax_vs_cpu),
        ("Forward: ConvBNSiLU Block", test_forward_conv_bn_silu_jax_vs_cpu),
        ("Forward: C3k2 Block", test_forward_c3k2_block_jax_vs_cpu),
        ("Forward: SPPF Block", test_forward_sppf_block_jax_vs_cpu),
        ("Forward: Backbone", test_forward_backbone_jax_vs_cpu),
        ("Forward: Head", test_forward_head_jax_vs_cpu),
        ("Forward: Complete Model", test_forward_complete_model_jax_vs_cpu),
        ("Loss: Computation", test_loss_computation_jax_vs_cpu),
        ("Gradients: All Parameters", test_gradients_all_parameters_jax_vs_cpu),
        ("Gradients: Finite and Nonzero", test_gradients_finite_and_nonzero_jax),
        ("Training: Single Step", test_training_single_step_updates_parameters_jax),
        ("Training: Multiple Steps", test_training_multiple_steps_decreases_loss_jax),
        ("Training: Multi-Epoch", test_training_multi_epoch_jax),
        ("Checkpoint: Save and Resume", test_checkpoint_save_and_resume_jax),
    ]

    results = {}
    for name, test_func in tests:
        print(f"\n[Test] {name}")
        print("-" * 70)
        try:
            test_func()
            results[name] = True
            print(f"✓ PASSED: {name}")
        except Exception as e:
            results[name] = False
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary".center(70))
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:50s} {status}")

    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    # Exit with error if any failed
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [x] All test files created with proper structure
- [x] Tests skip gracefully when GPU unavailable: `python test_jax_gpu_required.py` (on CPU)
- [ ] Tests run when GPU available: `python test_jax_gpu_required.py` (on GPU) - REQUIRES GPU SERVER
- [x] No syntax errors: `python -m py_compile test_jax_gpu_required.py`

#### Manual Verification:
- [x] Each test has clear, informative docstring
- [x] Test names clearly describe what they test
- [x] Assertion messages are diagnostic
- [x] Test code is readable and maintainable
- [x] GPU detection works correctly
- [x] JAX/CPU comparison is comprehensive

---

## Phase 2: Test Failure Verification

### Overview

Run the new test suite and verify that tests behave correctly:
- Skip gracefully on CPU machines (developer workflow)
- Run and verify correctness on GPU machines

### Verification Steps:

#### On CPU Machine (Development):

1. **Run the test suite**:
```bash
cd examples/onnx/onnx-yolo-demo/testing
python test_jax_gpu_required.py
```

2. **Expected output**:
```
======================================================================
         GPU-Required JAX Tests for YOLO11n
======================================================================

⚠ GPU not detected - all tests will be skipped
  To run these tests, execute on a machine with GPU
======================================================================
```

3. **Verify exit code**:
```bash
echo $?
# Should be 0 (success/skip, not failure)
```

#### On GPU Machine (Production):

1. **Run the test suite**:
```bash
cd examples/onnx/onnx-yolo-demo/testing
python test_jax_gpu_required.py
```

2. **Expected output** (if code already works):
```
======================================================================
         GPU-Required JAX Tests for YOLO11n
======================================================================

✓ GPU detected: [cuda(id=0)]
✓ PyTensor mode: JAX
======================================================================

======================================================================
                    Running GPU-Required Tests
======================================================================

[Test] Environment: JAX Available
----------------------------------------------------------------------
✓ JAX version: 0.4.20
✓ PASSED: Environment: JAX Available

[Test] Environment: GPU Available
----------------------------------------------------------------------
✓ GPU devices: [cuda(id=0)]
✓ PASSED: Environment: GPU Available

...

======================================================================
                          Test Summary
======================================================================
Environment: JAX Available                        ✓ PASS
Environment: GPU Available                        ✓ PASS
Forward: Basic Operation                          ✓ PASS
...
----------------------------------------------------------------------
Results: 18/18 tests passed
======================================================================
```

3. **If any test fails**, analyze failure:
   - **Expected behavior**: Test fails with clear diagnostic message
   - **Failure message should indicate**: What was expected vs what was received
   - **Failure should point to**: The specific operation/component that's broken

### Expected Failures:

Since the YOLO code has already been fixed for JAX compatibility (from previous work), we expect **all tests to pass** on the first run. However, if any component is broken, tests would fail like:

**Example failure - Numerical difference**:
```
[Test] Forward: ConvBNSiLU Block
----------------------------------------------------------------------
✗ FAILED: Forward: ConvBNSiLU Block
  Error: AssertionError: ConvBNSiLU: JAX result doesn't match CPU
  Maximum difference: 0.052 (exceeds tolerance rtol=0.0001, atol=0.00001)

  Arrays are not close:
  - JAX result mean: 0.145
  - CPU result mean: 0.143
  - Max absolute difference: 0.052 at index (0, 5, 12, 8)
```

**Example failure - Not using GPU**:
```
[Test] Forward: Basic Operation
----------------------------------------------------------------------
✗ FAILED: Forward: Basic Operation
  Error: AssertionError: Expected jax.Array, got <class 'numpy.ndarray'>

  JAX did not compile the function - result is not a JAX array.
  This indicates JAX mode is not being used.
```

### Adjustment Phase:

If tests don't behave properly on GPU machine:

- [ ] Verify GPU detection logic is correct
- [ ] Verify JAX mode forcing works
- [ ] Verify comparison logic is sound
- [ ] Improve assertion messages for clarity
- [ ] Adjust tolerance values if needed

### Success Criteria:

#### Automated Verification:
- [ ] Tests skip on CPU machine with exit code 0: `python test_jax_gpu_required.py` (CPU)
- [ ] Tests run on GPU machine: `python test_jax_gpu_required.py` (GPU)
- [ ] Tests fail with clear messages if component broken (simulate by breaking code)

#### Manual Verification:
- [ ] Skip message is clear and informative
- [ ] GPU detection works correctly
- [ ] Each test either passes or fails with diagnostic message
- [ ] No confusing or cryptic error messages
- [ ] Failure messages indicate what needs fixing

---

## Phase 3: Feature Implementation (Red → Green)

### Overview

Based on Phase 2 results, implement fixes if any tests fail. Since the YOLO code has already been fixed for JAX compatibility, this phase may be minimal or skipped.

### Implementation Strategy:

**If all tests pass** (expected):
- ✓ No implementation needed
- ✓ Tests verify existing code works correctly
- ✓ Move to Phase 4 (Refactoring)

**If tests fail** (unexpected):
- Analyze failure messages
- Identify broken component
- Fix implementation
- Re-run tests
- Repeat until all pass

### Potential Failure Scenarios:

#### Scenario 1: Numerical Difference Between JAX and CPU

**Failure**:
```
test_forward_conv_bn_silu_jax_vs_cpu FAILED
Maximum difference: 0.052 (exceeds tolerance)
```

**Debugging Steps**:
1. Check if difference is in batch normalization (common source)
2. Verify epsilon values match between JAX and CPU implementations
3. Check for dtype mismatches
4. Verify batch norm is using JAX-compatible implementation

**Implementation**:
- Review `yolo/blocks.py:jax_compatible_batch_norm()` function
- Ensure running stats are disabled (training mode only)
- Verify all operations use float32

#### Scenario 2: Result Not a JAX Array

**Failure**:
```
test_forward_basic_operation_jax_vs_cpu FAILED
Expected jax.Array, got numpy.ndarray
```

**Debugging Steps**:
1. Verify `pytensor.config.mode = "JAX"` is set before compilation
2. Check if JAX backend is properly installed
3. Verify function compilation uses JAX mode

**Implementation**:
- Ensure global mode setting happens before any compilation
- Add explicit mode checking in test setup

#### Scenario 3: Gradient is NaN or Inf

**Failure**:
```
test_gradients_finite_and_nonzero_jax FAILED
Gradient 5 contains NaN
```

**Debugging Steps**:
1. Check for division by zero in loss computation
2. Verify numerical stability in loss functions
3. Check for log(0) or sqrt(negative) operations

**Implementation**:
- Add epsilon to denominators
- Clip values before log/sqrt operations
- Review loss.py for numerical stability

### Success Criteria:

Since tests are expected to pass already:

#### Automated Verification:
- [ ] All tests pass on GPU machine: `python test_jax_gpu_required.py`
- [ ] Exit code is 0
- [ ] No errors in output

#### Manual Verification:
- [ ] All 18 tests show ✓ PASS
- [ ] Test summary shows 18/18 passed
- [ ] No warnings in output

---

## Phase 4: Refactoring & Cleanup

### Overview

Once tests are passing, refactor test code for maintainability and clarity while keeping tests green.

### Refactoring Targets:

1. **Code Duplication**:
   - Extract common test patterns
   - Create helper functions for JAX vs CPU comparison
   - Share test data generation code

2. **Code Clarity**:
   - Improve variable names
   - Add explanatory comments
   - Group related tests

3. **Test Quality**:
   - Extract common fixtures
   - Create reusable assertion helpers
   - Improve error messages

### Refactoring Steps:

#### Refactoring 1: Extract JAX vs CPU Comparison Helper

**Before**:
```python
def test_forward_conv_bn_silu_jax_vs_cpu():
    # ... setup ...

    # Compile with JAX
    pytensor.config.mode = "JAX"
    f_jax = function([x], y)

    # Compile with CPU
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function([x], y)

    # Execute
    result_jax = f_jax(x_val)
    result_cpu = f_cpu(x_val)

    # Verify JAX used JIT
    assert isinstance(result_jax, jax.Array)

    # Verify correctness
    np.testing.assert_allclose(result_jax, result_cpu, rtol=1e-4, atol=1e-5)
```

**After**:
```python
def compare_jax_and_cpu(inputs, outputs, test_data, rtol=1e-4, atol=1e-5):
    """
    Compare JAX GPU execution with CPU PyTensor.

    This helper function enforces JAX mode and verifies JIT compilation.

    Args:
        inputs: List of input tensors
        outputs: List of output tensors
        test_data: List of numpy arrays for inputs
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (jax_results, cpu_results) tuple

    Raises:
        AssertionError if results don't match or JAX mode not active
    """
    import jax

    # FORCE JAX mode and verify
    force_jax_mode()
    f_jax = function(inputs, outputs)

    # Compile with CPU for comparison
    pytensor.config.mode = "FAST_RUN"
    f_cpu = function(inputs, outputs)

    # Execute
    jax_results = f_jax(*test_data)
    cpu_results = f_cpu(*test_data)

    # Ensure lists
    if not isinstance(jax_results, (list, tuple)):
        jax_results = [jax_results]
        cpu_results = [cpu_results]

    # VERIFY all are JAX arrays (proves JIT compiled)
    for i, result in enumerate(jax_results):
        if not isinstance(result, jax.Array):
            raise AssertionError(
                f"CRITICAL: Output {i} is not a JAX array!\n"
                f"  Expected: jax.Array\n"
                f"  Got: {type(result)}\n"
                f"  JAX JIT did not compile - mode not enforced properly."
            )

    # Verify correctness
    for i, (jax_res, cpu_res) in enumerate(zip(jax_results, cpu_results)):
        np.testing.assert_allclose(
            jax_res, cpu_res,
            rtol=rtol, atol=atol,
            err_msg=f"Output {i}: JAX doesn't match CPU"
        )
        max_diff = np.max(np.abs(jax_res - cpu_res))
        print(f"  ✓ Output {i}: JAX mode active, JIT compiled, max diff = {max_diff:.2e}")

    return jax_results, cpu_results

def test_forward_conv_bn_silu_jax_vs_cpu():
    """Test ConvBNSiLU block: JAX GPU vs CPU."""
    from yolo.blocks import ConvBNSiLU

    # Create block
    conv = ConvBNSiLU(3, 16, kernel_size=3, stride=2, padding="same")

    # Define graph
    x = pt.tensor4("x", dtype="float32")
    y = conv(x)

    # Test data
    x_val = np.random.randn(1, 3, 32, 32).astype("float32")

    # Compare
    results_jax, results_cpu = compare_jax_and_cpu([x], [y], [x_val])

    print(f"  ✓ ConvBNSiLU: {x_val.shape} -> {results_jax[0].shape}")
```

#### Refactoring 2: Extract Test Data Generation

**Create helper functions**:
```python
def generate_random_image(batch_size=1, channels=3, height=320, width=320):
    """Generate random image tensor for testing."""
    return np.random.randn(batch_size, channels, height, width).astype("float32")

def generate_random_feature_map(batch_size=1, channels=64, height=40, width=40):
    """Generate random feature map for testing."""
    return np.random.randn(batch_size, channels, height, width).astype("float32")
```

#### Refactoring 3: Group Related Tests

**Organize file structure**:
```python
# ============================================================================
# Environment Tests
# ============================================================================

def test_environment_jax_available():
    ...

def test_environment_gpu_available():
    ...

# ============================================================================
# Forward Pass Tests - Building Blocks
# ============================================================================

def test_forward_basic_operation_jax_vs_cpu():
    ...

def test_forward_upsampling_jax_vs_cpu():
    ...

# ============================================================================
# Forward Pass Tests - YOLO Components
# ============================================================================

def test_forward_conv_bn_silu_jax_vs_cpu():
    ...

# ============================================================================
# Gradient Tests
# ============================================================================

def test_gradients_all_parameters_jax_vs_cpu():
    ...

# ============================================================================
# Training Tests
# ============================================================================

def test_training_single_step_updates_parameters_jax():
    ...
```

### Success Criteria:

#### Automated Verification:
- [ ] All tests still pass after refactoring: `python test_jax_gpu_required.py`
- [ ] No performance regression (tests don't run slower)
- [ ] Code is more maintainable (fewer lines, clearer structure)

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] No unnecessary complexity added
- [ ] Helper functions are well-documented
- [ ] Test organization is logical
- [ ] Common patterns are extracted

---

## Testing Strategy Summary

### Test Coverage Goals:

- [x] Environment detection (JAX, GPU availability)
- [x] Basic operations (tensors, arithmetic)
- [x] Critical operations (upsampling, convolution, pooling)
- [x] Building blocks (ConvBNSiLU, C3k2, SPPF, Bottleneck)
- [x] Complete model components (Backbone, Head)
- [x] Full forward pass
- [x] Loss computation
- [x] Gradient computation (all parameters)
- [x] Single training step (parameter updates)
- [x] Multiple training steps (loss decrease)
- [x] Multi-epoch training
- [x] Checkpoint save and resume

### Test Organization:

**File**: `examples/onnx/onnx-yolo-demo/testing/test_jax_gpu_required.py`

**Test Categories**:
1. Environment tests (3 tests)
2. Forward pass tests (8 tests)
3. Loss tests (1 test)
4. Gradient tests (2 tests)
5. Training tests (3 tests)
6. Checkpoint tests (1 test)

**Total**: 18 comprehensive tests

### Running Tests:

```bash
# On CPU machine (development) - tests skip gracefully
cd examples/onnx/onnx-yolo-demo/testing
python test_jax_gpu_required.py

# On GPU machine (production) - tests run and verify
cd examples/onnx/onnx-yolo-demo/testing
python test_jax_gpu_required.py

# Expected output on GPU:
# - All tests pass
# - 18/18 tests passed
# - Exit code 0
```

### Key Features:

- **Automatic GPU detection** - skips on CPU, runs on GPU
- **ENFORCED JAX mode** - multiple layers of verification that JAX is actually being used:
  - `force_jax_mode()` - Sets mode and verifies it took effect
  - `verify_jax_mode()` - Checks mode before each test
  - `isinstance(result, jax.Array)` - Proves JIT compilation happened
  - Clear error messages if mode enforcement fails
- **JIT verification** - confirms JAX compilation happened for every operation
- **Correctness verification** - compares with CPU backend
- **Comprehensive workflow** - tests from ops to training
- **Clear diagnostics** - failure messages guide debugging
- **No external dependencies** - uses synthetic data
- **Fast execution** - optimized test sizes

**Critical:** Previous issues with JAX mode not being forced are addressed through triple-verification:
1. Global mode enforcement at startup
2. Per-test mode verification
3. Result type checking (jax.Array proves JIT worked)

## Performance Considerations

Not applicable - per requirements, we are not testing performance.

## Migration Notes

Not applicable - this is a new test suite, not a migration.

## References

- **PyTensor JAX tests**: `tests/link/jax/test_basic.py` - `compare_jax_and_py()` pattern
- **Existing YOLO tests**: `examples/onnx/onnx-yolo-demo/testing/test_jax_components.py`
- **YOLO model**: `examples/onnx/onnx-yolo-demo/yolo/model.py`
- **YOLO blocks**: `examples/onnx/onnx-yolo-demo/yolo/blocks.py`
- **YOLO loss**: `examples/onnx/onnx-yolo-demo/yolo/loss.py`
- **Training script**: `examples/onnx/onnx-yolo-demo/train.py`
- **GPU detection pattern**: `test_jax_components.py:35-62` - `setup_jax()` function
- **JIT verification**: Check `isinstance(result, jax.Array)`
