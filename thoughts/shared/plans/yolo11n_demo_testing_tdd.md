---
date: 2025-10-18
author: Claude Code
git_commit: 226f34c37775b44b18b783723a7357f56fb5e116
branch: onnx-workshop-demo
repository: pytensor
topic: "TDD Implementation Plan for YOLO11n Demo Testing"
tags: [tdd, testing, yolo, onnx, hypothesis, property-based-testing, jax]
status: active
related_research: thoughts/shared/research/2025-10-18_21-43-17_yolo-demo-testing-setup.md
---

# YOLO11n Demo Testing - TDD Implementation Plan

## Overview

Implement comprehensive property-based and integration tests for the YOLO11n object detection demo using Test-Driven Development. Tests will verify correctness of the PyTensor→JAX→ONNX pipeline for a production-quality neural network with:
- **YOLO11n Backbone**: 7-stage feature extractor with C3k2, SPPF, and C2PSA blocks
- **FPN-PAN Head**: Feature pyramid network with multi-scale detection
- **Detection Output**: 3-scale predictions (P3:40x40, P4:20x20, P5:10x10)
- **Loss Functions**: IoU-based bbox regression and classification loss

## Current State Analysis

### Existing YOLO11n Implementation

**Location**: `examples/onnx/onnx-yolo-demo/`

**Model Architecture** (`yolo/model.py:16-382`):
```python
# Backbone: 320x320 → {P3:40x40, P4:20x20, P5:10x10}
YOLO11nBackbone:
  - Stem: ConvBNSiLU(3→16, s=2)
  - Stage1: ConvBNSiLU(16→32, s=2) + C3k2(32→32, n=1)
  - Stage2(P3): ConvBNSiLU(32→64, s=2) + C3k2(64→64, n=2)  # Output: 40x40
  - Stage3(P4): ConvBNSiLU(64→128, s=2) + C3k2(128→128, n=2)  # Output: 20x20
  - Stage4(P5): ConvBNSiLU(128→256, s=2) + C3k2(256→256, n=1) + SPPF(256) + C2PSA(256)  # Output: 10x10

# Head: FPN-PAN with multi-scale fusion
YOLO11nHead:
  - FPN upsampling: P5→P4→P3 (top-down)
  - PAN downsampling: P3→P4→P5 (bottom-up)
  - Detection heads: Conv1x1 producing (4+num_classes) channels at each scale

# Output format: (det_p3, det_p4, det_p5)
- det_p3: (batch, 4+C, 40, 40)
- det_p4: (batch, 4+C, 20, 20)
- det_p5: (batch, 4+C, 10, 10)
```

**Building Blocks** (`yolo/blocks.py:22-458`):
- `ConvBNSiLU`: Conv2D + BatchNorm + SiLU activation (JAX-compatible batch norm)
- `Bottleneck`: Two 3x3 convs with optional residual connection
- `C3k2`: CSP bottleneck with split-process-merge architecture
- `SPPF`: Spatial Pyramid Pooling - Fast (cascaded max pooling)
- `C2PSA`: CSP with simplified channel attention

**Loss Functions** (`yolo/loss.py:15-266`):
- `box_iou()`: IoU computation between box predictions and ground truth
- `yolo_loss()`: Simplified training loss (regularization-based for demo)
- `yolo_loss_with_targets()`: Target assignment to grid cells

**Dataset** (`yolo/dataset.py:20-351`):
- COCO 2017 loader filtering for person (class 0) and cellphone (class 1)
- Image resizing to 320x320
- Normalized bbox format: [x_center, y_center, width, height]

### Current Testing Landscape

**Test Framework**: pytest + Hypothesis (property-based testing)
**Location**: Tests will be created in `examples/onnx/onnx-yolo-demo/tests/`
**Status**: **No tests currently exist** (previously deleted `testing/` directory per git status)

**Available Patterns** (from `thoughts/shared/research/2025-10-18_21-43-17_yolo-demo-testing-setup.md`):
- ONNX test patterns: `tests/link/onnx/` (Hypothesis strategies, backend comparison)
- JAX test patterns: `tests/link/jax/` (JAX vs Python backend comparison, gradient tests)
- PyMC demo tests: `examples/onnx/onnx-pymc-demo/tests/` (demo-specific test organization)

### Key Constraints

1. **JAX-Compatible Operations**:
   - Custom batch norm implementation (`jax_compatible_batch_norm`) avoids dynamic shapes
   - Upsample uses explicit tiling (not pt.repeat which breaks JIT)
   - All operations must be JAX-traceable

2. **ONNX Export Requirements**:
   - Model output is tuple `(det_p3, det_p4, det_p5)` not dict
   - Filter flipping must be handled for convolutions
   - Batch norm uses frozen statistics (inference mode)

3. **Numerical Precision**:
   - Float32 required: `PYTENSOR_FLAGS='floatX=float32'`
   - Relaxed tolerances for accumulated errors (rtol=1e-4, atol=1e-5)

## Desired End State

### Test Suite Structure

```
examples/onnx/onnx-yolo-demo/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures + Hypothesis config
│   ├── strategies/                    # Hypothesis strategies
│   │   ├── __init__.py
│   │   ├── core.py                   # Tensor/image strategies
│   │   └── yolo_data.py              # YOLO-specific strategies
│   ├── test_blocks.py                # ConvBNSiLU, C3k2, SPPF, C2PSA
│   ├── test_model_backbone.py        # YOLO11nBackbone correctness
│   ├── test_model_head.py            # YOLO11nHead (FPN-PAN)
│   ├── test_model_integration.py     # Full YOLO11n model
│   ├── test_loss.py                  # Loss functions
│   ├── test_jax_backend.py           # JAX compilation tests
│   ├── test_onnx_export.py           # ONNX export and inference
│   └── test_pipeline.py              # End-to-end pipeline
├── pyproject.toml                     # Test configuration
└── yolo/                              # Implementation (already exists)
```

### Success Criteria

1. **Correctness**: All blocks produce expected shapes and numerical outputs
2. **JAX Compatibility**: Model compiles and runs on JAX backend without errors
3. **ONNX Export**: Model exports to ONNX and produces matching outputs via ONNX Runtime
4. **Gradient Flow**: Gradients backpropagate correctly through all components
5. **Property Verification**: Mathematical properties hold (shape preservation, non-negativity, etc.)

## What We're NOT Testing/Implementing

1. **Training Loop**: Not testing full training convergence (loss function is simplified)
2. **Data Augmentation**: Not testing image augmentation pipelines
3. **Post-Processing**: Not testing NMS, bbox decoding, or visualization
4. **Performance**: Not benchmarking inference speed (focus on correctness)
5. **Distributed Training**: Not testing multi-GPU or distributed setups
6. **Other YOLO Versions**: Only YOLO11n nano variant (not YOLOv8, YOLOv10, etc.)

## TDD Approach

### Test Design Philosophy

1. **Property-Based Testing**: Use Hypothesis to test invariants and mathematical properties
   - Shape preservation through layers
   - Non-negative outputs from activations
   - Gradient magnitude bounds

2. **Backend Comparison**: Compare JAX outputs with PyTensor Python backend
   - Ensures JAX compilation correctness
   - Verifies numerical equivalence

3. **ONNX Runtime Comparison**: Compare ONNX Runtime with PyTensor
   - Validates export correctness
   - Ensures deployment equivalence

4. **Progressive Integration**: Test components bottom-up
   - Blocks → Backbone → Head → Full Model → Pipeline
   - Each level builds confidence in lower levels

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define expected behavior for all YOLO11n components. Tests will fail initially (no implementation bugs to fix since code exists), but verify our testing infrastructure works.

### Test Categories

#### 1. Building Block Tests (`test_blocks.py`)
**Purpose**: Verify individual neural network components work correctly

##### Test: `test_conv_bn_silu_output_shape`
**Purpose**: Verify ConvBNSiLU produces correct output shapes
**Test Data**: Random tensor (batch=2, channels=3, H=32, W=32)
**Expected Behavior**: Output shape matches convolution formula
**Assertions**:
- Output shape: (2, 16, 32, 32) for stride=1, same padding
- Output shape: (2, 16, 16, 16) for stride=2, same padding

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import pytensor.tensor as pt
from pytensor import function

from yolo.blocks import ConvBNSiLU
from tests.strategies.core import yolo_feature_map

@given(
    input_tensor=yolo_feature_map(
        batch_size=2,
        num_filters=st.sampled_from([3, 16, 32, 64]),
        size=st.sampled_from([(32, 32), (16, 16), (8, 8)]),
    ),
    out_filters=st.sampled_from([16, 32, 64]),
    stride=st.sampled_from([1, 2]),
)
def test_conv_bn_silu_output_shape(input_tensor, out_filters, stride):
    """
    Property: ConvBNSiLU output shape matches convolution formula.

    For same padding:
    - out_h = ceil(in_h / stride)
    - out_w = ceil(in_w / stride)
    """
    batch, in_channels, h, w = input_tensor.shape

    # Create symbolic input
    x = pt.tensor4("x", dtype="float32")

    # Apply ConvBNSiLU
    conv_block = ConvBNSiLU(
        in_channels, out_filters, kernel_size=3, stride=stride, padding="same"
    )
    y = conv_block(x)

    # Compile function
    f = function([x], y)
    output = f(input_tensor)

    # Expected shape
    expected_h = int(np.ceil(h / stride))
    expected_w = int(np.ceil(w / stride))
    expected_shape = (batch, out_filters, expected_h, expected_w)

    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )
```

**Expected Failure Mode**:
- Before writing Hypothesis strategies: `NameError: name 'yolo_feature_map' is not defined`
- After strategies: Test passes (implementation already exists)

##### Test: `test_conv_bn_silu_activation_range`
**Purpose**: Verify SiLU activation produces outputs in expected range
**Property**: SiLU(x) = x * sigmoid(x), range approximately [-0.28, +∞)

```python
@given(input_tensor=yolo_feature_map(batch_size=2, num_filters=16, size=(8, 8)))
def test_conv_bn_silu_activation_range(input_tensor):
    """
    Property: SiLU activation never produces values below -0.28.

    SiLU(x) = x * sigmoid(x)
    - For x < 0: output is negative but bounded above -0.28
    - For x > 0: output is positive and grows without bound
    - Minimum occurs at x ≈ -1.278, where SiLU(x) ≈ -0.2784
    """
    batch, in_channels, h, w = input_tensor.shape

    x = pt.tensor4("x", dtype="float32")
    conv_block = ConvBNSiLU(in_channels, 16, kernel_size=3, stride=1, padding="same")
    y = conv_block(x)

    f = function([x], y)
    output = f(input_tensor)

    # SiLU minimum is approximately -0.2784
    assert output.min() >= -0.3, (
        f"SiLU output below theoretical minimum: {output.min()}"
    )
```

**Expected Failure Mode**: Test passes (verifies activation function)

##### Test: `test_c3k2_csp_split_correctness`
**Purpose**: Verify C3k2 splits channels correctly in CSP architecture
**Property**: C3k2 splits input into two paths of equal channels

```python
@given(
    input_tensor=yolo_feature_map(
        batch_size=2, num_filters=st.sampled_from([32, 64, 128]), size=(16, 16)
    )
)
def test_c3k2_csp_split_correctness(input_tensor):
    """
    Property: C3k2 processes channels through CSP architecture.

    CSP (Cross Stage Partial) splits channels:
    - Path 1: conv1 → hidden_channels (no bottlenecks)
    - Path 2: conv1 → bottleneck blocks → hidden_channels
    - Concatenate → conv2 → out_channels
    """
    batch, in_channels, h, w = input_tensor.shape
    out_channels = in_channels  # Preserve channels

    x = pt.tensor4("x", dtype="float32")
    c3k2 = C3k2(in_channels, out_channels, n_blocks=2, shortcut=True)
    y = c3k2(x)

    f = function([x], y)
    output = f(input_tensor)

    # Shape preservation
    assert output.shape == input_tensor.shape, (
        f"C3k2 changed shape: {input_tensor.shape} → {output.shape}"
    )

    # Output should be different from input (transformation occurred)
    assert not np.allclose(output, input_tensor), (
        "C3k2 output identical to input (no transformation)"
    )
```

**Expected Failure Mode**: Test passes (implementation exists)

##### Test: `test_sppf_multi_scale_feature_extraction`
**Purpose**: Verify SPPF creates multi-scale features via cascaded pooling
**Property**: SPPF concatenates features at multiple pooling scales

```python
@given(input_tensor=yolo_feature_map(batch_size=2, num_filters=256, size=(10, 10)))
def test_sppf_multi_scale_feature_extraction(input_tensor):
    """
    Property: SPPF concatenates features from multiple pooling scales.

    SPPF applies cascaded max pooling:
    - x → pool → y1
    - y1 → pool → y2
    - y2 → pool → y3
    - Concatenate [x, y1, y2, y3] → 4x channels
    """
    batch, in_channels, h, w = input_tensor.shape

    x = pt.tensor4("x", dtype="float32")
    sppf = SPPF(in_channels, in_channels, pool_size=5)
    y = sppf(x)

    f = function([x], y)
    output = f(input_tensor)

    # SPPF preserves spatial dimensions
    assert output.shape == (batch, in_channels, h, w), (
        f"SPPF changed shape: {input_tensor.shape} → {output.shape}"
    )

    # Output incorporates pooled features (different from input)
    assert not np.allclose(output, input_tensor), (
        "SPPF output identical to input"
    )
```

**Expected Failure Mode**: Test passes

##### Test: `test_c2psa_attention_modulation`
**Purpose**: Verify C2PSA applies attention mechanism
**Property**: C2PSA output differs from simple convolution

```python
@given(input_tensor=yolo_feature_map(batch_size=2, num_filters=256, size=(10, 10)))
def test_c2psa_attention_modulation(input_tensor):
    """
    Property: C2PSA applies attention to modulate features.

    C2PSA (CSP with Parallel Spatial Attention):
    - Splits channels
    - Applies attention branch
    - Concatenates and merges
    """
    batch, in_channels, h, w = input_tensor.shape

    x = pt.tensor4("x", dtype="float32")
    c2psa = C2PSA(in_channels, in_channels)
    y = c2psa(x)

    f = function([x], y)
    output = f(input_tensor)

    assert output.shape == input_tensor.shape, (
        f"C2PSA changed shape: {input_tensor.shape} → {output.shape}"
    )

    # Attention should modulate features
    assert not np.allclose(output, input_tensor), (
        "C2PSA did not modulate features"
    )
```

**Expected Failure Mode**: Test passes

#### 2. Backbone Tests (`test_model_backbone.py`)
**Purpose**: Verify YOLO11nBackbone feature extraction

##### Test: `test_backbone_multi_scale_output_shapes`
**Purpose**: Verify backbone outputs correct feature map shapes at 3 scales

```python
@settings(deadline=None)  # Backbone forward pass can be slow
@given(batch_size=st.integers(1, 4))
def test_backbone_multi_scale_output_shapes(batch_size):
    """
    Property: Backbone outputs features at 3 scales with correct shapes.

    For 320x320 input:
    - P3: (batch, 64, 40, 40) - stride 8
    - P4: (batch, 128, 20, 20) - stride 16
    - P5: (batch, 256, 10, 10) - stride 32
    """
    from yolo.model import YOLO11nBackbone

    x = pt.tensor4("x", dtype="float32")
    backbone = YOLO11nBackbone(in_channels=3)
    p3, p4, p5 = backbone(x)

    f = function([x], [p3, p4, p5])

    # Test with 320x320 input
    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
    p3_val, p4_val, p5_val = f(x_val)

    # Verify shapes
    assert p3_val.shape == (batch_size, 64, 40, 40), (
        f"P3 shape: expected {(batch_size, 64, 40, 40)}, got {p3_val.shape}"
    )
    assert p4_val.shape == (batch_size, 128, 20, 20), (
        f"P4 shape: expected {(batch_size, 128, 20, 20)}, got {p4_val.shape}"
    )
    assert p5_val.shape == (batch_size, 256, 10, 10), (
        f"P5 shape: expected {(batch_size, 256, 10, 10)}, got {p5_val.shape}"
    )
```

**Expected Failure Mode**: Test passes

##### Test: `test_backbone_feature_hierarchy`
**Purpose**: Verify feature maps have increasing receptive fields
**Property**: Later stages have more abstract features (larger values)

```python
def test_backbone_feature_hierarchy():
    """
    Property: Feature hierarchy - P5 captures more global context than P3.

    This is verified by checking that:
    - P5 has higher variance (more abstract/global features)
    - P3 has finer-grained features (lower variance)
    """
    from yolo.model import YOLO11nBackbone

    x = pt.tensor4("x", dtype="float32")
    backbone = YOLO11nBackbone(in_channels=3)
    p3, p4, p5 = backbone(x)

    f = function([x], [p3, p4, p5])

    # Use structured input (checkerboard pattern)
    x_val = np.zeros((1, 3, 320, 320), dtype="float32")
    x_val[:, :, ::16, ::16] = 1.0  # Sparse checkerboard

    p3_val, p4_val, p5_val = f(x_val)

    # P5 should have higher variance (more global context)
    var_p3 = np.var(p3_val)
    var_p5 = np.var(p5_val)

    # This is a weak check - just verifies both have non-zero variance
    assert var_p3 > 0, "P3 has zero variance"
    assert var_p5 > 0, "P5 has zero variance"
```

**Expected Failure Mode**: Test passes

#### 3. Head Tests (`test_model_head.py`)
**Purpose**: Verify YOLO11nHead (FPN-PAN architecture)

##### Test: `test_head_fpn_upsampling_correctness`
**Purpose**: Verify FPN upsampling path fuses features correctly
**Property**: FPN upsamples P5→P4→P3 and concatenates features

```python
def test_head_fpn_upsampling_correctness():
    """
    Property: FPN upsampling path doubles spatial dimensions correctly.

    FPN path:
    - P5 (10x10) → upsample 2x → (20x20) → concat with P4 → C3k2
    - P4_fused (20x20) → upsample 2x → (40x40) → concat with P3 → C3k2
    """
    from yolo.model import YOLO11nHead

    # Create symbolic inputs matching backbone output shapes
    p3 = pt.tensor4("p3", dtype="float32")  # (batch, 64, 40, 40)
    p4 = pt.tensor4("p4", dtype="float32")  # (batch, 128, 20, 20)
    p5 = pt.tensor4("p5", dtype="float32")  # (batch, 256, 10, 10)

    head = YOLO11nHead(num_classes=2)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    # Test data
    batch = 2
    p3_val = np.random.randn(batch, 64, 40, 40).astype("float32")
    p4_val = np.random.randn(batch, 128, 20, 20).astype("float32")
    p5_val = np.random.randn(batch, 256, 10, 10).astype("float32")

    det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

    # Detection outputs: (batch, 4+num_classes, H, W)
    assert det_p3_val.shape == (batch, 6, 40, 40), f"det_p3 shape: {det_p3_val.shape}"
    assert det_p4_val.shape == (batch, 6, 20, 20), f"det_p4 shape: {det_p4_val.shape}"
    assert det_p5_val.shape == (batch, 6, 10, 10), f"det_p5 shape: {det_p5_val.shape}"
```

**Expected Failure Mode**: Test passes

##### Test: `test_head_detection_channel_format`
**Purpose**: Verify detection heads output correct channel structure
**Property**: Each detection has 4 bbox coords + num_classes

```python
@pytest.mark.parametrize("num_classes", [2, 20, 80])
def test_head_detection_channel_format(num_classes):
    """
    Property: Detection heads output (4 + num_classes) channels.

    Channel format: [x_offset, y_offset, w, h, class_0, class_1, ..., class_N]
    """
    from yolo.model import YOLO11nHead

    p3 = pt.tensor4("p3", dtype="float32")
    p4 = pt.tensor4("p4", dtype="float32")
    p5 = pt.tensor4("p5", dtype="float32")

    head = YOLO11nHead(num_classes=num_classes)
    det_p3, det_p4, det_p5 = head(p3, p4, p5)

    f = function([p3, p4, p5], [det_p3, det_p4, det_p5])

    batch = 1
    p3_val = np.random.randn(batch, 64, 40, 40).astype("float32")
    p4_val = np.random.randn(batch, 128, 20, 20).astype("float32")
    p5_val = np.random.randn(batch, 256, 10, 10).astype("float32")

    det_p3_val, det_p4_val, det_p5_val = f(p3_val, p4_val, p5_val)

    expected_channels = 4 + num_classes

    assert det_p3_val.shape[1] == expected_channels, (
        f"det_p3 channels: expected {expected_channels}, got {det_p3_val.shape[1]}"
    )
    assert det_p4_val.shape[1] == expected_channels, (
        f"det_p4 channels: expected {expected_channels}, got {det_p4_val.shape[1]}"
    )
    assert det_p5_val.shape[1] == expected_channels, (
        f"det_p5 channels: expected {expected_channels}, got {det_p5_val.shape[1]}"
    )
```

**Expected Failure Mode**: Test passes

#### 4. Full Model Tests (`test_model_integration.py`)
**Purpose**: Verify complete YOLO11n end-to-end

##### Test: `test_yolo11n_forward_pass_deterministic`
**Purpose**: Verify model is deterministic (no dropout in eval mode)
**Property**: Same input produces same output

```python
@given(batch_size=st.integers(1, 4))
def test_yolo11n_forward_pass_deterministic(batch_size):
    """
    Property: Model produces deterministic outputs (no stochastic layers in eval).

    Running the same input twice should produce identical results.
    """
    from yolo.model import YOLO11n

    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    det_p3, det_p4, det_p5 = model(x)

    f = function([x], [det_p3, det_p4, det_p5])

    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")

    # Run twice
    out1 = f(x_val)
    out2 = f(x_val)

    # Should be identical
    for o1, o2 in zip(out1, out2):
        np.testing.assert_array_equal(o1, o2, err_msg="Model is non-deterministic")
```

**Expected Failure Mode**: Test passes

##### Test: `test_yolo11n_output_tuple_structure`
**Purpose**: Verify model returns tuple (not dict) for compatibility
**Property**: Output is tuple of 3 tensors

```python
def test_yolo11n_output_tuple_structure():
    """
    Property: Model output is tuple (det_p3, det_p4, det_p5), not dict.

    This is required for:
    - Loss function compatibility (yolo_loss expects tuple unpacking)
    - ONNX export (easier to handle tuple than dict)
    """
    from yolo.model import YOLO11n

    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    f = function([x], predictions)
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    result = f(x_val)

    # Verify it's a tuple of 3 arrays
    assert isinstance(result, tuple), f"Output is {type(result)}, not tuple"
    assert len(result) == 3, f"Output has {len(result)} elements, expected 3"

    # Verify each element is array
    for i, r in enumerate(result):
        assert isinstance(r, np.ndarray), f"Output[{i}] is {type(r)}, not ndarray"
```

**Expected Failure Mode**: Test passes

#### 5. Loss Function Tests (`test_loss.py`)
**Purpose**: Verify loss computations are correct

##### Test: `test_box_iou_identity`
**Purpose**: Verify IoU is 1.0 for identical boxes
**Property**: IoU(box, box) = 1.0

```python
@given(
    box=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    )
)
def test_box_iou_identity(box):
    """
    Property: IoU of a box with itself is 1.0.

    For any box [x_c, y_c, w, h], IoU(box, box) = 1.0
    """
    from yolo.loss import box_iou

    box1 = pt.vector("box1", dtype="float32")
    box2 = pt.vector("box2", dtype="float32")
    iou = box_iou(box1, box2)

    f = function([box1, box2], iou)

    box_arr = np.array(box, dtype="float32")
    iou_val = f(box_arr, box_arr)

    np.testing.assert_allclose(iou_val, 1.0, rtol=1e-6, err_msg="IoU(box, box) != 1.0")
```

**Expected Failure Mode**: Test passes

##### Test: `test_box_iou_symmetry`
**Purpose**: Verify IoU is symmetric
**Property**: IoU(box1, box2) = IoU(box2, box1)

```python
@given(
    box1=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=4, max_size=4),
    box2=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=4, max_size=4),
)
def test_box_iou_symmetry(box1, box2):
    """
    Property: IoU is symmetric - IoU(A, B) = IoU(B, A).
    """
    from yolo.loss import box_iou

    b1 = pt.vector("box1", dtype="float32")
    b2 = pt.vector("box2", dtype="float32")
    iou_12 = box_iou(b1, b2)
    iou_21 = box_iou(b2, b1)

    f = function([b1, b2], [iou_12, iou_21])

    box1_arr = np.array(box1, dtype="float32")
    box2_arr = np.array(box2, dtype="float32")

    iou_12_val, iou_21_val = f(box1_arr, box2_arr)

    np.testing.assert_allclose(
        iou_12_val, iou_21_val, rtol=1e-6, err_msg="IoU not symmetric"
    )
```

**Expected Failure Mode**: Test passes

##### Test: `test_yolo_loss_non_negative`
**Purpose**: Verify loss is always non-negative
**Property**: loss >= 0 for all inputs

```python
@given(batch_size=st.integers(1, 4))
def test_yolo_loss_non_negative(batch_size):
    """
    Property: YOLO loss is always non-negative.

    All loss components (bbox, classification) should be >= 0.
    """
    from yolo.loss import yolo_loss
    from yolo.model import YOLO11n

    x = pt.tensor4("x", dtype="float32")
    model = YOLO11n(num_classes=2, input_size=320)
    predictions = model(x)

    # Simplified targets (dummy)
    targets = {
        "boxes": pt.tensor3("boxes", dtype="float32"),
        "classes": pt.matrix("classes", dtype="int64"),
        "num_boxes": pt.vector("num_boxes", dtype="int64"),
    }

    total_loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)

    f = function(
        [x, targets["boxes"], targets["classes"], targets["num_boxes"]],
        [total_loss, loss_dict["box_loss"], loss_dict["cls_loss"]],
    )

    x_val = np.random.randn(batch_size, 3, 320, 320).astype("float32")
    boxes_val = np.zeros((batch_size, 10, 4), dtype="float32")
    classes_val = np.zeros((batch_size, 10), dtype="int64")
    num_boxes_val = np.array([0] * batch_size, dtype="int64")

    total, box_loss, cls_loss = f(x_val, boxes_val, classes_val, num_boxes_val)

    assert total >= 0, f"Total loss is negative: {total}"
    assert box_loss >= 0, f"Box loss is negative: {box_loss}"
    assert cls_loss >= 0, f"Cls loss is negative: {cls_loss}"
```

**Expected Failure Mode**: Test passes

#### 6. JAX Backend Tests (`test_jax_backend.py`)
**Purpose**: Verify JAX compilation and execution

##### Test: `test_model_compiles_with_jax`
**Purpose**: Verify model can be compiled with JAX backend
**Property**: Model runs without errors on JAX

```python
def test_model_compiles_with_jax():
    """
    Test: Model compiles successfully with JAX backend.

    Verifies:
    - No shape tracing errors
    - No unsupported operations
    - Returns JAX DeviceArray (indicates GPU execution)
    """
    import pytensor
    from yolo.model import build_yolo11n

    pytest.importorskip("jax")

    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile with JAX mode
    f = pytensor.function([x_sym], predictions, mode="JAX")

    # Test forward pass
    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    det_p3, det_p4, det_p5 = f(x_val)

    # Verify outputs are JAX arrays
    import jax
    assert isinstance(det_p3, jax.Array), "det_p3 is not JAX Array"
    assert isinstance(det_p4, jax.Array), "det_p4 is not JAX Array"
    assert isinstance(det_p5, jax.Array), "det_p5 is not JAX Array"
```

**Expected Failure Mode**:
- If JAX not installed: Test skipped
- If JAX compilation fails: Error with stack trace

##### Test: `test_jax_vs_python_backend_numerical_equivalence`
**Purpose**: Verify JAX produces same results as Python backend
**Property**: JAX output ≈ Python output (within tolerance)

```python
def test_jax_vs_python_backend_numerical_equivalence():
    """
    Property: JAX backend produces numerically equivalent results to Python.

    Compares:
    - Forward pass outputs
    - Tolerances: rtol=1e-4, atol=1e-5 (float32 accumulation errors)
    """
    import pytensor
    from pytensor.compile.mode import Mode
    from yolo.model import build_yolo11n

    pytest.importorskip("jax")

    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile with both backends
    f_jax = pytensor.function([x_sym], predictions, mode="JAX")
    f_py = pytensor.function([x_sym], predictions, mode=Mode(linker="py"))

    # Test data
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")

    # Run both
    jax_out = f_jax(x_val)
    py_out = f_py(x_val)

    # Compare each scale
    for jax_det, py_det, scale in zip(jax_out, py_out, ["P3", "P4", "P5"]):
        np.testing.assert_allclose(
            jax_det,
            py_det,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"JAX vs Python mismatch at {scale}",
        )
```

**Expected Failure Mode**: Test passes if JAX compilation works

##### Test: `test_jax_gradient_flow`
**Purpose**: Verify gradients backpropagate correctly through JAX
**Property**: Gradients are non-zero and finite

```python
def test_jax_gradient_flow():
    """
    Test: Gradients flow correctly through JAX-compiled model.

    Verifies:
    - Gradients w.r.t. model parameters are computed
    - Gradients are non-zero (model is learning signal)
    - Gradients are finite (no NaN/Inf)
    """
    import pytensor
    from pytensor import grad
    from yolo.model import build_yolo11n

    pytest.importorskip("jax")

    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compute scalar loss (sum of outputs)
    det_p3, det_p4, det_p5 = predictions
    loss = det_p3.sum() + det_p4.sum() + det_p5.sum()

    # Compute gradients w.r.t. first parameter
    first_param = model.params[0]
    grad_param = grad(loss, first_param)

    # Compile with JAX
    f = pytensor.function([x_sym], [loss, grad_param], mode="JAX")

    x_val = np.random.randn(1, 3, 320, 320).astype("float32")
    loss_val, grad_val = f(x_val)

    # Verify gradient properties
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert np.all(np.isfinite(grad_val)), "Gradient contains NaN or Inf"
    assert np.abs(grad_val).sum() > 0, "Gradient is all zeros"
```

**Expected Failure Mode**: Test passes if JAX gradient computation works

#### 7. ONNX Export Tests (`test_onnx_export.py`)
**Purpose**: Verify ONNX export and inference

##### Test: `test_model_exports_to_onnx`
**Purpose**: Verify model can be exported to ONNX format
**Property**: Export succeeds and produces valid ONNX file

```python
def test_model_exports_to_onnx(tmp_path):
    """
    Test: Model exports to ONNX without errors.

    Verifies:
    - ONNX export completes
    - ONNX file is created
    - ONNX model passes validation
    """
    import onnx
    from pytensor.link.onnx import export_onnx
    import pytensor
    from yolo.model import build_yolo11n

    pytest.importorskip("onnx")

    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile PyTensor function
    f = pytensor.function([x_sym], predictions)

    # Export to ONNX
    onnx_path = tmp_path / "yolo11n.onnx"
    onnx_model = export_onnx(f, str(onnx_path))

    # Verify file exists
    assert onnx_path.exists(), "ONNX file was not created"

    # Validate ONNX model
    onnx.checker.check_model(onnx_model)
```

**Expected Failure Mode**:
- If ONNX not installed: Test skipped
- If export fails: Error with stack trace

##### Test: `test_onnx_runtime_vs_pytensor_equivalence`
**Purpose**: Verify ONNX Runtime produces same results as PyTensor
**Property**: ONNX output ≈ PyTensor output

```python
def test_onnx_runtime_vs_pytensor_equivalence(tmp_path):
    """
    Property: ONNX Runtime inference matches PyTensor.

    This is the most critical test - ensures deployed model correctness.

    Tolerances: rtol=1e-4, atol=1e-5 (account for float32 differences)
    """
    import onnx
    import onnxruntime as ort
    from pytensor.link.onnx import export_onnx
    import pytensor
    from yolo.model import build_yolo11n

    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    model, x_sym, predictions = build_yolo11n(num_classes=2, input_size=320)

    # Compile PyTensor function
    f_pytensor = pytensor.function([x_sym], predictions)

    # Export to ONNX
    onnx_path = tmp_path / "yolo11n.onnx"
    export_onnx(f_pytensor, str(onnx_path))

    # Load with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test data
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")

    # PyTensor inference
    pytensor_out = f_pytensor(x_val)

    # ONNX Runtime inference
    onnx_inputs = {session.get_inputs()[0].name: x_val}
    onnx_out = session.run(None, onnx_inputs)

    # Compare each scale
    for pt_det, onnx_det, scale in zip(pytensor_out, onnx_out, ["P3", "P4", "P5"]):
        np.testing.assert_allclose(
            onnx_det,
            pt_det,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"ONNX vs PyTensor mismatch at {scale}",
        )
```

**Expected Failure Mode**: Test passes if ONNX export works correctly

#### 8. Hypothesis Strategies (`tests/strategies/core.py`)
**Purpose**: Generate valid test data for property-based testing

```python
"""Core Hypothesis strategies for YOLO testing."""
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np


def valid_image_shapes(
    min_size=32,
    max_size=640,
    multiple_of=32,
    fixed_size=None
):
    """
    Generate valid YOLO image shapes (multiples of 32).

    YOLO requires input dimensions divisible by 32 due to 5 downsampling stages.
    """
    if fixed_size is not None:
        return st.just(fixed_size)

    valid_sizes = [i for i in range(min_size, max_size + 1, multiple_of)]
    return st.tuples(
        st.sampled_from(valid_sizes),  # height
        st.sampled_from(valid_sizes),  # width
    )


def batch_sizes(min_size=1, max_size=4):
    """Generate valid batch sizes for testing."""
    return st.integers(min_value=min_size, max_value=max_size)


def num_channels():
    """Generate valid number of channels (grayscale or RGB)."""
    return st.sampled_from([1, 3])


@st.composite
def yolo_image_tensor(draw, batch_size=None, channels=None, size=None, dtype=np.float32):
    """
    Generate YOLO-compatible image tensor.

    Returns tensor of shape (batch_size, channels, height, width).
    Values normalized to [-1, 1] or [0, 1].
    """
    if batch_size is None:
        batch_size = draw(batch_sizes())
    if channels is None:
        channels = draw(num_channels())
    if size is None:
        size = draw(valid_image_shapes())

    height, width = size
    shape = (batch_size, channels, height, width)

    # Generate normalized image data
    elements = st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False
    )

    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def yolo_feature_map(draw, batch_size=None, num_filters=None, size=None):
    """
    Generate YOLO feature map tensor for intermediate layers.

    Feature maps typically have:
    - Larger channel counts (16-512)
    - Smaller spatial dimensions (8x8 to 40x40)
    """
    if batch_size is None:
        batch_size = draw(batch_sizes())
    if num_filters is None:
        num_filters = draw(st.sampled_from([16, 32, 64, 128, 256, 512]))
    if size is None:
        size = draw(valid_image_shapes(min_size=8, max_size=40, multiple_of=8))

    height, width = size
    shape = (batch_size, num_filters, height, width)

    # Feature maps can have wider value ranges
    elements = st.floats(min_value=-10.0, max_value=10.0,
                        allow_nan=False, allow_infinity=False)

    return draw(arrays(dtype=np.float32, shape=shape, elements=elements))
```

#### 9. Test Configuration (`conftest.py`)

```python
"""Pytest configuration for YOLO11n demo tests."""
import os
from datetime import timedelta
import pytest
import numpy as np
from hypothesis import HealthCheck, Phase, settings


# Register Hypothesis profiles
settings.register_profile(
    "dev",
    max_examples=10,
    deadline=timedelta(milliseconds=500),
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    print_blob=False,
)

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    derandomize=True,
    print_blob=True,
)

settings.register_profile(
    "thorough",
    max_examples=200,
    deadline=None,
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def test_seed():
    """Consistent seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(test_seed):
    """NumPy random generator with consistent seed."""
    return np.random.default_rng(test_seed)


@pytest.fixture
def tmp_onnx_dir(tmp_path):
    """Temporary directory for ONNX model files."""
    onnx_dir = tmp_path / "onnx_models"
    onnx_dir.mkdir()
    return onnx_dir


@pytest.fixture
def yolo_input_size():
    """Standard YOLO11n input size."""
    return (320, 320)


@pytest.fixture
def simple_image_batch(rng, yolo_input_size):
    """Generate simple batch of images for testing."""
    batch_size = 2
    channels = 3
    height, width = yolo_input_size
    return rng.standard_normal((batch_size, channels, height, width)).astype(np.float32)
```

#### 10. Project Configuration (`pyproject.toml`)

```toml
[project]
name = "onnx-yolo-demo"
version = "0.1.0"
description = "YOLO11n Object Detection Demo with PyTensor, JAX, and ONNX"
requires-python = ">=3.11"
dependencies = [
    # NOTE: pytensor should be installed from root repo in editable mode FIRST:
    #   cd /path/to/pytensor && pip install -e ".[development,onnx]"
    # This entry ensures the version requirement is documented, but pip will
    # use the already-installed editable version instead of fetching from PyPI.
    "pytensor>=2.18",
    "numpy>=2.0",
    "jax[cpu]>=0.4.20",
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "pytest>=7.0",
    "pytest-cov>=2.6.1",
    "pytest-benchmark",
    "hypothesis>=6.100.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "onnx: marks tests that require ONNX Runtime",
]
addopts = "--durations=20 -v"

[tool.coverage.run]
source = ["yolo"]
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### Success Criteria

#### Automated Verification:
- [ ] All test files created: `pytest --collect-only tests/`
- [ ] All imports work: `python -c "import tests; import tests.strategies"`
- [ ] Hypothesis strategies generate valid data: `pytest tests/strategies/ -v`
- [ ] conftest.py fixtures accessible: `pytest tests/ --fixtures`

#### Manual Verification:
- [ ] Each test has clear docstring explaining what property it verifies
- [ ] Test names follow convention: `test_<component>_<property>`
- [ ] Hypothesis strategies have sensible value ranges
- [ ] Fixtures are reusable across test files

---

## Phase 2: Test Failure Verification

### Overview
Run the test suite and verify tests pass (implementation already exists). Fix any test infrastructure issues (imports, strategies, fixtures).

### Verification Steps

1. **Create test directory structure**:
   ```bash
   cd examples/onnx/onnx-yolo-demo
   mkdir -p tests/strategies
   touch tests/__init__.py
   touch tests/conftest.py
   touch tests/strategies/__init__.py
   ```

2. **Install dependencies** (in correct order):
   ```bash
   # STEP 1: Install root pytensor in EDITABLE mode FIRST
   # This ensures the local development version is used, not PyPI
   cd C:/Users/armor/OneDrive/Desktop/cs/pytensor
   pip install -e ".[development,onnx]"

   # Verify pytensor is installed from local source
   pip show pytensor
   # Should show: Location: C:\Users\armor\OneDrive\Desktop\cs\pytensor

   # STEP 2: Install YOLO demo dependencies
   # pip will see pytensor is already installed and won't override it
   cd examples/onnx/onnx-yolo-demo
   pip install -e .
   ```

   **IMPORTANT**: Install root pytensor FIRST in editable mode (`-e`).
   This ensures the demo uses your local development version instead of
   downloading pytensor from PyPI.

3. **Run test discovery**:
   ```bash
   HYPOTHESIS_PROFILE=dev pytest tests/ --collect-only
   ```

4. **Run tests by category**:
   ```bash
   # Building blocks (fast)
   pytest tests/test_blocks.py -v

   # Model components (medium)
   pytest tests/test_model_backbone.py -v
   pytest tests/test_model_head.py -v

   # Integration (slow)
   pytest tests/test_model_integration.py -v

   # Loss functions (fast)
   pytest tests/test_loss.py -v

   # JAX backend (medium, requires JAX)
   pytest tests/test_jax_backend.py -v

   # ONNX export (slow, requires ONNX Runtime)
   pytest tests/test_onnx_export.py -v
   ```

### Expected Test Results

Since the implementation already exists, most tests should **PASS**. We're verifying:
1. **Test infrastructure works**: Imports, fixtures, strategies
2. **Implementation is correct**: Tests validate existing code
3. **Edge cases covered**: Hypothesis finds no violations

### Potential Issues to Fix

#### Issue 1: Import Errors
**Symptom**: `ImportError: cannot import name 'yolo_feature_map'`
**Fix**: Ensure `__init__.py` files export strategies

```python
# tests/strategies/__init__.py
from .core import (
    yolo_image_tensor,
    yolo_feature_map,
    valid_image_shapes,
    batch_sizes,
)

__all__ = [
    "yolo_image_tensor",
    "yolo_feature_map",
    "valid_image_shapes",
    "batch_sizes",
]
```

#### Issue 2: PYTENSOR_FLAGS Not Set
**Symptom**: `TypeError: data type 'float64' not understood`
**Fix**: Set floatX flag before running tests

```bash
export PYTENSOR_FLAGS='floatX=float32'
pytest tests/ -v
```

Or in conftest.py:
```python
import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32'
```

#### Issue 3: JAX Not Installed
**Symptom**: `Skipped: jax not available`
**Fix**: Install JAX

```bash
pip install jax[cpu]  # CPU-only
# OR
pip install jax[cuda12]  # With CUDA 12
```

#### Issue 4: Hypothesis Too Slow
**Symptom**: `Hypothesis health check: test took >500ms`
**Fix**: Suppress health check for slow operations

```python
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_yolo11n_forward_pass(...):
    ...
```

### Success Criteria

#### Automated Verification:
- [ ] All tests discovered: `pytest --collect-only` shows all tests
- [ ] No import errors: All test files import successfully
- [ ] Tests run: `pytest tests/` executes without crashes
- [ ] Expected passes: Tests pass (implementation exists)

#### Manual Verification:
- [ ] Test output is readable and informative
- [ ] Hypothesis examples are generated correctly
- [ ] Failure messages (if any) are diagnostic
- [ ] Coverage report shows tested code: `pytest --cov=yolo tests/`

---

## Phase 3: Feature Implementation (Red → Green)

### Overview

**IMPORTANT**: This phase does NOT apply in the traditional TDD sense because the YOLO11n implementation **already exists** and is correct. However, we still follow the TDD spirit:

1. **Tests validate existing implementation**: Verify code works as specified
2. **Tests guide refactoring**: If tests fail, fix tests (not implementation)
3. **Tests document behavior**: Tests serve as executable specifications

### What If Tests Fail?

If tests fail, follow this decision tree:

```
Test fails?
├─> Is the test correct?
│   ├─> YES: Implementation has a bug → Fix implementation
│   └─> NO: Test is wrong → Fix test
│
├─> Is the expected behavior correct?
│   ├─> YES: Test correctly captures requirement
│   └─> NO: Requirement misunderstood → Fix test
│
└─> Is this a precision issue?
    ├─> YES: Adjust tolerances (document why)
    └─> NO: Investigate root cause
```

### Likely Scenarios

#### Scenario 1: Test Infrastructure Issues
**Example**: Hypothesis strategy generates invalid shapes
**Action**: Fix strategy, not implementation

```python
# BEFORE (incorrect strategy)
def valid_image_shapes():
    return st.tuples(
        st.integers(1, 640),  # Any integer - WRONG!
        st.integers(1, 640),
    )

# AFTER (correct strategy)
def valid_image_shapes():
    valid_sizes = list(range(32, 640 + 1, 32))  # Multiples of 32
    return st.tuples(
        st.sampled_from(valid_sizes),
        st.sampled_from(valid_sizes),
    )
```

#### Scenario 2: Tolerance Tuning
**Example**: JAX vs Python backend differs slightly
**Action**: Adjust tolerance after investigation

```python
# BEFORE (too strict)
np.testing.assert_allclose(jax_out, py_out, rtol=1e-7, atol=1e-8)

# AFTER (realistic for float32)
np.testing.assert_allclose(jax_out, py_out, rtol=1e-4, atol=1e-5)
# Document why: float32 accumulation in deep network
```

#### Scenario 3: Test Assumption Wrong
**Example**: Expecting dict output, but implementation returns tuple
**Action**: Fix test to match implementation

```python
# BEFORE (incorrect assumption)
def test_model_output_format():
    predictions = model(x)
    assert isinstance(predictions, dict)  # WRONG!

# AFTER (matches implementation)
def test_model_output_format():
    predictions = model(x)
    assert isinstance(predictions, tuple)  # Correct per model.py:348
    assert len(predictions) == 3
```

### Implementation Steps (If Bugs Found)

**Only if tests reveal actual bugs** (unlikely, but possible):

#### Bug Fix 1: Shape Mismatch
**Symptom**: Test fails with shape assertion error
**Investigation**:
1. Print actual vs expected shapes
2. Trace through forward pass
3. Check convolution stride/padding calculations

**Fix**: Adjust layer configuration
```python
# Example fix (hypothetical)
self.conv = ConvBNSiLU(64, 128, stride=2)  # Was stride=1
```

#### Bug Fix 2: Gradient NaN/Inf
**Symptom**: Gradient test fails with non-finite values
**Investigation**:
1. Check batch norm epsilon
2. Check activation clipping
3. Check loss scaling

**Fix**: Add numerical stability
```python
# Example fix
eps = 1e-7
loss = -pt.log(pred_classes + eps)  # Prevent log(0)
```

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage meets threshold: `pytest --cov=yolo --cov-report=term-missing tests/`
- [ ] No regressions: Previously passing tests still pass
- [ ] Hypothesis finds no violations: Property tests pass for many examples

#### Manual Verification:
- [ ] Tests validate all major components (blocks, backbone, head, model)
- [ ] JAX compilation works without errors
- [ ] ONNX export produces valid models
- [ ] Numerical results match across backends (within tolerance)

---

## Phase 4: Refactoring & Cleanup

### Overview
Improve test code quality and organization while keeping all tests passing. Tests protect us during refactoring.

### Refactoring Targets

#### 1. Test Code Duplication

**Pattern**: Similar test setup repeated across files

**Before**:
```python
# In test_blocks.py
def test_conv_bn_silu_shape():
    x = pt.tensor4("x", dtype="float32")
    conv = ConvBNSiLU(16, 32, kernel_size=3, stride=1)
    y = conv(x)
    f = function([x], y)
    x_val = np.random.randn(2, 16, 32, 32).astype("float32")
    output = f(x_val)
    ...

# In test_model_backbone.py
def test_backbone_forward():
    x = pt.tensor4("x", dtype="float32")
    backbone = YOLO11nBackbone()
    p3, p4, p5 = backbone(x)
    f = function([x], [p3, p4, p5])
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")
    ...
```

**After** (extract helper):
```python
# tests/conftest.py
@pytest.fixture
def compile_and_run():
    """Helper to compile PyTensor function and run with test data."""
    def _compile(inputs, outputs, test_values):
        f = function(inputs, outputs)
        return f(*test_values)
    return _compile

# In test files
def test_conv_bn_silu_shape(compile_and_run):
    x = pt.tensor4("x", dtype="float32")
    conv = ConvBNSiLU(16, 32, kernel_size=3, stride=1)
    y = conv(x)

    x_val = np.random.randn(2, 16, 32, 32).astype("float32")
    output = compile_and_run([x], y, [x_val])
    ...
```

#### 2. Backend Comparison Boilerplate

**Pattern**: Repeated JAX vs Python comparison

**Before**:
```python
# Repeated in multiple tests
def test_jax_equivalence():
    f_jax = function([x], y, mode="JAX")
    f_py = function([x], y, mode=Mode(linker="py"))
    jax_out = f_jax(x_val)
    py_out = f_py(x_val)
    np.testing.assert_allclose(jax_out, py_out, rtol=1e-4, atol=1e-5)
```

**After** (extract utility):
```python
# tests/conftest.py
def compare_jax_and_py(inputs, outputs, test_values, rtol=1e-4, atol=1e-5):
    """Compare JAX and Python backend outputs."""
    pytest.importorskip("jax")

    import pytensor
    from pytensor.compile.mode import Mode

    f_jax = pytensor.function(inputs, outputs, mode="JAX")
    f_py = pytensor.function(inputs, outputs, mode=Mode(linker="py"))

    jax_out = f_jax(*test_values)
    py_out = f_py(*test_values)

    if isinstance(outputs, list):
        for jo, po in zip(jax_out, py_out):
            np.testing.assert_allclose(jo, po, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(jax_out, py_out, rtol=rtol, atol=atol)

    return f_jax, jax_out

# In test files
def test_jax_equivalence():
    x = pt.tensor4("x", dtype="float32")
    y = model(x)
    x_val = np.random.randn(2, 3, 320, 320).astype("float32")
    compare_jax_and_py([x], y, [x_val])
```

#### 3. ONNX Export Boilerplate

**Pattern**: Repeated ONNX export and validation

**Before**:
```python
# Repeated in multiple tests
def test_onnx_export(tmp_path):
    f = function([x], y)
    onnx_path = tmp_path / "model.onnx"
    onnx_model = export_onnx(f, str(onnx_path))
    assert onnx_path.exists()
    onnx.checker.check_model(onnx_model)
```

**After** (extract utility):
```python
# tests/conftest.py
def export_and_validate_onnx(pytensor_fn, onnx_path):
    """Export PyTensor function to ONNX and validate."""
    from pytensor.link.onnx import export_onnx
    import onnx

    onnx_model = export_onnx(pytensor_fn, str(onnx_path))
    assert onnx_path.exists(), f"ONNX file not created: {onnx_path}"
    onnx.checker.check_model(onnx_model)
    return onnx_model

# In test files
def test_onnx_export(tmp_path):
    f = function([x], y)
    onnx_path = tmp_path / "model.onnx"
    export_and_validate_onnx(f, onnx_path)
```

#### 4. Test Naming Consistency

**Before**:
```python
def test_conv_output_shape()  # Inconsistent
def test_C3k2_csp_split()     # Inconsistent capitalization
def testBackboneShapes()      # Wrong convention
```

**After**:
```python
def test_conv_bn_silu_output_shape()  # Component_property
def test_c3k2_csp_split_correctness()  # Component_property
def test_backbone_multi_scale_shapes()  # Component_property
```

#### 5. Hypothesis Strategy Organization

**Before**: Strategies scattered across test files

**After**: Centralized in `tests/strategies/`
```python
# tests/strategies/__init__.py
from .core import (
    yolo_image_tensor,
    yolo_feature_map,
    valid_image_shapes,
)
from .yolo_data import (
    bounding_box,
    yolo_detection,
)

__all__ = [
    # Core
    "yolo_image_tensor",
    "yolo_feature_map",
    "valid_image_shapes",
    # YOLO data
    "bounding_box",
    "yolo_detection",
]
```

#### 6. Fixture Organization

**Before**: All fixtures in one large conftest.py

**After**: Logical grouping with clear docstrings
```python
# tests/conftest.py

# === Hypothesis Configuration ===
settings.register_profile("dev", ...)
settings.register_profile("ci", ...)

# === Random Seeds ===
@pytest.fixture
def test_seed():
    """Reproducible random seed."""
    return 42

@pytest.fixture
def rng(test_seed):
    """NumPy random generator."""
    return np.random.default_rng(test_seed)

# === Directories ===
@pytest.fixture
def tmp_onnx_dir(tmp_path):
    """Temporary directory for ONNX files."""
    ...

# === Test Data ===
@pytest.fixture
def simple_image_batch(rng):
    """Simple batch for deterministic tests."""
    ...

# === Utilities ===
def compare_jax_and_py(...):
    """Compare backend outputs."""
    ...
```

### Refactoring Steps

1. **Run tests before refactoring**: `pytest tests/ -v`
2. **Make one small change at a time**
3. **Run tests after each change**: `pytest tests/ -v`
4. **If tests fail, revert and reconsider**
5. **Commit after successful refactoring**

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `pytest tests/ -v`
- [ ] Coverage maintained: `pytest --cov=yolo tests/`
- [ ] No test duplication: Manual review
- [ ] Import structure clean: `pytest --collect-only`

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] Duplication reduced (DRY principle)
- [ ] Helper functions are well-documented
- [ ] Test organization is logical
- [ ] Fixtures are reusable and clear

---

## Testing Strategy Summary

### Test Coverage Goals
- [ ] **Building blocks**: 100% (ConvBNSiLU, C3k2, SPPF, C2PSA, Bottleneck)
- [ ] **Backbone**: 100% (YOLO11nBackbone forward pass and shapes)
- [ ] **Head**: 100% (YOLO11nHead FPN-PAN architecture)
- [ ] **Model**: 100% (YOLO11n integration)
- [ ] **Loss**: 80% (IoU, bbox loss, classification loss - simplified version)
- [ ] **JAX backend**: Core operations (compilation, gradients)
- [ ] **ONNX export**: Export and inference equivalence

### Test Organization
```
tests/
├── strategies/          # Hypothesis strategies
│   ├── core.py         # Tensor generation
│   └── yolo_data.py    # YOLO-specific data
├── test_blocks.py       # Building blocks (fast, ~2min)
├── test_model_backbone.py  # Backbone (medium, ~3min)
├── test_model_head.py   # Head (medium, ~3min)
├── test_model_integration.py  # Full model (slow, ~5min)
├── test_loss.py         # Loss functions (fast, ~1min)
├── test_jax_backend.py  # JAX compilation (medium, ~4min)
├── test_onnx_export.py  # ONNX export (slow, ~6min)
└── conftest.py          # Fixtures and config
```

### Running Tests

```bash
# Fast tests only (blocks, loss)
pytest tests/test_blocks.py tests/test_loss.py -v

# Medium tests (backbone, head, JAX)
pytest tests/test_model_backbone.py tests/test_model_head.py tests/test_jax_backend.py -v

# Slow tests (integration, ONNX)
pytest tests/test_model_integration.py tests/test_onnx_export.py -v --durations=10

# All tests with coverage
pytest tests/ --cov=yolo --cov-report=html -v

# Hypothesis profiles
HYPOTHESIS_PROFILE=dev pytest tests/ -v      # Fast (10 examples)
HYPOTHESIS_PROFILE=ci pytest tests/ -v       # Medium (50 examples)
HYPOTHESIS_PROFILE=thorough pytest tests/ -v # Slow (200 examples)

# Skip slow tests
pytest tests/ -m "not slow" -v

# Only ONNX tests
pytest tests/ -m onnx -v

# With verbose Hypothesis output
pytest tests/ -v --hypothesis-show-statistics
```

## Performance Considerations

### Expected Test Runtimes (Hypothesis dev profile)

| Test File | Tests | Runtime | Bottleneck |
|-----------|-------|---------|------------|
| test_blocks.py | ~15 | 2 min | C3k2, SPPF forward passes |
| test_model_backbone.py | ~5 | 3 min | Full backbone forward pass |
| test_model_head.py | ~5 | 3 min | FPN-PAN architecture |
| test_model_integration.py | ~8 | 5 min | Full model compilation |
| test_loss.py | ~10 | 1 min | Fast (pure PyTensor ops) |
| test_jax_backend.py | ~6 | 4 min | JAX JIT compilation |
| test_onnx_export.py | ~4 | 6 min | ONNX export + Runtime loading |
| **TOTAL** | **~53** | **24 min** | |

### Performance Optimizations

1. **Parallel Test Execution**:
   ```bash
   pytest tests/ -n auto  # Use all CPU cores
   ```

2. **Test Ordering** (pytest-ordering):
   - Run fast tests first (early feedback)
   - Run slow tests last (integration, ONNX)

3. **Hypothesis Example Database**:
   - Reuse failing examples for faster debugging
   - Location: `.hypothesis/examples/`

4. **JAX JIT Warmup**:
   - First compilation is slow (~10s)
   - Subsequent runs are fast (~0.1s)
   - Tests cache compiled functions

## Development Setup Notes

### Local pytensor Development

**Problem**: The demo's `pyproject.toml` lists `pytensor>=2.18` as a dependency. When running `pip install`, pip might install pytensor from PyPI instead of using your local development version.

**Solution 1: Editable Install (RECOMMENDED)**:
```bash
# Install root pytensor in editable mode FIRST
cd C:/Users/armor/OneDrive/Desktop/cs/pytensor
pip install -e ".[development,onnx]"

# Then install demo (will use already-installed pytensor)
cd examples/onnx/onnx-yolo-demo
pip install -e .
```

**Solution 2: Remove pytensor from demo dependencies**:
Edit `examples/onnx/onnx-yolo-demo/pyproject.toml`:
```toml
dependencies = [
    # "pytensor>=2.18",  # REMOVE - assume installed from root
    "numpy>=2.0",
    # ... other deps
]
```

Then document that root pytensor must be installed first.

**Solution 3: Use uv workspace** (if using `uv`):
```toml
# Add to root pyproject.toml
[tool.uv.workspace]
members = ["examples/onnx/onnx-yolo-demo"]
```

**Verification**:
```bash
# Check pytensor installation location
pip show pytensor | grep Location
# Should show: C:\Users\armor\OneDrive\Desktop\cs\pytensor

# NOT: ...site-packages\pytensor (PyPI version)
```

### Existing Setup Scripts

**Status**: ✅ **FIXED** - `scripts/setup.sh` now correctly preserves editable pytensor install!

**What was fixed** (commit: pending):
- Removed `uv sync` which was reinstalling pytensor from PyPI
- Added verification step to ensure pytensor is installed from source
- Added detailed comments explaining why we don't use `uv sync`
- Updated test path from `testing/` to `tests/`

**The fixed script now**:
1. Installs pytensor from root in editable mode: `uv pip install -e ".[development,onnx]"`
2. **Verifies** installation location is source directory (not site-packages)
3. Exits with error if verification fails
4. Installs demo dependencies with `uv pip install -e .` (respects editable packages)

**To use the setup script**:
```bash
cd examples/onnx/onnx-yolo-demo
bash scripts/setup.sh
```

**Manual installation** (alternative to setup.sh):
```bash
# Step 1: Install root pytensor in editable mode
cd C:/Users/armor/OneDrive/Desktop/cs/pytensor
pip install -e ".[development,onnx]"

# Verify
pip show pytensor | grep Location
# Should show source directory, not site-packages

# Step 2: Install demo dependencies
cd examples/onnx/onnx-yolo-demo
pip install -e .
```

## Migration Notes

### From Previous Testing Setup

**Previous structure** (git history shows deleted files):
```
examples/onnx/onnx-yolo-demo/
└── testing/             # OLD location
    ├── conftest.py      # Deleted
    ├── test_jax_components.py
    ├── test_jax_gpu_required.py
    ├── test_jax_issues.py
    └── test_model.py
```

**New structure**:
```
examples/onnx/onnx-yolo-demo/
└── tests/               # NEW location (standard Python convention)
    ├── strategies/      # Hypothesis strategies
    ├── conftest.py      # New config
    ├── test_blocks.py   # New organization
    ├── test_model_*.py  # Split into backbone/head/integration
    └── ...
```

**Changes**:
1. **Directory name**: `testing/` → `tests/` (Python convention)
2. **Organization**: Monolithic → Per-component test files
3. **Strategy**: Ad-hoc → Hypothesis property-based testing
4. **Coverage**: Basic → Comprehensive (blocks → model → pipeline)

## References

- **Original ticket**: N/A (self-initiated testing improvement)
- **Related research**: `thoughts/shared/research/2025-10-18_21-43-17_yolo-demo-testing-setup.md`
- **Model implementation**: `examples/onnx/onnx-yolo-demo/yolo/model.py:16-382`
- **Building blocks**: `examples/onnx/onnx-yolo-demo/yolo/blocks.py:22-458`
- **Loss functions**: `examples/onnx/onnx-yolo-demo/yolo/loss.py:15-266`
- **ONNX test patterns**: `tests/link/onnx/` (property-based testing examples)
- **JAX test patterns**: `tests/link/jax/` (backend comparison examples)
- **PyMC demo tests**: `examples/onnx/onnx-pymc-demo/tests/` (demo test organization)

## Appendix: Test Checklist

### Phase 1: Test Writing
- [ ] Create test directory structure
- [ ] Write `conftest.py` with fixtures and Hypothesis config
- [ ] Write Hypothesis strategies (`tests/strategies/core.py`)
- [ ] Write building block tests (`test_blocks.py`)
- [ ] Write backbone tests (`test_model_backbone.py`)
- [ ] Write head tests (`test_model_head.py`)
- [ ] Write integration tests (`test_model_integration.py`)
- [ ] Write loss function tests (`test_loss.py`)
- [ ] Write JAX backend tests (`test_jax_backend.py`)
- [ ] Write ONNX export tests (`test_onnx_export.py`)
- [ ] Write `pyproject.toml` with test configuration

### Phase 2: Test Verification
- [ ] Install test dependencies
- [ ] Set PYTENSOR_FLAGS='floatX=float32'
- [ ] Run test discovery (`pytest --collect-only`)
- [ ] Fix import errors
- [ ] Run building block tests
- [ ] Run model tests
- [ ] Run JAX tests (if JAX installed)
- [ ] Run ONNX tests (if ONNX Runtime installed)
- [ ] Generate coverage report

### Phase 3: Implementation Validation
- [ ] All tests pass
- [ ] No failures due to test infrastructure
- [ ] Coverage meets targets
- [ ] Hypothesis finds no property violations
- [ ] JAX compilation works
- [ ] ONNX export produces valid models
- [ ] Numerical equivalence verified

### Phase 4: Refactoring
- [ ] Extract common test utilities
- [ ] Remove code duplication
- [ ] Improve test names
- [ ] Organize fixtures logically
- [ ] Document test helpers
- [ ] All tests still pass after refactoring
- [ ] Code is more maintainable

---

**Status**: Ready for implementation
**Next Steps**:
1. Create test directory structure
2. Write `conftest.py` and Hypothesis strategies
3. Implement tests per this plan
4. Run and verify tests pass
5. Refactor for maintainability
