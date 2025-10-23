---
date: 2025-10-22T17:38:47-05:00
researcher: Claude Code
git_commit: 2f418100f2c66feedfee53af670389c5d9309071
branch: onnx-workshop-demo
repository: pytensor
topic: "Comprehensive List of All Operations Used in YOLO11 - Op Level, Block Level, and Higher Level"
tags: [research, codebase, yolo11, operations, pytorch, tensor-ops, architecture, onnx]
status: complete
last_updated: 2025-10-22
last_updated_by: Claude Code
---

# Research: Comprehensive List of All Operations Used in YOLO11

**Date**: 2025-10-22T17:38:47-05:00
**Researcher**: Claude Code
**Git Commit**: 2f418100f2c66feedfee53af670389c5d9309071
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question
Create a comprehensive list of all operations used in the YOLO11 implementation at three levels:
1. Op level (tensor operations)
2. Block level (building blocks and modules)
3. Higher level (architectural components)

## Summary
The YOLO11n implementation uses 31+ distinct PyTensor/NumPy operations organized into 5 building blocks (ConvBNSiLU, Bottleneck, C3k2, SPPF, C2PSA), which form 3 major architectural components (Backbone, Neck/Head, Loss). The model is designed for JAX JIT compatibility and ONNX export, using static shape operations throughout.

## Detailed Findings

### Level 1: Op Level - Tensor Operations

#### 1.1 Convolution and Pooling Operations

| Operation | Import/Module | Location | Purpose |
|-----------|--------------|----------|---------|
| `conv2d` | `pytensor.tensor.conv.abstract_conv` | `blocks.py:163-169` | 2D convolution for feature extraction |
| `pool_2d` | `pytensor.tensor.pool` | `blocks.py:362-382` | Max pooling for SPPF multi-scale features |

#### 1.2 Activation Functions

| Operation | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `pt.sigmoid` | `pytensor.tensor` | `blocks.py:177`, `loss.py:122,125` | Sigmoid activation for SiLU and output normalization |

#### 1.3 Mathematical Operations

| Operation | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `pt.sqrt` | `pytensor.tensor` | `blocks.py:56` | Square root for batch normalization |
| `pt.maximum` | `pytensor.tensor` | `loss.py:44-45,49` | Maximum for IoU intersection bounds |
| `pt.minimum` | `pytensor.tensor` | `loss.py:46-47` | Minimum for IoU intersection bounds |
| `pt.log` | `pytensor.tensor` | `loss.py:148,238` | Logarithm for binary cross-entropy |
| `pt.mean` | `pytensor.tensor` | `loss.py:141,148,233,238` | Mean reduction for loss aggregation |
| `pt.constant` | `pytensor.tensor` | `loss.py:152` | Constant tensor creation |

#### 1.4 Shape Manipulation Operations

| Operation | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `pt.concatenate` | `pytensor.tensor` | `model.py:233,238,244,249`, `blocks.py:307,385,454` | Channel-wise feature fusion |
| `.dimshuffle` | Tensor method | `blocks.py:50-53`, `loss.py:114,209` | Dimension permutation and broadcasting |
| `resize` | `pytensor.tensor.resize` | `model.py:268` | Nearest neighbor upsampling |

#### 1.5 Tensor Creation and Utility Operations

| Operation | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `pt.tensor4` | `pytensor.tensor` | `model.py:357` | Create 4D symbolic input tensor |
| `pt.zeros_like` | `pytensor.tensor` | `train.py:247` | Create zero tensor matching shape |
| `pt.as_tensor_variable` | `pytensor.tensor` | `train.py:268-270` | Convert numpy to PyTensor tensor |
| `pt.cast` | `pytensor.tensor` | `train.py:276` | Type casting (to float32) |

#### 1.6 Arithmetic Operations (Element-wise)

| Operation | Type | Location | Purpose |
|-----------|------|----------|---------|
| Addition (`+`) | operator | `blocks.py:56,59,229`, `loss.py:35-36,54` | Batch norm, residuals, union area |
| Subtraction (`-`) | operator | `blocks.py:56`, `loss.py:33-34` | Mean centering, box conversion |
| Multiplication (`*`) | operator | `blocks.py:59,177`, `loss.py:49,52-53` | Scaling, SiLU gating, area computation |
| Division (`/`) | operator | `blocks.py:56`, `loss.py:33-41,57` | Normalization, IoU ratio |
| Power (`**`) | operator | `loss.py:141,233` | L2 regularization (squared) |

#### 1.7 Gradient and Optimization Operations

| Operation | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `pytensor.grad` | `pytensor` | `train.py:243` | Compute gradients of loss |
| `pytensor.function` | `pytensor` | `train.py:292-298` | Compile training function |

#### 1.8 NumPy Operations (Data Preprocessing)

| Operation | Module | Location | Purpose |
|-----------|--------|----------|---------|
| `np.array` | `numpy` | `dataset.py:239,245-246` | Convert to numpy array |
| `np.zeros` | `numpy` | `dataset.py:227-228` | Create zero arrays |
| `np.random.randn` | `numpy.random` | `dataset.py:225` | Generate random normal data |
| `np.random.shuffle` | `numpy.random` | `dataset.py:280` | Shuffle dataset indices |
| `np.stack` | `numpy` | `dataset.py:295` | Stack batch images |
| `.transpose` | numpy method | `dataset.py:242` | Reorder dimensions (H,W,C) → (C,H,W) |
| `.astype` | numpy method | `dataset.py:225` | Type conversion to float32 |

### Level 2: Block Level - Building Blocks and Modules

#### 2.1 ConvBNSiLU Block
**Location**: `blocks.py:64-179`
**Components**:
- Conv2D operation (configurable kernel size, stride, padding)
- JAX-compatible batch normalization
- SiLU activation (x * sigmoid(x))

**Parameters**:
- Trainable: W (conv weights), gamma (BN scale), beta (BN shift)
- Non-trainable: bn_mean, bn_var

**Usage**: Fundamental atomic block used 22+ times throughout the model

#### 2.2 Bottleneck Block
**Location**: `blocks.py:182-231`
**Components**:
- Two sequential ConvBNSiLU blocks (3x3 kernels)
- Optional residual connection (skip)

**Pattern**:
```
input → conv1(3x3) → conv2(3x3) → (+identity if shortcut) → output
```

**Usage**: Core component of C3k2 blocks for feature extraction

#### 2.3 C3k2 Block (CSP Bottleneck with 2 Convolutions)
**Location**: `blocks.py:234-310`
**Components**:
- Split conv (1x1): in_channels → hidden_channels
- N bottleneck blocks in sequence (N=1 or 2)
- Concatenation along channel axis
- Merge conv (1x1): 2*hidden_channels → out_channels

**CSP Pattern**:
```
input → conv1(1x1) → split ─────────────────┐
                         └→ bottlenecks → concat → conv2(1x1) → output
```

**Usage**: 10 instances across backbone and neck

#### 2.4 SPPF Block (Spatial Pyramid Pooling - Fast)
**Location**: `blocks.py:313-388`
**Components**:
- Initial conv (1x1): in_channels → hidden_channels
- 3 cascaded max pooling (5x5, stride=1, padded)
- 4-way concatenation [x, pool1, pool2, pool3]
- Final conv (1x1): 4*hidden_channels → out_channels

**Pyramid Pattern**:
```
x → pool → y1 → pool → y2 → pool → y3
|        |           |           |
└────────┴───────────┴───────────┴→ concat → conv
```

**Usage**: 1 instance in backbone stage 4 (P5)

#### 2.5 C2PSA Block (CSP with Parallel Spatial Attention)
**Location**: `blocks.py:391-457`
**Components**:
- Split conv (1x1): in_channels → hidden_channels
- Attention branch: 3x3 conv (simplified attention)
- Concatenation of direct and attention paths
- Merge conv (1x1): 2*hidden_channels → out_channels

**Pattern**:
```
input → conv1(1x1) → split ────────────┐
                         └→ attn_conv → concat → conv2(1x1) → output
```

**Usage**: 1 instance in backbone stage 4 (P5)

### Level 3: Higher Level - Architectural Components

#### 3.1 YOLO11nBackbone
**Location**: `model.py:26-128`
**Purpose**: Multi-scale feature extraction

**Architecture Pipeline**:
```
Input (3, 320, 320)
    ↓
Stem: ConvBNSiLU(3→16, s=2)         → (16, 160, 160)
    ↓
Stage 1: ConvBNSiLU(16→32, s=2)      → (32, 80, 80)
         C3k2(32→32, n=1)
    ↓
Stage 2: ConvBNSiLU(32→64, s=2)      → (64, 40, 40) = P3
         C3k2(64→64, n=2)
    ↓
Stage 3: ConvBNSiLU(64→128, s=2)     → (128, 20, 20) = P4
         C3k2(128→128, n=2)
    ↓
Stage 4: ConvBNSiLU(128→256, s=2)    → (256, 10, 10) = P5
         C3k2(256→256, n=1)
         SPPF(256→256)
         C2PSA(256→256)
```

**Outputs**: (P3, P4, P5) - Features at 3 scales

#### 3.2 YOLO11nHead
**Location**: `model.py:131-270`
**Purpose**: Feature fusion and detection

**FPN (Feature Pyramid Network) - Top-Down Path**:
```
P5 (256, 10, 10) → upsample(2x) → concat with P4 → C3k2 → P4_out (128, 20, 20)
P4_out → upsample(2x) → concat with P3 → C3k2 → P3_out (64, 40, 40)
```

**PAN (Path Aggregation Network) - Bottom-Up Path**:
```
P3_out → ConvBNSiLU(s=2) → concat with P4_out → C3k2 → P4_final (128, 20, 20)
P4_final → ConvBNSiLU(s=2) → concat with P5 → C3k2 → P5_final (256, 10, 10)
```

**Detection Heads**:
- P3: ConvBNSiLU(64→4+C, 1x1) → (4+C, 40, 40) - Small objects
- P4: ConvBNSiLU(128→4+C, 1x1) → (4+C, 20, 20) - Medium objects
- P5: ConvBNSiLU(256→4+C, 1x1) → (4+C, 10, 10) - Large objects

#### 3.3 YOLO11n Model (Complete)
**Location**: `model.py:273-333`
**Components**:
- Backbone: YOLO11nBackbone instance
- Head: YOLO11nHead instance
- Parameters: Combined from both

**Forward Pass Flow**:
```
Input(batch, 3, 320, 320)
    → Backbone
    → (P3, P4, P5) features
    → Head
    → (det_P3, det_P4, det_P5) predictions
```

#### 3.4 Loss Functions
**Location**: `loss.py:62-247`

**Components**:
1. **Box IoU Computation** (`box_iou` function):
   - Center format to corner format conversion
   - Intersection area calculation
   - Union area calculation
   - IoU score computation

2. **YOLO Loss** (`yolo_loss` function):
   - Box regression loss (L2)
   - Classification loss (BCE)
   - Objectness loss (placeholder)
   - Weighted sum combination

**Loss Operations Flow**:
```
Predictions → Sigmoid normalization
           → Box loss (L2 regularization)
           → Class loss (Binary Cross-Entropy)
           → Weighted sum → Total loss
```

#### 3.5 Training System
**Location**: `train.py:150-350`

**Components**:
1. **Gradient Computation**:
   - `pytensor.grad` for each parameter
   - Fallback to zeros for disconnected params

2. **SGD with Momentum Optimizer**:
   - Velocity initialization
   - Momentum update: v = momentum * v - lr * grad
   - Weight decay: v = v - lr * weight_decay * param
   - Parameter update: param = param + v

3. **Compiled Training Function**:
   - Single `pytensor.function` with:
     - Input: images
     - Outputs: [total_loss, box_loss, cls_loss]
     - Updates: parameter and velocity updates
   - All operations (forward, backward, update) in one call

**Configuration**:
- Float32 enforcement for ONNX
- JAX mode with "shape_unsafe" exclusions
- Custom batch norm for JAX compatibility

### Operation Count Summary

#### Total Unique Operations by Category:
- **PyTensor Tensor Ops**: 17
  - Convolution/Pooling: 2
  - Activations: 1
  - Math: 6
  - Shape: 3
  - Creation/Utility: 5

- **Arithmetic Operators**: 5
  - +, -, *, /, **

- **NumPy Operations**: 8
  - Array manipulation: 5
  - Random: 2
  - Shape: 1

- **Gradient/Optimization**: 2
  - grad, function compilation

- **PIL Operations**: 3
  - Image loading, color conversion, resize

**Total**: 31+ distinct operations

#### Building Blocks: 5
1. ConvBNSiLU (atomic unit)
2. Bottleneck (residual block)
3. C3k2 (CSP bottleneck)
4. SPPF (spatial pyramid)
5. C2PSA (attention block)

#### Major Components: 5
1. YOLO11nBackbone (feature extraction)
2. YOLO11nHead (FPN/PAN fusion + detection)
3. YOLO11n (complete model)
4. Loss Functions (IoU + YOLO loss)
5. Training System (gradient + SGD + compilation)

## Code References

### Op Level Operations
- Convolution: `blocks.py:163-169`
- Pooling: `blocks.py:362-382`
- Activation: `blocks.py:177`, `loss.py:122,125`
- Math ops: `blocks.py:56`, `loss.py:44-49,57,141,148`
- Shape ops: `model.py:233,238,244,249,268`, `blocks.py:50-53,307,385,454`

### Block Level Components
- ConvBNSiLU: `blocks.py:64-179`
- Bottleneck: `blocks.py:182-231`
- C3k2: `blocks.py:234-310`
- SPPF: `blocks.py:313-388`
- C2PSA: `blocks.py:391-457`

### Higher Level Architecture
- Backbone: `model.py:26-128`
- Head: `model.py:131-270`
- Model: `model.py:273-333`
- Loss: `loss.py:15-247`
- Training: `train.py:150-350`

## Architecture Insights

### Design Patterns

1. **CSP (Cross Stage Partial)**:
   - Used in C3k2 and C2PSA blocks
   - Split-process-merge pattern for gradient flow
   - Reduces computation while maintaining feature richness

2. **Feature Pyramid (FPN/PAN)**:
   - Bidirectional feature fusion
   - Top-down semantic enrichment (FPN)
   - Bottom-up localization refinement (PAN)

3. **Multi-Scale Detection**:
   - 3 detection scales (40x40, 20x20, 10x10)
   - Specialized heads for different object sizes
   - Shared feature extraction backbone

4. **JAX Compatibility Strategy**:
   - Static shape operations throughout
   - Custom batch norm avoiding dynamic shapes
   - Optimizer exclusions for shape_unsafe rewrites
   - Explicit float32 typing

### Key Implementation Decisions

1. **SiLU over ReLU**: Smoother gradients, better performance
2. **SPPF over SPP**: Cascaded pooling more efficient than parallel
3. **1x1 Convolutions**: Efficient channel manipulation
4. **Residual Connections**: Gradient flow in deep networks
5. **He Initialization**: Appropriate for ReLU-like activations
6. **Float32 Enforcement**: ONNX export compatibility

## Historical Context (from thoughts/)

Previous research on JAX JIT issues (`thoughts/shared/research/2025-10-22_17-09-01_jax-jit-yolo-compilation-issues.md`) revealed:
- Dynamic shape arithmetic creates JAX tracers
- Shape_unsafe optimizer rewrites cause compilation failures
- Solution: Static operations + optimizer exclusions
- Performance impact: 5-10% reduction for JAX compatibility

## Related Research
- [JAX JIT Compilation Issues in YOLO Model](thoughts/shared/research/2025-10-22_17-09-01_jax-jit-yolo-compilation-issues.md)

## Open Questions

1. **Potential Optimizations**:
   - Could SPPF use different pool sizes for richer features?
   - Would grouped convolutions reduce parameters?
   - Can C2PSA implement full spatial attention?

2. **JAX Improvements**:
   - Can tile() operation be made more JAX-friendly?
   - Should Alloc add shape validation like Reshape?
   - Could shape_unsafe rewrites be selectively enabled?

3. **Architecture Extensions**:
   - Support for dynamic input sizes?
   - Additional detection scales (P2, P6)?
   - Alternative neck architectures (BiFPN)?

This comprehensive analysis provides a complete understanding of all operations used in the YOLO11n implementation, from low-level tensor operations through building blocks to high-level architectural components, all designed for efficient training and ONNX deployment.