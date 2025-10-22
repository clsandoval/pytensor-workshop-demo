---
date: 2025-10-22T10:39:50-05:00
researcher: Claude
git_commit: a0f629a39240f04cf68bafd89aed73686efc184e
branch: onnx-workshop-demo
repository: pytensor
topic: "Training Issues in ONNX YOLO GPU Demo"
tags: [research, codebase, yolo, gpu, training, jax, onnx, pytorch]
status: complete
last_updated: 2025-10-22
last_updated_by: Claude
---

# Research: Training Issues in ONNX YOLO GPU Demo

**Date**: 2025-10-22T10:39:50-05:00
**Researcher**: Claude
**Git Commit**: a0f629a39240f04cf68bafd89aed73686efc184e
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question
Find possible training issues in the ONNX YOLO GPU demo implementation at `examples/onnx/onnx-yolo-demo-gpu/`

## Summary

The YOLO11n GPU demo implementation has **fundamental training issues** that prevent it from functioning as a real object detector. While the PyTensor→ONNX export pipeline works correctly for inference, the training implementation is a **demonstration-only placeholder** that cannot learn to detect objects. Critical issues include:

1. **Ground truth annotations are never used** - loaded but ignored in loss computation
2. **Loss function is pure regularization** - no actual detection loss
3. **Upsampling creates JAX JIT tracers** - blocks GPU training (documented fix exists)
4. **No data augmentation** despite claims
5. **Synchronous single-threaded data loading** - severe GPU underutilization

## Detailed Findings

### 1. Critical Training Blockers

#### A. Ground Truth Completely Ignored
**Location**: `examples/onnx/onnx-yolo-demo-gpu/train.py:184-186`
```python
self.loss, self.loss_dict = yolo_loss(
    self.predictions, targets=None, num_classes=args.num_classes
)
```
- The `targets=None` parameter means ground truth is never passed to loss
- COCO annotations are loaded at `train.py:414-424` but discarded
- Training loop at `train.py:311-314` only passes images, not targets
- **Impact**: Model cannot learn what objects look like or where they are

#### B. Placeholder Loss Function
**Location**: `examples/onnx/onnx-yolo-demo-gpu/yolo/loss.py:139-155`
```python
# Box loss - just L2 regularization
box_loss = pt.mean(pred_boxes_norm**2) * 0.5

# Classification loss - BCE against zeros (background)
cls_loss = -pt.log(1 - pred_classes_sig + eps).mean()

# Objectness loss - placeholder
obj_loss = pt.constant(0.0)
```
- Box loss pushes all predictions toward zero
- Classification loss trains everything as background
- Objectness (confidence) is completely disabled
- Authors explicitly document this at `loss.py:250-265` as intentional simplification
- **Impact**: Loss will decrease but model won't detect objects

#### C. JAX JIT Dynamic Shape Issue
**Location**: `examples/onnx/onnx-yolo-demo-gpu/yolo/model.py:288-292`
```python
# Problem: arithmetic on symbolic shapes creates tracers
out_height = height * scale  # Creates JitTracer
out_width = width * scale    # Creates JitTracer
x_upsampled = x_rearranged.reshape((batch_size, channels, out_height, out_width))
```
- JAX requires concrete integer shapes for `reshape()`
- Shape arithmetic creates abstract tracer objects
- **Solution documented**: `thoughts/shared/research/2025-01-15_jax-jit-issues-yolo-gpu-training.md`
- **Fix**: Set `optimizer_excluding='shape_unsafe'` in PyTensor config
- **Impact**: Without fix, GPU training fails after first batch

### 2. Performance Bottlenecks

#### A. No Data Augmentation
**Location**: `examples/onnx/onnx-yolo-demo-gpu/yolo/dataset.py` (entire file)
- Comment claims "Basic data augmentation" but none implemented
- No horizontal flips, color jitter, random crops, mosaic, mixup
- Fixed aspect ratio resize distorts images at line 236
- **Impact**: Poor generalization, easy overfitting

#### B. Synchronous Single-Threaded Loading
**Location**: `examples/onnx/onnx-yolo-demo-gpu/yolo/dataset.py:282-301`
```python
def create_dataloader(dataset, batch_size=8, shuffle=True):
    # Simple Python generator, no multiprocessing
    for idx in batch_indices:
        sample = dataset[idx]  # Blocking I/O
        batch_images.append(sample["image"])
```
- No prefetching or parallel loading
- Each image loaded from disk sequentially
- Estimated loading time: 150-250ms per batch
- GPU training step: ~50-100ms for YOLO11n
- **Impact**: GPU utilization only 20-40%, 3-5x slower training

#### C. Memory Inefficiencies
**Locations**: Multiple in `dataset.py`
- Line 153: Entire COCO JSON loaded into memory (~500-800MB)
- Line 233: No image caching, reload from disk every epoch
- Lines 239, 242: Multiple array copies without buffer reuse
- **Impact**: High memory usage, slower data pipeline

### 3. Architectural Issues

#### A. Model/Loss Type Mismatch (FIXED)
**Previous Issue**: Model returned dict, loss expected tuple
**Current Status**: Fixed - model returns tuple at `model.py:356-358`
```python
return det_p3, det_p4, det_p5  # Correct tuple format
```
- Loss unpacks tuple at `loss.py:106`
- Tests validate at `test_model_integration.py:40-42`

#### B. Single-Scale Loss Only
**Location**: `examples/onnx/onnx-yolo-demo-gpu/yolo/loss.py:106-110`
```python
_pred_p3, pred_p4, _pred_p5 = predictions
# Only uses P4 scale (20x20), ignores P3 (40x40) and P5 (10x10)
```
- Wastes 2/3 of model outputs
- Loses multi-scale detection capability
- **Impact**: Can't detect small (P3) or large (P5) objects effectively

#### C. Batch Norm in Inference Mode
**Location**: `examples/onnx/onnx-yolo-demo-gpu/yolo/blocks.py:171-174`
- Always uses running statistics, never updates during training
- JAX-compatible implementation prevents dynamic shapes
- **Impact**: Statistics remain at initialization values

### 4. Configuration and Setup

#### A. Working GPU Configuration
**Location**: `examples/onnx/onnx-yolo-demo-gpu/train.py:47-76`
```python
# Required environment settings
PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"

# JAX backend auto-detection
if device_type == "gpu":
    pytensor.config.optimizer_excluding = "shape_unsafe"
    pytensor.config.mode = "JAX"
```

#### B. Float32 Enforcement
**Location**: `examples/onnx/onnx-yolo-demo-gpu/train.py:38-45`
- **Critical**: Must use float32 for ONNX export compatibility
- RuntimeError raised if not set before import
- Mixed precision creates invalid ONNX models

### 5. Test Coverage Analysis

#### What's Tested (65+ tests)
- ✅ All building blocks forward pass (`test_blocks.py`)
- ✅ Model architecture correctness (`test_model_*.py`)
- ✅ Loss mathematical properties (`test_loss.py`)
- ✅ JAX compilation and gradients (`test_jax_backend.py`)
- ✅ ONNX export and inference (`test_onnx_export.py`)
- ✅ Browser WebGPU/WASM runtime (`test_browser_*.py`)

#### What's NOT Tested
- ❌ Actual training convergence
- ❌ Multi-epoch training loops
- ❌ Loss decreasing over time
- ❌ Detection accuracy metrics
- ❌ GPU training performance
- ❌ Data augmentation effects

#### Potential Test Issue
**Location**: `examples/onnx/onnx-yolo-demo-gpu/tests/test_jax_backend.py:112`
```python
if len(params) == 0:
    pytest.skip("No parameters found in model")
```
- Gradient test might skip if parameter extraction fails
- Could indicate deeper parameter registration issue

## Code References

### Core Implementation Files
- `examples/onnx/onnx-yolo-demo-gpu/train.py:147` - Trainer class
- `examples/onnx/onnx-yolo-demo-gpu/train.py:184-186` - Loss initialization (targets=None bug)
- `examples/onnx/onnx-yolo-demo-gpu/train.py:311-314` - Training step (ignores targets)
- `examples/onnx/onnx-yolo-demo-gpu/yolo/model.py:298` - YOLO11n class
- `examples/onnx/onnx-yolo-demo-gpu/yolo/model.py:288-292` - Upsampling (JAX tracer issue)
- `examples/onnx/onnx-yolo-demo-gpu/yolo/blocks.py:64` - ConvBNSiLU implementation
- `examples/onnx/onnx-yolo-demo-gpu/yolo/blocks.py:171-174` - Batch norm (inference mode only)
- `examples/onnx/onnx-yolo-demo-gpu/yolo/loss.py:62` - yolo_loss function
- `examples/onnx/onnx-yolo-demo-gpu/yolo/loss.py:139-155` - Placeholder loss implementation
- `examples/onnx/onnx-yolo-demo-gpu/yolo/dataset.py:35` - COCODataset class
- `examples/onnx/onnx-yolo-demo-gpu/yolo/dataset.py:282-301` - Synchronous dataloader

### Test Files
- `examples/onnx/onnx-yolo-demo-gpu/tests/conftest.py:8` - Float32 enforcement
- `examples/onnx/onnx-yolo-demo-gpu/tests/test_jax_backend.py:86-128` - Gradient flow test
- `examples/onnx/onnx-yolo-demo-gpu/tests/test_model_integration.py:40-42` - Tuple return validation
- `examples/onnx/onnx-yolo-demo-gpu/tests/test_loss.py:213-274` - Loss with targets test

## Architecture Insights

### Design Philosophy
This implementation is a **technical demonstration** of PyTensor's capabilities:
- ✅ Proves PyTensor can build CNN architectures
- ✅ Shows JAX backend GPU acceleration works
- ✅ Validates ONNX export pipeline
- ✅ Demonstrates browser inference via WebGPU/WASM
- ❌ NOT a functional object detection trainer

### Key Patterns

1. **Hierarchical Parameter Collection**
   - Each module collects params from sub-modules
   - Separate lists for trainable params vs batch norm stats
   - Enables checkpointing and optimizer setup

2. **JAX-Compatible Operations**
   - Avoid dynamic shapes in graph optimizations
   - Use explicit broadcasting via `dimshuffle`
   - Exclude `shape_unsafe` optimizer passes

3. **Float32 Everywhere**
   - Required for ONNX export
   - Explicit casting throughout codebase
   - Environment variable enforcement

4. **Fallback Strategies**
   - Synthetic data when COCO unavailable
   - Graceful test skipping without dependencies
   - Multiple runtime backends (JAX/ONNX/WebGPU/WASM)

## Historical Context (from thoughts/)

### JAX JIT Issues Research
**Document**: `thoughts/shared/research/2025-01-15_jax-jit-issues-yolo-gpu-training.md`

Key findings:
- Two critical issues identified and fixed:
  1. Model returning dict instead of tuple (2-minute fix)
  2. JAX tracer errors from shape arithmetic (2-minute fix)
- Solutions tested and verified working
- Performance overhead ~5% acceptable for compatibility
- Compilation time: 45 seconds first run, cached afterward
- GPU utilization: 85-90% with fixes applied

### Implementation Plans
**Documents**:
- `thoughts/shared/plans/2025-01-15_tdd-plan-fix-jax-jit-yolo-training.md`
- `thoughts/shared/plans/yolo_jax_gpu_training_test_suite_tdd.md`

Key insights:
- TDD approach enabled 5-minute fix implementation
- Triple verification pattern for JAX mode (set, verify, type check)
- Test suite design for GPU-specific functionality
- Patterns for detecting JIT compilation via `isinstance(result, jax.Array)`

## Related Research

- `thoughts/shared/research/2025-01-15_jax-jit-issues-yolo-gpu-training.md` - Complete JAX diagnostic
- `thoughts/shared/plans/2025-01-15_tdd-plan-fix-jax-jit-yolo-training.md` - Fix implementation plan
- `thoughts/shared/plans/jax-cnn-ops-implementation.md` - JAX backend CNN ops specs

## Open Questions

1. **Why was this designed as demo-only?**
   - Was full training ever intended?
   - Is there a separate full implementation elsewhere?

2. **Parameter extraction in tests**
   - Why might `test_jax_backend.py:112` fail to find parameters?
   - Are parameters properly registered for gradients?

3. **Performance targets**
   - What's the expected GPU utilization with proper data loading?
   - How much would multiprocess loading improve throughput?

4. **Future development**
   - Are there plans to implement real training?
   - Will data augmentation be added?
   - Is async data loading planned?

## Recommendations for Fixing Training

### Priority 1: Make Training Functional
1. **Pass targets to loss function**
   ```python
   # train.py:184-186
   self.loss, self.loss_dict = yolo_loss(
       self.predictions,
       targets=(target_boxes, target_classes, target_num_boxes),  # Add this
       num_classes=args.num_classes
   )
   ```

2. **Implement proper target assignment in loss**
   - Match ground truth boxes to grid cells
   - Compute IoU-based box loss
   - Use actual class labels for classification

3. **Apply JAX fix for GPU training**
   ```bash
   export PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run,optimizer_excluding=shape_unsafe"
   ```

### Priority 2: Improve Performance
1. **Add multiprocess data loading**
   - Use `torch.utils.data.DataLoader` or similar
   - Implement prefetching with 2-4 workers
   - Target: 80%+ GPU utilization

2. **Implement data augmentation**
   - Horizontal flips
   - Color jitter
   - Random crops/scales
   - Mosaic augmentation for YOLO

3. **Fix batch normalization**
   - Update running statistics during training
   - Separate train/eval modes

### Priority 3: Add Training Tests
1. **Convergence test**
   - Train on tiny dataset (10 images)
   - Verify loss decreases
   - Check model can overfit

2. **GPU training test**
   - Verify JAX arrays used throughout
   - Measure GPU utilization
   - Compare speed vs CPU

3. **End-to-end training test**
   - Train for 1 epoch
   - Save checkpoint
   - Load and verify inference

## Conclusion

The ONNX YOLO GPU demo successfully demonstrates PyTensor's capability to:
- Build modern CNN architectures
- Compile with JAX for GPU acceleration
- Export to ONNX for browser deployment
- Run inference via WebGPU/WASM

However, it **cannot perform real object detection training** due to fundamental issues:
- Ground truth is never used
- Loss function is placeholder only
- No data augmentation
- Severe data loading bottlenecks

These appear to be **intentional simplifications** for a technical demo, not bugs. The implementation achieves its stated goal of demonstrating the PyTensor→ONNX pipeline, but users expecting functional YOLO training will be disappointed. The fixes are well-documented and straightforward to implement if real training is desired.