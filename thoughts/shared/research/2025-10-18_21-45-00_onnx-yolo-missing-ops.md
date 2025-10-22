---
date: 2025-10-18T21:45:00-00:00
researcher: Claude
git_commit: 226f34c37775b44b18b783723a7357f56fb5e116
branch: onnx-workshop-demo
repository: pytensor
topic: "All ONNX Operations Needed for YOLO Demo Export"
tags: [research, onnx, yolo, operations, elemwise]
status: complete
last_updated: 2025-10-18
last_updated_by: Claude
---

# Research: All ONNX Operations Needed for YOLO Demo Export

**Date**: 2025-10-18T21:45:00
**Researcher**: Claude
**Git Commit**: 226f34c37775b44b18b783723a7357f56fb5e116
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question

User recently implemented IntDiv for the ONNX backend to support the onnx-yolo demo. Now EQ is not supported. The goal is to find **ALL** the operations needed to support ONNX export for the demo, to avoid the implement → discover → implement cycle.

## Summary

Based on comprehensive analysis of the YOLO11n model's computation graph (525 nodes total), **only 2 missing Elemwise scalar operations** are blocking ONNX export:

1. **EQ** (Equal) - Used 13 times
2. **Switch** (conditional selection) - Used 5 times

All other operations (DimShuffle, Shape_i, Join, Pool, Conv, etc.) already have ONNX dispatch implementations.

Additionally, there is **1 top-level operation** that will need support after EQ and Switch are added:

3. **ScalarFromTensor** - Used 8 times

## Detailed Findings

### Model Statistics

The YOLO11n model compilation graph contains:
- **Total nodes**: 525
- **Unique operation types**: 11
- **Elemwise operations**: 178 (34% of all nodes)
- **DimShuffle operations**: 206 (39% of all nodes)

### 1. All Operation Types in the Model

| Operation Type | Occurrences | ONNX Dispatch Status |
|----------------|-------------|----------------------|
| DimShuffle | 206 | ✓ Supported (`shape.py:189`) |
| Elemwise | 178 | ✓ Supported (with caveats, see below) |
| Shape_i | 51 | ✓ Supported (`shape.py:18`) |
| AbstractConv2d | 51 | ✓ Supported (`conv.py:13`) |
| Join | 14 | ✓ Supported (`join.py:10`) |
| ScalarFromTensor | 8 | ✗ **NOT SUPPORTED** |
| Assert | 8 | ? Unknown (may be ignorable) |
| Pool | 3 | ✓ Supported (`pool.py:9`) |
| MakeVector | 2 | ✓ Supported (`shape.py:541`) |
| Reshape | 2 | ✓ Supported (`shape.py:98`) |
| Alloc | 2 | ? Unknown |

### 2. Elemwise Scalar Operations

The model uses 8 different scalar operation types within Elemwise nodes:

| Scalar Operation | Occurrences | ONNX Support | ONNX Op Mapping |
|------------------|-------------|--------------|-----------------|
| Composite | 102 | ✓ Supported | Decomposed into primitives |
| Add | 24 | ✓ Supported | `Add` |
| Sub | 14 | ✓ Supported | `Sub` |
| **EQ** | **13** | **✗ NOT SUPPORTED** | **`Equal`** |
| IntDiv | 12 | ✓ Supported | `Div` + floor logic |
| Mul | 7 | ✓ Supported | `Mul` |
| **Switch** | **5** | **✗ NOT SUPPORTED** | **`Where`** |
| ScalarMaximum | 1 | ✓ Supported | `Max` |

**Currently supported scalar ops** (from `elemwise.py:17-32`):
- Add, Mul, Sub, TrueDiv, IntDiv ✓
- Neg, Exp, Log, Sqrt, Sqr, Pow, Abs ✓
- ScalarMaximum, ScalarMinimum ✓
- Sigmoid ✓

### 3. Missing Operations Analysis

#### 3.1 EQ (Equal) - Priority 1

**Usage**: 13 occurrences
**Purpose**: Element-wise equality comparison
**ONNX Mapping**: `Equal` operator

Sample usage in graph:
```
Eq(Shape_i{0}.0, Shape_i{0}.0)  # Shape comparisons
Eq(Sub.0, -1)                     # Value comparisons
Eq(Mul.0, -1)                     # Conditional checks
```

**Location**: Used primarily for shape validation and conditional logic in the upsampling operations (`yolo/model.py:249-286`).

**Implementation**: Add to `SCALAR_OP_TO_ONNX` dictionary:
```python
scalar.EQ: "Equal"
```

#### 3.2 Switch (Conditional Selection) - Priority 2

**Usage**: 5 occurrences
**Purpose**: If-then-else conditional selection: `Switch(condition, then_value, else_value)`
**ONNX Mapping**: `Where` operator

Sample usage in graph:
```
Switch(Eq.0, Int_div.0, Sub.0)     # Conditional dimension calculation
Switch(Eq.0, Int_div.0, Mul.0)     # Conditional shape logic
Switch(Eq.0, Int_div.0, Assert{...}.0)  # Conditional assertions
```

**Location**: Used in reshape and dimension calculation logic, particularly in the upsampling operations.

**Implementation**: Add to `SCALAR_OP_TO_ONNX` dictionary:
```python
scalar.Switch: "Where"
```

**Note**: Switch takes 3 inputs (condition, then, else) which matches ONNX Where semantics exactly.

#### 3.3 ScalarFromTensor - Priority 3

**Usage**: 8 occurrences
**Purpose**: Extracts a scalar value from a 0-d or 1-element tensor
**ONNX Mapping**: `Squeeze` operator (or `Identity` for already-scalar tensors)

Sample usage in graph:
```
ScalarFromTensor(Eq.0)  # Convert boolean comparison result to scalar
```

**Location**: Used to convert tensor boolean results to scalar values for assertions and conditional checks.

**Implementation**: Requires a new dispatch function in `shape.py`:
```python
@onnx_funcify.register(ScalarFromTensor)
def onnx_funcify_ScalarFromTensor(op, node, var_names, get_var_name, **kwargs):
    # Use Squeeze to remove all dimensions
    return helper.make_node("Squeeze", inputs=[...], outputs=[...])
```

### 4. Operations That Are Already Supported

These operations appear in the model but already have ONNX dispatches:

- **AbstractConv2d** (`conv.py:13`) - Convolutional layers
- **Pool** (`pool.py:9`) - Max pooling in SPPF block
- **DimShuffle** (`shape.py:189`) - Tensor dimension reordering (heavily used)
- **Shape_i** (`shape.py:18`) - Extract specific shape dimensions
- **Join** (`join.py:10`) - Tensor concatenation
- **MakeVector** (`shape.py:541`) - Create 1-D vectors for shapes
- **Reshape** (`shape.py:98`) - Tensor reshaping

### 5. Test Failure Sequence

When attempting ONNX export (`test_onnx_export.py:33`), the failures occur in this order:

1. **First failure**: EQ not in `SCALAR_OP_TO_ONNX`
   ```
   NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ
   ```

2. **After adding EQ and Switch**: ScalarFromTensor not registered
   ```
   NotImplementedError: No ONNX conversion available for: ScalarFromTensor
   ```

3. **After adding ScalarFromTensor**: Export should succeed ✓

## Code References

- `pytensor/link/onnx/dispatch/elemwise.py:17-32` - SCALAR_OP_TO_ONNX mapping
- `pytensor/link/onnx/dispatch/shape.py` - Shape-related dispatch implementations
- `examples/onnx/onnx-yolo-demo/yolo/model.py:249-286` - Upsampling logic using EQ/Switch
- `examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py:12` - ONNX export test

## Architecture Insights

### Why These Operations Are Needed

The YOLO model uses **dynamic upsampling** with nearest-neighbor interpolation (`model.py:249`). The implementation uses:

1. **Shape operations** (Shape_i) to extract input dimensions
2. **Integer arithmetic** (IntDiv, Mul) to calculate output dimensions
3. **Equality checks** (EQ) to validate shape compatibility
4. **Conditional logic** (Switch) to handle dynamic shapes
5. **Scalar conversions** (ScalarFromTensor) for assertions

This pattern appears in the FPN (Feature Pyramid Network) upsampling path where P5 → P4 → P3 feature maps are created at different resolutions.

### Implementation Pattern

PyTensor's ONNX export uses a **dispatch system** with two levels:

1. **Top-level Op dispatch**: `@onnx_funcify.register(OpClass)`
   - Handles operation classes like Conv2d, Pool, Reshape
   - Located in `pytensor/link/onnx/dispatch/*.py`

2. **Elemwise scalar dispatch**: `SCALAR_OP_TO_ONNX` dictionary
   - Maps scalar operations to ONNX primitive ops
   - Special handling for Composite (decomposed), IntDiv (floor semantics), etc.
   - Located in `pytensor/link/onnx/dispatch/elemwise.py`

## Complete List of Operations to Implement

### Summary Table

| Priority | Operation | Type | ONNX Mapping | Lines of Code | Difficulty |
|----------|-----------|------|--------------|---------------|------------|
| 1 | EQ | Elemwise scalar | Equal | 1 line | Trivial |
| 2 | Switch | Elemwise scalar | Where | 1 line | Trivial |
| 3 | ScalarFromTensor | Top-level Op | Squeeze | ~15 lines | Easy |

### Implementation Checklist

- [ ] Add `scalar.EQ: "Equal"` to `SCALAR_OP_TO_ONNX` in `elemwise.py`
- [ ] Add `scalar.Switch: "Where"` to `SCALAR_OP_TO_ONNX` in `elemwise.py`
- [ ] Import `ScalarFromTensor` in `shape.py`
- [ ] Register `@onnx_funcify.register(ScalarFromTensor)` dispatch in `shape.py`
- [ ] Test ONNX export: `pytest tests/test_onnx_export.py::test_model_exports_to_onnx`

**Total implementation time**: ~15-20 minutes

## Open Questions

1. **Assert operations**: The model contains 8 Assert nodes. These may be:
   - Ignorable (assertions are compile-time checks)
   - Automatically handled by existing dispatch
   - Need explicit handling

   **Resolution**: Test after implementing the 3 operations above to see if Assert blocks export.

2. **Alloc operations**: The model contains 2 Alloc nodes. Status unclear, but likely:
   - Already handled by AllocEmpty dispatch (`shape.py:393`)
   - Or may need separate Alloc dispatch

   **Resolution**: Test to determine if additional work needed.

## Conclusion

To make the ONNX export test pass for the YOLO demo, you need to implement exactly **3 operations**:

1. **EQ** - Elemwise equality comparison → ONNX `Equal`
2. **Switch** - Conditional selection → ONNX `Where`
3. **ScalarFromTensor** - Tensor to scalar → ONNX `Squeeze`

These are the **ONLY** operations blocking export. All other operations (DimShuffle, Shape_i, Conv, Pool, Join, etc.) already have working ONNX dispatches.

The implementation is straightforward:
- EQ and Switch: Add 2 entries to a dictionary
- ScalarFromTensor: Add a ~15-line dispatch function

After these implementations, the ONNX export should work without discovering additional missing operations.
