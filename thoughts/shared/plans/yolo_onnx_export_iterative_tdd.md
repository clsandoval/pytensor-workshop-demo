# YOLO ONNX Export - Iterative TDD Plan

## Overview

This plan systematically passes all 8 ONNX export tests in `examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py` by discovering and fixing missing operations iteratively. Each iteration follows: **Run Test → Fix Blocker → Verify → Clear Context → Next Iteration**.

The key challenge: Operations are only discovered when they fail during export, so we need an iterative approach rather than knowing all missing ops upfront.

## Current State

**Test File**: `examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py`
- 8 test functions (lines 12-269)
- All tests marked with `@pytest.mark.onnx`
- Tests require: `onnx`, `onnxruntime`, `pytensor.link.onnx`

**Already Implemented Operations**:
- Elemwise: Add, Mul, Sub, Div, IntDiv, Neg, Abs, Pow, Exp, Log, Sqrt, Sigmoid, SiLU, EQ, Switch, Cast
- Shape: Shape_i, Reshape, DimShuffle, AllocEmpty, MakeVector, ScalarFromTensor
- Conv: AbstractConv2d
- Pool: Pool (MaxPool)
- Linear Algebra: Dot, Dot22, Gemv
- Join: Join (Concat)
- BatchNorm: BatchNormalization
- Resize: Resize
- Special: Softmax

**Known Missing Operations** (from research):
- Assert / CheckAndRaise (COp operations, not Elemwise)

## Desired End State

All 8 ONNX export tests pass:
- ✓ `test_model_exports_to_onnx` - Basic export succeeds
- ✓ `test_onnx_runtime_vs_pytensor_equivalence` - Numerical correctness
- ✓ `test_onnx_model_metadata` - Correct inputs/outputs
- ✓ `test_onnx_export_different_batch_sizes` - Dynamic batching
- ✓ `test_onnx_output_shapes` - Correct output shapes
- ✓ `test_onnx_deterministic_inference` - Deterministic execution
- ✓ `test_onnx_model_opset_version` - Opset >= 11
- ✓ `test_onnx_no_training_ops` - No training-only ops

## What We're NOT Implementing

- Operations for other models (only what YOLO needs)
- Training-mode export (inference only)
- Optimizations (focus on correctness)

## TDD Approach

**Iterative Discovery Pattern**:
1. Run the test suite
2. Identify the first failing operation
3. Implement ONNX dispatcher for that operation
4. Verify with tests
5. Clear context to manage token usage
6. Repeat until all tests pass

**Why Iterative**: PyTensor's graph rewriting may insert operations not visible in source code (e.g., Assert from `local_merge_alloc` rewriter). We discover these only when export fails.

---

## Iteration Template

Each iteration follows this structure:

### Iteration N: [Operation Name]

#### 1. Discovery (Run Tests)

```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py::test_model_exports_to_onnx -v
```

**Expected Failure Pattern**:
```
NotImplementedError: [OpType] operation not supported for ONNX export
```

#### 2. Analysis

**Questions to Answer**:
- [ ] What is the operation type? (Elemwise, COp, etc.)
- [ ] Where is it defined in PyTensor? (`pytensor/*.py`)
- [ ] What does it do? (Mathematical operation, control flow, etc.)
- [ ] What's the ONNX equivalent? (Check ONNX operator docs)
- [ ] Can it be decomposed into multiple ONNX nodes?

**Research Files**:
- Operation definition: Find with `Grep` in `pytensor/`
- ONNX mapping: Check ONNX docs at https://onnx.ai/onnx/operators/
- Similar implementations: Review `pytensor/link/onnx/dispatch/*.py`

#### 3. Implementation

**Dispatcher Location**: Determine correct dispatch file
- Elemwise scalar op → `pytensor/link/onnx/dispatch/elemwise.py`
- Shape manipulation → `pytensor/link/onnx/dispatch/shape.py`
- COp (like Assert) → `pytensor/link/onnx/dispatch/basic.py`
- New category → Create new dispatch file

**Implementation Pattern**:

**Pattern A: Direct 1:1 Mapping**
```python
@onnx_funcify.register(OpType)
def onnx_funcify_operation(op, node, var_names, get_var_name, **kwargs):
    """[Operation description]"""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    return helper.make_node(
        "OnnxOperator",
        inputs=input_names,
        outputs=output_names,
        name=f"OpName_{output_names[0]}",
    )
```

**Pattern B: Multi-Node Decomposition**
```python
@onnx_funcify.register(OpType)
def onnx_funcify_operation(op, node, var_names, get_var_name, **kwargs):
    """[Operation description - decomposes to multiple ONNX nodes]"""
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    nodes = []

    # Node 1: First transformation
    intermediate_1 = f"stage1_{output_names[0]}"
    nodes.append(helper.make_node(
        "OnnxOp1",
        inputs=input_names,
        outputs=[intermediate_1],
        name=f"Stage1_{output_names[0]}",
    ))

    # Node 2: Second transformation
    nodes.append(helper.make_node(
        "OnnxOp2",
        inputs=[intermediate_1],
        outputs=output_names,
        name=f"Stage2_{output_names[0]}",
    ))

    return nodes
```

**Pattern C: Elemwise Scalar Op**
```python
# In elemwise.py:36-67 SCALAR_OP_TO_ONNX mapping
SCALAR_OP_TO_ONNX = {
    # ... existing ops ...
    scalar.NewOp: "OnnxOperator",  # Add new mapping
}
```

#### 4. Testing

**Unit Test** (if new operation type):

Create test in appropriate file:
- `tests/link/onnx/test_elemwise.py` for elemwise ops
- `tests/link/onnx/test_shape.py` for shape ops
- `tests/link/onnx/test_basic.py` for other ops

```python
def test_new_operation_basic(tmp_path):
    """
    Test: [Operation] exports to ONNX correctly.

    Verifies:
    - ONNX export completes without error
    - Output matches PyTensor computation
    - Handles edge cases
    """
    x = pt.vector("x", dtype="float32")
    y = pt.new_operation(x)

    x_val = np.array([test, values], dtype="float32")

    from tests.link.onnx.utils import compare_onnx_and_py
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Integration Test** (YOLO export):

```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py::test_model_exports_to_onnx -v
```

#### 5. Verification Checklist

- [ ] Unit test passes (if created)
- [ ] YOLO export test progresses past this operation
- [ ] No linting errors: `cd pytensor && make lint` (or skip if too slow)
- [ ] Implementation follows existing patterns
- [ ] Code has clear comments explaining the mapping

#### 6. Documentation

Update this plan with what was implemented:

**Operation**: [Name]
**Type**: [Elemwise/COp/etc.]
**ONNX Mapping**: [PyTensor Op] → [ONNX Op(s)]
**Implementation**: `[file.py:line]`
**Test**: `[test_file.py:line]`
**Notes**: [Any special considerations]

#### 7. Clear Context (Optional)

If context is getting large (>150k tokens), clear and restart:
- Save progress in this plan (check completed boxes)
- Exit and re-enter Claude Code
- Resume at next iteration

---

## Iteration 1: Assert / CheckAndRaise

**Status**: ✅ Completed

### 1. Discovery

```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py::test_model_exports_to_onnx -xvs
```

**Expected Error**:
```
NotImplementedError: [Operation type] not supported for ONNX export
```

### 2. Analysis

**From Research Document** (`thoughts/shared/research/2025-10-19_assert-operations-analysis.md`):

- [x] **Operation Type**: COp (C-implemented Op), not Elemwise
- [x] **Definition**: `pytensor/raise_op.py:145-182` (Assert class)
- [x] **Purpose**: Runtime assertion - raises AssertionError if condition fails
- [x] **ONNX Equivalent**: **None** - ONNX has no exception handling
- [x] **Solution**: Map to Identity (pass-through) - assertions are validation only

**Why Assert Appears**: Automatically inserted by PyTensor's graph rewriting:
- `pytensor/tensor/rewriting/basic.py:1226-1231` - `local_merge_alloc` rewriter
- Used to validate dimension compatibility during broadcasting
- Not written explicitly in YOLO model code

**Why find_missing_ops.py Didn't Catch It**:
- Script only checks Elemwise operations (`find_missing_ops.py:66`)
- Assert is a COp, separate from Elemwise hierarchy

### 3. Implementation

**File**: `pytensor/link/onnx/dispatch/basic.py`

**Add After Line 70** (after onnx_funcify and onnx_typify definitions):

```python
from pytensor.raise_op import Assert, CheckAndRaise

@onnx_funcify.register(Assert)
@onnx_funcify.register(CheckAndRaise)
def onnx_funcify_assert(op, node, var_names, get_var_name, **kwargs):
    """
    Assert and CheckAndRaise operations are skipped during ONNX export.

    These operations perform runtime validation and can raise exceptions,
    but ONNX has no exception handling mechanism. For ONNX export, we
    assume shapes are validated at compile time, so assertions are
    converted to pass-through (Identity) operations.

    Node structure:
    - inputs[0]: The value to return if assertion passes
    - inputs[1:]: Boolean conditions that must all be True

    We only pass through the value, ignoring the conditions.
    """
    from onnx import helper

    # First input is the value, rest are conditions
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    return helper.make_node(
        "Identity",
        inputs=[input_name],
        outputs=[output_name],
        name=f"assert_passthrough_{output_name}",
    )
```

**Rationale**:
- Assert's first input is the value it returns if conditions pass
- Remaining inputs are boolean conditions (must all be True)
- ONNX has no way to raise exceptions
- For inference, assertions are compile-time checks
- Identity node passes value through unchanged

### 4. Testing

**No Dedicated Unit Test Needed**: Assert is a meta-operation for validation, not computation

**Integration Test**: YOLO export test will verify Assert doesn't block export

```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py::test_model_exports_to_onnx -xvs
```

**Expected Result**: Test progresses past Assert operations (should discover next missing op, if any)

### 5. Verification Checklist

- [x] YOLO export test runs without NotImplementedError for Assert
- [x] Test progresses to next missing operation (Alloc)
- [x] Implementation added to `pytensor/link/onnx/dispatch/basic.py`
- [x] Import statement added for Assert and CheckAndRaise

### 6. Documentation

**Operation**: Assert / CheckAndRaise
**Type**: COp (C-implemented operation)
**ONNX Mapping**: Assert(value, conditions...) → Identity(value)
**Implementation**: `pytensor/link/onnx/dispatch/basic.py` (to be added after line 70)
**Test**: `examples/onnx/onnx-yolo-demo/tests/test_onnx_export.py` (integration test)
**Notes**:
- Assertions are skipped in ONNX export (no exception mechanism)
- Value passes through unchanged via Identity node
- Automatically inserted by `local_merge_alloc` rewriter, not in user code

### 7. Actual Outcome

✅ **Iteration 1 Complete!**

Test progressed past Assert operation. Discovered next missing operation: **Alloc**

Proceeding to Iteration 2...

---

## Iteration 2: Alloc

**Status**: 🔄 In Progress

Discovered after Iteration 1. The Alloc operation creates a tensor by broadcasting a value to a specified shape.

### 1. Discovery

```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py::test_model_exports_to_onnx -xvs
```

**Actual Error**:
```
NotImplementedError: No ONNX conversion available for: Alloc
Op: Alloc
Node: Alloc(ExpandDims{axes=[4, 5]}.0, Shape_i{0}.0, Assert{...}.0, Sub.0, Sub.0, 2, 2)
```

### 2. Analysis

- [ ] What is the operation type?
- [ ] Where is it defined?
- [ ] What does it do?
- [ ] What's the ONNX equivalent?
- [ ] Single node or multi-node decomposition?

### 3. Implementation

[Fill in after analysis]

### 4. Testing

[Fill in based on operation type]

### 5. Verification Checklist

- [ ] Unit test passes (if applicable)
- [ ] Integration test progresses
- [ ] No regressions in existing tests

### 6. Documentation

[Fill in after implementation]

### 7. Next Steps

[Continue to Iteration 3 or declare victory]

---

## Iteration 3+: [To Be Determined]

Follow the same template for each subsequent missing operation.

---

## Progress Tracking

### Operations Implemented

| Iteration | Operation | Type | ONNX Mapping | Status |
|-----------|-----------|------|--------------|--------|
| 1 | Assert | COp | Identity | ✅ Complete |
| 2 | Alloc | Shape Op | Expand | 🔄 In Progress |
| 3 | TBD | - | - | ⏸️ Waiting |

### Test Progress

| Test Name | Status | Notes |
|-----------|--------|-------|
| `test_model_exports_to_onnx` | 🔄 | Blocked by: Alloc |
| `test_onnx_runtime_vs_pytensor_equivalence` | 🔲 | Not reached |
| `test_onnx_model_metadata` | 🔲 | Not reached |
| `test_onnx_export_different_batch_sizes` | 🔲 | Not reached |
| `test_onnx_output_shapes` | 🔲 | Not reached |
| `test_onnx_deterministic_inference` | 🔲 | Not reached |
| `test_onnx_model_opset_version` | 🔲 | Not reached |
| `test_onnx_no_training_ops` | 🔲 | Not reached |

---

## Quick Reference Commands

### Run Single Test (Fast Feedback)
```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py::test_model_exports_to_onnx -xvs
```

### Run All ONNX Export Tests
```bash
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py -v
```

### Run Analysis Script (Identify Operations)
```bash
cd examples/onnx/onnx-yolo-demo
uv run python find_all_missing_ops.py
```

### Check PyTensor ONNX Tests (Verify No Regressions)
```bash
cd ../../../
pytest tests/link/onnx/test_elemwise.py -v
pytest tests/link/onnx/test_shape.py -v
```

### Lint Check (Optional, Slow on Windows)
```bash
cd pytensor
make lint
```

---

## Context Management Strategy

**When to Clear Context**:
- After completing each iteration (implementation + verification)
- When token usage exceeds 150k tokens
- When starting a complex new operation that requires extensive research

**What to Save Before Clearing**:
1. Update this plan with completed checkboxes
2. Document what was implemented in "Operations Implemented" table
3. Note current blocker in "Test Progress" table
4. Commit code changes (optional but recommended)

**How to Resume**:
1. Read this plan file
2. Check "Operations Implemented" table for what's done
3. Check "Test Progress" table for current blocker
4. Jump to the next iteration
5. Run the test to discover the next missing operation

---

## Success Criteria

### Per-Iteration Success

- [ ] Operation dispatcher implemented
- [ ] Code follows existing patterns
- [ ] YOLO export test progresses past this operation
- [ ] No new errors introduced
- [ ] Progress documented in this plan

### Final Success

- [ ] All 8 tests in `test_onnx_export.py` pass
- [ ] No regressions in PyTensor's core ONNX tests
- [ ] YOLO model exports to ONNX successfully
- [ ] ONNX Runtime can execute the exported model
- [ ] Numerical outputs match PyTensor within tolerance (rtol=1e-4, atol=1e-5)

---

## Troubleshooting

### Error: "NotImplementedError: [Op] not supported"

**Solution**: This is expected! Follow the iteration template:
1. Note the operation name
2. Research its definition and purpose
3. Determine ONNX equivalent
4. Implement dispatcher
5. Test and verify

### Error: "ONNX model validation failed"

**Possible Causes**:
- Incorrect ONNX node structure
- Wrong attribute types or values
- Mismatched tensor shapes
- Invalid opset version

**Debug Strategy**:
1. Check the ONNX error message for specifics
2. Verify node creation with ONNX docs
3. Compare with working examples in existing dispatchers
4. Use `onnx.checker.check_model()` for detailed validation

### Error: "Numerical mismatch between ONNX and PyTensor"

**Possible Causes**:
- Wrong ONNX operator (e.g., truncation vs floor division)
- Incorrect decomposition order
- Dtype mismatch
- Broadcast shape issues

**Debug Strategy**:
1. Test the operation in isolation (unit test)
2. Print intermediate values in both PyTensor and ONNX
3. Check if decomposition order matters (e.g., SiLU)
4. Verify dtypes match throughout pipeline

### Test Hangs or Times Out

**Possible Causes**:
- Infinite loop in ONNX graph
- Memory allocation issue (AllocEmpty with huge size)
- Extremely slow operation

**Debug Strategy**:
1. Add timeout: `pytest --timeout=30 tests/test_onnx_export.py`
2. Run with simpler inputs (smaller model size)
3. Check for circular dependencies in ONNX graph
4. Use ONNX visualization tools to inspect graph

---

## Resources

### PyTensor ONNX Dispatch Examples

- **Elemwise**: `pytensor/link/onnx/dispatch/elemwise.py`
  - Simple mapping: EQ → Equal (line 56)
  - Multi-node: SiLU decomposition (lines 181-207, 319-346)
  - Complex: IntDiv decomposition (lines 209-283, 348-416)

- **Shape**: `pytensor/link/onnx/dispatch/shape.py`
  - Identity: ScalarFromTensor → Identity (lines 145-177)
  - Multi-node: Shape_i decomposition (lines 65-142)
  - Complex: DimShuffle decomposition (lines 271-472)

- **Basic**: `pytensor/link/onnx/dispatch/basic.py`
  - Graph conversion: FunctionGraph handler (lines 152-320)
  - Registration: onnx_funcify decorator (line 29)

### ONNX Operator Documentation

- **ONNX Operators**: https://onnx.ai/onnx/operators/
- **Opset 18**: https://onnx.ai/onnx/operators/onnx__Identity.html
- **ONNX Python API**: https://onnx.ai/onnx/api/

### Research Documents

- **Assert Analysis**: `thoughts/shared/research/2025-10-19_assert-operations-analysis.md`
  - Why Assert appears in YOLO graph
  - Why find_missing_ops.py didn't catch it
  - Detailed COp vs Elemwise explanation

### Test Utilities

- **Compare Helper**: `tests/link/onnx/utils.py::compare_onnx_and_py()`
  - Exports PyTensor function to ONNX
  - Runs both PyTensor and ONNX Runtime
  - Compares outputs with tolerance

### Analysis Scripts

- **Comprehensive**: `examples/onnx/onnx-yolo-demo/find_all_missing_ops.py`
- **Elemwise Focus**: `examples/onnx/onnx-yolo-demo/find_missing_ops.py`
- **Operation Check**: `examples/onnx/onnx-yolo-demo/check_all_ops.py`

---

## Notes

**Important**: This plan is designed to be **iterative and discoverable**. We can't know all missing operations upfront because:

1. PyTensor's graph rewriting inserts operations not visible in source code
2. The compilation phase may introduce operations based on optimization passes
3. Different input shapes or configurations may trigger different rewrites

**The solution**: Run the test, discover what's missing, implement it, repeat.

**Token Management**: Clear context between iterations if needed. This plan serves as the source of truth for progress.

**Flexibility**: If a new operation is complex, break it into its own sub-iteration with detailed analysis before implementation.

