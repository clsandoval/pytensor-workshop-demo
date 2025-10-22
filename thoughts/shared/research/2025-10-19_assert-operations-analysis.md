---
date: 2025-10-19T00:00:00-08:00
researcher: Claude
git_commit: 226f34c37775b44b18b783723a7357f56fb5e116
branch: onnx-workshop-demo
repository: pytensor
topic: "Why Assert operations appear in YOLO11n graph and why find_missing_ops.py didn't catch them"
tags: [research, codebase, onnx, assert, graph-rewriting, elemwise]
status: complete
last_updated: 2025-10-19
last_updated_by: Claude
---

# Research: Why Assert Operations Appear in YOLO11n Graph

**Date**: 2025-10-19T00:00:00-08:00
**Researcher**: Claude
**Git Commit**: 226f34c37775b44b18b783723a7357f56fb5e116
**Branch**: onnx-workshop-demo
**Repository**: pytensor

## Research Question

Why does ONNX export fail with "Assert not supported" when testing the YOLO11n model, and why didn't the `find_missing_ops.py` script catch this issue?

## Summary

**The Core Issue**: `find_missing_ops.py` only analyzes **Elemwise scalar operations** but Assert is a **COp (C-implemented Op)**, not an Elemwise operation. This is a fundamental category mismatch.

**Key Findings**:

1. **Assert is a COp, not Elemwise**: Assert operations (`pytensor/raise_op.py:145-182`) inherit from `CheckAndRaise` → `COp` → `Op`, completely separate from the Elemwise operation hierarchy
2. **Automatically inserted by rewrites**: Assert operations are NOT in the user's YOLO model code - they're automatically inserted by PyTensor's graph rewriting passes during compilation
3. **find_missing_ops.py scope limitation**: The script explicitly only checks `Elemwise` operations and their scalar ops (lines 66-68), missing other Op types like COps
4. **No ONNX dispatcher**: There is no (and typically shouldn't be) an ONNX dispatcher for Assert operations since ONNX has no exception handling mechanism

## Detailed Findings

### 1. Assert Operation Type - It's a COp, Not Elemwise

**Location**: `pytensor/raise_op.py:145-182`

**Class Hierarchy**:
```
Op (base class)
  └─ COp (C-implemented Op)
      └─ CheckAndRaise (lines 26-143)
          └─ Assert (lines 145-180)
```

**Key Characteristics**:
- Assert is a **COp** - a C-implemented operation with custom logic
- It is **NOT** a scalar operation wrapped in Elemwise
- It is **NOT** in `pytensor.scalar` module
- It has its own module: `pytensor.raise_op`

**Code Reference**:
```python
# pytensor/raise_op.py:145-180
class Assert(CheckAndRaise):
    """
    Implements assertion in computation graph.
    """
    def __init__(self, msg: str = ""):
        super().__init__(AssertionError, msg)

assert_op = Assert()  # Singleton instance
```

### 2. Why Assert Operations Appear in the Graph

**Critical Discovery**: Assert operations are **NOT explicitly written** in the YOLO model code at `examples/onnx/onnx-yolo-demo/yolo/model.py`.

**Where They Come From**: Automatically inserted by **PyTensor's graph rewriting passes** during compilation.

#### Primary Source: Alloc Merging Rewrite

**Location**: `pytensor/tensor/rewriting/basic.py:1197-1232`

**Rewriter**: `local_merge_alloc`

**Purpose**: When merging nested `Alloc` operations (used for broadcasting), validates dimension compatibility

**Code**:
```python
@register_canonicalize
@node_rewriter([Alloc])
def local_merge_alloc(fgraph, node):
    """
    This rewriter takes care of:
        Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) -> Alloc(m, x, assert(y1, y1==y2), z, w)
    """
    # ... setup code ...
    for i, dim_inner in enumerate(reversed(dims_inner)):
        dim_outer = dims_outer[-1 - i]
        if dim_inner == dim_outer:
            continue
        if isinstance(dim_inner, Constant) and dim_inner.data == 1:
            continue
        dims_outer[-1 - i] = Assert(
            "You have a shape error in your graph..."
        )(dim_outer, eq(dim_outer, dim_inner))  # <-- Assert with EQ here!
    return [alloc(inputs_inner[0], *dims_outer)]
```

**When Triggered**:
- YOLO model uses broadcasting in batch normalization and attention mechanisms
- PyTensor's `Alloc` op is used internally for broadcasting
- The rewriter merges nested allocations and adds Assert to validate shapes

**Line Reference**: `pytensor/tensor/rewriting/basic.py:1226-1231`

#### Secondary Sources

**Split Simplification** (`pytensor/tensor/rewriting/basic.py:1084-1100`):
```python
@register_canonicalize
@node_rewriter([Split])
def local_useless_split(fgraph, node):
    if node.op.len_splits == 1:
        x, axis, splits = node.inputs
        out = assert_op(x, eq(splits.shape[0], 1))
        out2 = assert_op(out, eq(x.shape[axis], splits[0]))
        return [out2]
```

**Advanced Indexing** (`pytensor/tensor/rewriting/subtensor.py:1240-1263`):
```python
cond = [pt_all(and_(lt(idx, x.shape[0]), ge(idx, -x.shape[0])))]
if not fgraph.shape_feature.same_shape(idx, y, 0, 0):
    cond.append(eq(idx.shape[0], y.shape[0]))
r = Assert("Bad indexing...")(y, *cond)
```

**Data Flow**:
1. User writes YOLO model (no explicit assertions)
2. PyTensor builds initial symbolic graph
3. Compilation with `pytensor.function()` triggers rewriting
4. **Canonicalization phase** runs `local_merge_alloc` and other rewrites
5. Rewrites **insert Assert operations with EQ comparisons**
6. Final compiled graph contains Assert ops (not in original code)

### 3. Why find_missing_ops.py Didn't Catch Assert

**Location**: `examples/onnx/onnx-yolo-demo/find_missing_ops.py:66-68`

**Code Analysis**:
```python
for node in graph_nodes:
    if type(node.op).__name__ == "Elemwise":  # <-- Only checks Elemwise!
        scalar_op_name = type(node.op.scalar_op).__name__
        elemwise_ops[scalar_op_name] += 1
```

**The Limitation**:
- Script explicitly filters for `type(node.op).__name__ == "Elemwise"`
- Assert operations have `type(node.op).__name__ == "Assert"` (or `"CheckAndRaise"`)
- **Assert ≠ Elemwise**, so it's completely skipped by the analysis

**Script Scope**:
```python
# What find_missing_ops.py DOES check:
- Elemwise operations (Add, Mul, Sub, etc.)
- Scalar ops inside Elemwise (EQ, Switch, etc.)

# What find_missing_ops.py DOESN'T check:
- COps like Assert, CheckAndRaise
- Other Op types like Reshape, Join, Dot
- Any non-Elemwise operations
```

**Why This Design Choice**:
The script was specifically designed to identify missing **Elemwise scalar operations** (like EQ, Switch) because:
1. That was the initial blocker for ONNX export
2. Elemwise ops are common and well-defined for ONNX mapping
3. The script's purpose was narrow: "Find missing scalar ops"

### 4. ONNX Dispatch System and Assert

**ONNX Dispatch Architecture**: `pytensor/link/onnx/dispatch/`

PyTensor uses `singledispatch` to register ONNX converters for each Op type:

**Dispatch Files**:
- `basic.py` - FunctionGraph conversion infrastructure
- `elemwise.py` - Elemwise operations and scalar ops (Add, Mul, EQ, Switch, etc.)
- `shape.py` - Shape manipulation (Reshape, DimShuffle, Shape_i, etc.)
- `special.py` - Special functions (Softmax)
- `join.py` - Concatenation (Join)
- `nlinalg.py` - Linear algebra (Dot, Gemv)
- `conv.py` - Convolution (AbstractConv2d)
- `pool.py` - Pooling (Pool)
- `resize.py` - Image resizing (Resize)
- `batchnorm.py` - Batch normalization

**Assert Dispatcher**: **NONE** - No registered converter for Assert or CheckAndRaise

**Why Assert is Missing from ONNX**:

1. **Semantic Incompatibility**:
   - PyTensor Assert can raise exceptions at runtime
   - ONNX is a pure dataflow graph with no exception handling
   - ONNX has no equivalent to "raise AssertionError"

2. **Export Philosophy**:
   - ONNX export is for inference deployment, not debugging
   - Assertions are development/validation tools
   - They should be validated at compile-time, not needed in runtime model

3. **Optimization Consideration**:
   - PyTensor documentation states Assert "can be removed from the graph because of optimizations" (`raise_op.py:153-155`)
   - If conditions are provably True, Assert can be eliminated

4. **No ONNX Equivalent**: ONNX simply doesn't have operations for conditional exception raising

## Code References

- `pytensor/raise_op.py:145-182` - Assert class definition (COp)
- `pytensor/tensor/rewriting/basic.py:1226-1231` - Alloc merging inserts Assert with EQ
- `pytensor/tensor/rewriting/basic.py:1093-1096` - Split simplification inserts Assert
- `pytensor/tensor/rewriting/subtensor.py:1247-1250` - Indexing validation inserts Assert
- `examples/onnx/onnx-yolo-demo/find_missing_ops.py:66-68` - Only checks Elemwise ops
- `pytensor/link/onnx/dispatch/` - ONNX dispatcher directory (no Assert handler)

## Architecture Insights

### Operation Type Hierarchy in PyTensor

```
Op (base class)
├─ COp (C-implemented ops)
│   ├─ CheckAndRaise
│   │   └─ Assert  ← THIS IS WHAT WE'RE DEALING WITH
│   └─ Other COps...
│
├─ Elemwise (element-wise operations)
│   └─ scalar_op (Add, Mul, EQ, Switch, etc.)  ← find_missing_ops.py checks these
│
└─ Other Op types (Reshape, Join, Dot, etc.)
```

**Key Insight**: Elemwise and COp are **sibling branches**, not parent-child. A script checking one won't find the other.

### Graph Rewriting Pipeline

```
User Code (model.py)
    ↓
Initial Symbolic Graph (no assertions)
    ↓
pytensor.function() compilation
    ↓
Canonicalization Phase
    ├─ local_merge_alloc rewrite  → Inserts Assert + EQ
    ├─ local_useless_split rewrite → Inserts Assert + EQ
    └─ Other rewrites...
    ↓
Specialization Phase
    └─ Subtensor rewrites → May insert Assert
    ↓
Final Compiled Graph (contains Assert operations)
    ↓
ONNX Export Attempt
    ↓
NotImplementedError: Assert not supported
```

## Historical Context (from thoughts/)

From `thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md`:
- Documents that YOLO model contains **8 Assert operations**
- Notes Assert operations "may be ignorable (assertions are compile-time checks)"
- Status marked as "? Unknown" - requires testing after EQ/Switch implementation

From `thoughts/shared/research/2025-10-15_onnx-open-questions-answers.md:303`:
- Lists `CheckAndRaise`, `Assert` as having "NO ONNX equivalent"

## Recommendations

### Immediate Action: Implement Assert Handler for ONNX Export

**Option 1: Pass-Through as Identity (Recommended)**

Add to `pytensor/link/onnx/dispatch/basic.py`:

```python
from pytensor.raise_op import Assert, CheckAndRaise

@onnx_funcify.register(Assert)
@onnx_funcify.register(CheckAndRaise)
def onnx_funcify_assert(op, node, var_names, get_var_name, **kwargs):
    """
    Assert operations are skipped during ONNX export.
    They pass through their input value unchanged.

    Assertions are for runtime validation and debugging.
    For ONNX export, shapes should be validated at compile time.
    """
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
- Remaining inputs are boolean conditions
- In ONNX export, we assume shapes are valid (validated at compile time)
- Identity node passes value through unchanged

### Long-term: Improve find_missing_ops.py

**Current Limitation**: Only checks Elemwise operations

**Enhancement**: Check all Op types in the graph

```python
# After line 72 in find_missing_ops.py, add:
print("\n" + "=" * 80)
print("ALL OPERATION TYPES IN GRAPH")
print("=" * 80)

from collections import defaultdict
all_ops = defaultdict(int)

for node in graph_nodes:
    op_type = type(node.op).__name__
    all_ops[op_type] += 1

print(f"\nAll operations in the model:")
for op_type, count in sorted(all_ops.items(), key=lambda x: -x[1]):
    print(f"  {op_type:30s} : {count:3d} times")

# Check which are supported in ONNX
print("\nChecking ONNX support for all operations...")
try:
    from pytensor.link.onnx.dispatch.basic import onnx_funcify

    supported = []
    unsupported = []

    for op_type in all_ops.keys():
        # Try to find the Op class and check if it has a dispatcher
        # This is a simplified check - actual implementation needs proper module lookup
        try:
            # Check if operation type is registered with onnx_funcify
            # (This requires introspecting the singledispatch registry)
            supported.append(op_type)  # Placeholder logic
        except Exception:
            unsupported.append((op_type, all_ops[op_type]))

    print(f"\n✗ UNSUPPORTED operations: {len(unsupported)}")
    for op, count in unsupported:
        print(f"    {op:30s} : used {count:3d} times")

except ImportError as e:
    print(f"  Could not check: {e}")
```

### Testing Strategy

1. **Implement Assert → Identity converter** in `pytensor/link/onnx/dispatch/basic.py`
2. **Run ONNX export test** on YOLO model
3. **Verify numerical correctness** - Assert pass-through shouldn't affect outputs
4. **Document behavior** - Note that runtime assertions are skipped in ONNX export

## Open Questions

1. **Should Assert conditions be validated during export?**
   - Pro: Catch shape errors before deployment
   - Con: Adds complexity, may fail on symbolic shapes
   - **Decision needed**: Pass-through (simple) vs. validation (safer)

2. **Are there other COps missing from ONNX export?**
   - Current analysis only checked Elemwise and Assert
   - Need comprehensive audit of all Op types in YOLO graph
   - Recommendation: Run enhanced find_missing_ops.py to catch all unsupported ops

3. **Can rewrite configuration reduce Assert usage?**
   - Try `optimizer='fast_compile'` to minimize rewrites
   - May reduce Assert insertions but could miss optimizations
   - Trade-off: Fewer assertions vs. potentially slower runtime

## Conclusion

**Why Assert shows up**: Automatically inserted by PyTensor's graph rewriting passes (specifically `local_merge_alloc`) during compilation, not present in original YOLO code.

**Why find_missing_ops.py didn't catch it**: Script only analyzes Elemwise scalar operations; Assert is a COp (C-implemented Op), completely separate category.

**Solution**: Implement an ONNX dispatcher for Assert that passes through the value as Identity, skipping runtime validation since ONNX has no exception mechanism.

**Next Steps**:
1. Add Assert → Identity converter to `pytensor/link/onnx/dispatch/basic.py`
2. Test ONNX export with the YOLO model
3. Enhance find_missing_ops.py to check all Op types, not just Elemwise
4. Document that ONNX export skips runtime assertions
