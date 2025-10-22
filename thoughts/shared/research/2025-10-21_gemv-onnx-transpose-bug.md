---
date: 2025-10-21T00:00:00Z
researcher: Claude Code
git_commit: 226f34c37775b44b18b783723a7357f56fb5e116
branch: onnx-workshop-demo
repository: pytensor-workshop-demo
topic: "Gemv ONNX Conversion Transpose Bug: Shape-Dependent Behavior"
tags: [research, codebase, onnx, gemv, transpose, bug, dimshuffle, property-testing]
status: complete
last_updated: 2025-10-21
last_updated_by: Claude Code
---

# Research: Gemv ONNX Conversion Transpose Bug

**Date**: 2025-10-21T00:00:00Z
**Researcher**: Claude Code
**Git Commit**: 226f34c37775b44b18b783723a7357f56fb5e116
**Branch**: onnx-workshop-demo
**Repository**: pytensor-workshop-demo

## Research Question

Why does PyTensor's ONNX export produce transposed output for dot operations when tensor shapes are known at graph construction time, while tests with unknown shapes pass correctly?

## Summary

The bug is caused by a **combination of three factors**:

1. **Optimizer Rewrites**: When tensor shapes are known, PyTensor's optimizer rewrites `Dot22` operations to `Gemv` (General Matrix-Vector multiplication) for performance optimization
2. **Complex DimShuffle Operations**: The Gemv transformation involves squeeze, transpose, and unsqueeze operations (`dimshuffle(1)`, `.T`, `dimshuffle("x", 0)`)
3. **Known DimShuffle Bugs**: The ONNX conversion for DimShuffle has **documented bugs** when combining squeeze+transpose and transpose+unsqueeze operations, which are exactly the patterns used in the Gemv transformation

The result is that **property tests with concrete shapes fail** because they trigger the buggy Gemv path, while **unit tests with symbolic shapes pass** because they use the simple Dot22 path.

## Root Cause Analysis

### The Two Execution Paths

#### Path 1: Unknown Shapes (✅ WORKS - test_nlinalg.py)

```python
# Test creates symbolic tensors WITHOUT concrete shapes
x = pt.matrix("x", dtype="float32")  # Shape: (None, None)
y = pt.matrix("y", dtype="float32")  # Shape: (None, None)
z = pt.dot(x, y)
```

**Optimization**: `Dot` → `Dot22` (2D matrix multiplication)
**ONNX Conversion**: Direct 1-to-1 mapping to ONNX `MatMul` node
**Result**: ✅ Correct output

**Code Reference**: `pytensor/link/onnx/dispatch/nlinalg.py:33-46`

#### Path 2: Known Shapes (❌ FAILS - test_properties.py)

```python
# Property test creates tensors WITH concrete shapes
x = pt.tensor("x", dtype="float32", shape=(1, 2))  # Shape: (1, 2)
y = pt.tensor("y", dtype="float32", shape=(2, 2))  # Shape: (2, 2)
z = pt.dot(x, y)
```

**Optimization**: `Dot` → `Dot22` → `Gemv` (optimized matrix-vector path)
**ONNX Conversion**: Complex multi-node graph with Squeeze, Transpose, MatMul, Unsqueeze
**Result**: ❌ Transposed output (produces `[[3, 4]]` instead of `[[4, 3]]`)

**Code Reference**: `pytensor/tensor/rewriting/blas.py:690-695`

### The Gemv Transformation Details

When the optimizer sees a (1,2) @ (2,2) multiplication with **known shapes**:

**Optimizer Logic** (`pytensor/tensor/rewriting/blas.py:690-695`):
```python
elif xb[0] and not yb[0] and not yb[1]:
    # x is row vector (1,k), y is matrix (k,n) so try gemv
    xv = x.dimshuffle(1)                              # Squeeze: (1,2) → (2,)
    zeros = ptb.AllocEmpty(x.dtype)(y.shape[1])       # zeros: (2,)
    rval = gemv_no_inplace(zeros, one, y.T, xv, zero) # MatMul: y.T @ xv
    new_out = [rval.dimshuffle("x", 0)]               # Unsqueeze: (2,) → (1,2)
```

**Mathematical Correctness**: ✅ This transformation is mathematically correct!
- Original: `x @ y = [[a, b]] @ [[c, d], [e, f]] = [[a*c + b*e, a*d + b*f]]`
- Transformed: `y.T @ xv = [[c, e], [d, f]] @ [a, b] = [c*a + e*b, d*a + f*b]`
- After unsqueeze: `[[c*a + e*b, d*a + f*b]]` ✅ Matches!

**The Bug**: The ONNX conversion of the DimShuffle operations (squeeze/transpose/unsqueeze) has **known bugs** that cause the output to be transposed.

### The DimShuffle ONNX Conversion Bug

**Critical Finding**: The ONNX conversion for DimShuffle has **three documented failing test cases** that exactly match the patterns used in Gemv transformation:

**Documented Bugs** (`tests/link/onnx/test_shape.py`):

1. **Squeeze + Transpose** (Line 96-104): ❌ FAILS
   - Pattern: `x.dimshuffle(2, 0)` on `(2, 1, 3)` → `(3, 2)`
   - Requires: Squeeze(axis=1) → Transpose(1,0)
   - **This is used in Gemv!**

2. **Transpose + Unsqueeze** (Line 85-94): ❌ FAILS
   - Pattern: `x.dimshuffle(1, "x", 0)` on `(2, 3)` → `(3, 1, 2)`
   - Requires: Transpose(1,0) → Unsqueeze(axis=1)
   - **This is used in Gemv!**

3. **Unsqueeze + Transpose** (Line 107-115): ❌ FAILS
   - Pattern: `x.dimshuffle("x", 1, 0)` on `(2, 3)` → `(1, 3, 2)`
   - Requires: Transpose(1,0) → Unsqueeze(axis=0)
   - **This is used in Gemv!**

**Root Cause**: The DimShuffle ONNX converter decomposes complex patterns into: Squeeze → Transpose → Unsqueeze sequence. However, there are bugs in handling these combinations, particularly with dimension index remapping after squeeze operations.

## Detailed Findings

### Component 1: Gemv Operation Implementation

**Core Implementation**: `pytensor/tensor/blas.py:153-256`

```python
class Gemv(Op):
    """General Matrix-Vector multiplication

    Computes: y = alpha * A @ x + beta * y
    """
```

**ONNX Conversion**: `pytensor/link/onnx/dispatch/nlinalg.py:49-113`

The Gemv ONNX converter decomposes the operation into **4 ONNX nodes**:
1. **MatMul**: `A @ x`
2. **Mul**: `alpha * (A @ x)`
3. **Mul**: `beta * y`
4. **Add**: `alpha * (A @ x) + beta * y`

**Input Signature**: `[y_in, alpha, A, x, beta]`

**Critical Note**: The Gemv ONNX converter itself is **mathematically correct**. The bug is in the **DimShuffle operations** that precede and follow the Gemv operation in the graph.

### Component 2: Dot22 Operation (Simple Path)

**Core Implementation**: `pytensor/tensor/blas.py:1123-1206`

```python
class Dot22(GemmRelated):
    """Optimized 2D x 2D matrix multiplication"""
```

**ONNX Conversion**: `pytensor/link/onnx/dispatch/nlinalg.py:33-46`

```python
@onnx_funcify.register(Dot22)
def onnx_funcify_Dot22(op, node, var_names, get_var_name, **kwargs):
    """Convert Dot22 to ONNX MatMul node."""
    onnx_node = helper.make_node(
        "MatMul",
        inputs=input_names,
        outputs=output_names,
        name=f"MatMul_{output_names[0]}",
    )
    return onnx_node
```

**Simplicity**: Direct 1-to-1 mapping with **no shape manipulations** → No bugs!

### Component 3: Optimizer Rewrite Rules

**Primary Rewrite Rule**: `pytensor/tensor/rewriting/blas.py:669-705`

**Function**: `local_dot22_to_ger_or_gemv`

**Shape Detection Logic**: Uses `broadcastable` attribute to detect row/column vectors
- `broadcastable == (True, False)` → Row vector (first dim is 1)
- `broadcastable == (False, True)` → Column vector (second dim is 1)

**Case 1: Row Vector @ Matrix** (Lines 690-695):
```python
elif xb[0] and not yb[0] and not yb[1]:
    # x is vector (1,k), y is matrix (k,n)
    xv = x.dimshuffle(1)  # (1,k) → (k,) - Squeeze dimension 0
    zeros = ptb.AllocEmpty(x.dtype)(y.shape[1])
    rval = gemv_no_inplace(zeros, one, y.T, xv, zero)  # y.T @ xv
    new_out = [rval.dimshuffle("x", 0)]  # (n,) → (1,n) - Unsqueeze dimension 0
```

**Case 2: Matrix @ Column Vector** (Lines 696-701):
```python
elif not xb[0] and not xb[1] and yb[1]:
    # x is matrix (m,k), y is vector (k,1)
    yv = y.dimshuffle(0)  # (k,1) → (k,) - Squeeze dimension 1
    zeros = ptb.AllocEmpty(x.dtype)(x.shape[0])
    rval = gemv_no_inplace(zeros, one, x, yv, zero)  # x @ yv
    new_out = [rval.dimshuffle(0, "x")]  # (m,) → (m,1) - Unsqueeze dimension 1
```

**Key Observation**: Both cases use **squeeze and unsqueeze operations**, and Case 1 also uses **transpose** (`.T`), which are the exact patterns with known DimShuffle bugs!

### Component 4: DimShuffle ONNX Conversion

**Implementation**: `pytensor/link/onnx/dispatch/shape.py:271-472`

**Decomposition Algorithm**: `shape.py:198-268` - `decompose_dimshuffle_pattern()`

**Strategy**: Decompose complex DimShuffle patterns into sequential operations:
1. **Squeeze**: Remove dimensions (axes with size 1)
2. **Transpose**: Reorder remaining dimensions
3. **Unsqueeze**: Add new dimensions (marked with 'x')

**Critical Bug Fix Attempt**: `shape.py:252-266`
```python
# CRITICAL: Adjust transpose permutation after squeeze
# After squeezing, dimension indices shift down
if result["squeeze_axes"] and result["transpose_perm"]:
    # Create mapping from original dims to post-squeeze dims
    dim_mapping = {}
    new_idx = 0
    for old_idx in range(input_ndim):
        if old_idx not in result["squeeze_axes"]:
            dim_mapping[old_idx] = new_idx
            new_idx += 1

    # Remap transpose permutation
    result["transpose_perm"] = [
        dim_mapping[old_dim] for old_dim in result["transpose_perm"]
    ]
```

**Purpose**: After squeezing dimension 1 from a 3D tensor, the remaining dimensions need their indices adjusted (e.g., dimension 2 becomes dimension 1).

**Bug Status**: Despite this fix attempt, the test suite still marks squeeze+transpose and transpose+unsqueeze combinations as failing.

### Component 5: Test Files

#### Unit Tests (PASS): `tests/link/onnx/test_nlinalg.py`

**Key Tests**:
- `test_dot_matrix_matrix()` (Line 46): Tests matrix-matrix multiplication
- `test_gemv_operation()` (Line 76-109): Tests Gemv with scaling factors
- `test_gemv_structure()` (Line 112-154): Validates 4-node ONNX structure

**Why They Pass**: All tests use **symbolic shapes** (e.g., `pt.matrix("x")` without shape parameter)
```python
x = pt.matrix("x", dtype="float32")  # Shape is (None, None) - symbolic!
y = pt.matrix("y", dtype="float32")
z = pt.dot(x, y)
```

**Result**: PyTensor cannot determine that x or y are vectors → Uses simple Dot22 path → No DimShuffle bugs

#### Property Tests (FAIL): `tests/link/onnx/test_properties.py`

**Key Test**: `test_onnx_matches_pytensor()` (Line 30-111)

**Strategy**: Uses Hypothesis to generate random test cases with **concrete shapes**
```python
# Property test creates tensors WITH known shapes
inputs_tuple = data.draw(op_config.input_strategy)  # Hypothesis generates shapes
x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
result = op_config.op_func(x, y)  # pt.dot(x, y)
```

**Input Strategy**: `matmul_inputs()` in `tests/link/onnx/strategies/operations.py:171-218`

**Generates Various Shapes**:
- Vector @ Vector: `(k,) @ (k,)`
- Vector @ Matrix: `(k,) @ (k, n)`
- Matrix @ Vector: `(m, k) @ (k,)`  ← **This triggers the bug!**
- Matrix @ Matrix: `(m, k) @ (k, n)`

**Why It Fails**: When Hypothesis generates shapes like `(1, 2) @ (2, 2)`:
1. PyTensor sees x.broadcastable = (True, False) → row vector
2. Optimizer rewrites to Gemv with squeeze + transpose + unsqueeze
3. DimShuffle ONNX conversion has bugs with these combinations
4. Output is transposed: produces `[[3, 4]]` instead of `[[4, 3]]`

**Failure Mode**: The compare_onnx_and_py() function detects numerical mismatch between PyTensor and ONNX outputs

#### DimShuffle Regression Tests: `tests/link/onnx/test_regressions.py`

**Documented Regressions** (Lines 39-87):

1. **Identity Node Misuse**:
   - Bug: Pattern `(1, 'x', 0)` on shape `(2,3)` incorrectly used Identity node
   - Result: Produced shape `(2,3)` instead of correct `(3,1,2)`
   - Fix: Added proper Squeeze→Transpose→Unsqueeze decomposition

2. **Case 3 Condition Bug**:
   - Bug: Case 3 (pure transpose) didn't check for `axes_to_add`
   - Pattern: `(2, 0)` on `(2,1,3)` incorrectly matched Case 3
   - Fix: Added proper condition checking at `shape.py:369-373`

## Code References

### Core Implementation Files

- `pytensor/tensor/blas.py:153-256` - Gemv class definition
- `pytensor/tensor/blas.py:1123-1206` - Dot22 class definition
- `pytensor/tensor/rewriting/blas.py:669-705` - local_dot22_to_ger_or_gemv optimizer
- `pytensor/link/onnx/dispatch/nlinalg.py:33-46` - Dot22 → ONNX MatMul
- `pytensor/link/onnx/dispatch/nlinalg.py:49-113` - Gemv → ONNX (4 nodes)
- `pytensor/link/onnx/dispatch/shape.py:198-268` - decompose_dimshuffle_pattern
- `pytensor/link/onnx/dispatch/shape.py:271-472` - onnx_funcify_DimShuffle

### Test Files

- `tests/link/onnx/test_nlinalg.py:46-174` - Unit tests (symbolic shapes)
- `tests/link/onnx/test_properties.py:30-111` - Property tests (concrete shapes)
- `tests/link/onnx/test_shape.py:85-115` - DimShuffle failing test cases
- `tests/link/onnx/test_regressions.py:39-87` - Regression tests
- `tests/link/onnx/strategies/operations.py:171-218` - matmul_inputs strategy

### Optimizer Files

- `pytensor/tensor/rewriting/blas.py:580-596` - local_dot_to_dot22
- `pytensor/tensor/rewriting/blas.py:622-635` - local_gemm_to_gemv
- `pytensor/tensor/rewriting/blas.py:714-758` - Optimizer registration

## Architecture Insights

### The Two-Path Design

PyTensor's ONNX backend has **two distinct paths** for matrix multiplication:

1. **Generic Path (Dot/Dot22)**:
   - Used when shapes are unknown or don't match special patterns
   - Direct mapping to ONNX MatMul
   - Simple, reliable, well-tested

2. **Optimized Path (Gemv)**:
   - Used when shapes indicate matrix-vector operations
   - Complex transformation with shape manipulations
   - Higher performance potential but buggy ONNX conversion

### Why Shape Knowledge Matters

The optimizer's `broadcastable` attribute is **only set when shapes are concrete**:
- `pt.matrix("x")` → broadcastable = (False, False) - unknown
- `pt.tensor("x", shape=(1, 2))` → broadcastable = (True, False) - first dim is 1

This creates a **shape-dependent behavior** where the same mathematical operation follows different code paths based on whether shapes are known at compile time.

### The DimShuffle Complexity

DimShuffle is a **powerful but complex** operation that can:
- Remove dimensions (squeeze)
- Add dimensions (unsqueeze)
- Reorder dimensions (transpose)
- Combine all three in a single operation

The ONNX backend must decompose this into **three sequential operations**, and the dimension indices must be carefully remapped at each step. Bugs in this remapping logic cause the transpose issues.

### Testing Gap

**Critical Gap**: No tests combine:
- Concrete shapes (triggering Gemv optimization)
- DimShuffle + MatMul patterns
- ONNX export validation

The unit tests use symbolic shapes (avoiding the bug), and the property tests use concrete shapes (triggering the bug) but without DimShuffle-specific validation.

## Historical Context (from thoughts/)

### Prior Research on ONNX Issues

**DimShuffle Silent Fallback Bug** (CRITICAL):
- `thoughts/shared/plans/onnx-backend-coverage-and-quality-improvements.md`
- Documents critical bug where complex DimShuffle operations fall back to Identity node
- Causes **silent data corruption** (no error, wrong output)

**Gemv Completely Untested**:
- `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md`
- Identifies Gemv as fully implemented (62 lines) but with **ZERO tests**
- Documents 4-node ONNX decomposition structure

**Transpose + Squeeze/Unsqueeze Issues**:
- `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md`
- Documents regression tests for DimShuffle bugs
- Specifically mentions Case 3 bug where pure transpose incorrectly matched patterns

### Implementation Plans

**Comprehensive Test Plan**:
- `thoughts/shared/plans/onnx-backend-coverage-and-quality-improvements.md`
- Contains test plan with `test_gemv_operation`, `test_gemv_structure`, `test_gemv_scaling_factors`
- Property-based testing strategy for Gemv with structure validation

**TDD Approach**:
- `thoughts/shared/plans/onnx-tier1-blockers-tdd.md`
- Multi-node Gemv identified as blocker (60 lines in `nlinalg.py:48-109`)

## Related Research

- `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md` - ONNX coverage gaps
- `thoughts/shared/research/2025-10-15_onnx-open-questions-answers.md` - ONNX operation mapping
- `thoughts/shared/plans/hypothesis-property-based-onnx-testing.md` - Property testing strategy

## Open Questions

### 1. What is the exact ONNX graph structure for failing cases?

**Action**: Export ONNX model for (1,2) @ (2,2) case and inspect the node sequence

**Expected**:
- Squeeze node: (1,2) → (2,)
- Transpose node: (2,2) → (2,2)
- MatMul node: (2,2) @ (2,) → (2,)
- Unsqueeze node: (2,) → (1,2)

**Question**: Which node is producing the transposed output?

### 2. Does ONNX MatMul handle 1D vectors correctly?

**ONNX MatMul Spec**: When one input is 1D, specific broadcasting rules apply:
- Left 1D: Treat as (1, N) → result has dimension prepended then removed
- Right 1D: Treat as (N, 1) → result has dimension appended then removed

**Question**: Is the bug in how PyTensor generates the ONNX MatMul, or in how ONNX Runtime interprets 1D inputs?

### 3. Why haven't the DimShuffle bug fixes worked?

**Observation**: `shape.py:252-266` has a fix for dimension index remapping, but tests still fail

**Possible Reasons**:
1. The fix is incomplete or incorrect
2. The fix is correct but not applied in all code paths
3. There are multiple bugs and only one was fixed
4. The tests are marking bugs that are actually fixed

**Action**: Run the failing DimShuffle tests and check if they actually fail or if test annotations are outdated

### 4. Can we bypass the Gemv optimization for ONNX export?

**Idea**: Add an optimization flag to disable Gemv rewrite when compiling for ONNX backend

**Pros**:
- Immediate workaround for the bug
- Uses well-tested Dot22 path
- No risk of regression

**Cons**:
- Performance penalty (though minimal for most cases)
- Doesn't fix the root cause
- Technical debt accumulation

### 5. Should we fix DimShuffle or rewrite Gemv transformation?

**Option A**: Fix the DimShuffle ONNX conversion
- Pro: Fixes the root cause, benefits all operations
- Con: Complex, risky, affects many operations

**Option B**: Rewrite Gemv transformation to avoid complex DimShuffle
- Pro: Targeted fix, less risk
- Con: Doesn't fix underlying DimShuffle bugs

**Option C**: Add Gemv as a native ONNX operation handler
- Pro: Avoid DimShuffle entirely for this case
- Con: ONNX doesn't have a native Gemv operation, would still need multi-node decomposition

### 6. What other operations might be affected?

**Pattern**: Any operation that:
1. Uses shape information to optimize
2. Transforms using DimShuffle (squeeze/transpose/unsqueeze)
3. Exports to ONNX

**Candidates to investigate**:
- Gemm (General Matrix Multiply)
- Ger (General Rank-1 update)
- Conv2D with shape transformations
- Pooling operations with shape manipulations

## Recommended Actions

### Immediate Actions

1. **Reproduce and Document**: Create minimal test case that demonstrates the transpose bug
2. **Inspect ONNX Graph**: Export ONNX model and visualize node structure using netron or onnx.checker
3. **Verify DimShuffle Tests**: Run the "failing" DimShuffle tests to confirm they actually fail

### Short-term Fixes

1. **Add Failing Test**: Add test case to `test_nlinalg.py` that uses concrete shapes
2. **Debug DimShuffle**: Step through DimShuffle ONNX conversion with failing case
3. **Fix or Workaround**: Either fix DimShuffle bugs or add workaround to bypass Gemv optimization

### Long-term Improvements

1. **Comprehensive Testing**: Add property-based tests for all ONNX operations with concrete shapes
2. **DimShuffle Refactor**: Rewrite DimShuffle ONNX conversion with comprehensive test coverage
3. **Shape-Aware Testing**: Ensure test suite covers both symbolic and concrete shape cases
4. **Documentation**: Document the two-path behavior and when each path is triggered

## Conclusion

The Gemv ONNX conversion bug is a **compound issue** resulting from:
1. Shape-dependent optimizer behavior (Dot22 vs Gemv)
2. Complex DimShuffle transformations in Gemv path
3. Known bugs in DimShuffle ONNX conversion for squeeze+transpose combinations

The bug manifests as **transposed output** when:
- Tensor shapes are known at graph construction time (concrete shapes)
- Shapes match vector patterns (e.g., (1,2) @ (2,2))
- Code follows the optimized Gemv path with DimShuffle operations

The fix requires either:
- **Fixing DimShuffle ONNX conversion** for squeeze+transpose patterns (root cause)
- **Rewriting Gemv transformation** to avoid problematic DimShuffle combinations (workaround)
- **Disabling Gemv optimization** for ONNX backend (temporary fix)

**Priority**: HIGH - This affects correctness of ONNX models and can cause silent data corruption in production deployments.
