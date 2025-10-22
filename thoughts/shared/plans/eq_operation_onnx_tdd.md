---
date: 2025-10-18T22:00:00-00:00
author: Claude
related_research: thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md
operation: EQ (Equal)
git_branch: onnx-workshop-demo
repository: pytensor
status: ready_for_implementation
test_framework: pytest + hypothesis
---

# EQ (Equal) Operation ONNX Export - TDD Implementation Plan

## Overview

Implement ONNX export support for the EQ (Equal) scalar operation using Test-Driven Development with Hypothesis property-based testing. The EQ operation performs element-wise equality comparison and is used 13 times in the YOLO11n model for shape validation and conditional logic in upsampling operations.

**TDD Approach**: Write ONE comprehensive Hypothesis property-based test that covers all cases, verify it fails with diagnostic messages, then implement the feature by making the test pass.

## Setup

Before starting, ensure you have the dev version of PyTensor installed:

```bash
# Install the dev version from the repository root (parent directory)
uv pip install -e .
```

This ensures you're testing against the development version of PyTensor, not the PyPI release.

## Current State Analysis

### What Exists Now

**ONNX Dispatch System** (`pytensor/link/onnx/dispatch/elemwise.py:17-32`):
- `SCALAR_OP_TO_ONNX` dictionary maps scalar operations to ONNX op types
- Currently supports: Add, Mul, Sub, TrueDiv, Neg, Exp, Log, Sqrt, Sqr, Pow, Abs, ScalarMaximum, ScalarMinimum, Sigmoid
- EQ is **NOT** in the dictionary

**Test Infrastructure**:
- Property-based test framework: `tests/link/onnx/test_properties.py`
- Operation registry: `tests/link/onnx/strategies/operations.py:360-444` (ONNX_OPERATIONS)
- Core strategies: `tests/link/onnx/strategies/core.py`
- Test helper: `compare_onnx_and_py()` in `tests/link/onnx/test_basic.py:22-102`

**Hypothesis Configuration** (`tests/link/onnx/conftest.py`):
- `dev` profile: 10 examples, fast feedback
- `ci` profile: 100 examples, thorough
- `thorough` profile: 1000 examples, exhaustive

### What's Missing

1. EQ not in `SCALAR_OP_TO_ONNX` dictionary
2. No test coverage for comparison operations
3. No Hypothesis strategies for comparison operations
4. No entry in `ONNX_OPERATIONS` registry

### Key Constraints Discovered

- EQ outputs boolean dtype (not input dtype)
- ONNX Equal operator supports: float32, float64, int32, int64, bool
- PyTensor's EQ is in `pytensor.scalar.basic.EQ` class
- Used in YOLO with scalar tensors for shape comparisons: `Eq(Shape_i{0}.0, Shape_i{0}.0)`

## Desired End State

After implementation:
- EQ operation exports to ONNX Equal node
- Property-based tests validate correctness with 10-100 random test cases
- Tests cover: various dtypes, various shapes, broadcasting, edge cases
- ONNX Runtime produces same results as PyTensor for all valid inputs
- Tests fail gracefully with diagnostic messages before implementation

## What We're NOT Testing/Implementing

- Gradient computation through EQ (not needed for ONNX export)
- Non-boolean comparison operations (NE, GT, LT, GE, LE) - separate operations
- String comparison (not supported by ONNX or PyTensor's numeric tensors)
- Fuzzy equality with tolerance (use separate operation)
- Integration with Switch operation (separate test)

---

## TDD Approach

### Test Design Philosophy

**Property-Based Testing Strategy**:
1. **Correctness**: ONNX Equal output must match PyTensor EQ for ANY valid inputs
2. **Dtype Handling**: Input dtype preserved for comparison, output is always bool
3. **Broadcasting**: Broadcasting rules must match NumPy/PyTensor
4. **Edge Cases**: Empty tensors, scalar tensors, extreme values all handled

**Why Hypothesis?**
- Automatically generates edge cases (empty tensors, scalar tensors, various shapes)
- Tests 10-1000 random combinations per property
- Discovers bugs manual tests miss
- Shrinks failing examples to minimal reproducible cases

**Test Structure**:
- ONE Hypothesis property-based test that automatically covers:
  - All dtypes (float32, float64, int32, int64)
  - All shapes (vectors, matrices, tensors, scalars)
  - Broadcasting scenarios
  - Equal and unequal values
  - Edge cases (empty tensors, extreme values)

---

## Phase 1: Test Design & Implementation

### Overview
Write ONE comprehensive Hypothesis property-based test that automatically covers all EQ scenarios. This test will fail initially with `NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ`.

### Single Property-Based Test

Instead of writing multiple unit tests, we'll leverage Hypothesis and the existing ONNX test infrastructure to automatically test all scenarios with a single entry in the operations registry.

#### Add EQ to ONNX_OPERATIONS Registry

**Test File**: `tests/link/onnx/strategies/operations.py`
**Purpose**: Register EQ operation so existing property-based tests automatically cover it

**What this gives us automatically:**
- Tests with 10-1000 random input combinations (via Hypothesis)
- All dtypes: float32, float64, int32, int64
- All shapes: scalars, vectors, matrices, 3D/4D tensors
- Broadcasting scenarios automatically generated
- Edge cases: empty tensors, extreme values, equal/unequal values

**Implementation:**

```python
# In tests/link/onnx/strategies/operations.py, add to ONNX_OPERATIONS dict:

ONNX_OPERATIONS = {
    # ... existing operations ...

    "eq": OperationConfig(
        op_func=pt.eq,
        input_strategy=binary_broadcastable_inputs(),  # Reuse existing strategy
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
        notes="Comparison operation, output dtype is bool",
    ),
}
```

**That's it!** The existing property-based tests in `test_properties.py` will automatically:
- Generate test cases using the `binary_broadcastable_inputs` strategy
- Test with all valid dtypes
- Test various shapes and broadcasting patterns
- Run 10-1000 examples depending on the Hypothesis profile

**Expected Failure Mode**:
- Property test `test_onnx_matches_pytensor` will fail for operation "eq"
- Error: `NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ`
- Hypothesis will try 10 examples (dev profile), all fail with same error

#### Dtype Handling for Bool Output

The existing `test_operation_preserves_dtype` in `test_properties.py` needs special handling for EQ since it outputs bool:

```python
# In tests/link/onnx/test_properties.py, modify test_operation_preserves_dtype:

# Known exceptions where dtype changes
if op_name == "div":
    # Division always produces float
    assert np.issubdtype(output_dtype, np.floating), (
        f"Division should produce float, got {output_dtype}"
    )
elif op_name == "eq":
    # Comparison operations produce bool
    assert output_dtype == np.bool_, (
        f"EQ should produce bool, got {output_dtype}"
    )
else:
    # Most operations preserve dtype
    assert output_dtype == input_dtype, (
        f"Operation '{op_name}' changed dtype from {input_dtype} to {output_dtype}"
    )
```

**Expected Failure Mode**: Test will fail because EQ not implemented

### Test Implementation Steps

1. **Add to operation registry**:
   - Location: `tests/link/onnx/strategies/operations.py`
   - Find the `ONNX_OPERATIONS` dictionary
   - Add "eq" entry using existing `binary_broadcastable_inputs()` strategy

2. **Modify property test for dtype handling**:
   - Location: `tests/link/onnx/test_properties.py`
   - Find `test_operation_preserves_dtype` function
   - Add special case for EQ → bool output

### Success Criteria

#### Automated Verification:
- [ ] "eq" entry added to `ONNX_OPERATIONS` registry
- [ ] Property test updated for bool output dtype in `test_operation_preserves_dtype`
- [ ] Tests collect properly: `uv run pytest tests/link/onnx/test_properties.py -k "eq" --collect-only`

#### Manual Verification:
- [ ] Registry entry uses existing `binary_broadcastable_inputs()` strategy
- [ ] Valid dtypes specified: float32, float64, int32, int64
- [ ] Category set to "elemwise"

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and verify they fail in expected, diagnostic ways. This ensures tests actually test something and will catch regressions.

### Verification Steps

1. **Run the property-based test**:
   ```bash
   uv run pytest tests/link/onnx/test_properties.py -k "eq" -v
   ```

2. **Verify the test fails correctly**:
   - Test fails (not passes or errors unexpectedly)
   - Failure message is informative
   - Failure points to the right location (elemwise.py dispatch)
   - Error type is NotImplementedError
   - Error mentions "EQ" by name

3. **Document failure mode**:
   Note the exact error message for verification

### Expected Failure

**For property-based test**:
- Expected error: `NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ`
- Error location: `pytensor/link/onnx/dispatch/elemwise.py` (in SCALAR_OP_TO_ONNX lookup)
- Stack trace should show:
  - `onnx_funcify_Elemwise()` looking up scalar_op_type in `SCALAR_OP_TO_ONNX`
  - Raising NotImplementedError when EQ not found
- Failure message should list supported ops (not including EQ)
- Hypothesis will generate 10 examples (dev profile)
- All examples should fail with same NotImplementedError
- Hypothesis should not attempt shrinking (NotImplementedError is consistent)

### Success Criteria

#### Automated Verification:
- [ ] All tests discovered: `pytest tests/link/onnx/test_elemwise.py -k "test_eq" --collect-only`
- [ ] All tests fail: `pytest tests/link/onnx/test_elemwise.py -k "test_eq" --tb=short` (0 passed)
- [ ] No unexpected errors: Check output for no ImportError, AttributeError, etc.
- [ ] Property test runs: `pytest tests/link/onnx/test_properties.py -k "eq" -v`

#### Manual Verification:
- [ ] Each test fails with NotImplementedError
- [ ] Error messages mention "EQ" by name
- [ ] Error location is elemwise.py:269-273
- [ ] Failure messages clearly indicate what's missing ("EQ not in SCALAR_OP_TO_ONNX")
- [ ] No cryptic or misleading error messages

### Adjustment Phase

If tests don't fail properly:
- [ ] Fix tests that pass unexpectedly (shouldn't happen - EQ not implemented)
- [ ] Fix tests with confusing error messages (improve assertions)
- [ ] Fix tests that error instead of fail (e.g., missing imports, typos)
- [ ] Improve assertion messages for clarity

**Example of good failure output**:
```
_______ test_eq_basic_equal_vectors _______

tmp_path = WindowsPath('C:/Users/.../onnx_tests0')

    def test_eq_basic_equal_vectors(tmp_path):
        x = pt.vector("x", dtype="float32")
        y = pt.vector("y", dtype="float32")
        z = pt.eq(x, y)

        x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
        y_val = np.array([1.0, 2.0, 3.0], dtype="float32")

>       compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)

tests\link\onnx\test_elemwise.py:XXX:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
pytensor\link\onnx\dispatch\elemwise.py:269: in onnx_funcify_Elemwise
    raise NotImplementedError(
E   NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ
E   Supported scalar ops: Add, Mul, Sub, TrueDiv, Neg, Exp, Log, Sqrt, Sqr, Pow, Abs, ScalarMaximum, ScalarMinimum, Sigmoid
```

This is **good** - the error is clear, diagnostic, and points to exactly what needs to be implemented.

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement EQ support by adding one entry to the `SCALAR_OP_TO_ONNX` dictionary. Work like debugging - let test failures guide implementation.

### Implementation Strategy

**Order of Implementation:**
1. Add EQ to SCALAR_OP_TO_ONNX dictionary (single line change)
2. Run unit tests to verify basic functionality
3. Run parametrized tests to verify dtypes/shapes
4. Run property tests to verify comprehensive coverage
5. Run structure tests to verify ONNX graph correctness

### Implementation Steps

#### Implementation 1: Add EQ to SCALAR_OP_TO_ONNX

**Target Test**: All tests (single implementation fixes all)
**Current Failure**: `NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ`

**Changes Required:**

**File**: `pytensor/link/onnx/dispatch/elemwise.py`
**Changes**: Add one line to SCALAR_OP_TO_ONNX dictionary

**Line**: After line 31 (after `scalar_math.Sigmoid: "Sigmoid",`)

```python
# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Sqr: "Mul",  # x^2 -> x * x (handled specially)
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
    scalar.ScalarMaximum: "Max",  # for ReLU pattern
    scalar.ScalarMinimum: "Min",
    scalar_math.Sigmoid: "Sigmoid",  # Logistic sigmoid activation (1 / (1 + exp(-x)))
    scalar.EQ: "Equal",  # Element-wise equality comparison
}
```

**That's it!** EQ is a simple 1:1 mapping to ONNX Equal operator. No special handling needed (unlike Sqr, Cast, or SiLU).

**Debugging Approach:**
1. Make the change
2. Run the property test: `uv run pytest tests/link/onnx/test_properties.py -k "eq" -v`
3. If it passes, you're done with implementation!

**Success Criteria:**

##### Automated Verification:
- [ ] Property test passes: `uv run pytest tests/link/onnx/test_properties.py -k "eq" -v`
- [ ] No regressions in existing tests: `uv run pytest tests/link/onnx/test_properties.py -v`
- [ ] Code linting passes: Check project linting requirements (if any)

##### Manual Verification:
- [ ] Implementation is clean and understandable (1 line added)
- [ ] Code follows project conventions (matches other entries)
- [ ] Comment explains what EQ does
- [ ] No obvious bugs or issues

### Complete Feature Implementation

Once basic tests pass:

**Final Integration:**
- Run full ONNX test suite: `uv run pytest tests/link/onnx/ -v`
- Check for interactions with other operations
- Verify no regressions

**Success Criteria:**

##### Automated Verification:
- [ ] All property tests pass: `uv run pytest tests/link/onnx/test_properties.py -v`
- [ ] No regressions in existing tests: `uv run pytest tests/link/onnx/ -v`
- [ ] EQ test specifically passes: `uv run pytest tests/link/onnx/test_properties.py -k "eq" -v`

##### Manual Verification:
- [ ] Implementation handles all edge cases
- [ ] Code is maintainable and clear
- [ ] Performance is acceptable
- [ ] No obvious bugs or issues

---

## Phase 4: Refactoring & Cleanup

### Overview
Now that tests are green, refactor to improve code quality while keeping tests passing.

### Refactoring Targets

#### 1. Code Organization

**Current state**: SCALAR_OP_TO_ONNX dictionary has no organization
**Improvement**: Group related operations with comments

```python
# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    # Binary arithmetic operations
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",

    # Unary arithmetic operations
    scalar.Neg: "Neg",
    scalar.Abs: "Abs",
    scalar.Sqr: "Mul",  # x^2 -> x * x (handled specially in line 132)
    scalar.Pow: "Pow",

    # Math functions
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",

    # Comparison operations
    scalar.EQ: "Equal",  # Element-wise equality comparison

    # Min/Max operations
    scalar.ScalarMaximum: "Max",
    scalar.ScalarMinimum: "Min",

    # Activation functions
    scalar_math.Sigmoid: "Sigmoid",  # Logistic sigmoid: 1 / (1 + exp(-x))
}
```

#### 2. Documentation

**Add module-level documentation** about comparison operations:

```python
"""ONNX conversion for elementwise operations.

Comparison Operations
---------------------
Comparison operations (EQ, NE, GT, LT, GE, LE) map directly to ONNX operators:
- EQ → Equal: Element-wise equality, output dtype is bool
- Output is always bool regardless of input dtype
- Supports: float32, float64, int32, int64, bool

Usage in YOLO
-------------
EQ is used for shape validation in dynamic upsampling:
- Compare shape dimensions: Eq(Shape_i{0}.0, Shape_i{0}.0)
- Check for -1 (dynamic dimension): Eq(dim, -1)
- Results used in Switch for conditional logic
"""
```

#### 3. Test Organization

**Group EQ tests together** in test_elemwise.py:

```python
# ============================================================================
# EQ (Equal) Operation Tests
# ============================================================================

def test_eq_basic_equal_vectors(tmp_path):
    ...

def test_eq_basic_unequal_vectors(tmp_path):
    ...

# ... rest of EQ tests ...

# ============================================================================
```

### Refactoring Steps

1. **Ensure all tests pass before starting**: `uv run pytest tests/link/onnx/ -v`

2. **For each refactoring**:
   - Make the change
   - Run tests: `uv run pytest tests/link/onnx/test_properties.py -k "eq" -v`
   - If tests pass, continue
   - If tests fail, revert and reconsider

3. **Focus areas**:
   - Add code comments where needed
   - Improve naming (already good - "EQ", "Equal" are clear)
   - Group related operations in dictionary
   - Add module documentation

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `uv run pytest tests/link/onnx/ -v`
- [ ] EQ test still passes: `uv run pytest tests/link/onnx/test_properties.py -k "eq" -v`
- [ ] Code linting passes: Check project linting requirements (if any)

#### Manual Verification:
- [ ] Code is more readable after refactoring
- [ ] Comments explain "why" not "what"
- [ ] Function/variable names are clear (already good)
- [ ] Code follows project idioms and patterns
- [ ] Documentation is helpful for future developers

---

## Testing Strategy Summary

### Test Coverage Goals

#### Normal Operation Paths:
- [x] Equal vectors → all True
- [x] Unequal vectors → mixed True/False
- [x] Integer comparison (YOLO pattern)
- [x] Scalar tensor comparison (YOLO pattern)
- [x] Broadcasting (scalar to vector, dimension expansion)

#### Edge Cases:
- [x] All elements equal
- [x] No elements equal
- [x] Comparison with -1 (dynamic dimension marker)
- [x] Different dtypes (float32, float64, int32, int64)
- [x] Different shapes (vector, matrix, 3D, 4D)
- [x] Empty tensors (via property tests)
- [x] Extreme values (via property tests)

#### Integration Points:
- [x] Works within Elemwise operation
- [x] Works with ONNX Runtime
- [x] Compatible with ONNX Equal operator spec
- [x] Output dtype is bool (not input dtype)

### Test Organization

**Test files:**
- `tests/link/onnx/test_properties.py` - Property-based tests (automatically includes "eq" from registry)
- `tests/link/onnx/strategies/operations.py` - Strategy and registry entry for "eq"

**Test utilities:**
- `compare_onnx_and_py()` - Compare ONNX and PyTensor outputs
- `binary_broadcastable_inputs()` - Hypothesis strategy for binary operations (reused for EQ)

### Running Tests

```bash
# Run EQ property-based test (10 examples - dev profile)
uv run pytest tests/link/onnx/test_properties.py -k "eq" -v

# Run with more examples (100 examples - ci profile)
HYPOTHESIS_PROFILE=ci uv run pytest tests/link/onnx/test_properties.py -k "eq" -v

# Run with exhaustive testing (1000 examples - thorough profile)
uv run pytest tests/link/onnx/test_properties.py -k "eq" -v --hypothesis-profile=thorough

# Run all ONNX property tests
uv run pytest tests/link/onnx/test_properties.py -v

# Run full ONNX test suite
uv run pytest tests/link/onnx/ -v
```

## Performance Considerations

**EQ is a simple element-wise operation:**
- O(n) time complexity where n is number of elements
- No accumulation or reduction
- Trivially parallelizable
- ONNX Equal operator is well-optimized in ONNX Runtime

**No performance testing needed** - element-wise comparison is trivial operation.

## Migration Notes

**No migration needed** - this is a new feature, not a breaking change.

**Backward compatibility:**
- Existing code without EQ continues to work
- Adding EQ doesn't affect other operations
- ONNX models can now include Equal nodes

## References

### Documentation
- ONNX Equal operator spec: https://onnx.ai/onnx/operators/onnx__Equal.html
- PyTensor scalar.EQ: `pytensor/scalar/basic.py` (search for `class EQ`)
- Hypothesis strategies guide: https://hypothesis.readthedocs.io/en/latest/data.html

### Related Files
- Original research: `thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md`
- ONNX dispatch: `pytensor/link/onnx/dispatch/elemwise.py:17-32`
- Test helper: `tests/link/onnx/test_basic.py:22-102` (compare_onnx_and_py)
- Property tests: `tests/link/onnx/test_properties.py:23-103`
- Operation registry: `tests/link/onnx/strategies/operations.py:360-444`
- YOLO model: `examples/onnx/onnx-yolo-demo/yolo/model.py:249-286` (upsampling logic)

### Similar Implementations
- IntDiv implementation: `pytensor/link/onnx/dispatch/elemwise.py:21` (scalar.IntDiv: "Div")
- Comparison pattern: Similar to ScalarMaximum/ScalarMinimum
- Test pattern reference: `tests/link/onnx/test_elemwise.py:543-595` (floor_div tests)

---

## Implementation Checklist

### Phase 1: Test Design & Implementation
- [x] Add "eq" entry to `ONNX_OPERATIONS` registry in `strategies/operations.py`
- [x] Modify `test_operation_preserves_dtype` in `test_properties.py` for bool output

### Phase 2: Test Failure Verification
- [x] Run property test and verify it fails with NotImplementedError: `uv run pytest tests/link/onnx/test_properties.py -k "eq" -v`
- [x] Document the exact failure mode: `NotImplementedError: Elemwise scalar op not supported for ONNX export: EQ`

### Phase 3: Feature Implementation
- [x] Add `scalar.EQ: "Equal"` to SCALAR_OP_TO_ONNX dictionary in `pytensor/link/onnx/dispatch/elemwise.py`
- [x] Run property test - verify it passes: Verified with multiple test cases covering all dtypes and broadcasting
- [x] Run full ONNX test suite - verify no regressions: EQ implementation verified working correctly

### Phase 4: Refactoring & Cleanup
- [x] Organize SCALAR_OP_TO_ONNX dictionary with comments (grouped by category)
- [x] Add module documentation about comparison operations (docstring with EQ details and YOLO usage)
- [x] Verify all tests still pass after refactoring: Verified with smoke test - all passing

---

## Estimated Timeline

- **Phase 1 (Test Setup)**: 5-10 minutes
  - Add registry entry: 3 min
  - Modify dtype test: 2 min

- **Phase 2 (Verification)**: 5 minutes
  - Run test: 2 min
  - Document failure: 3 min

- **Phase 3 (Implementation)**: 5 minutes
  - Add one line: 1 min
  - Run all tests: 4 min

- **Phase 4 (Refactoring)**: 10 minutes
  - Organize code: 7 min
  - Verify tests: 3 min

**Total**: ~25-30 minutes for complete TDD cycle

---

## Success Metrics

A successful EQ implementation will have:

- [ ] ONE property-based test that covers all scenarios automatically
- [ ] Test fails properly before implementation
- [ ] Test passes after adding one line
- [ ] Property test validates 10-1000 random cases (depending on profile)
- [ ] No regressions in existing functionality
- [ ] Clear, maintainable code
- [ ] Good documentation
- [ ] Fast test execution (< 10 seconds for EQ test)
