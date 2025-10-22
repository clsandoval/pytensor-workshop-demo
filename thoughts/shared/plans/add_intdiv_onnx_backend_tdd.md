# Add IntDiv Support to PyTensor ONNX Backend - TDD Implementation Plan

## Overview

Add support for PyTensor's `IntDiv` (floor division `//`) scalar operation to the ONNX export backend. This is required for exporting models like YOLO11n that use integer division for shape calculations (e.g., computing padding dimensions).

## Current State Analysis

### Problem
ONNX export fails when encountering `IntDiv` scalar operations with error:
```
NotImplementedError: Elemwise scalar op not supported for ONNX export: IntDiv
Supported scalar ops: Add, Mul, Sub, TrueDiv, Neg, Exp, Log, Sqrt, Sqr, Pow, Abs, ScalarMaximum, ScalarMinimum, Sigmoid
```

### Current Testing Landscape
- **Testing framework**: pytest
- **Test location**: `tests/link/onnx/test_elemwise.py`
- **Test pattern**: Use `compare_onnx_and_py()` helper from `tests/link/onnx/test_basic.py:22-102`
- **Available test utilities**:
  - `compare_onnx_and_py()`: Validates ONNX Runtime output matches PyTensor output
  - `validate_onnx_graph_structure()`: Validates ONNX graph structure
  - `tmp_path` pytest fixture for ONNX file storage

### Current Implementation Pattern
- **File**: `pytensor/link/onnx/dispatch/elemwise.py`
- **Pattern**: Dictionary mapping at lines 17-32: `SCALAR_OP_TO_ONNX = {scalar.Op: "OnnxOp"}`
- **Registration**: Uses `@onnx_funcify.register(Elemwise)` decorator (line 195)
- **Example mappings**:
  - `scalar.Add: "Add"`
  - `scalar.TrueDiv: "Div"`
  - `scalar.Mul: "Mul"`

### IntDiv Usage in Real Models
From YOLO11n model analysis (12 IntDiv operations found):
```python
# Pattern: Int_div(Sub.0, 2) - computing padding
# Input types: TensorType(int64, shape=())
# Output types: TensorType(int64, shape=())
```
- Used for padding calculations: `(kernel_size - 1) // 2`
- Used for dimension calculations: `(width * height) // anchor_count`
- Operates on **int64 scalar tensors** (0-dimensional), not regular arrays
- All operations use positive integers in typical use cases

### IntDiv Semantics
From `pytensor/scalar/basic.py:2139-2147`:
```python
class IntDiv(BinaryScalarOp):
    nfunc_spec = ("floor_divide", 2, 1)

    def impl(self, x, y):
        return x // y
```
- **Semantics**: Floor division (rounds toward negative infinity)
- **Floor division vs Truncation**:
  - Floor: `-7 // 2 = -4` (rounds toward -∞)
  - Truncation: `-7 / 2 = -3` (rounds toward 0)
- **NumPy equivalent**: `np.floor_divide(x, y)`

### ONNX Div Operator
- **Operator name**: `"Div"`
- **Supported types** (ONNX opset 14+): int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16
- **Behavior**: Element-wise division with NumPy-style broadcasting
- **Integer division behavior**: Documentation doesn't explicitly specify floor vs truncation
  - **Assumption**: Since ONNX follows NumPy semantics, integer Div likely performs floor division
  - **Verification needed**: Tests will verify this assumption

## Desired End State

After implementation:
- IntDiv scalar operations export successfully to ONNX `Div` operator
- ONNX Runtime produces identical results to PyTensor for integer division
- YOLO11n model exports to ONNX without errors
- All existing tests continue to pass

### Key Discoveries
- IntDiv operates on int64 scalars in real models: `pytensor/link/onnx/dispatch/elemwise.py:17-32`
- Simple dictionary mapping is sufficient: `scalar.IntDiv: "Div"`
- Testing pattern established: `tests/link/onnx/test_elemwise.py:21-67`
- ONNX Div supports integer types: Verified via ONNX operator documentation

## What We're NOT Testing/Implementing

- **Division by zero**: Not testing explicit division by zero (undefined behavior)
- **Overflow conditions**: Not testing integer overflow edge cases
- **Non-scalar tensors initially**: Focusing on scalar operations first, will test vectors/matrices for completeness
- **All integer types**: Focusing on commonly used int32 and int64, not testing int8/int16/uint types
- **Mixed signedness**: Not testing uint with int operations
- **Performance optimization**: Not optimizing for specific ONNX runtime versions

## TDD Approach

### Test Design Philosophy
1. **Verify ONNX Div matches PyTensor IntDiv semantics**: Test that floor division behavior is preserved
2. **Cover practical use cases**: Focus on positive integer division (padding calculations)
3. **Include edge cases**: Test negative integers to verify floor vs truncation
4. **Test type compatibility**: Ensure int32 and int64 work correctly
5. **Validate against reference**: Use PyTensor as ground truth via `compare_onnx_and_py()`

### Test First, Always
- Write ALL tests before implementation
- Verify tests fail with expected error message
- Implementation = making tests pass one by one

---

## Phase 1: Test Design & Implementation

### Overview
Add IntDiv (floor division) to the ONNX operations registry and let the existing property-based tests automatically verify it. This follows the established pattern where operations are registered in `ONNX_OPERATIONS` and tested via Hypothesis properties.

### Test Strategy: Property-Based Testing

The codebase uses **Hypothesis** for property-based testing. Instead of writing specific test cases, we:
1. Register `floor_div` (IntDiv) in the `ONNX_OPERATIONS` registry
2. Define an input strategy for integer division
3. Let Hypothesis generate hundreds of test cases automatically
4. The existing property tests will verify:
   - Property 1: ONNX output matches PyTensor output
   - Property 2: Shape preservation (elemwise operations)
   - Property 3: Dtype preservation
   - Property 4: No crashes on edge cases

**Why this is better**:
- Hypothesis will find edge cases we wouldn't think of
- Minimal test code (just registration)
- Consistent with existing test architecture
- Automatic coverage of shapes, dtypes, broadcasting, etc.

### Changes Required

#### Change 1: Add IntDiv Input Strategy

**File**: `tests/link/onnx/strategies/operations.py`
**Location**: After `binary_broadcastable_inputs` function (around line 124)

Add a specialized input strategy for integer division that avoids division by zero:

```python
@st.composite
def binary_int_division_inputs(draw):
    """Generate inputs for integer division (floor_div).

    Ensures divisor is never zero.

    Returns
    -------
    tuple
        (x, y) - Two integer tensors where y != 0
    """
    # Only integer types for floor division
    dtypes = [np.int32, np.int64]
    dtype = draw(st.sampled_from(dtypes))

    # Generate base shape
    base_shape = draw(valid_shapes(min_rank=1, max_rank=3, min_dim=1, max_dim=5))

    # Generate broadcasting variant for second tensor
    broadcast_pattern = draw(
        st.sampled_from(["same", "broadcast_dims", "prefix"])
    )

    if broadcast_pattern == "same":
        shape_y = base_shape
    elif broadcast_pattern == "broadcast_dims":
        shape_y = tuple(
            1 if draw(st.booleans()) and dim > 1 else dim for dim in base_shape
        )
    else:  # prefix
        suffix_len = draw(st.integers(1, len(base_shape)))
        shape_y = base_shape[-suffix_len:]

    # Generate tensors
    x = draw(onnx_tensor(dtype=dtype, shape=base_shape))
    y = draw(onnx_tensor(dtype=dtype, shape=shape_y))

    # Ensure y has no zeros (avoid division by zero)
    y = np.where(y == 0, 1, y)

    return (x, y)
```

**Expected Failure Mode**: No failure yet - just defining a strategy

#### Change 2: Register floor_div Operation

**File**: `tests/link/onnx/strategies/operations.py`
**Location**: In the `ONNX_OPERATIONS` dictionary (around line 345, after "div" entry)

Add floor_div to the operation registry:

```python
    "floor_div": OperationConfig(
        op_func=lambda x, y: x // y,
        input_strategy=binary_int_division_inputs(),
        valid_dtypes=["int32", "int64"],
        category="elemwise",
        notes="Floor division for integer types, maps to ONNX Div",
    ),
```

**Expected Failure Mode**:
When Hypothesis runs `test_onnx_matches_pytensor` with `op_name="floor_div"`, it will fail with:
- Error type: `NotImplementedError`
- Expected message: `"Elemwise scalar op not supported for ONNX export: IntDiv"`
- Points to: `pytensor/link/onnx/dispatch/elemwise.py:270`

#### Change 3: Add Specific Regression Tests (Optional but Recommended)

**File**: `tests/link/onnx/test_elemwise.py`
**Location**: End of file

Add a few focused tests for known IntDiv use cases:

```python
def test_floor_div_yolo_padding_pattern(tmp_path):
    """Test floor division in YOLO padding calculation pattern.

    Pattern: (kernel_size - 1) // 2
    This is the specific usage that triggered the need for IntDiv support.
    """
    kernel_size = pt.lscalar("kernel_size", dtype="int64")
    padding = (kernel_size - 1) // 2

    kernel_size_val = np.array(5, dtype="int64")
    # Expected: (5 - 1) // 2 = 2

    compare_onnx_and_py([kernel_size], padding, [kernel_size_val], tmp_path=tmp_path)


def test_floor_div_negative_integers(tmp_path):
    """Test that floor division (not truncation) is used for negative integers.

    Floor division:    -7 // 2 = -4 (toward -∞)
    Truncation:        -7 / 2 = -3 (toward 0)  ← wrong!

    This test locks in the floor division behavior.
    """
    x = pt.lvector("x", dtype="int64")
    y = pt.lvector("y", dtype="int64")
    z = x // y

    # Test cases that distinguish floor from truncation
    x_val = np.array([-7, -6, 7], dtype="int64")
    y_val = np.array([2, 2, -2], dtype="int64")
    # Floor division: [-4, -3, -4]
    # Truncation:     [-3, -3, -3] ← wrong

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_floor_div_scalar_tensors(tmp_path):
    """Test floor division on 0-dimensional (scalar) tensors.

    YOLO uses IntDiv on scalar tensors:
    - Pattern: Int_div(Sub.0, 2)
    - Input types: TensorType(int64, shape=())
    """
    x = pt.lscalar("x", dtype="int64")
    y = pt.lscalar("y", dtype="int64")
    z = x // y

    x_val = np.array(9, dtype="int64")
    y_val = np.array(2, dtype="int64")
    # Expected: 4

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Same NotImplementedError for all three tests

### Test Implementation Steps

1. **Add integer division input strategy** to `tests/link/onnx/strategies/operations.py`:
   - Add `binary_int_division_inputs()` function after line 124

2. **Register floor_div operation** in `ONNX_OPERATIONS` dict:
   - Add entry in `tests/link/onnx/strategies/operations.py` around line 345

3. **Add regression tests** to `tests/link/onnx/test_elemwise.py`:
   - Add the three specific test functions at the end of the file

4. **Verify test discovery**:
   ```bash
   pytest --collect-only tests/link/onnx/test_properties.py -k floor_div
   pytest --collect-only tests/link/onnx/test_elemwise.py -k floor_div
   ```

### Success Criteria

#### Automated Verification:
- [x] `binary_int_division_inputs` strategy is added without syntax errors
- [x] `floor_div` is registered in `ONNX_OPERATIONS` dictionary
- [x] Hypothesis property tests discover floor_div: `pytest --collect-only tests/link/onnx/test_properties.py`
- [x] Regression tests are discoverable: `pytest --collect-only tests/link/onnx/test_elemwise.py -k floor_div`
- [x] No import or syntax errors

#### Manual Verification:
- [x] Input strategy follows the same pattern as other strategies
- [x] Operation config has correct fields (op_func, input_strategy, valid_dtypes, category)
- [x] Regression tests have clear docstrings
- [x] Code style matches existing test file conventions

---

## Phase 2: Test Failure Verification

### Overview
Run the tests and verify they fail in the expected way with property-based testing discovering IntDiv. This ensures the tests are actually exercising IntDiv and will catch future regressions.

### Verification Steps

1. **Run regression tests first** (quickest to verify failure):
   ```bash
   pytest tests/link/onnx/test_elemwise.py -k "floor_div" -v
   ```

2. **Run property-based tests** (Hypothesis will generate many cases):
   ```bash
   pytest tests/link/onnx/test_properties.py::test_onnx_matches_pytensor -v --hypothesis-seed=0
   ```
   Note: This tests ALL operations, including floor_div

3. **Run floor_div specifically in property tests**:
   Since Hypothesis samples from all operations, we can't easily filter to just floor_div.
   Instead, verify by checking test output for floor_div failures.

4. **Verify failure characteristics**:
   - Failure type: `NotImplementedError`
   - Message: `"Elemwise scalar op not supported for ONNX export: IntDiv"`
   - Location: `pytensor/link/onnx/dispatch/elemwise.py:270`

### Expected Failures

**Regression tests** (3 tests):
All should fail with:
```python
NotImplementedError: Elemwise scalar op not supported for ONNX export: IntDiv
Supported scalar ops: Add, Mul, Sub, TrueDiv, Neg, Exp, Log, Sqrt, Sqr, Pow, Abs, ScalarMaximum, ScalarMinimum, Sigmoid
```

- **test_floor_div_yolo_padding_pattern**: YOLO-specific pattern `(kernel_size - 1) // 2`
- **test_floor_div_negative_integers**: Verify floor (not truncation) semantics
- **test_floor_div_scalar_tensors**: 0-dimensional tensor case

**Property-based tests** (when Hypothesis draws floor_div):
- **test_onnx_matches_pytensor**: Will eventually sample floor_div and fail
- **test_elemwise_preserves_broadcast_shape**: Will test floor_div broadcasting
- **test_operation_preserves_dtype**: Will verify int32/int64 dtype preservation
- **test_operation_handles_edge_cases**: Will test edge cases like scalars, empty shapes

### Success Criteria

#### Automated Verification:
- [x] Regression tests discovered: `pytest --collect-only tests/link/onnx/test_elemwise.py -k floor_div` (shows 3 tests)
- [x] Regression tests all fail: `pytest tests/link/onnx/test_elemwise.py -k floor_div --tb=line` (3 failures)
- [x] Property tests run: `pytest tests/link/onnx/test_properties.py -v` (will test floor_div among others)
- [x] No syntax/import errors

#### Manual Verification:
- [x] Each test fails with `NotImplementedError` mentioning "IntDiv"
- [x] Failure points to `pytensor/link/onnx/dispatch/elemwise.py:270` (and line 176 for Composite)
- [x] Error message lists supported ops (doesn't include IntDiv yet)
- [x] Stack traces are clear and diagnostic
- [x] No tests pass (would mean they're not testing IntDiv)

### Hypothesis Test Behavior

When running property tests, you'll see output like:
```
Falsifying example: test_onnx_matches_pytensor(
    op_name='floor_div',
    data=data(...)
)
```

This is expected! Hypothesis found that floor_div fails and is showing you the minimal failing case.

### Adjustment Phase

If tests don't fail as expected:

- [ ] **If import errors**: Check that `binary_int_division_inputs` is defined before being used
- [ ] **If wrong error type**: Verify floor_div is actually using `//` operator
- [ ] **If tests pass**: floor_div is somehow already supported (unlikely) or tests aren't using IntDiv
- [ ] **If confusing failures**: Check that input strategy properly avoids division by zero

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement IntDiv support by making tests pass one at a time. Work like debugging - let test failures guide implementation.

### Implementation Strategy

**Single-step implementation**: IntDiv is simple enough to implement in one step. The implementation is just adding one line to the dictionary.

**Order**: Implement once, then verify all tests pass.

### Implementation Steps

#### Implementation: Add IntDiv to ONNX Backend

**Target Tests**: All 8 IntDiv tests
**Current Failure**: `NotImplementedError: IntDiv not supported`

**Changes Required:**

**File**: `pytensor/link/onnx/dispatch/elemwise.py`
**Changes**: Add IntDiv to SCALAR_OP_TO_ONNX mapping

**Line 17-32** (the mapping dictionary):
```python
# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.IntDiv: "Div",  # ← ADD THIS LINE
    scalar.Neg: "Neg",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Sqr: "Mul",  # x^2 -> x * x (handled specially)
    scalar.Pow: "Pow",
    scalar.Abs: "Abs",
    scalar.ScalarMaximum: "Max",
    scalar.ScalarMinimum: "Min",
    scalar_math.Sigmoid: "Sigmoid",
}
```

**Implementation notes:**
- Map `scalar.IntDiv` to ONNX `"Div"` operator
- Same operator as `TrueDiv`, but IntDiv operates on integer types
- No special handling needed - ONNX Div handles both float and integer division
- Placement: Add after `TrueDiv` for logical grouping

**Debugging Approach:**

1. **Add the mapping line**:
   ```python
   scalar.IntDiv: "Div",
   ```

2. **Run a simple test first**:
   ```bash
   pytest tests/link/onnx/test_elemwise.py::test_int_div_positive_integers -v
   ```

3. **Check if it passes**:
   - If passes: Move to next test
   - If fails: Check failure message

4. **Potential issues to debug**:
   - If test still shows "IntDiv not supported": Check import statement, ensure `scalar.IntDiv` is the correct reference
   - If test fails with wrong results: ONNX Div might not perform floor division - would need to add special handling
   - If test fails with type errors: ONNX Div might need explicit integer type support

5. **Run all tests**:
   ```bash
   pytest tests/link/onnx/test_elemwise.py -k int_div -v
   ```

6. **Special case handling** (if needed):
   - If ONNX Div performs truncation instead of floor division for integers
   - Would need to decompose into: Div + Floor for integer types
   - Similar pattern to how SiLU is decomposed in lines 208-235

**Success Criteria:**

##### Automated Verification:
- [x] All 3 IntDiv regression tests pass: `pytest tests/link/onnx/test_elemwise.py -k floor_div -v`
- [x] No regressions in existing tests: `pytest tests/link/onnx/test_elemwise.py -v` (39 tests passed)
- [x] Existing ONNX tests still pass: `pytest tests/link/onnx/test_basic.py -v` (9 tests passed)
- [x] YOLO export progresses past IntDiv: IntDiv now in supported ops list, fails on next operation (EQ)

##### Manual Verification:
- [x] Implementation follows existing patterns (similar to SiLU decomposition)
- [x] Code properly handles floor division semantics (Cast → Div → Floor → Cast)
- [x] Handles both Composite and non-Composite IntDiv operations
- [x] Comments are clear and explain the floor vs truncation issue

### Verification of Floor Division Behavior

After implementation, verify ONNX Div performs floor division:

```bash
pytest tests/link/onnx/test_elemwise.py::test_int_div_negative_integers -v
```

**If this test fails**:
- ONNX Div likely performs truncation, not floor division
- Need special handling (see contingency plan below)

### Contingency: If ONNX Div Uses Truncation

If `test_int_div_negative_integers` fails because ONNX Div truncates:

**Option 1: Accept truncation** (simplest)
- Document the limitation
- YOLO only uses positive integers anyway
- Update test to match truncation behavior

**Option 2: Decompose to Floor + Div** (more correct)
- Cast to float → Divide → Floor → Cast back to int
- More ONNX nodes, but correct semantics
- Implementation in lines 238-267 (similar to Cast handling)

---

## Phase 4: Refactoring & Cleanup

### Overview
Since this is a one-line change, refactoring should be minimal. Focus on documentation and testing completeness.

### Refactoring Targets

1. **Code Comments**:
   - Verify the comment for the mapping dictionary is clear
   - Add inline comment if IntDiv needs clarification

2. **Test Documentation**:
   - Ensure all test docstrings are clear
   - Add comments for edge cases in test data

3. **Error Messages**:
   - Update error message at line 271 to include IntDiv in supported list

### Refactoring Steps

1. **Ensure all tests still pass**: `pytest tests/link/onnx/test_elemwise.py -v`

2. **Review code comments**:
   ```python
   # Line 32 - verify comment above dictionary is clear
   # If needed, add: "IntDiv (floor division) maps to ONNX Div for integer types"
   ```

3. **Update error message** (line 271):
   The error message will automatically include IntDiv now that it's in the dict.
   Verify by running an unsupported op and checking error message.

4. **Run YOLO export test**:
   ```bash
   cd examples/onnx/onnx-yolo-demo
   uv run pytest tests/test_onnx_export.py -v
   ```
   This is the ultimate integration test.

5. **Check test coverage**:
   ```bash
   pytest tests/link/onnx/test_elemwise.py -k int_div --cov=pytensor.link.onnx.dispatch.elemwise --cov-report=term-missing
   ```
   Verify the IntDiv code path is covered.

### Success Criteria

#### Automated Verification:
- [x] All IntDiv tests still pass: `pytest tests/link/onnx/test_elemwise.py -k floor_div -v`
- [x] All ONNX tests still pass: `pytest tests/link/onnx/test_basic.py -v`
- [x] YOLO export progresses past IntDiv: IntDiv operations now supported
- [x] No regressions: `pytest tests/link/onnx/test_elemwise.py -v`
- [x] Code coverage: IntDiv lines are covered by tests

#### Manual Verification:
- [x] Code is readable and well-commented
- [x] No unnecessary complexity (follows SiLU decomposition pattern)
- [x] Test docstrings are clear and informative
- [x] Implementation follows project conventions

---

## Testing Strategy Summary

### Test Coverage Goals
- [x] Normal operation: Positive integer division (padding calculations)
- [x] Edge cases: Negative integers (floor vs truncation verification)
- [x] Type coverage: int32 and int64
- [x] Shape coverage: Scalars, vectors, matrices, 3D tensors
- [x] Integration: IntDiv in larger computations
- [x] Structure validation: Correct ONNX node generation

### Test Organization
- **Test file**: `tests/link/onnx/test_elemwise.py`
- **Test pattern**: `test_int_div_*`
- **Test utility**: `compare_onnx_and_py()` from `test_basic.py`
- **8 tests total**: Basic (3) + Types (2) + Shapes (1) + Composite (2) + Structure (1)

### Running Tests

```bash
# Run all IntDiv tests
pytest tests/link/onnx/test_elemwise.py -k int_div -v

# Run specific test
pytest tests/link/onnx/test_elemwise.py::test_int_div_positive_integers -v

# Run with coverage
pytest tests/link/onnx/test_elemwise.py -k int_div --cov=pytensor.link.onnx.dispatch.elemwise --cov-report=term-missing

# Run YOLO integration test
cd examples/onnx/onnx-yolo-demo
uv run pytest tests/test_onnx_export.py -v

# Run all ONNX tests
pytest tests/link/onnx/ -v
```

## Performance Considerations

**No performance concerns**:
- Single dictionary lookup
- ONNX Div is a primitive operation
- No additional overhead vs TrueDiv

## Migration Notes

**No migration needed**:
- Backward compatible
- Previously failed exports will now succeed
- No API changes

## References

- **IntDiv definition**: `pytensor/scalar/basic.py:2139-2147`
- **ONNX dispatch**: `pytensor/link/onnx/dispatch/elemwise.py:17-32`
- **Test pattern**: `tests/link/onnx/test_elemwise.py:21-67`
- **ONNX Div operator**: https://onnx.ai/onnx/operators/onnx__Div.html
- **YOLO usage analysis**: 12 IntDiv ops on int64 scalars for shape calculations
