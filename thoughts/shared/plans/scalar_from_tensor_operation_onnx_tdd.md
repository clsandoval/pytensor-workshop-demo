---
date: 2025-10-18T23:00:00-00:00
author: Claude
related_research: thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md
operation: ScalarFromTensor
git_branch: onnx-workshop-demo
repository: pytensor
status: ready_for_implementation
test_framework: pytest + hypothesis
dependencies: EQ and Switch operations (ScalarFromTensor uses their outputs)
---

# ScalarFromTensor Operation ONNX Export - TDD Implementation Plan

## Overview

Implement ONNX export support for the ScalarFromTensor operation using Test-Driven Development with Hypothesis property-based testing. ScalarFromTensor extracts a scalar value from a 0-dimensional tensor or 1-element tensor, and is used 8 times in the YOLO11n model to convert boolean comparison results to scalar values for assertions and conditional checks.

**TDD Approach**: Write comprehensive tests (unit + property-based) first, verify they fail with diagnostic messages, then implement the feature by making tests pass.

## Setup

Before starting, ensure you have the dev version of PyTensor installed:

```bash
# Install the dev version from the repository root (parent directory)
uv pip install -e .
```

This ensures you're testing against the development version of PyTensor, not the PyPI release.

## Current State Analysis

### What Exists Now

**ONNX Dispatch System** (`pytensor/link/onnx/dispatch/`):
- Top-level Op dispatch via `@onnx_funcify.register(OpClass)`
- Shape operations dispatched in `shape.py`
- Currently has: Shape_i, Reshape, DimShuffle, MakeVector, AllocEmpty
- ScalarFromTensor is **NOT** registered

**Test Infrastructure**:
- Property-based test framework: `tests/link/onnx/test_properties.py`
- Shape operation tests: `tests/link/onnx/test_shape.py`
- Core strategies: `tests/link/onnx/strategies/core.py`
- Test helper: `compare_onnx_and_py()` in `tests/link/onnx/test_basic.py:22-102`

**ScalarFromTensor Operation in PyTensor**:
- Location: `pytensor.tensor.basic.ScalarFromTensor`
- Purpose: Convert 0-D tensor to scalar type
- Input: TensorType with ndim=0 (scalar tensor)
- Output: Scalar value
- Used after EQ operations to get scalar boolean for conditionals

### What's Missing

1. No `@onnx_funcify.register(ScalarFromTensor)` dispatch function
2. No test coverage for ScalarFromTensor
3. No Hypothesis strategies for scalar tensor operations
4. ScalarFromTensor not imported in dispatch/shape.py

### Key Constraints Discovered

**ONNX Representation**:
- ONNX does **NOT** distinguish between scalars and 0-D tensors
- A scalar in ONNX **IS** a 0-dimensional tensor
- No conversion needed in most cases → Use `Identity` node
- Alternative: Use `Squeeze` to remove all dimensions (but 0-D has no dimensions to remove)
- **Decision**: Use `Identity` node (simplest, clearest intent)

**PyTensor ScalarFromTensor Behavior**:
- Input: 0-D tensor (shape `()`)
- Output: Scalar type (not a tensor)
- Type conversion only (tensor → scalar), no data change
- Validates input has ndim=0 at graph construction time

**YOLO Usage Patterns**:
```
ScalarFromTensor(Eq.0)  # Convert boolean comparison to scalar boolean
```

Pattern: `scalar_bool = ScalarFromTensor(tensor_bool)` for use in assertions/conditions

**Test Strategy Considerations**:
- 0-D tensors are rare edge cases
- Must test with bool, int, and float dtypes
- Integration with EQ operation important (main use case)
- Hypothesis needs strategy for generating 0-D tensors

## Desired End State

After implementation:
- ScalarFromTensor operation exports to ONNX Identity node
- Property-based tests validate correctness with 10-100 random test cases
- Tests cover: 0-D tensors with various dtypes, integration with EQ
- ONNX Runtime produces same results as PyTensor for all inputs
- Tests fail gracefully with diagnostic messages before implementation

## What We're NOT Testing/Implementing

- 1-element tensors (shape `(1,)`) - different from scalar tensors (shape `()`)
- Multi-element tensors (should fail at graph construction time in PyTensor)
- Gradient computation through ScalarFromTensor
- Type conversion beyond tensor→scalar (PyTensor handles this)
- N-dimensional tensor squeezing (different operation)

---

## TDD Approach

### Test Design Philosophy

**Property-Based Testing Strategy**:
1. **Correctness**: ONNX Identity output must match PyTensor ScalarFromTensor
2. **Dtype Preservation**: Output dtype same as input dtype
3. **Shape Handling**: Input must be 0-D, output is scalar value
4. **Edge Cases**: Various dtypes (bool, int, float), extreme values

**Why Hypothesis?**
- Generates random 0-D tensors with various dtypes
- Tests with extreme values (large floats, negative integers, etc.)
- Validates both true/false for boolean scalars
- Ensures no crashes with edge cases

**Test Structure**:
- Unit tests: Basic ScalarFromTensor with concrete examples
- Property tests: Automated testing with random 0-D tensors
- Integration tests: ScalarFromTensor after EQ operation (YOLO pattern)
- Structure tests: Validate ONNX graph has Identity node

**Key Difference from EQ/Switch**:
- ScalarFromTensor is a **top-level Op**, not a scalar op in Elemwise
- Requires `@onnx_funcify.register()` decorator, not dictionary entry
- Tests go in `test_shape.py` (shape operations), not `test_elemwise.py`

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define ScalarFromTensor behavior completely. These tests will fail initially with `NotImplementedError: No ONNX conversion available for: ScalarFromTensor`.

### Test Categories

#### 1. Property-Based Tests with Hypothesis - Core Functionality

**Test File**: `tests/link/onnx/test_shape.py`
**Purpose**: Test ScalarFromTensor with random 0-D tensors across all dtypes and edge cases

**First: Create Strategy for 0-D Tensors**

**File**: `tests/link/onnx/strategies/core.py`

Add strategy for generating scalar tensors:

```python
@st.composite
def scalar_tensor(draw, dtype=None, value_range=None):
    """Generate 0-dimensional (scalar) tensor.

    Parameters
    ----------
    dtype : numpy dtype or None
        Tensor dtype. If None, randomly chosen from onnx_dtypes()
    value_range : tuple or None
        (min, max) for numeric values. If None, uses safe defaults

    Returns
    -------
    numpy.ndarray
        0-D tensor (shape ())

    Examples
    --------
    >>> scalar_tensor().example()  # doctest: +SKIP
    array(3.14, dtype=float32)

    >>> scalar_tensor(dtype=np.bool_).example()  # doctest: +SKIP
    array(True)
    """
    if dtype is None:
        dtype = draw(onnx_dtypes())

    shape = ()  # 0-D tensor

    # Generate value based on dtype
    if dtype == np.bool_ or dtype == "bool":
        value = draw(st.booleans())
        return np.array(value, dtype=np.bool_)
    elif np.issubdtype(dtype, np.floating):
        if value_range is None:
            value_range = (-1e3, 1e3)
        value = draw(st.floats(
            min_value=value_range[0],
            max_value=value_range[1],
            allow_nan=False,
            allow_infinity=False,
        ))
        return np.array(value, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        if value_range is None:
            value_range = (-100, 100)
        value = draw(st.integers(min_value=value_range[0], max_value=value_range[1]))
        return np.array(value, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
```

##### Test: `test_scalar_from_tensor_property`
**Purpose**: Property test - ScalarFromTensor preserves value for ANY 0-D tensor
**Test Data**: Random 0-D tensors (all dtypes, edge cases included automatically)
**Expected Behavior**: Value preserved exactly for all inputs
**Assertions**: ONNX output matches PyTensor output

```python
from hypothesis import given, settings
from tests.link.onnx.strategies.core import scalar_tensor

@settings(deadline=None, max_examples=100)
@given(
    data=st.data(),
)
def test_scalar_from_tensor_property(tmp_path, data):
    """Property: ScalarFromTensor preserves value for any 0-D tensor.

    For ANY 0-D tensor (regardless of dtype or value), ScalarFromTensor
    should preserve the value exactly.

    This property test automatically covers:
    - All dtypes (bool, float32, float64, int32, int64)
    - Edge cases (zero, negative values, extreme values)
    - Boolean True/False
    - Normal values

    Hypothesis will generate 100 random test cases covering the entire
    input space, which is more comprehensive than manual edge case tests.
    """
    from pytensor.tensor.basic import scalar_from_tensor

    # Generate random 0-D tensor
    x_val = data.draw(scalar_tensor())
    dtype = x_val.dtype

    # Create symbolic computation
    x = pt.scalar("x", dtype=dtype)
    y = scalar_from_tensor(x)

    # Compare ONNX and PyTensor
    compare_onnx_and_py([x], y, [x_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: `"No ONNX conversion available for: ScalarFromTensor"` or similar
- Points to: Dispatch system unable to find registered converter
- Hypothesis will report: "Falsifying example" with the specific input that failed

**Why Property Tests Instead of Edge Case Tests?**

Property-based testing with Hypothesis is superior for this operation because:

1. **Comprehensive Coverage**: Automatically tests 100+ random cases including edge cases (zero, negative, extreme values, True/False)
2. **Less Code**: One property test replaces 7+ individual edge case tests
3. **Better Edge Case Discovery**: Hypothesis finds edge cases we might not think of
4. **Clearer Intent**: States the invariant "ScalarFromTensor preserves all values" directly
5. **Automatic Shrinking**: When a test fails, Hypothesis finds the minimal failing example
6. **Future-Proof**: Automatically tests new dtypes if added to `onnx_dtypes()`

#### 2. Integration Tests - With EQ Operation

**Test File**: `tests/link/onnx/test_shape.py`
**Purpose**: Test ScalarFromTensor in YOLO pattern (after EQ comparison)

##### Test: `test_scalar_from_tensor_after_eq`
**Purpose**: Test ScalarFromTensor integrated with EQ operation
**Test Data**: Two scalars compared with EQ, result converted with ScalarFromTensor
**Expected Behavior**: Boolean comparison result converted to scalar
**Assertions**: Integration works correctly

```python
def test_scalar_from_tensor_after_eq(tmp_path):
    """Test ScalarFromTensor after EQ comparison (YOLO pattern).

    YOLO pattern:
    1. Compare dimensions: eq_result = Eq(dim1, dim2)  # TensorType(bool, shape=())
    2. Convert to scalar: scalar_bool = ScalarFromTensor(eq_result)
    3. Use for assertions/conditionals

    This tests the complete pattern.
    """
    from pytensor.tensor.basic import scalar_from_tensor

    x = pt.lscalar("x")
    y = pt.lscalar("y")

    # Compare with EQ
    eq_result = pt.eq(x, y)  # bool tensor

    # Convert to scalar
    scalar_bool = scalar_from_tensor(eq_result)

    # Test case: equal values
    x_val = np.array(20, dtype="int64")
    y_val = np.array(20, dtype="int64")
    # Expected: True (scalar bool)

    compare_onnx_and_py([x, y], scalar_bool, [x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError for ScalarFromTensor (EQ already implemented)

##### Test: `test_scalar_from_tensor_after_eq_false`
**Purpose**: Test integration with EQ returning False
**Test Data**: Unequal scalars
**Expected Behavior**: False scalar boolean
**Assertions**: False result handled correctly

```python
def test_scalar_from_tensor_after_eq_false(tmp_path):
    """Test ScalarFromTensor after EQ comparison returning False.

    Ensures False boolean path works in integration.
    """
    from pytensor.tensor.basic import scalar_from_tensor

    x = pt.lscalar("x")
    y = pt.lscalar("y")

    eq_result = pt.eq(x, y)
    scalar_bool = scalar_from_tensor(eq_result)

    # Test case: unequal values
    x_val = np.array(20, dtype="int64")
    y_val = np.array(10, dtype="int64")
    # Expected: False

    compare_onnx_and_py([x, y], scalar_bool, [x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError for ScalarFromTensor

##### Test: `test_scalar_from_tensor_in_computation_chain`
**Purpose**: Test ScalarFromTensor in multi-operation chain
**Test Data**: EQ comparison, ScalarFromTensor, then use in Switch
**Expected Behavior**: Full YOLO pattern works end-to-end
**Assertions**: Complete integration

```python
def test_scalar_from_tensor_in_computation_chain(tmp_path):
    """Test ScalarFromTensor in complete YOLO computation chain.

    Full pattern:
    1. Compare: eq_result = Eq(dim1, dim2)
    2. Convert: is_match = ScalarFromTensor(eq_result)
    3. Conditional: result = Switch(is_match, val1, val2)

    This tests that all three operations work together.
    """
    from pytensor.tensor.basic import scalar_from_tensor

    dim1 = pt.lscalar("dim1")
    dim2 = pt.lscalar("dim2")
    val1 = pt.lscalar("val1")
    val2 = pt.lscalar("val2")

    # YOLO pattern
    eq_result = pt.eq(dim1, dim2)
    is_match = scalar_from_tensor(eq_result)
    result = pt.switch(is_match, val1, val2)

    # Test case: dims match
    dim1_val = np.array(20, dtype="int64")
    dim2_val = np.array(20, dtype="int64")
    val1_val = np.array(10, dtype="int64")  # Will be selected
    val2_val = np.array(99, dtype="int64")
    # Expected: 10 (val1 selected because dims match)

    compare_onnx_and_py(
        [dim1, dim2, val1, val2],
        result,
        [dim1_val, dim2_val, val1_val, val2_val],
        tmp_path=tmp_path
    )
```

**Expected Failure Mode**: NotImplementedError for ScalarFromTensor (EQ and Switch already implemented)

#### 3. Structure Validation Tests

**Test File**: `tests/link/onnx/test_shape.py`
**Purpose**: Verify ONNX graph structure (Identity node created)

##### Test: `test_scalar_from_tensor_onnx_structure`
**Purpose**: Validate ONNX graph contains Identity node
**Test Data**: Simple ScalarFromTensor operation
**Expected Behavior**: Graph has one Identity node
**Assertions**: Node type, node count, inputs, outputs

```python
def test_scalar_from_tensor_onnx_structure(tmp_path):
    """Test that ScalarFromTensor generates correct ONNX Identity node.

    Verifies:
    - One Identity node created
    - Node has one input (scalar tensor)
    - Node has one output (scalar)
    - No unnecessary nodes
    """
    from pytensor.link.onnx import export_onnx
    from pytensor.tensor.basic import scalar_from_tensor

    x = pt.scalar("x", dtype="float32")
    y = scalar_from_tensor(x)

    f = pytensor.function([x], y)

    model_path = tmp_path / "test_scalar_from_tensor.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    validate_onnx_graph_structure(
        model,
        expected_node_types=["Identity"],
        expected_node_count=1,
    )

    # Check Identity node structure
    identity_node = model.graph.node[0]
    assert identity_node.op_type == "Identity"
    assert len(identity_node.input) == 1
    assert len(identity_node.output) == 1
```

**Expected Failure Mode**: NotImplementedError when trying to export

### Test Implementation Steps

1. **Add scalar tensor strategy**:
   - Location: `tests/link/onnx/strategies/core.py`
   - Add `scalar_tensor()` composite strategy
   - This enables property-based testing

2. **Write property test** (PRIMARY TEST):
   - Location: `tests/link/onnx/test_shape.py`
   - Add `test_scalar_from_tensor_property()` with Hypothesis
   - This single test replaces 7+ individual edge case tests
   - Automatically covers all dtypes and edge cases

3. **Write integration tests**:
   - Location: `tests/link/onnx/test_shape.py`
   - Add `test_scalar_from_tensor_after_eq()`
   - Add `test_scalar_from_tensor_after_eq_false()`
   - Add `test_scalar_from_tensor_in_computation_chain()`

4. **Write structure validation test**:
   - Location: `tests/link/onnx/test_shape.py`
   - Add `test_scalar_from_tensor_onnx_structure()`

5. **Add test documentation**:
   - Ensure each test has clear docstring
   - Document expected failure modes
   - Reference YOLO patterns

### Success Criteria

#### Automated Verification:
- [ ] All test files created with proper structure
- [ ] Tests use `compare_onnx_and_py` correctly
- [ ] Tests follow project conventions: `uv run pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" --collect-only`
- [ ] Strategy for 0-D tensors created
- [ ] Tests in correct file (test_shape.py, not test_elemwise.py)

#### Manual Verification:
- [ ] Each test has clear, informative docstring
- [ ] Test names clearly describe what they test
- [ ] Assertion messages are diagnostic
- [ ] Test code is readable
- [ ] YOLO patterns documented with references

---

## Phase 2: Test Failure Verification

### Overview
Run tests and verify they fail in expected, diagnostic ways.

### Verification Steps

1. **Run test suite**:
   ```bash
   uv run pytest tests/link/onnx/test_shape.py::test_scalar_from_tensor_float -v
   uv run pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" -v
   ```

2. **For each test, verify**:
   - Test fails with NotImplementedError
   - Error message mentions "ScalarFromTensor"
   - Points to dispatch system

### Expected Failures

**For unit tests**:
- Expected: `NotImplementedError` from dispatch system
- Message: Something like "No ONNX conversion available for: ScalarFromTensor"
- Location: Dispatch system (`onnx_funcify` base function)

**For property tests**:
- Hypothesis generates 10 examples (dev profile)
- All fail with same NotImplementedError

### Success Criteria

#### Automated Verification:
- [ ] All tests discovered: `uv run pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" --collect-only`
- [ ] All tests fail: `uv run pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" --tb=short` (0 passed)
- [ ] No unexpected errors (import errors, etc.)

#### Manual Verification:
- [ ] Each test fails with NotImplementedError
- [ ] Error messages mention "ScalarFromTensor"
- [ ] Failure messages are clear

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement ScalarFromTensor support by registering a dispatch function.

### Implementation Strategy

**Order:**
1. Import ScalarFromTensor in dispatch/shape.py
2. Register dispatch function
3. Run unit tests
4. Run integration tests
5. Run property tests

### Implementation Steps

#### Implementation: Register ScalarFromTensor Dispatch

**Target Test**: All tests
**Current Failure**: `NotImplementedError: No ONNX conversion available for: ScalarFromTensor`

**File**: `pytensor/link/onnx/dispatch/shape.py`

**Step 1: Import ScalarFromTensor**

At top of file with other imports:

```python
from pytensor.tensor.basic import ScalarFromTensor
```

**Step 2: Register dispatch function**

Add after other shape operations (e.g., after Shape_i, before or after DimShuffle):

```python
@onnx_funcify.register(ScalarFromTensor)
def onnx_funcify_ScalarFromTensor(op, node, var_names, get_var_name, **kwargs):
    """Convert ScalarFromTensor to ONNX Identity node.

    ScalarFromTensor extracts a scalar value from a 0-dimensional tensor.
    In ONNX, scalars ARE 0-dimensional tensors, so no conversion needed.
    We use Identity to represent this no-op transformation clearly.

    Parameters
    ----------
    op : ScalarFromTensor
        The operation instance
    node : Apply
        The apply node
    var_names : dict
        Variable name mapping
    get_var_name : callable
        Function to get variable names

    Returns
    -------
    NodeProto
        ONNX Identity node
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    return helper.make_node(
        "Identity",
        inputs=input_names,
        outputs=output_names,
        name=f"Identity_{output_names[0]}",
    )
```

**That's it!** ScalarFromTensor maps to ONNX Identity (no-op in ONNX since scalars are 0-D tensors).

**Debugging Approach:**
1. Add the import and function
2. Run: `uv run pytest tests/link/onnx/test_shape.py::test_scalar_from_tensor_float -v`
3. If passes, continue with other tests

**Success Criteria:**

##### Automated Verification:
- [ ] Unit tests pass: `uv run pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" -v`
- [ ] Integration tests pass (with EQ and Switch)
- [ ] Property tests pass (if implemented)
- [ ] No regressions: `uv run pytest tests/link/onnx/test_shape.py -v`

##### Manual Verification:
- [ ] Implementation is clean (~20 lines)
- [ ] Follows project conventions
- [ ] Comments explain why Identity is used

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor to improve code quality while keeping tests passing.

### Refactoring Targets

#### 1. Code Organization

Ensure ScalarFromTensor is in logical location in shape.py:

```python
# In shape.py, organize shape-related operations together:

@onnx_funcify.register(Shape_i)
def onnx_funcify_Shape_i(...):
    ...

@onnx_funcify.register(ScalarFromTensor)
def onnx_funcify_ScalarFromTensor(...):
    """Convert ScalarFromTensor to ONNX Identity node."""
    ...

@onnx_funcify.register(DimShuffle)
def onnx_funcify_DimShuffle(...):
    ...
```

#### 2. Documentation

Add module-level documentation about ScalarFromTensor:

```python
"""
ONNX conversion for shape-related operations.

...

Scalar Tensor Operations
------------------------
ScalarFromTensor:
- Converts 0-dimensional tensor to scalar value
- In ONNX, scalars ARE 0-D tensors (no distinction)
- Maps to Identity node (no-op, but makes intent clear)
- Used in YOLO after EQ comparisons for conditional logic

Usage in YOLO
-------------
Pattern: ScalarFromTensor(Eq(dim1, dim2))
- Compare dimensions with EQ → boolean 0-D tensor
- Convert to scalar boolean with ScalarFromTensor
- Use scalar boolean in Switch or Assert operations
"""
```

#### 3. Test Organization

Group ScalarFromTensor tests:

```python
# In test_shape.py

# ============================================================================
# ScalarFromTensor Operation Tests
# ============================================================================

def test_scalar_from_tensor_float(tmp_path):
    ...

# ... rest of ScalarFromTensor tests ...

# ============================================================================
```

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `uv run pytest tests/link/onnx/test_shape.py -v`
- [ ] No regressions

#### Manual Verification:
- [ ] Code is more readable
- [ ] Documentation is helpful
- [ ] Comments explain "why"

---

## Testing Strategy Summary

### Test Coverage Goals

#### Normal Operation:
- [x] Float scalar tensor
- [x] Integer scalar tensor
- [x] Boolean scalar tensor (True and False)
- [x] Zero value
- [x] Negative value
- [x] Multiple dtypes (bool, float32, float64, int32, int64)

#### Edge Cases:
- [x] Zero value (can cause issues)
- [x] Negative values (YOLO uses -1)
- [x] Extreme values (via property tests)
- [x] Both True and False for booleans

#### Integration:
- [x] After EQ comparison (YOLO pattern)
- [x] In complete computation chain (EQ → ScalarFromTensor → Switch)
- [x] With various dtypes

### Test Organization

**Test files:**
- `tests/link/onnx/test_shape.py` - Property tests, integration tests, structure tests
- `tests/link/onnx/strategies/core.py` - scalar_tensor() strategy

**Test utilities:**
- `compare_onnx_and_py()` - Compare outputs
- `validate_onnx_graph_structure()` - Validate graph
- `scalar_tensor()` - Hypothesis strategy for 0-D tensors

**Test Count:**
- 1 property test (100 examples) - Covers all dtypes and edge cases
- 3 integration tests - EQ pattern tests
- 1 structure test - ONNX graph validation
- **Total: 5 tests (but 100+ cases via property testing)**

### Running Tests

```bash
# All ScalarFromTensor tests
pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" -v

# Property test (runs 100 examples)
pytest tests/link/onnx/test_shape.py::test_scalar_from_tensor_property -v

# With coverage
pytest tests/link/onnx/test_shape.py -k "scalar_from_tensor" --cov=pytensor.link.onnx.dispatch.shape --cov-report=term-missing

# Integration with full YOLO pattern
pytest tests/link/onnx/test_shape.py::test_scalar_from_tensor_in_computation_chain -v

# Structure validation
pytest tests/link/onnx/test_shape.py::test_scalar_from_tensor_onnx_structure -v
```

## Performance Considerations

ScalarFromTensor is an Identity operation - O(1) time, no special performance concerns.

## Migration Notes

No migration needed - new feature, backward compatible.

## References

### Documentation
- ONNX Identity operator: https://onnx.ai/onnx/operators/onnx__Identity.html
- PyTensor ScalarFromTensor: `pytensor/tensor/basic.py` (search for `class ScalarFromTensor`)
- ONNX scalar representation: https://onnx.ai/onnx/intro/concepts.html#scalars

### Related Files
- Research: `thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md`
- ONNX dispatch: `pytensor/link/onnx/dispatch/shape.py`
- Test helper: `tests/link/onnx/test_basic.py:22-102`
- Shape tests: `tests/link/onnx/test_shape.py`
- YOLO model: `examples/onnx/onnx-yolo-demo/yolo/model.py:249-286`

### Similar Implementations
- Shape_i dispatch: `pytensor/link/onnx/dispatch/shape.py:18-95` (multi-node decomposition example)
- DimShuffle dispatch: `pytensor/link/onnx/dispatch/shape.py:189-390` (complex shape operation)
- Identity pattern: Similar to no-op transformations in other frameworks

---

## Implementation Checklist

### Phase 1: Test Design & Implementation
- [x] Create `scalar_tensor()` strategy in `strategies/core.py`
- [x] Write `test_scalar_from_tensor_property` property test (PRIMARY - replaces 7+ edge case tests)
- [x] Write `test_scalar_from_tensor_after_eq` integration test
- [x] Write `test_scalar_from_tensor_after_eq_false` integration test
- [x] Write `test_scalar_from_tensor_in_computation_chain` full pattern test
- [x] Write `test_scalar_from_tensor_onnx_structure` structure validation test

### Phase 2: Test Failure Verification
- [x] Verify all tests fail with NotImplementedError
- [x] Document failure modes

### Phase 3: Feature Implementation
- [x] Import ScalarFromTensor in `dispatch/shape.py`
- [x] Register `@onnx_funcify.register(ScalarFromTensor)` dispatch
- [x] Implement dispatch function (~20 lines)
- [x] Run and pass all tests

### Phase 4: Refactoring
- [x] Organize shape.py dispatch functions
- [x] Add module documentation
- [x] Group tests in test_shape.py

---

## Estimated Timeline

- **Phase 1**: 30-35 minutes
  - Strategy: 10 min
  - Property test: 10 min (replaces 7+ individual tests)
  - Integration tests: 10 min
  - Structure test: 5 min

- **Phase 2**: 5 minutes
- **Phase 3**: 15 minutes (more complex than EQ/Switch - dispatch function)
- **Phase 4**: 10 minutes

**Total**: ~60-65 minutes

**Time saved by property testing**: 30 minutes (no need to write 7+ individual edge case tests)

---

## Success Metrics

- [ ] Comprehensive test coverage (5 tests, 100+ cases via property testing)
- [ ] All tests fail before implementation
- [ ] All tests pass after implementation (~20 lines)
- [ ] Property test validates 100 random 0-D tensors (all dtypes, edge cases)
- [ ] YOLO integration patterns working
- [ ] No regressions
- [ ] Clean code and docs
- [ ] Identity node correctly used

**Key Improvement**: Property-based testing provides:
- 100+ test cases from 1 test function
- Automatic edge case discovery
- Better coverage with less code
- Easier maintenance (1 test vs 7+ tests)
