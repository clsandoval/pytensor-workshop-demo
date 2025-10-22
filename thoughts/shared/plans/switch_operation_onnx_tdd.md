---
date: 2025-10-18T22:30:00-00:00
author: Claude
related_research: thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md
operation: Switch (Conditional Selection)
git_branch: onnx-workshop-demo
repository: pytensor
status: ready_for_implementation
test_framework: pytest + hypothesis
dependencies: EQ operation (for conditional tests)
---

# Switch (Conditional Selection) Operation ONNX Export - TDD Implementation Plan

## Overview

Implement ONNX export support for the Switch scalar operation using Test-Driven Development with Hypothesis property-based testing. The Switch operation performs conditional selection (`if condition then x else y`) and is used 5 times in the YOLO11n model for dynamic dimension calculation and shape logic in upsampling operations.

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
- Switch is **NOT** in the dictionary

**Test Infrastructure**:
- Property-based test framework: `tests/link/onnx/test_properties.py`
- Operation registry: `tests/link/onnx/strategies/operations.py:360-444` (ONNX_OPERATIONS)
- Core strategies: `tests/link/onnx/strategies/core.py`
- Test helper: `compare_onnx_and_py()` in `tests/link/onnx/test_basic.py:22-102`

**Switch Operation in PyTensor**:
- Location: `pytensor.scalar.basic.Switch`
- Signature: `Switch(condition, then_value, else_value)`
- Behavior: If condition is True/non-zero, return then_value, else return else_value
- Used in Elemwise context for element-wise conditional selection

### What's Missing

1. Switch not in `SCALAR_OP_TO_ONNX` dictionary
2. No test coverage for conditional operations
3. No Hypothesis strategies for ternary operations (3 inputs)
4. No entry in `ONNX_OPERATIONS` registry for Switch

### Key Constraints Discovered

**ONNX Where Operator**:
- Signature: `Where(condition, X, Y)`
- Semantics: Exactly matches PyTensor Switch!
- Condition dtype: bool (ONNX spec), but PyTensor accepts any dtype (0 = false, non-zero = true)
- Output dtype: Same as X and Y (must match)
- Broadcasting: All three inputs broadcast to common shape

**PyTensor Switch Behavior**:
- Condition can be bool or numeric (0 = false, non-zero = true)
- X and Y can have different dtypes (PyTensor casts)
- Lazy evaluation: Only selected branch evaluated
- **CRITICAL**: For ONNX export, we lose lazy evaluation (both branches evaluated, then selected)

**YOLO Usage Patterns**:
```
Switch(Eq.0, Int_div.0, Sub.0)     # Conditional dimension calculation
Switch(Eq.0, Int_div.0, Mul.0)     # Conditional shape logic
Switch(Eq.0, Int_div.0, Assert{...}.0)  # Conditional assertions
```

Pattern: `if (condition from EQ) then (dimension_calc_1) else (dimension_calc_2)`

## Desired End State

After implementation:
- Switch operation exports to ONNX Where node
- Property-based tests validate correctness with 10-100 random test cases
- Tests cover: boolean conditions, various dtypes, broadcasting, YOLO patterns
- ONNX Runtime produces same results as PyTensor for all valid inputs
- Tests fail gracefully with diagnostic messages before implementation

## What We're NOT Testing/Implementing

- Lazy evaluation (ONNX doesn't support this - both branches evaluated)
- Non-boolean conditions (will cast to bool for ONNX)
- Gradient computation through Switch
- Nested Switch operations (separate test, not blocking)
- Integration with Assert operation (may not work in ONNX)
- Multi-output Switch (PyTensor feature, rare)

---

## TDD Approach

### Test Design Philosophy

**Property-Based Testing Strategy**:
1. **Correctness**: ONNX Where output must match PyTensor Switch for ANY valid inputs
2. **Condition Handling**: Boolean conditions work correctly, numeric conditions cast to bool
3. **Broadcasting**: Three-way broadcasting (condition, X, Y) follows ONNX/NumPy rules
4. **Dtype Preservation**: Output dtype matches then/else branches (must be same)
5. **Edge Cases**: All-true conditions, all-false conditions, mixed conditions

**Why Hypothesis?**
- Generates random boolean masks for conditions
- Tests various broadcasting scenarios automatically
- Discovers edge cases with mixed True/False conditions
- Tests with different shapes for condition, X, Y

**Test Structure**:
- ONE Hypothesis property-based test that automatically covers:
  - All dtypes (float32, float64, int32, int64)
  - All shapes (vectors, matrices, tensors, scalars)
  - Broadcasting scenarios (condition, then_value, else_value)
  - Mixed True/False conditions
  - Edge cases (all-true, all-false conditions)

---

## Phase 1: Test Design & Implementation

### Overview
Write comprehensive tests that define Switch behavior completely. These tests will fail initially with `NotImplementedError: Elemwise scalar op not supported for ONNX export: Switch`.

### Test Categories

#### 1. Unit Tests - Basic Switch Functionality

**Test File**: `tests/link/onnx/test_elemwise.py`
**Purpose**: Validate basic Switch operation with concrete, hand-crafted examples

**Test Cases to Write:**

##### Test: `test_switch_all_true_condition`
**Purpose**: Verify Switch with all-true condition selects then_value
**Test Data**: Boolean condition (all True), two value tensors
**Expected Behavior**: Output equals then_value
**Assertions**: Output matches then_value exactly

```python
def test_switch_all_true_condition(tmp_path):
    """Test Switch operation with all-true condition.

    Switch(True, x, y) should return x.

    This test verifies:
    - Switch recognizes True condition
    - Then-branch (x) is selected
    - Else-branch (y) is ignored
    - ONNX Where matches PyTensor Switch
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, True, True], dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([9.0, 9.0, 9.0], dtype="float32")
    # Expected: [1.0, 2.0, 3.0] (x values)

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**:
- Error type: `NotImplementedError`
- Expected message: `"Elemwise scalar op not supported for ONNX export: Switch"`
- Points to: `pytensor/link/onnx/dispatch/elemwise.py:269-273`

##### Test: `test_switch_all_false_condition`
**Purpose**: Verify Switch with all-false condition selects else_value
**Test Data**: Boolean condition (all False), two value tensors
**Expected Behavior**: Output equals else_value
**Assertions**: Output matches else_value exactly

```python
def test_switch_all_false_condition(tmp_path):
    """Test Switch operation with all-false condition.

    Switch(False, x, y) should return y.

    This test verifies:
    - Switch recognizes False condition
    - Else-branch (y) is selected
    - Then-branch (x) is ignored
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([False, False, False], dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([9.0, 8.0, 7.0], dtype="float32")
    # Expected: [9.0, 8.0, 7.0] (y values)

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Same NotImplementedError

##### Test: `test_switch_mixed_condition`
**Purpose**: Verify Switch with mixed True/False condition
**Test Data**: Boolean condition with both True and False
**Expected Behavior**: Element-wise selection from x or y
**Assertions**: Output is element-wise combination of x and y

```python
def test_switch_mixed_condition(tmp_path):
    """Test Switch operation with mixed True/False condition.

    This is the most common case - element-wise conditional selection.

    Condition: [True, False, True, False]
    X:         [1,    2,     3,    4]
    Y:         [10,   20,    30,   40]
    Output:    [1,    20,    3,    40]
                ↑ from X     ↑ from X
                      ↑ from Y     ↑ from Y
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, False, True, False], dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    y_val = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    # Expected: [1.0, 20.0, 3.0, 40.0]

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Same NotImplementedError

##### Test: `test_switch_integers`
**Purpose**: Verify Switch works with integer dtypes
**Test Data**: int64 tensors
**Expected Behavior**: Integer selection works correctly
**Assertions**: Integer values selected correctly

```python
def test_switch_integers(tmp_path):
    """Test Switch operation on integer tensors.

    Integer selection is used in YOLO for dimension calculations.
    This verifies Switch works with int32/int64 dtypes.
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.lvector("x")  # int64
    y = pt.lvector("y")
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, False, True], dtype="bool")
    x_val = np.array([10, 20, 30], dtype="int64")
    y_val = np.array([100, 200, 300], dtype="int64")
    # Expected: [10, 200, 30]

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Same NotImplementedError

##### Test: `test_switch_scalar_condition`
**Purpose**: Test Switch with scalar condition (broadcasts to all elements)
**Test Data**: Scalar boolean condition, vector values
**Expected Behavior**: Scalar condition applies to all elements
**Assertions**: All elements from same branch

```python
def test_switch_scalar_condition(tmp_path):
    """Test Switch with scalar condition (broadcasts to all elements).

    If condition is scalar True, all elements come from x.
    If condition is scalar False, all elements come from y.
    """
    condition = pt.scalar("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    # Test with True condition
    condition_val = np.array(True, dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_val = np.array([9.0, 9.0, 9.0], dtype="float32")
    # Expected: [1.0, 2.0, 3.0] (all from x)

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Same NotImplementedError

##### Test: `test_switch_scalar_tensors`
**Purpose**: Test Switch with 0-dimensional scalar tensors (YOLO pattern)
**Test Data**: Scalar condition, scalar values
**Expected Behavior**: Scalar output
**Assertions**: Scalar conditional selection works

```python
def test_switch_scalar_tensors(tmp_path):
    """Test Switch on 0-dimensional (scalar) tensors.

    YOLO uses Switch on scalar tensors:
    - Pattern: Switch(Eq.0, Int_div.0, Sub.0)
    - Input types: TensorType(int64, shape=())

    This is critical for YOLO's dynamic dimension logic.
    """
    condition = pt.scalar("condition", dtype="bool")
    x = pt.lscalar("x")
    y = pt.lscalar("y")
    z = pt.switch(condition, x, y)

    # Test true condition
    condition_val = np.array(True, dtype="bool")
    x_val = np.array(10, dtype="int64")
    y_val = np.array(20, dtype="int64")
    # Expected: 10 (from x)

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: Same NotImplementedError

#### 2. Parametrized Tests - Multiple Dtypes and Shapes

**Test File**: `tests/link/onnx/test_elemwise.py`
**Purpose**: Systematically test all supported dtypes and shapes

##### Test: `test_switch_multiple_dtypes`
**Purpose**: Verify Switch works with all ONNX-supported dtypes
**Test Data**: Parametrized over float32, float64, int32, int64
**Expected Behavior**: Works for all numeric dtypes
**Assertions**: Dtype-agnostic conditional selection

```python
@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64", "int32", "int64"],
)
def test_switch_multiple_dtypes(tmp_path, dtype):
    """Test Switch operation with all supported dtypes.

    ONNX Where supports: float32, float64, int32, int64, bool
    This test verifies all numeric dtypes work correctly.
    """
    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    z = pt.switch(condition, x, y)

    condition_val = np.array([True, False, True, False], dtype="bool")
    rng = np.random.default_rng(42)
    if dtype.startswith("float"):
        x_val = rng.random(4).astype(dtype) * 10
        y_val = rng.random(4).astype(dtype) * 100
    else:
        x_val = rng.integers(0, 10, size=4).astype(dtype)
        y_val = rng.integers(100, 200, size=4).astype(dtype)

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError for all dtypes

##### Test: `test_switch_different_shapes`
**Purpose**: Test Switch with various tensor ranks
**Test Data**: Vector, matrix, 3D tensor, 4D tensor
**Expected Behavior**: Shape-agnostic conditional selection
**Assertions**: Works for all ranks

```python
@pytest.mark.parametrize(
    "shape",
    [
        (5,),          # vector
        (3, 4),        # matrix
        (2, 3, 4),     # 3D tensor
        (2, 3, 4, 5),  # 4D tensor (CNN feature maps)
    ],
)
def test_switch_different_shapes(tmp_path, shape):
    """Test Switch operation with different tensor shapes.

    Switch should work for any tensor shape (element-wise operation).
    """
    condition = pt.tensor("condition", dtype="bool", shape=shape)
    x = pt.tensor("x", dtype="float32", shape=shape)
    y = pt.tensor("y", dtype="float32", shape=shape)
    z = pt.switch(condition, x, y)

    rng = np.random.default_rng(42)
    condition_val = rng.random(shape) > 0.5  # Random True/False
    x_val = rng.random(shape).astype("float32")
    y_val = rng.random(shape).astype("float32") + 10.0  # Different values

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError for all shapes

#### 3. Broadcasting Tests

**Test File**: `tests/link/onnx/test_elemwise.py`
**Purpose**: Verify three-way broadcasting rules match ONNX/NumPy

##### Test: `test_switch_broadcast_condition_scalar`
**Purpose**: Test scalar condition broadcasting to vector values
**Test Data**: Scalar condition, vector x and y
**Expected Behavior**: Condition broadcasts to all elements
**Assertions**: Correct broadcasting shape

```python
def test_switch_broadcast_condition_scalar(tmp_path):
    """Test Switch with scalar condition broadcasting.

    Broadcasting: condition (scalar) broadcasts to x and y (vectors).
    Result shape: same as x and y.
    """
    condition = pt.scalar("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    # True condition → all from x
    condition_val = np.array(True, dtype="bool")
    x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    y_val = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    # Expected: [1.0, 2.0, 3.0, 4.0]

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError

##### Test: `test_switch_broadcast_dimensions`
**Purpose**: Test broadcasting with dimension expansion
**Test Data**: condition (3, 1), x (1, 4), y (3, 4) → broadcasts to (3, 4)
**Expected Behavior**: Three-way broadcasting follows NumPy rules
**Assertions**: Output shape matches np.broadcast_shapes()

```python
def test_switch_broadcast_dimensions(tmp_path):
    """Test Switch with three-way dimension broadcasting.

    Broadcasting example:
    - condition: shape (3, 1)
    - x: shape (1, 4)
    - y: shape (3, 4)
    - Output: shape (3, 4)
    """
    condition = pt.matrix("condition", dtype="bool")
    x = pt.matrix("x", dtype="float32")
    y = pt.matrix("y", dtype="float32")
    z = pt.switch(condition, x, y)

    condition_val = np.array([[True], [False], [True]], dtype="bool")  # (3, 1)
    x_val = np.array([[1.0, 2.0, 3.0, 4.0]], dtype="float32")  # (1, 4)
    y_val = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
        [90.0, 100.0, 110.0, 120.0]
    ], dtype="float32")  # (3, 4)

    # Broadcasts to (3, 4):
    # Row 0: True  → [1, 2, 3, 4] from x
    # Row 1: False → [50, 60, 70, 80] from y
    # Row 2: True  → [1, 2, 3, 4] from x

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError

##### Test: `test_switch_broadcast_complex`
**Purpose**: Test complex three-way broadcasting scenario
**Test Data**: Different shapes for condition, x, y that broadcast together
**Expected Behavior**: Broadcasting matches NumPy's np.where()
**Assertions**: Output shape and values correct

```python
def test_switch_broadcast_complex(tmp_path):
    """Test Switch with complex three-way broadcasting.

    This tests the full generality of ONNX Where broadcasting.
    """
    condition = pt.tensor("condition", dtype="bool", shape=(2, 1, 4))
    x = pt.tensor("x", dtype="float32", shape=(1, 3, 1))
    y = pt.tensor("y", dtype="float32", shape=(2, 3, 4))
    z = pt.switch(condition, x, y)

    rng = np.random.default_rng(42)
    condition_val = rng.random((2, 1, 4)) > 0.5
    x_val = rng.random((1, 3, 1)).astype("float32")
    y_val = rng.random((2, 3, 4)).astype("float32") + 10.0

    # Broadcasts to (2, 3, 4)

    compare_onnx_and_py([condition, x, y], z, [condition_val, x_val, y_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError

#### 4. Property-Based Tests with Hypothesis

**Test File**: `tests/link/onnx/test_properties.py`
**Purpose**: Automatically test with 10-100 random input combinations

##### Test: Add Switch to ONNX_OPERATIONS Registry

**File**: `tests/link/onnx/strategies/operations.py:360-444`

First, create a strategy for ternary operations (3 inputs):

```python
@st.composite
def switch_inputs(draw, dtypes=None):
    """Generate inputs for Switch operation (ternary: condition, then, else).

    Generates:
    - Boolean condition tensor
    - Two value tensors (then and else) with same dtype
    - Compatible broadcasting shapes

    Parameters
    ----------
    dtypes : list or None
        Allowed dtypes for value tensors. If None, uses all ONNX dtypes

    Returns
    -------
    tuple
        (condition, then_value, else_value)
    """
    if dtypes is None:
        dtypes = [np.float32, np.float64, np.int32, np.int64]

    # Generate compatible dtype for value tensors
    dtype = draw(st.sampled_from(dtypes))

    # Generate base shape for values
    base_shape = draw(valid_shapes(min_rank=1, max_rank=3, min_dim=1, max_dim=5))

    # Generate then_value with base shape
    then_val = draw(onnx_tensor(dtype=dtype, shape=base_shape))

    # Generate else_value with potentially different but compatible shape
    broadcast_pattern = draw(
        st.sampled_from(["same", "broadcast_dims", "prefix"])
    )

    if broadcast_pattern == "same":
        else_shape = base_shape
    elif broadcast_pattern == "broadcast_dims":
        # Randomly make some dimensions 1
        else_shape = tuple(
            1 if draw(st.booleans()) and dim > 1 else dim for dim in base_shape
        )
    else:  # prefix
        # Take suffix of base_shape
        suffix_len = draw(st.integers(1, len(base_shape)))
        else_shape = base_shape[-suffix_len:]

    else_val = draw(onnx_tensor(dtype=dtype, shape=else_shape))

    # Generate condition with compatible broadcasting shape
    # Condition can be scalar, same as then/else, or broadcast-compatible
    condition_pattern = draw(
        st.sampled_from(["same", "scalar", "broadcast_dims"])
    )

    if condition_pattern == "same":
        condition_shape = base_shape
    elif condition_pattern == "scalar":
        condition_shape = ()
    else:  # broadcast_dims
        condition_shape = tuple(
            1 if draw(st.booleans()) and dim > 1 else dim for dim in base_shape
        )

    # Generate boolean condition
    condition_val = draw(
        arrays(
            dtype=np.bool_,
            shape=condition_shape,
            elements=st.booleans()
        )
    )

    return (condition_val, then_val, else_val)
```

Then add Switch to the registry:

```python
ONNX_OPERATIONS = {
    # ... existing operations ...

    "switch": OperationConfig(
        op_func=lambda cond, x, y: pt.switch(cond, x, y),
        input_strategy=switch_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
        notes="Conditional selection (ternary), maps to ONNX Where",
    ),
}
```

**Modification needed in test_properties.py**:

The property tests assume binary operations. Need to handle ternary:

```python
# In test_onnx_matches_pytensor, modify:

# Create symbolic variables
if len(inputs_tuple) == 1:
    # Unary operation
    x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
    symbolic_inputs = [x]
    test_values = [inputs_tuple[0]]
    result = op_config.op_func(x)
elif len(inputs_tuple) == 2:
    # Binary operation
    x = pt.tensor("x", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
    if isinstance(inputs_tuple[1], tuple):
        # Second argument is a shape (e.g., reshape)
        symbolic_inputs = [x]
        test_values = [inputs_tuple[0]]
        result = op_config.op_func(x, inputs_tuple[1])
    else:
        y = pt.tensor("y", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
        symbolic_inputs = [x, y]
        test_values = [inputs_tuple[0], inputs_tuple[1]]
        result = op_config.op_func(x, y)
elif len(inputs_tuple) == 3:
    # Ternary operation (switch)
    cond = pt.tensor("cond", dtype=inputs_tuple[0].dtype, shape=inputs_tuple[0].shape)
    x = pt.tensor("x", dtype=inputs_tuple[1].dtype, shape=inputs_tuple[1].shape)
    y = pt.tensor("y", dtype=inputs_tuple[2].dtype, shape=inputs_tuple[2].shape)
    symbolic_inputs = [cond, x, y]
    test_values = [inputs_tuple[0], inputs_tuple[1], inputs_tuple[2]]
    result = op_config.op_func(cond, x, y)
else:
    raise NotImplementedError(
        f"Operations with {len(inputs_tuple)} inputs not yet supported"
    )
```

**Expected Failure Mode**:
- Property test `test_onnx_matches_pytensor` will fail for operation "switch"
- Error: `NotImplementedError: Elemwise scalar op not supported for ONNX export: Switch`
- Hypothesis will try 10-100 examples, all fail with same error

#### 5. Structure Validation Tests

**Test File**: `tests/link/onnx/test_elemwise.py`
**Purpose**: Verify ONNX graph structure (Where node created)

##### Test: `test_switch_onnx_structure`
**Purpose**: Validate ONNX graph contains Where node
**Test Data**: Simple Switch operation
**Expected Behavior**: Graph has one Where node
**Assertions**: Node type, node count, inputs, outputs

```python
def test_switch_onnx_structure(tmp_path):
    """Test that Switch generates correct ONNX Where node.

    Verifies:
    - One Where node created
    - Node has three inputs (condition, X, Y)
    - Node has one output
    - No unnecessary nodes
    """
    from pytensor.link.onnx import export_onnx

    condition = pt.vector("condition", dtype="bool")
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = pt.switch(condition, x, y)

    f = pytensor.function([condition, x, y], z)

    model_path = tmp_path / "test_switch.onnx"
    model = export_onnx(f, model_path)

    # Validate structure
    from tests.link.onnx.test_basic import validate_onnx_graph_structure

    validate_onnx_graph_structure(
        model,
        expected_node_types=["Where"],
        expected_node_count=1,
    )

    # Check Where node structure
    where_node = model.graph.node[0]
    assert where_node.op_type == "Where"
    assert len(where_node.input) == 3  # condition, X, Y
    assert len(where_node.output) == 1
```

**Expected Failure Mode**: NotImplementedError when trying to export

#### 6. YOLO Pattern Tests

**Test File**: `tests/link/onnx/test_elemwise.py`
**Purpose**: Test specific patterns from YOLO model

##### Test: `test_switch_yolo_dimension_calculation_pattern`
**Purpose**: Test YOLO's conditional dimension calculation
**Test Data**: Switch based on EQ result, dimension calculations
**Expected Behavior**: Correct dimension selected based on condition
**Assertions**: Integration with EQ operation

```python
def test_switch_yolo_dimension_calculation_pattern(tmp_path):
    """Test Switch in YOLO dimension calculation pattern.

    YOLO pattern (from model.py:249-286):
    - Compare dimensions with EQ: is_match = Eq(dim1, dim2)
    - Conditionally select dimension: Switch(is_match, calc1, calc2)

    This simulates: if dims match, use floor division, else use subtraction.
    Pattern: Switch(Eq.0, Int_div.0, Sub.0)
    """
    # Inputs: dimension values
    dim1 = pt.lscalar("dim1")
    dim2 = pt.lscalar("dim2")

    # Compare dimensions
    is_match = pt.eq(dim1, dim2)

    # Conditional dimension calculations
    if_match = dim1 // 2  # Use floor division if match
    if_no_match = dim1 - 1  # Use subtraction if no match

    # Switch based on condition
    result_dim = pt.switch(is_match, if_match, if_no_match)

    # Test case 1: dims match (20 == 20)
    dim1_val = np.array(20, dtype="int64")
    dim2_val = np.array(20, dtype="int64")
    # is_match = True → use floor division: 20 // 2 = 10

    compare_onnx_and_py([dim1, dim2], result_dim, [dim1_val, dim2_val], tmp_path=tmp_path)
```

**Expected Failure Mode**: NotImplementedError (Switch not implemented)

##### Test: `test_switch_yolo_negative_one_pattern`
**Purpose**: Test conditional logic for -1 (dynamic dimension marker)
**Test Data**: Switch based on comparison with -1
**Expected Behavior**: Different calculation for dynamic vs static dimensions
**Assertions**: Correct branch selected

```python
def test_switch_yolo_negative_one_pattern(tmp_path):
    """Test Switch for -1 (dynamic dimension) handling.

    YOLO uses -1 to indicate dynamic dimensions:
    - Check if dimension is dynamic: is_dynamic = Eq(dim, -1)
    - Use different calculation: Switch(is_dynamic, dynamic_calc, static_calc)

    Pattern: Switch(Eq(dim, -1), calc_dynamic, calc_static)
    """
    dim = pt.lscalar("dim")
    neg_one = pt.lscalar("neg_one")

    # Check if dynamic
    is_dynamic = pt.eq(dim, neg_one)

    # Conditional calculations
    # If dynamic (-1), use special value (e.g., 0 or another marker)
    # If static, use the actual dimension
    dynamic_value = pt.lscalar("dynamic_value")
    static_value = dim

    result = pt.switch(is_dynamic, dynamic_value, static_value)

    # Test case: static dimension (not -1)
    dim_val = np.array(10, dtype="int64")
    neg_one_val = np.array(-1, dtype="int64")
    dynamic_value_val = np.array(0, dtype="int64")
    # is_dynamic = False → use static_value: 10

    compare_onnx_and_py(
        [dim, neg_one, dynamic_value],
        result,
        [dim_val, neg_one_val, dynamic_value_val],
        tmp_path=tmp_path
    )
```

**Expected Failure Mode**: NotImplementedError

### Test Implementation Steps

1. **Create test strategy**:
   - Location: `tests/link/onnx/strategies/operations.py`
   - Add `switch_inputs()` strategy

2. **Add to operation registry**:
   - Location: `tests/link/onnx/strategies/operations.py:360-444`
   - Add "switch" entry to `ONNX_OPERATIONS` dict

3. **Modify property test for ternary operations**:
   - Location: `tests/link/onnx/test_properties.py:23-103`
   - Add handling for len(inputs_tuple) == 3

4. **Write unit tests**:
   - Location: `tests/link/onnx/test_elemwise.py`
   - Add all test functions above

5. **Add test documentation**:
   - Ensure each test has clear docstring
   - Document expected failure modes

### Success Criteria

#### Automated Verification:
- [ ] All test files created with proper structure
- [ ] Tests use `compare_onnx_and_py` correctly
- [ ] Tests follow project conventions: `uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" --collect-only`
- [ ] Strategy added to `ONNX_OPERATIONS` registry
- [ ] Property test handles ternary operations

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
   uv run pytest tests/link/onnx/test_elemwise.py::test_switch_all_true_condition -v
   uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" -v
   uv run pytest tests/link/onnx/test_properties.py -k "switch" -v
   ```

2. **For each test, verify**:
   - Test fails with NotImplementedError
   - Error message mentions "Switch"
   - Points to elemwise.py:269

### Expected Failures

**For unit tests**:
- Expected: `NotImplementedError: Elemwise scalar op not supported for ONNX export: Switch`
- Location: `pytensor/link/onnx/dispatch/elemwise.py:269-273`

**For property tests**:
- Hypothesis generates 10 examples (dev profile)
- All fail with same NotImplementedError

### Success Criteria

#### Automated Verification:
- [ ] All tests discovered: `uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" --collect-only`
- [ ] All tests fail: `uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" --tb=short` (0 passed)
- [ ] No unexpected errors
- [ ] Property test runs: `uv run pytest tests/link/onnx/test_properties.py -k "switch" -v`

#### Manual Verification:
- [ ] Each test fails with NotImplementedError
- [ ] Error messages mention "Switch"
- [ ] Error location is elemwise.py:269
- [ ] Failure messages are clear

---

## Phase 3: Feature Implementation (Red → Green)

### Overview
Implement Switch support by adding one entry to `SCALAR_OP_TO_ONNX` dictionary.

### Implementation Strategy

**Order:**
1. Add Switch to SCALAR_OP_TO_ONNX
2. Run unit tests
3. Run parametrized tests
4. Run property tests
5. Run YOLO pattern tests

### Implementation Steps

#### Implementation: Add Switch to SCALAR_OP_TO_ONNX

**Target Test**: All tests
**Current Failure**: `NotImplementedError: Elemwise scalar op not supported for ONNX export: Switch`

**File**: `pytensor/link/onnx/dispatch/elemwise.py`
**Changes**: Add one line to SCALAR_OP_TO_ONNX dictionary

```python
# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    # ... existing entries ...
    scalar.EQ: "Equal",  # Element-wise equality comparison
    scalar.Switch: "Where",  # Conditional selection: if cond then x else y
}
```

**That's it!** Switch maps directly to ONNX Where operator. Both have signature (condition, then_value, else_value).

**Debugging Approach:**
1. Add the line
2. Run: `uv run pytest tests/link/onnx/test_elemwise.py::test_switch_all_true_condition -v`
3. If passes, continue with other tests

**Success Criteria:**

##### Automated Verification:
- [ ] Unit tests pass: `uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" -v`
- [ ] Property tests pass: `uv run pytest tests/link/onnx/test_properties.py -k "switch" -v`
- [ ] No regressions: `uv run pytest tests/link/onnx/test_elemwise.py -v`

##### Manual Verification:
- [ ] Implementation is clean (1 line)
- [ ] Follows project conventions
- [ ] Comment is clear

---

## Phase 4: Refactoring & Cleanup

### Overview
Refactor to improve code quality while keeping tests passing.

### Refactoring Targets

#### 1. Code Organization

Group comparison and conditional operations:

```python
SCALAR_OP_TO_ONNX = {
    # ... other operations ...

    # Comparison operations
    scalar.EQ: "Equal",  # Element-wise equality comparison

    # Conditional operations
    scalar.Switch: "Where",  # if cond then x else y

    # ... rest ...
}
```

#### 2. Documentation

Add documentation about Switch:

```python
"""
Conditional Operations
----------------------
Switch maps to ONNX Where operator:
- Signature: Switch(condition, then_value, else_value)
- ONNX Where: Where(condition, X, Y)
- Semantics: Element-wise selection based on condition
- Output dtype: Same as then_value and else_value (must match)
- Broadcasting: All three inputs broadcast to common shape

Note: ONNX doesn't support lazy evaluation - both branches are evaluated.

Usage in YOLO
-------------
Switch is used for conditional dimension calculations:
- Switch(Eq(dim1, dim2), floor_div_calc, sub_calc)
- Switch(Eq(dim, -1), dynamic_handling, static_handling)
"""
```

#### 3. Test Organization

Group Switch tests:

```python
# ============================================================================
# Switch (Conditional Selection) Operation Tests
# ============================================================================

def test_switch_all_true_condition(tmp_path):
    ...

# ... rest of Switch tests ...

# ============================================================================
```

### Success Criteria

#### Automated Verification:
- [ ] All tests still pass: `uv run pytest tests/link/onnx/ -v`
- [ ] No regressions

#### Manual Verification:
- [ ] Code is more readable
- [ ] Documentation is helpful
- [ ] Comments explain "why"

---

## Testing Strategy Summary

### Test Coverage Goals

#### Normal Operation:
- [x] All-true condition
- [x] All-false condition
- [x] Mixed condition
- [x] Integer values
- [x] Scalar condition broadcasting
- [x] Scalar tensors (YOLO)

#### Edge Cases:
- [x] Different dtypes (float32, float64, int32, int64)
- [x] Different shapes (vector, matrix, 3D, 4D)
- [x] Broadcasting (scalar condition, dimension expansion, complex)
- [x] Empty tensors (via property tests)
- [x] Extreme values (via property tests)

#### Integration:
- [x] Integration with EQ operation (YOLO pattern)
- [x] Dimension calculation pattern
- [x] Dynamic dimension handling (-1 pattern)

### Test Organization

**Test files:**
- `tests/link/onnx/test_elemwise.py` - Unit, parametrized, pattern tests
- `tests/link/onnx/test_properties.py` - Property tests
- `tests/link/onnx/strategies/operations.py` - Strategy and registry

**Test utilities:**
- `compare_onnx_and_py()` - Compare outputs
- `validate_onnx_graph_structure()` - Validate graph
- `switch_inputs()` - Hypothesis strategy

### Running Tests

```bash
# All Switch tests
uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" -v

# Specific test
uv run pytest tests/link/onnx/test_elemwise.py::test_switch_all_true_condition -v

# With coverage
uv run pytest tests/link/onnx/test_elemwise.py -k "test_switch" --cov=pytensor.link.onnx.dispatch.elemwise --cov-report=term-missing

# Property tests (dev)
uv run pytest tests/link/onnx/test_properties.py -k "switch" -v

# Property tests (ci - 100 examples)
HYPOTHESIS_PROFILE=ci uv run pytest tests/link/onnx/test_properties.py -k "switch" -v
```

## Performance Considerations

Switch is element-wise selection - O(n) time, no special performance concerns.

## Migration Notes

No migration needed - new feature, backward compatible.

## References

### Documentation
- ONNX Where operator: https://onnx.ai/onnx/operators/onnx__Where.html
- PyTensor scalar.Switch: `pytensor/scalar/basic.py`
- NumPy where: https://numpy.org/doc/stable/reference/generated/numpy.where.html

### Related Files
- Research: `thoughts/shared/research/2025-10-18_21-45-00_onnx-yolo-missing-ops.md`
- ONNX dispatch: `pytensor/link/onnx/dispatch/elemwise.py:17-32`
- Test helper: `tests/link/onnx/test_basic.py:22-102`
- Property tests: `tests/link/onnx/test_properties.py:23-103`
- YOLO model: `examples/onnx/onnx-yolo-demo/yolo/model.py:249-286`

---

## Implementation Checklist

### Phase 1: Test Design & Implementation
- [x] Create `switch_inputs()` strategy
- [x] Add "switch" to `ONNX_OPERATIONS` registry
- [x] Modify property test for ternary operations
- [x] Write unit tests (all 6 tests)
- [x] Write parametrized tests (dtypes, shapes)
- [x] Write broadcasting tests (3 tests)
- [x] Write structure validation test
- [x] Write YOLO pattern tests (2 tests)

### Phase 2: Test Failure Verification
- [x] Verify all tests fail with NotImplementedError
- [x] Document failure modes

### Phase 3: Feature Implementation
- [x] Add `scalar.Switch: "Where"` to dictionary
- [x] Run and pass all tests

### Phase 4: Refactoring
- [x] Organize dictionary with comments
- [x] Add documentation
- [x] Group tests

---

## Estimated Timeline

- **Phase 1**: 60-75 minutes
- **Phase 2**: 10 minutes
- **Phase 3**: 5 minutes
- **Phase 4**: 15 minutes

**Total**: ~90-105 minutes

---

## Success Metrics

- [ ] Comprehensive test coverage (15+ tests)
- [ ] All tests fail before implementation
- [ ] All tests pass after one line
- [ ] Property tests validate 10-1000 cases
- [ ] YOLO patterns working
- [ ] No regressions
- [ ] Clean code and docs
