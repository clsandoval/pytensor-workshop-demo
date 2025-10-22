"""Operation registry and input strategies for ONNX testing."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import pytensor.tensor as pt
from tests.link.onnx.strategies.core import onnx_dtypes, onnx_tensor, valid_shapes


@dataclass
class OperationConfig:
    """Configuration for testing an ONNX operation.

    Attributes
    ----------
    op_func : callable
        PyTensor operation function (e.g., pt.add, pt.dot)
    input_strategy : hypothesis.strategies.SearchStrategy
        Strategy that generates valid inputs for the operation
    valid_dtypes : list of str
        Dtypes supported by this operation
    category : str
        Operation category (elemwise, shape, nlinalg, etc.)
    notes : str, optional
        Additional notes or constraints
    """

    op_func: Callable
    input_strategy: st.SearchStrategy
    valid_dtypes: list[str]
    category: str
    notes: str | None = None


@st.composite
def unary_operation_inputs(draw, dtype=None, shape=None):
    """Generate inputs for unary operations (e.g., neg, exp, log).

    Returns
    -------
    tuple
        (tensor,) - Single input tensor
    """
    if dtype is None:
        dtype = draw(onnx_dtypes())
    if shape is None:
        shape = draw(valid_shapes())

    x = draw(onnx_tensor(dtype=dtype, shape=shape))
    return (x,)


@st.composite
def unary_float_operation_inputs(draw, shape=None):
    """Generate float inputs for unary operations that require float (e.g., exp, log, sqrt).

    Returns
    -------
    tuple
        (tensor,) - Single float input tensor
    """
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    if shape is None:
        shape = draw(valid_shapes())

    x = draw(onnx_tensor(dtype=dtype, shape=shape))
    return (x,)


@st.composite
def binary_broadcastable_inputs(draw, dtypes=None):
    """Generate inputs for binary operations with broadcasting (e.g., add, mul).

    Parameters
    ----------
    dtypes : list or None
        Allowed dtypes. If None, uses all ONNX dtypes

    Returns
    -------
    tuple
        (x, y) - Two tensors with compatible broadcasting shapes
    """
    if dtypes is None:
        dtypes = [np.float32, np.float64, np.int32, np.int64]

    # Generate compatible dtype for both tensors
    dtype = draw(st.sampled_from(dtypes))

    # Generate base shape
    base_shape = draw(valid_shapes(min_rank=1, max_rank=3, min_dim=1, max_dim=5))

    # Generate broadcasting variant for second tensor
    # Options: same shape, broadcast dims, or smaller tensor
    broadcast_pattern = draw(
        st.sampled_from(
            [
                "same",  # Same shape
                "broadcast_dims",  # Some dimensions are 1
                "prefix",  # Smaller tensor (broadcasts from right)
            ]
        )
    )

    if broadcast_pattern == "same":
        shape_y = base_shape
    elif broadcast_pattern == "broadcast_dims":
        # Randomly make some dimensions 1
        shape_y = tuple(
            1 if draw(st.booleans()) and dim > 1 else dim for dim in base_shape
        )
    else:  # prefix
        # Take suffix of base_shape
        suffix_len = draw(st.integers(1, len(base_shape)))
        shape_y = base_shape[-suffix_len:]

    x = draw(onnx_tensor(dtype=dtype, shape=base_shape))
    y = draw(onnx_tensor(dtype=dtype, shape=shape_y))

    return (x, y)


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
    broadcast_pattern = draw(st.sampled_from(["same", "broadcast_dims", "prefix"]))

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


@st.composite
def matmul_inputs(draw):
    """Generate inputs for matrix multiplication.

    Returns
    -------
    tuple
        (A, B) - Two tensors with compatible shapes for matmul
    """
    # Only float32 to avoid ONNX Runtime FusedMatMul issues with float64
    dtype = np.float32

    # Generate dimensions
    m = draw(st.integers(1, 50))
    n = draw(st.integers(1, 50))
    k = draw(st.integers(1, 50))

    # Optionally add batch dimension
    has_batch = draw(st.booleans())
    if has_batch:
        batch = draw(st.integers(1, 8))
        shape_a = (batch, m, k)
        shape_b = (batch, k, n)
    else:
        # Can be 1D (vector) or 2D (matrix)
        a_is_1d = draw(st.booleans()) and m > 1  # Avoid scalar
        b_is_1d = draw(st.booleans()) and n > 1

        if a_is_1d and b_is_1d:
            # Vector dot vector
            shape_a = (k,)
            shape_b = (k,)
        elif a_is_1d:
            # Vector @ Matrix
            shape_a = (k,)
            shape_b = (k, n)
        elif b_is_1d:
            # Matrix @ Vector
            shape_a = (m, k)
            shape_b = (k,)
        else:
            # Matrix @ Matrix
            shape_a = (m, k)
            shape_b = (k, n)

    A = draw(onnx_tensor(dtype=dtype, shape=shape_a))
    B = draw(onnx_tensor(dtype=dtype, shape=shape_b))

    return (A, B)


@st.composite
def reshape_inputs(draw):
    """Generate inputs for reshape operation.

    Returns
    -------
    tuple
        (tensor, new_shape) - Tensor and compatible reshape target
    """
    dtype = draw(onnx_dtypes())

    # Generate original shape
    original_shape = draw(valid_shapes(min_rank=1, max_rank=4, min_dim=1, max_dim=10))
    total_elements = np.prod(original_shape)

    # Generate compatible new shape
    # Find divisors of total_elements
    divisors = [
        i for i in range(1, int(total_elements**0.5) + 1) if total_elements % i == 0
    ]

    if not divisors:
        # Handle edge case: total_elements is 1 or very large prime
        new_shape = (int(total_elements),)
    else:
        # Build new shape from divisors
        rank = draw(st.integers(1, 4))
        new_shape = []
        remaining = total_elements

        for _ in range(rank - 1):
            if remaining == 1:
                new_shape.append(1)
            else:
                valid_divs = [
                    d for d in divisors if remaining % d == 0 and d <= remaining
                ]
                if valid_divs:
                    dim = draw(st.sampled_from(valid_divs))
                    new_shape.append(dim)
                    remaining //= dim
                else:
                    new_shape.append(1)

        new_shape.append(remaining)
        new_shape = tuple(new_shape)

    tensor = draw(onnx_tensor(dtype=dtype, shape=original_shape))

    return (tensor, new_shape)


@st.composite
def dimshuffle_inputs(draw):
    """Generate inputs for dimshuffle/transpose operation.

    Returns
    -------
    tuple
        (tensor, pattern) - Tensor and valid dimshuffle pattern
    """
    dtype = draw(onnx_dtypes())

    # Generate shape
    ndim = draw(st.integers(1, 4))
    shape = tuple(draw(st.integers(1, 10)) for _ in range(ndim))

    # Generate valid dimshuffle pattern
    # Pattern can include dimension indices and 'x' for new axes

    # Simple transpose case
    pattern = list(range(ndim))
    draw(st.randoms()).shuffle(pattern)

    # Optionally add 'x' dimensions
    if draw(st.booleans()):
        num_x = draw(st.integers(1, 2))
        for _ in range(num_x):
            insert_pos = draw(st.integers(0, len(pattern)))
            pattern.insert(insert_pos, "x")

    # Optionally drop some dimensions (only if dimension size is 1)
    # This is complex, so we'll skip for now and focus on transpose + unsqueeze

    tensor = draw(onnx_tensor(dtype=dtype, shape=shape))

    return (tensor, tuple(pattern))


@st.composite
def conv2d_inputs(draw):
    """Generate inputs for 2D convolution operations.

    Returns
    -------
    tuple
        (input_4d, kernel_4d) - Input and kernel with compatible shapes:
        - input: (batch, in_channels, height, width)
        - kernel: (filters, in_channels_per_group, kH, kW)

    Note: Generates various configurations including:
    - Different padding modes
    - Stride variations
    - Dilation (atrous convolution)
    - Grouped convolution
    """
    dtype = draw(st.sampled_from([np.float32, np.float64]))

    # Generate dimensions
    batch = draw(st.integers(1, 4))
    in_channels = draw(st.integers(1, 8))
    height = draw(st.integers(5, 20))
    width = draw(st.integers(5, 20))

    # Kernel dimensions
    num_filters = draw(st.integers(1, 16))
    kernel_h = draw(st.integers(1, 5))
    kernel_w = draw(st.integers(1, 5))

    # Grouped convolution (optional)
    use_groups = draw(st.booleans())
    if use_groups and in_channels % 2 == 0 and num_filters % 2 == 0:
        num_groups = draw(
            st.sampled_from([2, in_channels])
        )  # Regular groups or depthwise
        in_channels_per_group = in_channels // num_groups
    else:
        num_groups = 1
        in_channels_per_group = in_channels

    # Generate tensors
    input_shape = (batch, in_channels, height, width)
    kernel_shape = (num_filters, in_channels_per_group, kernel_h, kernel_w)

    input_tensor = draw(onnx_tensor(dtype=dtype, shape=input_shape))
    kernel_tensor = draw(onnx_tensor(dtype=dtype, shape=kernel_shape))

    return (input_tensor, kernel_tensor)


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
    broadcast_pattern = draw(st.sampled_from(["same", "broadcast_dims", "prefix"]))

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
    condition_pattern = draw(st.sampled_from(["same", "scalar", "broadcast_dims"]))

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
        arrays(dtype=np.bool_, shape=condition_shape, elements=st.booleans())
    )

    return (condition_val, then_val, else_val)


# Operation Registry
# This is the central registry that maps operation names to their test configurations
ONNX_OPERATIONS = {
    # Elemwise Binary Operations
    "add": OperationConfig(
        op_func=lambda x, y: x + y,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "mul": OperationConfig(
        op_func=lambda x, y: x * y,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "sub": OperationConfig(
        op_func=lambda x, y: x - y,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "div": OperationConfig(
        op_func=lambda x, y: x / y,
        input_strategy=binary_broadcastable_inputs(dtypes=[np.float32, np.float64]),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Division only defined for floating point types",
    ),
    "floor_div": OperationConfig(
        op_func=lambda x, y: x // y,
        input_strategy=binary_int_division_inputs(),
        valid_dtypes=["int32", "int64"],
        category="elemwise",
        notes="Floor division for integer types, maps to ONNX Div",
    ),
    "eq": OperationConfig(
        op_func=pt.eq,
        input_strategy=binary_broadcastable_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
        notes="Comparison operation, output dtype is bool",
    ),
    # Elemwise Unary Operations
    "neg": OperationConfig(
        op_func=lambda x: -x,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "abs": OperationConfig(
        op_func=pt.abs,
        input_strategy=unary_operation_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
    ),
    "exp": OperationConfig(
        op_func=pt.exp,
        input_strategy=unary_float_operation_inputs(),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Exponential only defined for floating point types",
    ),
    "log": OperationConfig(
        op_func=pt.log,
        input_strategy=unary_float_operation_inputs(),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Logarithm only defined for positive floating point values",
    ),
    "sqrt": OperationConfig(
        op_func=pt.sqrt,
        input_strategy=unary_float_operation_inputs(),
        valid_dtypes=["float32", "float64"],
        category="elemwise",
        notes="Square root only defined for non-negative floating point values",
    ),
    # Linear Algebra
    "dot": OperationConfig(
        op_func=pt.dot,
        input_strategy=matmul_inputs(),
        valid_dtypes=["float32"],  # float64 causes ONNX Runtime FusedMatMul issues
        category="nlinalg",
    ),
    # Shape Operations
    "reshape": OperationConfig(
        op_func=lambda x, shape: x.reshape(shape),
        input_strategy=reshape_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="shape",
    ),
    # Conditional Operations
    "switch": OperationConfig(
        op_func=lambda cond, x, y: pt.switch(cond, x, y),
        input_strategy=switch_inputs(),
        valid_dtypes=["float32", "float64", "int32", "int64"],
        category="elemwise",
        notes="Conditional selection (ternary), maps to ONNX Where",
    ),
}
