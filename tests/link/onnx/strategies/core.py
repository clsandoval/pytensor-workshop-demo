"""Core Hypothesis strategies for ONNX tensor generation."""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def onnx_dtypes():
    """Strategy for ONNX-supported dtypes.

    Returns dtypes that are commonly supported across:
    - PyTensor
    - ONNX
    - ONNX Runtime
    """
    return st.sampled_from(
        [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ]
    )


def valid_shapes(min_rank=1, max_rank=4, min_dim=0, max_dim=10):
    """Generate valid tensor shapes for ONNX.

    Parameters
    ----------
    min_rank : int
        Minimum number of dimensions (default: 1)
    max_rank : int
        Maximum number of dimensions (default: 4)
    min_dim : int
        Minimum size per dimension (default: 0, allows empty tensors)
    max_dim : int
        Maximum size per dimension (default: 10)

    Returns
    -------
    strategy
        Generates tuples of integers representing valid shapes

    Examples
    --------
    >>> valid_shapes().example()  # doctest: +SKIP
    (3, 5, 2)
    >>> valid_shapes(min_rank=2, max_rank=2).example()  # doctest: +SKIP
    (4, 7)
    """
    return st.lists(
        st.integers(min_value=min_dim, max_value=max_dim),
        min_size=min_rank,
        max_size=max_rank,
    ).map(tuple)


def _safe_float_elements(dtype):
    """Generate safe float elements for a dtype.

    Avoids infinities, NaNs, and extreme values that cause numerical issues.
    Uses conservative ranges to ensure numerical precision in operations like
    matrix multiplication where errors can accumulate.
    """
    if dtype in (np.float32, "float32"):
        # Float32 range: approximately ±3.4e38
        # Use smaller range to avoid numerical precision issues
        # Limit to ±1000 to ensure rtol=1e-4 works even with accumulated errors
        # Exclude values very close to zero to avoid atol comparison issues
        return st.one_of(
            st.floats(
                min_value=-1e3,
                max_value=-0.1,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            st.floats(
                min_value=0.1,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            # Include exact zero occasionally
            st.just(0.0),
        )
    elif dtype in (np.float64, "float64"):
        # Float64 range: approximately ±1.8e308
        # Use smaller range to avoid numerical precision issues
        return st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    else:
        raise ValueError(f"Unsupported float dtype: {dtype}")


def _safe_integer_elements(dtype):
    """Generate safe integer elements for a dtype."""
    if dtype in (np.int32, "int32"):
        # int32 range: -2^31 to 2^31-1
        return st.integers(min_value=-100, max_value=100)
    elif dtype in (np.int64, "int64"):
        # int64 range: -2^63 to 2^63-1
        return st.integers(min_value=-1000, max_value=1000)
    else:
        raise ValueError(f"Unsupported integer dtype: {dtype}")


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

    # Generate value based on dtype (0-D tensor)
    if dtype == np.bool_ or dtype == "bool":
        value = draw(st.booleans())
        return np.array(value, dtype=np.bool_)
    elif np.issubdtype(dtype, np.floating):
        if value_range is None:
            value_range = (-1e3, 1e3)
        value = draw(
            st.floats(
                min_value=value_range[0],
                max_value=value_range[1],
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return np.array(value, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        if value_range is None:
            value_range = (-100, 100)
        value = draw(st.integers(min_value=value_range[0], max_value=value_range[1]))
        return np.array(value, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@st.composite
def onnx_tensor(draw, dtype=None, shape=None, elements=None):
    """Generate ONNX-compatible tensor.

    Parameters
    ----------
    dtype : numpy dtype or None
        Tensor dtype. If None, randomly chosen from onnx_dtypes()
    shape : tuple or None
        Tensor shape. If None, randomly generated
    elements : strategy or None
        Strategy for generating element values. If None, uses safe defaults

    Returns
    -------
    numpy.ndarray
        Tensor compatible with ONNX operations

    Examples
    --------
    >>> # Random tensor
    >>> onnx_tensor().example()  # doctest: +SKIP
    array([[1.2, 3.4], [5.6, 7.8]], dtype=float32)

    >>> # Specific dtype
    >>> onnx_tensor(dtype=np.int32).example()  # doctest: +SKIP
    array([10, 20, 30], dtype=int32)

    >>> # Specific shape
    >>> onnx_tensor(shape=(2, 3)).example()  # doctest: +SKIP
    array([[...]], dtype=float32)
    """
    # Generate dtype if not provided
    if dtype is None:
        dtype = draw(onnx_dtypes())

    # Generate shape if not provided
    if shape is None:
        shape = draw(valid_shapes())

    # Generate elements strategy if not provided
    if elements is None:
        if np.issubdtype(dtype, np.floating):
            elements = _safe_float_elements(dtype)
        elif np.issubdtype(dtype, np.integer):
            elements = _safe_integer_elements(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    # Generate array
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))
