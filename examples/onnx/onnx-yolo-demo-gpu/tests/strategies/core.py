"""Core Hypothesis strategies for YOLO testing."""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def valid_image_shapes(min_size=32, max_size=640, multiple_of=32, fixed_size=None):
    """
    Generate valid YOLO image shapes (multiples of 32).

    YOLO requires input dimensions divisible by 32 due to 5 downsampling stages.
    """
    if fixed_size is not None:
        return st.just(fixed_size)

    valid_sizes = list(range(min_size, max_size + 1, multiple_of))
    return st.tuples(
        st.sampled_from(valid_sizes),  # height
        st.sampled_from(valid_sizes),  # width
    )


def batch_sizes(min_size=1, max_size=4):
    """Generate valid batch sizes for testing."""
    return st.integers(min_value=min_size, max_value=max_size)


def num_channels():
    """Generate valid number of channels (grayscale or RGB)."""
    return st.sampled_from([1, 3])


@st.composite
def yolo_image_tensor(
    draw, batch_size=None, channels=None, size=None, dtype=np.float32
):
    """
    Generate YOLO-compatible image tensor.

    Returns tensor of shape (batch_size, channels, height, width).
    Values normalized to [-1, 1] or [0, 1].
    """
    if batch_size is None:
        batch_size = draw(batch_sizes())
    if channels is None:
        channels = draw(num_channels())
    if size is None:
        size = draw(valid_image_shapes())

    height, width = size
    shape = (batch_size, channels, height, width)

    # Generate normalized image data
    elements = st.floats(
        min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )

    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def yolo_feature_map(draw, batch_size=None, num_filters=None, size=None):
    """
    Generate YOLO feature map tensor for intermediate layers.

    Feature maps typically have:
    - Larger channel counts (16-512)
    - Smaller spatial dimensions (8x8 to 40x40)
    """
    if batch_size is None:
        batch_size = draw(batch_sizes())
    if num_filters is None:
        num_filters = draw(st.sampled_from([16, 32, 64, 128, 256, 512]))

    # Handle size parameter - can be int, tuple, or strategy
    if size is None:
        size = draw(valid_image_shapes(min_size=8, max_size=40, multiple_of=8))
    elif isinstance(size, tuple):
        # If size is already a tuple, use it directly
        pass
    else:
        # If size is a strategy, draw from it
        size = draw(size)

    height, width = size
    shape = (batch_size, num_filters, height, width)

    # Feature maps can have wider value ranges
    elements = st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    )

    return draw(arrays(dtype=np.float32, shape=shape, elements=elements))
