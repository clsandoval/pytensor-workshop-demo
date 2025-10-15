"""JAX dispatch for resize operations.

This module implements JAX backend support for PyTensor's Resize operation,
which is critical for deep learning architectures like YOLO11n that use
Feature Pyramid Networks (FPN) with spatial upsampling/downsampling.

Implementation Notes:
---------------------
- **Nearest Neighbor**: Exact match with NumPy backend using floor-based
  coordinate mapping. All numerical results match precisely.

- **Bilinear Interpolation**: Functional match with NumPy/scipy backend,
  but numerical results differ due to different interpolation algorithms.
  JAX's image.resize and scipy's ndimage.zoom use different coordinate
  conventions and kernel functions. This is documented and acceptable
  for practical use.

- **Gradients**: Implemented via inverse resize operation. Works correctly
  for all nearest neighbor cases and most bilinear cases. One known limitation:
  bilinear downsample gradients with symbolic shapes trigger JAX JIT tracing
  issues.

Primary Use Case:
----------------
YOLO11n FPN upsampling: 2x nearest neighbor upsampling to match feature
map resolutions before concatenation. This is fully supported and tested.
"""

import jax.image
import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.resize import Resize


@jax_funcify.register(Resize)
def jax_funcify_Resize(op, node, **kwargs):
    """
    Convert PyTensor Resize to JAX operations that match NumPy backend.

    The PyTensor NumPy backend uses:
    - Nearest: floor-based coordinate mapping
    - Linear: scipy.ndimage.zoom with order=1

    We replicate this behavior in JAX for consistency.

    Parameters from op:
    - scale_factor: (scale_h, scale_w)
    - mode: 'nearest' or 'linear' (bilinear)

    Returns:
        Function that performs resizing using JAX
    """
    scale_h, scale_w = op.scale_factor
    mode = op.mode

    if mode == "nearest":

        def resize_nearest(input):
            """Perform nearest neighbor resize matching NumPy backend.

            Uses floor-based coordinate mapping: floor(out_idx / scale)
            """
            _batch, _channels, height, width = input.shape

            # Calculate output dimensions
            out_height = int(height * scale_h)
            out_width = int(width * scale_w)

            # Create coordinate mappings (matching NumPy implementation)
            # For each output pixel, find nearest input pixel
            out_h_coords = jnp.floor(jnp.arange(out_height) / scale_h).astype(jnp.int32)
            out_w_coords = jnp.floor(jnp.arange(out_width) / scale_w).astype(jnp.int32)

            # Clip to valid range
            out_h_coords = jnp.clip(out_h_coords, 0, height - 1)
            out_w_coords = jnp.clip(out_w_coords, 0, width - 1)

            # Index into input using nearest neighbor
            # Use advanced indexing: input[:, :, h_coords[:, None], w_coords]
            result = input[:, :, out_h_coords[:, None], out_w_coords]

            return result

        return resize_nearest

    elif mode == "linear":

        def resize_linear(input):
            """Perform bilinear interpolation using JAX's image.resize.

            Note: JAX's bilinear interpolation may differ slightly from scipy.ndimage.zoom
            due to different boundary handling and interpolation algorithms.
            We use antialias=False to get closer to scipy's behavior.
            """
            batch, channels, height, width = input.shape

            # Calculate output dimensions
            out_height = int(height * scale_h)
            out_width = int(width * scale_w)

            # JAX image.resize expects NHWC format, but PyTensor uses NCHW
            # Transpose: NCHW → NHWC
            input_nhwc = jnp.transpose(input, (0, 2, 3, 1))

            # Resize using JAX with antialias=False for closer scipy match
            resized_nhwc = jax.image.resize(
                input_nhwc,
                shape=(batch, out_height, out_width, channels),
                method="bilinear",
                antialias=False,  # Disable antialiasing to match scipy better
            )

            # Transpose back: NHWC → NCHW
            result = jnp.transpose(resized_nhwc, (0, 3, 1, 2))

            return result

        return resize_linear

    else:
        raise ValueError(f"Unsupported resize mode: {mode}")
