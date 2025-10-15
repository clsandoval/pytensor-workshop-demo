"""JAX dispatch for pooling operations."""

import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.pool import MaxPoolGrad, Pool


@jax_funcify.register(Pool)
def jax_funcify_Pool(op, node, **kwargs):
    """
    Convert PyTensor Pool to JAX reduce_window.

    Parameters from op:
    - ws: (pool_h, pool_w) - window size
    - stride: (stride_h, stride_w) - stride
    - padding: (pad_h, pad_w) - padding
    - mode: 'max' or 'average'

    Returns:
        Function that performs pooling using JAX
    """
    ws = op.ws
    stride = op.stride if op.stride else ws  # Default stride = ws
    padding = op.padding if op.padding else (0, 0)
    mode = op.mode

    # Set up for max pooling
    if mode == "max":
        init_value = -jnp.inf
        reducer = jax.lax.max
    else:
        raise NotImplementedError(f"Pooling mode '{mode}' not yet supported")

    # Convert padding to JAX format
    # PyTensor: (pad_h, pad_w)
    # JAX: [(pad_batch_before, pad_batch_after), (pad_channel_before, pad_channel_after),
    #       (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]
    jax_padding = [
        (0, 0),  # No padding on batch
        (0, 0),  # No padding on channel
        (padding[0], padding[0]),  # Symmetric H padding
        (padding[1], padding[1]),  # Symmetric W padding
    ]

    def pool(input):
        """
        Perform max pooling using JAX.

        Args:
            input: (N, C, H, W)

        Returns:
            output: (N, C, H', W')
        """
        # Window dimensions: (batch, channels, pool_h, pool_w)
        window_dims = (1, 1, ws[0], ws[1])

        # Window strides: (batch, channels, stride_h, stride_w)
        window_strides = (1, 1, stride[0], stride[1])

        # Apply pooling
        output = jax.lax.reduce_window(
            operand=input,
            init_value=init_value,
            computation=reducer,
            window_dimensions=window_dims,
            window_strides=window_strides,
            padding=jax_padding,
        )

        return output

    return pool


@jax_funcify.register(MaxPoolGrad)
def jax_funcify_MaxPoolGrad(op, node, **kwargs):
    """
    Convert PyTensor MaxPoolGrad to JAX gradient computation.

    JAX's automatic differentiation handles the gradient of reduce_window
    automatically when we use jax.grad. However, this dispatch is called
    when the gradient op is explicitly in the graph.

    We implement the gradient by finding which positions had the max values
    and routing gradients only to those positions.
    """
    ws = op.ws
    stride = op.stride
    padding = op.padding

    def maxpool_grad(x, gz):
        """
        Compute gradient of max pooling.

        Args:
            x: Original input to forward pass (N, C, H, W)
            gz: Gradient with respect to output (N, C, H', W')

        Returns:
            gx: Gradient with respect to input (N, C, H, W)
        """
        # Get shapes
        _batch, _channels, _in_h, _in_w = x.shape
        _, _, out_h, out_w = gz.shape

        # Apply padding to input if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = jnp.pad(
                x,
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode="constant",
                constant_values=-jnp.inf,
            )
        else:
            x_padded = x

        # Initialize gradient with zeros
        gx_padded = jnp.zeros_like(x_padded)

        # For each output position, find the input position that had the max
        # and add the gradient there
        def process_position(i, j, gx_padded):
            h_start = i * stride[0]
            w_start = j * stride[1]
            h_end = h_start + ws[0]
            w_end = w_start + ws[1]

            # Get the window
            window = x_padded[:, :, h_start:h_end, w_start:w_end]

            # Find max value in window
            max_val = jnp.max(window, axis=(2, 3), keepdims=True)  # Shape: (N, C, 1, 1)

            # Create mask for positions equal to max
            # This handles ties by distributing gradient equally
            mask = (window == max_val).astype(x.dtype)
            mask = mask / jnp.sum(mask, axis=(2, 3), keepdims=True)

            # Distribute gradient
            grad_contribution = gz[:, :, i : i + 1, j : j + 1] * mask

            # Add to the appropriate window in gx_padded
            return gx_padded.at[:, :, h_start:h_end, w_start:w_end].add(
                grad_contribution
            )

        # Process all output positions
        for i in range(out_h):
            for j in range(out_w):
                gx_padded = process_position(i, j, gx_padded)

        # Remove padding if it was applied
        if padding[0] > 0 or padding[1] > 0:
            gx = gx_padded[:, :, padding[0] : -padding[0], padding[1] : -padding[1]]
        else:
            gx = gx_padded

        return gx

    return maxpool_grad
