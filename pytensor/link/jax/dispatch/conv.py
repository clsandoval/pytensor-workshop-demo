"""JAX dispatch for convolution operations."""

import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.conv.abstract_conv import (
    AbstractConv_gradInputs,
    AbstractConv_gradWeights,
    BaseAbstractConv,
)


def _convert_border_mode_to_jax_padding(border_mode, filter_shape):
    """
    Convert PyTensor border_mode to JAX padding format.

    Args:
        border_mode: 'valid', 'full', 'half', or tuple of ints/tuples
        filter_shape: (kH, kW) - kernel height and width

    Returns:
        JAX padding: 'VALID', 'SAME', or list of tuples [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]
    """
    if border_mode == "valid":
        return "VALID"
    elif border_mode == "half":
        # Half padding: pad with filter_size // 2
        # For odd-sized filters, this gives output_size == input_size (with stride=1)
        return "SAME"
    elif border_mode == "full":
        # Full padding: pad with (filter_size - 1)
        # Output is larger than input
        pad_h = filter_shape[0] - 1
        pad_w = filter_shape[1] - 1
        return [(pad_h, pad_h), (pad_w, pad_w)]
    elif isinstance(border_mode, int):
        # Single int: apply same padding to both dimensions
        return [(border_mode, border_mode), (border_mode, border_mode)]
    elif isinstance(border_mode, (tuple, list)):
        # Convert tuple padding to JAX format
        result = []
        for pad_spec in border_mode:
            if isinstance(pad_spec, int):
                # Single int: symmetric padding
                result.append((pad_spec, pad_spec))
            elif isinstance(pad_spec, (tuple, list)) and len(pad_spec) == 2:
                # Tuple of (before, after) or (left, right)
                result.append(tuple(pad_spec))
            else:
                raise ValueError(f"Invalid padding specification: {pad_spec}")
        return result
    else:
        raise ValueError(f"Unsupported border_mode: {border_mode}")


@jax_funcify.register(BaseAbstractConv)
def jax_funcify_BaseAbstractConv(op, node, **kwargs):
    """
    Convert PyTensor Conv2D to JAX conv_general_dilated.

    Parameters from op:
    - convdim: Number of convolution dimensions (2 or 3)
    - subsample: (stride_h, stride_w) - stride
    - border_mode: 'valid', 'full', 'half', or tuple - padding mode
    - filter_dilation: (dilation_h, dilation_w) - dilation rate
    - filter_flip: bool - True for convolution, False for cross-correlation
    - num_groups: int - for grouped/depthwise convolution

    Returns:
        Function that performs convolution using JAX
    """
    # Extract op attributes
    convdim = op.convdim
    subsample = op.subsample
    border_mode = op.border_mode
    filter_dilation = op.filter_dilation
    num_groups = op.num_groups
    filter_flip = op.filter_flip

    # Only support 2D convolution for now
    if convdim != 2:
        raise NotImplementedError(
            f"Only 2D convolution supported, got convdim={convdim}"
        )

    # Dimension numbers: PyTensor uses NCHW format for both input and filters
    # JAX conv_general_dilated expects:
    # - lhs (input): batch, spatial..., features (but we'll use NCHW)
    # - rhs (kernel): spatial..., in_features, out_features (but we'll use OIHW)
    # The dimension_numbers parameter specifies the actual layout
    dimension_numbers = ("NCHW", "OIHW", "NCHW")

    def conv2d(input, filters):
        """
        Perform convolution using JAX.

        Args:
            input: (N, C_in, H, W)
            filters: (C_out, C_in, kH, kW)

        Returns:
            output: (N, C_out, H', W')
        """
        # Get filter shape for padding calculation
        filter_shape = filters.shape[-2:]  # (kH, kW)

        # Convert border_mode to JAX padding
        padding = _convert_border_mode_to_jax_padding(border_mode, filter_shape)

        # Handle filter flip
        if filter_flip:
            # Flip kernel spatially for true convolution
            # JAX uses cross-correlation by default, so we flip for convolution
            filters = jnp.flip(filters, axis=(-2, -1))

        # Call JAX convolution
        output = jax.lax.conv_general_dilated(
            lhs=input,
            rhs=filters,
            window_strides=subsample,
            padding=padding,
            lhs_dilation=(1, 1),  # Input dilation (not used in standard conv)
            rhs_dilation=filter_dilation,  # Filter/kernel dilation (atrous convolution)
            dimension_numbers=dimension_numbers,
            feature_group_count=num_groups,  # For grouped/depthwise convolution
        )

        return output

    return conv2d


@jax_funcify.register(AbstractConv_gradInputs)
def jax_funcify_AbstractConv_gradInputs(op, node, **kwargs):
    """
    Convert PyTensor Conv2D gradient w.r.t. inputs to JAX.

    This computes the gradient of the convolution output with respect to the input.

    The gradient computation follows the PyTensor reference:
    1. Transpose filters from (C_out, C_in, kH, kW) to (C_in, C_out, kH, kW)
    2. Use "full" convolution mode
    3. Handle stride > 1 by upsampling grad_output
    4. Handle filter_flip appropriately

    Parameters from op:
    - Same as BaseAbstractConv

    The function receives:
    - filters: (C_out, C_in, kH, kW)
    - grad_output: (N, C_out, H', W') - gradient flowing back from output
    - shape: Expected output shape (input shape)

    Returns:
        grad_input: (N, C_in, H, W)
    """
    # Extract op attributes
    convdim = op.convdim
    subsample = op.subsample
    border_mode = op.border_mode
    filter_dilation = op.filter_dilation
    num_groups = op.num_groups
    filter_flip = op.filter_flip

    # Only support 2D convolution for now
    if convdim != 2:
        raise NotImplementedError(
            f"Only 2D convolution supported, got convdim={convdim}"
        )

    dimension_numbers = ("NCHW", "OIHW", "NCHW")

    def conv2d_grad_inputs(filters, grad_output, output_shape):
        """
        Compute gradient w.r.t. inputs using full convolution.

        Args:
            filters: (C_out, C_in, kH, kW)
            grad_output: (N, C_out, H', W')
            output_shape: Shape of the input

        Returns:
            grad_input: (N, C_in, H, W)
        """
        # Get filter shape
        filter_shape = filters.shape[-2:]  # (kH, kW)

        # Calculate dilated filter shape
        dil_shape = (
            (filter_shape[0] - 1) * filter_dilation[0] + 1,
            (filter_shape[1] - 1) * filter_dilation[1] + 1,
        )

        # Get padding for the forward pass
        padding = _convert_border_mode_to_jax_padding(border_mode, dil_shape)

        # Handle stride > 1 by upsampling grad_output
        if any(s > 1 for s in subsample):
            # Calculate padded input size
            if isinstance(padding, str):
                if padding == "VALID":
                    pad_h = pad_w = 0
                elif padding == "SAME":
                    pad_h = dil_shape[0] // 2
                    pad_w = dil_shape[1] // 2
            else:
                pad_h = padding[0][0] + padding[0][1]
                pad_w = padding[1][0] + padding[1][1]

            # Expected size after padding
            padded_h = output_shape[-2] + pad_h
            padded_w = output_shape[-1] + pad_w

            # Size before pooling/stride
            pre_stride_h = (padded_h - dil_shape[0]) + 1
            pre_stride_w = (padded_w - dil_shape[1]) + 1

            # Create upsampled gradient by inserting zeros
            new_shape = (
                grad_output.shape[0],
                grad_output.shape[1],
                pre_stride_h,
                pre_stride_w,
            )
            grad_output_upsampled = jnp.zeros(new_shape, dtype=grad_output.dtype)
            grad_output_upsampled = grad_output_upsampled.at[
                :, :, :: subsample[0], :: subsample[1]
            ].set(grad_output)
            grad_output = grad_output_upsampled

        # Transpose filters: (C_out, C_in, kH, kW) -> (C_in, C_out, kH, kW)
        filters_transposed = jnp.swapaxes(filters, 0, 1)

        # For cross-correlation (filter_flip=False), we need to flip the filters
        # For true convolution (filter_flip=True), filters are already flipped in forward,
        # so we need to flip topgrad and result instead
        if filter_flip:
            # Forward used true convolution, so flip topgrad for backward
            grad_output = jnp.flip(grad_output, axis=(-2, -1))
        else:
            # Forward used cross-correlation, so flip filters for backward
            filters_transposed = jnp.flip(filters_transposed, axis=(-2, -1))

        # Use "full" convolution for gradient computation
        # Full padding = (filter_size - 1)
        full_padding = [
            (dil_shape[0] - 1, dil_shape[0] - 1),
            (dil_shape[1] - 1, dil_shape[1] - 1),
        ]

        # Compute gradient using regular convolution with full padding
        grad_input = jax.lax.conv_general_dilated(
            lhs=grad_output,
            rhs=filters_transposed,
            window_strides=(1, 1),  # Always stride=1 for gradient
            padding=full_padding,
            lhs_dilation=(1, 1),
            rhs_dilation=filter_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=num_groups,
        )

        # If filter_flip=True, flip result
        if filter_flip:
            grad_input = jnp.flip(grad_input, axis=(-2, -1))

        return grad_input

    return conv2d_grad_inputs


@jax_funcify.register(AbstractConv_gradWeights)
def jax_funcify_AbstractConv_gradWeights(op, node, **kwargs):
    """
    Convert PyTensor Conv2D gradient w.r.t. weights/filters to JAX.

    This computes the gradient of the convolution output with respect to the filters.

    Parameters from op:
    - Same as BaseAbstractConv

    The function receives:
    - input: (N, C_in, H, W)
    - grad_output: (N, C_out, H', W') - gradient flowing back from output
    - shape: Expected output shape (filter shape)

    Returns:
        grad_filters: (C_out, C_in, kH, kW)
    """
    # Extract op attributes
    convdim = op.convdim
    subsample = op.subsample
    border_mode = op.border_mode
    filter_dilation = op.filter_dilation
    filter_flip = op.filter_flip

    # Only support 2D convolution for now
    if convdim != 2:
        raise NotImplementedError(
            f"Only 2D convolution supported, got convdim={convdim}"
        )

    def conv2d_grad_weights(input, grad_output, output_shape):
        """
        Compute gradient w.r.t. filters.

        The gradient of conv(input, filters) w.r.t. filters is computed by
        convolving the input with the gradient output.

        Args:
            input: (N, C_in, H, W)
            grad_output: (N, C_out, H', W')
            output_shape: Shape of the filters - (C_out, C_in, kH, kW)

        Returns:
            grad_filters: (C_out, C_in, kH, kW) - same dtype as grad_output
        """
        # Ensure input matches grad_output dtype
        input_converted = input.astype(grad_output.dtype)

        # Extract filter dimensions from output_shape
        # output_shape is (C_out, C_in, kH, kW)
        kH = (
            output_shape[2]
            if hasattr(output_shape, "__getitem__") and len(output_shape) > 2
            else output_shape[-2]
        )
        kW = (
            output_shape[3]
            if hasattr(output_shape, "__getitem__") and len(output_shape) > 3
            else output_shape[-1]
        )

        # Compute padding for the input
        dil_shape = (
            (kH - 1) * filter_dilation[0] + 1,
            (kW - 1) * filter_dilation[1] + 1,
        )
        padding_tuple = _convert_border_mode_to_jax_padding(border_mode, dil_shape)

        # Apply padding to input if needed
        if isinstance(padding_tuple, str) and padding_tuple == "VALID":
            input_padded = input_converted
        elif isinstance(padding_tuple, str) and padding_tuple == "SAME":
            # For SAME padding, compute the padding amounts
            pad_h = (dil_shape[0] - 1) // 2
            pad_w = (dil_shape[1] - 1) // 2
            input_padded = jnp.pad(
                input_converted,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )
        elif isinstance(padding_tuple, list):
            # Explicit padding
            input_padded = jnp.pad(
                input_converted,
                ((0, 0), (0, 0), padding_tuple[0], padding_tuple[1]),
                mode="constant",
            )
        else:
            input_padded = input_converted

        # If stride > 1, we need to upsample grad_output by inserting zeros
        if any(s > 1 for s in subsample):
            # Create upsampled gradient
            new_shape = (
                grad_output.shape[0],
                grad_output.shape[1],
                (input_padded.shape[2] - dil_shape[0] + 1),
                (input_padded.shape[3] - dil_shape[1] + 1),
            )
            grad_output_upsampled = jnp.zeros(new_shape, dtype=grad_output.dtype)
            # Insert the grad_output values at strided positions
            grad_output_upsampled = grad_output_upsampled.at[
                :, :, :: subsample[0], :: subsample[1]
            ].set(grad_output)
            grad_output_to_use = grad_output_upsampled
        else:
            grad_output_to_use = grad_output

        # Transpose: swap batch and channel dimensions
        # Input: (N, C_in, H, W) -> (C_in, N, H, W)
        # Grad: (N, C_out, H', W') -> (C_out, N, H', W')
        input_transposed = jnp.swapaxes(input_padded, 0, 1)
        grad_transposed = jnp.swapaxes(grad_output_to_use, 0, 1)

        # Flip grad_output spatially for the convolution
        grad_flipped = jnp.flip(grad_transposed, axis=(-2, -1))

        # Compute gradient as convolution with mode='valid'
        # Result: (C_in, C_out, kH, kW)
        grad_filters = jax.lax.conv_general_dilated(
            lhs=input_transposed,  # (C_in, N, H, W)
            rhs=grad_flipped,  # (C_out, N, H', W')
            window_strides=(1, 1),  # Always use stride=1 for weight gradient
            padding="VALID",
            lhs_dilation=(1, 1),
            rhs_dilation=(1, 1),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=1,
        )

        # Transpose to get (C_out, C_in, kH, kW)
        grad_filters = jnp.swapaxes(grad_filters, 0, 1)

        # Handle filter dilation by subsampling
        if any(d > 1 for d in filter_dilation):
            grad_filters = grad_filters[
                :, :, :: filter_dilation[0], :: filter_dilation[1]
            ]

        # Handle filter flip
        if filter_flip:
            grad_filters = jnp.flip(grad_filters, axis=(-2, -1))

        # Ensure output matches grad_output dtype
        grad_filters = grad_filters.astype(grad_output.dtype)

        return grad_filters

    return conv2d_grad_weights
