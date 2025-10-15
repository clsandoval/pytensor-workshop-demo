"""Pooling operations for PyTensor."""

import numpy as np

import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType


class MaxPoolGrad(Op):
    """
    Gradient of max pooling operation.

    This operation computes the gradient with respect to the input of a max pooling
    operation. The gradient is routed back to the positions that contained the
    maximum values during the forward pass.

    Parameters
    ----------
    ws : tuple of int
        Window size (kernel size) for pooling
    stride : tuple of int
        Stride for pooling window
    padding : tuple of int
        Padding that was applied to input
    """

    __props__ = ("ws", "stride", "padding")

    def __init__(self, ws, stride, padding=(0, 0)):
        self.ws = tuple(ws)
        self.stride = tuple(stride)
        self.padding = tuple(padding)

    def make_node(self, x, gz):
        """Create an Apply node for this gradient operation."""
        x = pt.as_tensor_variable(x)
        gz = pt.as_tensor_variable(gz)

        # Output gradient has same shape as input
        output_type = TensorType(dtype=x.type.dtype, shape=(None,) * 4)

        return Apply(self, [x, gz], [output_type()])

    def perform(self, node, inputs, output_storage):
        """Execute the gradient computation using NumPy."""
        x, gz = inputs

        # Compute gradient
        gx = self._perform_max_pool_grad(x, gz)

        output_storage[0][0] = gx

    def _perform_max_pool_grad(self, x, gz):
        """Perform max pooling gradient computation."""
        batch, channels, _height, _width = x.shape
        pool_h, pool_w = self.ws
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Apply padding if needed (same as forward pass)
        if pad_h > 0 or pad_w > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=-np.inf,
            )
        else:
            x_padded = x

        # Initialize gradient with respect to input (same shape as padded input)
        gx_padded = np.zeros_like(x_padded)

        # Calculate output dimensions
        out_height, out_width = gz.shape[2], gz.shape[3]

        # Distribute gradient to max positions
        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        h_end = h_start + pool_h
                        w_end = w_start + pool_w

                        # Extract pool region
                        pool_region = x_padded[b, c, h_start:h_end, w_start:w_end]

                        # Find position of maximum value
                        max_val = np.max(pool_region)
                        # Create mask for positions that equal the max
                        # (handles ties by distributing gradient equally)
                        mask = (pool_region == max_val).astype(x.dtype)

                        # Distribute gradient to max positions
                        # If there are ties, gradient is split equally
                        gx_padded[b, c, h_start:h_end, w_start:w_end] += (
                            gz[b, c, i, j] * mask / np.sum(mask)
                        )

        # Remove padding if it was applied
        if pad_h > 0 or pad_w > 0:
            gx = gx_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            gx = gx_padded

        return gx

    def infer_shape(self, fgraph, node, input_shapes):
        """Infer output shape - same as input x shape."""
        return [input_shapes[0]]


class Pool(Op):
    """
    Pooling operation for tensors.

    Applies a pooling function (max, average, etc.) over spatial dimensions.

    Parameters
    ----------
    ws : tuple of int
        Window size (kernel size) for pooling. For 2D: (height, width).
    stride : tuple of int, optional
        Stride for pooling window. Defaults to ws (non-overlapping).
    padding : tuple of int, optional
        Padding to add to input. For 2D: (pad_h, pad_w). Defaults to (0, 0).
    mode : {'max', 'average'}
        Pooling mode. Currently only 'max' is implemented.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x")
    >>> y = pool_2d(x, ws=(2, 2), mode="max")
    """

    __props__ = ("ws", "stride", "padding", "mode")

    def __init__(self, ws, stride=None, padding=(0, 0), mode="max"):
        self.ws = tuple(ws)
        self.stride = tuple(stride) if stride is not None else self.ws
        self.padding = tuple(padding)
        self.mode = mode

        if mode != "max":
            raise NotImplementedError(f"Only 'max' pooling is implemented, got: {mode}")

    def make_node(self, x):
        """Create an Apply node for this operation."""
        x = pt.as_tensor_variable(x)

        # Validate input
        if x.type.ndim != 4:
            raise ValueError(
                f"Pool requires 4D input (NCHW format), got {x.type.ndim}D tensor"
            )

        # Output has same type as input
        output_type = TensorType(dtype=x.type.dtype, shape=(None,) * 4)

        return Apply(self, [x], [output_type()])

    def perform(self, node, inputs, output_storage):
        """Execute the pooling operation using NumPy."""
        (x,) = inputs

        if self.mode == "max":
            result = self._perform_max_pool(x)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        output_storage[0][0] = result

    def _perform_max_pool(self, x):
        """Perform max pooling using NumPy."""
        batch, channels, height, width = x.shape
        pool_h, pool_w = self.ws
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=-np.inf,  # Max pooling ignores -inf
            )
            height += 2 * pad_h
            width += 2 * pad_w

        # Calculate output dimensions
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1

        # Initialize output
        output = np.zeros((batch, channels, out_height, out_width), dtype=x.dtype)

        # Perform max pooling
        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        h_end = h_start + pool_h
                        w_end = w_start + pool_w

                        # Extract pool region and compute max
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(pool_region)

        return output

    def infer_shape(self, fgraph, node, input_shapes):
        """Infer output shape from input shape."""
        (x_shape,) = input_shapes

        batch, channels, height, width = x_shape
        pool_h, pool_w = self.ws
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Calculate output shape
        if height is not None:
            out_height = (height + 2 * pad_h - pool_h) // stride_h + 1
        else:
            out_height = None

        if width is not None:
            out_width = (width + 2 * pad_w - pool_w) // stride_w + 1
        else:
            out_width = None

        return [(batch, channels, out_height, out_width)]

    def grad(self, inputs, output_grads):
        """Compute gradient of pooling operation.

        For max pooling, gradient flows only to the positions that contained
        the maximum value in each pooling window. All other positions get zero gradient.

        Parameters
        ----------
        inputs : list
            List containing the input tensor x
        output_grads : list
            List containing the gradient with respect to the output

        Returns
        -------
        list
            List containing the gradient with respect to the input
        """
        (x,) = inputs
        (gz,) = output_grads

        if self.mode == "max":
            return [
                MaxPoolGrad(ws=self.ws, stride=self.stride, padding=self.padding)(x, gz)
            ]
        else:
            raise NotImplementedError(f"Gradient not implemented for mode: {self.mode}")


def pool_2d(input, ws, stride=None, padding=(0, 0), mode="max"):
    """
    Apply 2D pooling to a 4D tensor.

    Parameters
    ----------
    input : TensorVariable
        4D tensor in NCHW format (batch, channels, height, width)
    ws : tuple of 2 ints
        Window size (kernel size): (height, width)
    stride : tuple of 2 ints, optional
        Stride for pooling window. Defaults to ws (non-overlapping).
    padding : tuple of 2 ints, optional
        Padding to add: (pad_height, pad_width). Defaults to (0, 0).
    mode : {'max', 'average'}
        Pooling mode. Currently only 'max' is supported.

    Returns
    -------
    TensorVariable
        Pooled tensor, same rank as input with reduced spatial dimensions.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x", dtype="float32")
    >>> # Max pool with 2x2 kernel
    >>> y = pool_2d(x, ws=(2, 2), mode="max")
    >>> # Max pool with 3x3 kernel and stride 1
    >>> y = pool_2d(x, ws=(3, 3), stride=(1, 1), mode="max")
    """
    return Pool(ws=ws, stride=stride, padding=padding, mode=mode)(input)
