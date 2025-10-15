"""Resize (upsample/downsample) operations for PyTensor."""

import numpy as np

import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType


class Resize(Op):
    """
    Resize operation for tensors (upsampling or downsampling).

    Supports multiple interpolation modes:
    - 'nearest': Nearest neighbor (fast, blocky)
    - 'linear': Bilinear interpolation (smooth)

    Parameters
    ----------
    scale_factor : tuple of float
        Scale factors for spatial dimensions. For 2D: (scale_h, scale_w).
        Values > 1 upsample, values < 1 downsample.
    mode : {'nearest', 'linear'}
        Interpolation mode.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x")
    >>> # 2x nearest neighbor upsampling
    >>> y = resize(x, scale_factor=(2, 2), mode="nearest")
    >>> # 1.5x bilinear upsampling
    >>> y = resize(x, scale_factor=(1.5, 1.5), mode="linear")
    """

    __props__ = ("scale_factor", "mode")

    def __init__(self, scale_factor, mode="nearest"):
        self.scale_factor = tuple(scale_factor)
        self.mode = mode

        if mode not in ("nearest", "linear"):
            raise ValueError(f"Unsupported mode: {mode}. Use 'nearest' or 'linear'.")

    def make_node(self, x):
        """Create an Apply node for this operation."""
        x = pt.as_tensor_variable(x)

        if x.type.ndim != 4:
            raise ValueError(
                f"Resize requires 4D input (NCHW format), got {x.type.ndim}D tensor"
            )

        # Output has same type as input (shape will be different)
        output_type = TensorType(dtype=x.type.dtype, shape=(None,) * 4)

        return Apply(self, [x], [output_type()])

    def perform(self, node, inputs, output_storage):
        """Execute the resize operation using NumPy."""
        (x,) = inputs

        if self.mode == "nearest":
            result = self._perform_nearest(x)
        elif self.mode == "linear":
            result = self._perform_linear(x)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        output_storage[0][0] = result

    def _perform_nearest(self, x):
        """Perform nearest neighbor resize using NumPy."""
        _batch, _channels, height, width = x.shape
        scale_h, scale_w = self.scale_factor

        # Calculate output dimensions
        out_height = int(height * scale_h)
        out_width = int(width * scale_w)

        # Create coordinate mappings
        # For each output pixel, find nearest input pixel
        out_h_coords = np.floor(np.arange(out_height) / scale_h).astype(np.int32)
        out_w_coords = np.floor(np.arange(out_width) / scale_w).astype(np.int32)

        # Clip to valid range
        out_h_coords = np.clip(out_h_coords, 0, height - 1)
        out_w_coords = np.clip(out_w_coords, 0, width - 1)

        # Index into input using nearest neighbor
        # Use advanced indexing: x[:, :, h_coords[:, None], w_coords]
        result = x[:, :, out_h_coords[:, None], out_w_coords]

        return result.astype(x.dtype)

    def _perform_linear(self, x):
        """Perform bilinear interpolation using NumPy."""
        _batch, _channels, _height, _width = x.shape
        scale_h, scale_w = self.scale_factor

        # Use scipy for bilinear interpolation
        # This is simpler than implementing bilinear from scratch
        from scipy.ndimage import zoom

        # Zoom operates on each batch and channel independently
        # zoom factors: [batch, channels, height, width]
        result = zoom(x, (1, 1, scale_h, scale_w), order=1)  # order=1 = bilinear

        return result.astype(x.dtype)

    def infer_shape(self, fgraph, node, input_shapes):
        """Infer output shape from input shape."""
        (x_shape,) = input_shapes

        batch, channels, height, width = x_shape
        scale_h, scale_w = self.scale_factor

        # Calculate output shape
        # Use symbolic multiplication and conversion to integer type
        if height is not None:
            out_height = (height * scale_h).astype("int64")
        else:
            out_height = None

        if width is not None:
            out_width = (width * scale_w).astype("int64")
        else:
            out_width = None

        return [(batch, channels, out_height, out_width)]

    def grad(self, inputs, output_grads):
        """Compute gradient of resize operation.

        The gradient of resize is implemented by applying the inverse scale factor:
        - If forward was upsample (scale > 1), gradient is downsample (scale < 1)
        - If forward was downsample (scale < 1), gradient is upsample (scale > 1)

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
        (_x,) = inputs
        (gz,) = output_grads

        # Inverse scale factors for gradient
        scale_h, scale_w = self.scale_factor
        inv_scale_h = 1.0 / scale_h
        inv_scale_w = 1.0 / scale_w

        # Apply resize with inverse scale to get gradient
        # Use same mode as forward pass
        grad_x = Resize(scale_factor=(inv_scale_h, inv_scale_w), mode=self.mode)(gz)

        return [grad_x]


def resize(input, scale_factor, mode="nearest"):
    """
    Resize a 4D tensor using interpolation.

    Parameters
    ----------
    input : TensorVariable
        4D tensor in NCHW format (batch, channels, height, width)
    scale_factor : tuple of 2 floats
        Scale factors for spatial dimensions: (scale_height, scale_width)
        Values > 1 upsample, values < 1 downsample
    mode : {'nearest', 'linear'}
        Interpolation mode:
        - 'nearest': Nearest neighbor (fast, blocky output)
        - 'linear': Bilinear interpolation (smooth output)

    Returns
    -------
    TensorVariable
        Resized tensor with shape (batch, channels, H*scale_h, W*scale_w)

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.tensor4("x", dtype="float32")
    >>> # 2x upsampling with nearest neighbor (YOLO11n FPN pattern)
    >>> y = resize(x, scale_factor=(2, 2), mode="nearest")
    >>> # 1.5x upsampling with bilinear interpolation
    >>> y = resize(x, scale_factor=(1.5, 1.5), mode="linear")
    """
    return Resize(scale_factor=scale_factor, mode=mode)(input)
