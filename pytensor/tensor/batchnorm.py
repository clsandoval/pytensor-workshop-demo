"""Batch normalization operations for PyTensor.

Batch normalization is a technique to improve training stability and speed
by normalizing layer inputs. This module implements inference-mode batch
normalization as commonly used in CNN architectures like YOLO.

References
----------
Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift. ICML 2015.
"""

import numpy as np

import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op


def _get_reduce_axes(x, param):
    """
    Determine which axes to sum over when computing parameter gradients.

    For batch normalization, gamma and beta are typically per-channel parameters.
    We need to reduce (sum) over all dimensions except the channel dimension.

    Parameters
    ----------
    x : TensorVariable
        Input tensor (e.g., 4D: NCHW or 2D: NC)
    param : TensorVariable
        Parameter tensor (e.g., 1D: C)

    Returns
    -------
    tuple
        Axes to reduce over

    Examples
    --------
    For 4D input (N, C, H, W) and 1D param (C,):
        Returns (0, 2, 3) - reduce over batch, height, width

    For 2D input (N, C) and 1D param (C,):
        Returns (0,) - reduce over batch only
    """
    if x.ndim == 4 and param.ndim == 1:
        # NCHW format: reduce over batch, height, width (keep channels)
        return (0, 2, 3)
    elif x.ndim == 2 and param.ndim == 1:
        # NC format: reduce over batch only
        return (0,)
    elif x.ndim == 1:
        # 1D: no reduction needed (element-wise)
        return ()
    else:
        # General case: reduce over all except param dimension
        # Assume param corresponds to dimension 1 (channels)
        return (0, *range(2, x.ndim))


class BatchNormalization(Op):
    """
    Batch Normalization operation (inference mode).

    Applies batch normalization to the input:
        y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    This is the inference-mode implementation where mean and variance are
    pre-computed (not calculated from the batch). This is the standard mode
    used for deployed models and during evaluation.

    Parameters
    ----------
    epsilon : float
        Small constant added to variance for numerical stability.
        Default: 1e-5

    Inputs
    ------
    x : TensorVariable
        Input tensor to normalize (typically 4D for CNNs: NCHW format)
    gamma : TensorVariable
        Scale parameter (learned during training)
    beta : TensorVariable
        Shift parameter (learned during training)
    mean : TensorVariable
        Running mean (computed during training)
    variance : TensorVariable
        Running variance (computed during training)

    Outputs
    -------
    y : TensorVariable
        Normalized output, same shape as input

    Notes
    -----
    - This implements inference-only mode (no training statistics)
    - Assumes NCHW format for 4D tensors (batch, channel, height, width)
    - gamma, beta, mean, variance should have shape (C,) for 4D input
    - Common in CNNs: Conv → BatchNorm → Activation

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> from pytensor.tensor.batchnorm import batch_normalization
    >>>
    >>> # 4D input (batch=2, channels=3, height=4, width=4)
    >>> x = pt.tensor4("x")
    >>> gamma = pt.vector("gamma")  # shape (3,)
    >>> beta = pt.vector("beta")  # shape (3,)
    >>> mean = pt.vector("mean")  # shape (3,)
    >>> var = pt.vector("var")  # shape (3,)
    >>>
    >>> y = batch_normalization(x, gamma, beta, mean, var)
    """

    __props__ = ("epsilon",)

    def __init__(self, epsilon=1e-5):
        """
        Initialize BatchNormalization op.

        Parameters
        ----------
        epsilon : float
            Small constant for numerical stability
        """
        self.epsilon = float(epsilon)

    def make_node(self, x, gamma, beta, mean, variance):
        """
        Create an Apply node for batch normalization.

        Parameters
        ----------
        x : Variable
            Input tensor
        gamma : Variable
            Scale parameter
        beta : Variable
            Shift parameter
        mean : Variable
            Running mean
        variance : Variable
            Running variance

        Returns
        -------
        Apply
            Apply node with 5 inputs and 1 output
        """
        x = pt.as_tensor_variable(x)
        gamma = pt.as_tensor_variable(gamma)
        beta = pt.as_tensor_variable(beta)
        mean = pt.as_tensor_variable(mean)
        variance = pt.as_tensor_variable(variance)

        # Output has same type as input
        output_type = x.type()

        return Apply(self, [x, gamma, beta, mean, variance], [output_type])

    def perform(self, node, inputs, outputs):
        """
        Compute batch normalization in Python.

        Parameters
        ----------
        node : Apply
            The Apply node
        inputs : list
            [x, gamma, beta, mean, variance]
        outputs : list
            [output storage]
        """
        x, gamma, beta, mean, variance = inputs

        # Compute normalization
        # y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

        # For broadcasting: gamma, beta, mean, variance are typically 1D (C,)
        # and need to broadcast over x which is typically 4D (N, C, H, W)
        # We reshape them to (1, C, 1, 1) for proper broadcasting

        x_dtype = x.dtype

        # Determine broadcast shape based on input dimensions
        if x.ndim == 1:
            # 1D input: no reshaping needed (element-wise operation)
            gamma_bc = gamma
            beta_bc = beta
            mean_bc = mean
            var_bc = variance
        elif x.ndim == 2:
            # NC format: broadcast over (1, C)
            reshape_dims = (1, -1)
            gamma_bc = gamma.reshape(reshape_dims)
            beta_bc = beta.reshape(reshape_dims)
            mean_bc = mean.reshape(reshape_dims)
            var_bc = variance.reshape(reshape_dims)
        elif x.ndim == 4:
            # NCHW format: broadcast over (1, C, 1, 1)
            reshape_dims = (1, -1, 1, 1)
            gamma_bc = gamma.reshape(reshape_dims)
            beta_bc = beta.reshape(reshape_dims)
            mean_bc = mean.reshape(reshape_dims)
            var_bc = variance.reshape(reshape_dims)
        else:
            # Default: broadcast over batch dimension
            reshape_dims = tuple([1] + [-1] + [1] * (x.ndim - 2))
            gamma_bc = gamma.reshape(reshape_dims)
            beta_bc = beta.reshape(reshape_dims)
            mean_bc = mean.reshape(reshape_dims)
            var_bc = variance.reshape(reshape_dims)

        # Normalize
        normalized = (x - mean_bc) / np.sqrt(var_bc + self.epsilon)

        # Scale and shift
        result = gamma_bc * normalized + beta_bc

        # Ensure output dtype matches input
        outputs[0][0] = result.astype(x_dtype)

    def infer_shape(self, fgraph, node, input_shapes):
        """
        Infer output shape.

        Output has the same shape as input x.
        """
        return [input_shapes[0]]

    def grad(self, inputs, output_grads):
        """
        Compute gradients for batch normalization.

        Implements inference-mode gradients where mean and variance are treated
        as constants (not computed from the current batch). This is appropriate
        for fine-tuning pre-trained models or when using fixed batch statistics.

        Parameters
        ----------
        inputs : list
            [x, gamma, beta, mean, variance]
        output_grads : list
            [dy] - gradient w.r.t. output

        Returns
        -------
        list
            [grad_x, grad_gamma, grad_beta, grad_mean, grad_variance]
            where grad_mean and grad_variance are zero (constants in inference mode)

        Notes
        -----
        In inference mode, the forward pass is:
            y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

        The gradients are:
            dy/dx = gamma * dy / sqrt(variance + epsilon)
            dy/dgamma = sum(dy * (x - mean) / sqrt(variance + epsilon))
            dy/dbeta = sum(dy)
            dy/dmean = 0 (constant in inference mode)
            dy/dvariance = 0 (constant in inference mode)

        References
        ----------
        Ioffe & Szegedy (2015): Batch Normalization paper
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        """
        x, gamma, beta, mean, variance = inputs
        dy = output_grads[0]  # Gradient w.r.t output

        # Compute intermediate values
        # std = sqrt(variance + epsilon)
        std = pt.sqrt(variance + self.epsilon)

        # Need to broadcast mean, std, gamma for proper element-wise operations
        if x.ndim == 4:
            # NCHW format: reshape (C,) -> (1, C, 1, 1)
            mean_bc = mean.dimshuffle("x", 0, "x", "x")
            std_bc = std.dimshuffle("x", 0, "x", "x")
            gamma_bc = gamma.dimshuffle("x", 0, "x", "x")
        elif x.ndim == 2:
            # NC format: reshape (C,) -> (1, C)
            mean_bc = mean.dimshuffle("x", 0)
            std_bc = std.dimshuffle("x", 0)
            gamma_bc = gamma.dimshuffle("x", 0)
        else:
            # Default: no reshape needed
            mean_bc = mean
            std_bc = std
            gamma_bc = gamma

        # Normalized input: x_norm = (x - mean) / std
        x_centered = x - mean_bc
        x_norm = x_centered / std_bc

        # Gradient w.r.t. gamma: sum over all dimensions except the channel dimension
        # For 4D input (N, C, H, W), sum over (N, H, W)
        # For 2D input (N, C), sum over N
        grad_gamma = (dy * x_norm).sum(axis=_get_reduce_axes(x, gamma))

        # Gradient w.r.t. beta: sum over all dimensions except the channel dimension
        grad_beta = dy.sum(axis=_get_reduce_axes(x, beta))

        # Gradient w.r.t. x (inference mode - mean/var are constants)
        # dy/dx = gamma * dy / std
        grad_x = gamma_bc * dy / std_bc

        # No gradients for mean and variance (treated as constants in inference mode)
        grad_mean = pt.zeros_like(mean)
        grad_variance = pt.zeros_like(variance)

        return [grad_x, grad_gamma, grad_beta, grad_mean, grad_variance]


# Convenience function
def batch_normalization(x, gamma, beta, mean, variance, epsilon=1e-5):
    """
    Apply batch normalization (inference mode).

    Normalizes input using pre-computed statistics:
        y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    Parameters
    ----------
    x : TensorVariable
        Input tensor to normalize
    gamma : TensorVariable
        Scale parameter (shape must be broadcastable to x)
    beta : TensorVariable
        Shift parameter (shape must be broadcastable to x)
    mean : TensorVariable
        Running mean (shape must be broadcastable to x)
    variance : TensorVariable
        Running variance (shape must be broadcastable to x)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-5)

    Returns
    -------
    TensorVariable
        Normalized output, same shape as input

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> from pytensor.tensor.batchnorm import batch_normalization
    >>>
    >>> # Typical CNN usage (NCHW format)
    >>> x = pt.tensor4("x", dtype="float32")  # (N, C, H, W)
    >>> gamma = pt.vector("gamma", dtype="float32")  # (C,)
    >>> beta = pt.vector("beta", dtype="float32")  # (C,)
    >>> mean = pt.vector("mean", dtype="float32")  # (C,)
    >>> var = pt.vector("var", dtype="float32")  # (C,)
    >>>
    >>> y = batch_normalization(x, gamma, beta, mean, var)
    >>>
    >>> # Common pattern: Conv → BatchNorm → Activation
    >>> from pytensor.tensor.nnet.conv import conv2d
    >>> kernel = pt.tensor4("kernel", dtype="float32")
    >>> conv_out = conv2d(x, kernel)
    >>> bn_out = batch_normalization(conv_out, gamma, beta, mean, var)
    >>> activated = pt.nn.relu(bn_out)
    """
    return BatchNormalization(epsilon=epsilon)(x, gamma, beta, mean, variance)


# Make import work: from pytensor.tensor.batchnorm import BatchNormalization
__all__ = ["BatchNormalization", "batch_normalization"]
