"""JAX dispatch for batch normalization operations."""

import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.batchnorm import BatchNormalization


@jax_funcify.register(BatchNormalization)
def jax_funcify_BatchNormalization(op, node, **kwargs):
    """
    Convert PyTensor BatchNormalization to JAX operations.

    Implements: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    This is inference-mode batch normalization where mean and variance are
    pre-computed statistics (not calculated from the current batch).

    Parameters from op:
    - epsilon: Small constant for numerical stability (prevents division by zero)

    Args (from node inputs):
    - x: Input tensor (1D, 2D, or 4D)
        - 1D (C,): Single sample, C features
        - 2D (N, C): N samples, C features (fully connected layers)
        - 4D (N, C, H, W): N samples, C channels, HxW spatial (CNNs)
    - gamma: Scale parameter (1D, shape matches feature dimension)
    - beta: Shift parameter (1D, shape matches feature dimension)
    - mean: Running mean (1D, shape matches feature dimension)
    - variance: Running variance (1D, shape matches feature dimension)

    Returns:
        Function that performs batch normalization using JAX operations.

    Notes:
        Broadcasting strategy:
        - 1D input (C,): No reshaping needed, direct element-wise operations
        - 2D input (N, C): Reshape params from (C,) to (1, C)
        - 4D input (N, C, H, W): Reshape params from (C,) to (1, C, 1, 1)

        This ensures parameters broadcast correctly across batch and spatial
        dimensions while applying normalization per-channel.
    """
    epsilon = op.epsilon

    def batchnorm(x, gamma, beta, mean, variance):
        """
        Perform batch normalization.

        The normalization formula is:
            1. Normalize: x_norm = (x - mean) / sqrt(variance + epsilon)
            2. Scale and shift: output = gamma * x_norm + beta

        Broadcasting is handled by reshaping gamma, beta, mean, variance
        to match the input dimensionality.
        """
        # Determine input dimensionality
        ndim = x.ndim

        # Reshape parameters for broadcasting
        if ndim == 1:
            # 1D input (C,): No reshaping needed
            # Parameters are already (C,), element-wise operations work directly
            gamma_bc = gamma
            beta_bc = beta
            mean_bc = mean
            variance_bc = variance
        elif ndim == 2:
            # 2D input (N, C): Reshape to (1, C) for broadcasting over batch dimension
            # This allows the same normalization to be applied to each sample
            gamma_bc = gamma.reshape(1, -1)
            beta_bc = beta.reshape(1, -1)
            mean_bc = mean.reshape(1, -1)
            variance_bc = variance.reshape(1, -1)
        elif ndim == 4:
            # 4D input (N, C, H, W): Reshape to (1, C, 1, 1)
            # This broadcasts over batch and spatial dimensions, per-channel
            # Each channel has its own statistics, applied uniformly across
            # all spatial positions and samples
            gamma_bc = gamma.reshape(1, -1, 1, 1)
            beta_bc = beta.reshape(1, -1, 1, 1)
            mean_bc = mean.reshape(1, -1, 1, 1)
            variance_bc = variance.reshape(1, -1, 1, 1)
        else:
            # Unsupported dimensionality
            raise NotImplementedError(
                f"BatchNormalization for {ndim}D input not supported. "
                f"Supported dimensions: 1D, 2D, 4D"
            )

        # Normalize: (x - mean) / sqrt(variance + epsilon)
        # epsilon is added for numerical stability (prevents division by zero)
        x_normalized = (x - mean_bc) / jnp.sqrt(variance_bc + epsilon)

        # Scale and shift: gamma * x_normalized + beta
        # gamma and beta are learned affine transformation parameters
        output = gamma_bc * x_normalized + beta_bc

        return output

    return batchnorm
