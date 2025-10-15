"""
YOLO11n building blocks for PyTensor.

Implements:
- ConvBNSiLU: Conv + BatchNorm + SiLU activation
- Bottleneck: Standard bottleneck block with two convolutions
- C3k2: CSP bottleneck with 2 convolutions
- SPPF: Spatial Pyramid Pooling - Fast
- C2PSA: CSP with Parallel Spatial Attention

IMPORTANT: Set PYTENSOR_FLAGS='floatX=float32' before importing this module!
"""

import numpy as np

import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.batchnorm import batch_normalization
from pytensor.tensor.conv.abstract_conv import conv2d
from pytensor.tensor.pool import pool_2d


class ConvBNSiLU:
    """
    Conv2D + BatchNorm + SiLU activation.

    The fundamental building block used throughout YOLO11n.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        name_prefix="conv",
    ):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        kernel_size : int
        stride : int
        padding : int or str
            If int: explicit padding
            If 'same': zero padding to maintain size
            If 'valid': no padding
        name_prefix : str
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.name = name_prefix

        # Initialize weights (He initialization for ReLU-like)
        self.W = self._init_weight(
            (out_channels, in_channels, kernel_size, kernel_size),
            name=f"{name_prefix}_W",
        )

        # BatchNorm parameters
        self.gamma = shared(
            np.ones(out_channels, dtype="float32"),
            name=f"{name_prefix}_gamma",
            borrow=True,
        )
        self.beta = shared(
            np.zeros(out_channels, dtype="float32"),
            name=f"{name_prefix}_beta",
            borrow=True,
        )
        self.bn_mean = shared(
            np.zeros(out_channels, dtype="float32"),
            name=f"{name_prefix}_bn_mean",
            borrow=True,
        )
        self.bn_var = shared(
            np.ones(out_channels, dtype="float32"),
            name=f"{name_prefix}_bn_var",
            borrow=True,
        )

        self.params = [self.W, self.gamma, self.beta]
        self.bn_stats = [self.bn_mean, self.bn_var]

    def _init_weight(self, shape, name):
        """He initialization."""
        fan_in = shape[1] * shape[2] * shape[3]  # in_channels * kh * kw
        std = np.sqrt(2.0 / fan_in)
        W_val = np.random.randn(*shape).astype("float32") * std
        return shared(W_val, name=name, borrow=True)

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : TensorVariable
            Input (batch, in_channels, height, width)

        Returns
        -------
        TensorVariable
            Output (batch, out_channels, height', width')
        """
        # Conv2D
        if self.padding == "same":
            # Calculate padding for 'same'
            pad_h = (self.kernel_size - 1) // 2
            pad_w = (self.kernel_size - 1) // 2
            border_mode = (pad_h, pad_w)
        elif self.padding == "valid":
            border_mode = "valid"
        else:
            border_mode = (self.padding, self.padding)

        conv_out = conv2d(
            x,
            self.W,
            border_mode=border_mode,
            subsample=(self.stride, self.stride),
            filter_flip=False,
        )

        # BatchNorm
        bn_out = batch_normalization(
            conv_out, self.gamma, self.beta, self.bn_mean, self.bn_var, epsilon=1e-5
        )

        # SiLU activation: SiLU(x) = x * sigmoid(x)
        silu_out = bn_out * pt.sigmoid(bn_out)

        return silu_out


class Bottleneck:
    """
    Standard bottleneck block with two convolutions.

    Used inside C3k2 blocks.
    """

    def __init__(self, in_channels, out_channels, shortcut=True, name_prefix="btlnk"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        shortcut : bool
            Whether to add residual connection
        """
        self.shortcut = shortcut and (in_channels == out_channels)

        # Two 3x3 convs
        self.conv1 = ConvBNSiLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            name_prefix=f"{name_prefix}_conv1",
        )
        self.conv2 = ConvBNSiLU(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            name_prefix=f"{name_prefix}_conv2",
        )

        self.params = self.conv1.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.conv2.bn_stats

    def __call__(self, x):
        """Forward pass."""
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.shortcut:
            out = out + residual

        return out


class C3k2:
    """
    C3k2 block: CSP Bottleneck with 2 convolutions.

    Key component of YOLO11n backbone.
    """

    def __init__(
        self, in_channels, out_channels, n_blocks=1, shortcut=True, name_prefix="c3k2"
    ):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        n_blocks : int
            Number of bottleneck blocks
        shortcut : bool
            Whether bottlenecks use residual connections
        """
        self.n_blocks = n_blocks
        hidden_channels = out_channels // 2

        # Split convolution
        self.conv1 = ConvBNSiLU(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix=f"{name_prefix}_conv1",
        )

        # Bottleneck blocks
        self.bottlenecks = []
        for i in range(n_blocks):
            self.bottlenecks.append(
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    name_prefix=f"{name_prefix}_btlnk{i}",
                )
            )

        # Merge convolution
        self.conv2 = ConvBNSiLU(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix=f"{name_prefix}_conv2",
        )

        # Collect params
        self.params = self.conv1.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.conv2.bn_stats
        for btlnk in self.bottlenecks:
            self.params.extend(btlnk.params)
            self.bn_stats.extend(btlnk.bn_stats)

    def __call__(self, x):
        """Forward pass."""
        # Split path
        x1 = self.conv1(x)

        # Bottleneck path
        x2 = x1
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)

        # Concatenate and merge
        x_cat = pt.concatenate([x1, x2], axis=1)  # Channel axis
        out = self.conv2(x_cat)

        return out


class SPPF:
    """
    Spatial Pyramid Pooling - Fast.

    Uses cascaded max pooling to create multi-scale features.
    Critical for YOLO11n's receptive field.
    """

    def __init__(self, in_channels, out_channels, pool_size=5, name_prefix="sppf"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        pool_size : int
            Max pool kernel size
        """
        hidden_channels = in_channels // 2

        self.conv1 = ConvBNSiLU(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix=f"{name_prefix}_conv1",
        )

        self.pool_size = pool_size

        self.conv2 = ConvBNSiLU(
            hidden_channels * 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix=f"{name_prefix}_conv2",
        )

        self.params = self.conv1.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.conv2.bn_stats

    def __call__(self, x):
        """Forward pass."""
        x = self.conv1(x)

        # Cascaded max pooling with padding to maintain spatial dimensions
        pad = self.pool_size // 2

        y1 = pool_2d(
            x,
            ws=(self.pool_size, self.pool_size),
            stride=(1, 1),
            mode="max",
            padding=(pad, pad),
        )
        y2 = pool_2d(
            y1,
            ws=(self.pool_size, self.pool_size),
            stride=(1, 1),
            mode="max",
            padding=(pad, pad),
        )
        y3 = pool_2d(
            y2,
            ws=(self.pool_size, self.pool_size),
            stride=(1, 1),
            mode="max",
            padding=(pad, pad),
        )

        # Concatenate all pooling outputs
        out = pt.concatenate([x, y1, y2, y3], axis=1)
        out = self.conv2(out)

        return out


class C2PSA:
    """
    C2PSA: CSP with Parallel Spatial Attention.

    Simplified implementation - uses channel attention.
    Full spatial attention can be added if needed.
    """

    def __init__(self, in_channels, out_channels, name_prefix="c2psa"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        """
        hidden_channels = out_channels // 2

        self.conv1 = ConvBNSiLU(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix=f"{name_prefix}_conv1",
        )

        # Attention module (simplified)
        self.attn_conv = ConvBNSiLU(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            name_prefix=f"{name_prefix}_attn",
        )

        self.conv2 = ConvBNSiLU(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix=f"{name_prefix}_conv2",
        )

        self.params = self.conv1.params + self.attn_conv.params + self.conv2.params
        self.bn_stats = (
            self.conv1.bn_stats + self.attn_conv.bn_stats + self.conv2.bn_stats
        )

    def __call__(self, x):
        """Forward pass."""
        # Split
        x1 = self.conv1(x)

        # Attention branch
        x2 = self.attn_conv(x1)

        # Apply attention (element-wise multiplication with sigmoid gating)
        # Simplified: just concatenate for now
        # Full version would compute attention weights

        # Concatenate and merge
        x_cat = pt.concatenate([x1, x2], axis=1)
        out = self.conv2(x_cat)

        return out
