"""
YOLO11n model architecture for PyTensor.

Implements full YOLO11n nano model for object detection.
Default input: (batch, 3, 320, 320)
Output: Detection predictions at 3 scales

IMPORTANT: Set PYTENSOR_FLAGS='floatX=float32' before importing this module!
"""

from blocks import C2PSA, SPPF, C3k2, ConvBNSiLU

import pytensor.tensor as pt


class YOLO11nBackbone:
    """
    YOLO11n backbone for feature extraction.

    Outputs features at 3 scales: P3, P4, P5
    For 320x320 input: P3 (40x40), P4 (20x20), P5 (10x10)
    """

    def __init__(self, in_channels=3):
        """Initialize backbone."""
        # Stem
        self.conv0 = ConvBNSiLU(
            3, 16, kernel_size=3, stride=2, padding="same", name_prefix="stem"
        )

        # Stage 1
        self.conv1 = ConvBNSiLU(
            16, 32, kernel_size=3, stride=2, padding="same", name_prefix="s1_conv"
        )
        self.c3k2_1 = C3k2(32, 32, n_blocks=1, name_prefix="s1_c3k2")

        # Stage 2 (P3)
        self.conv2 = ConvBNSiLU(
            32, 64, kernel_size=3, stride=2, padding="same", name_prefix="s2_conv"
        )
        self.c3k2_2 = C3k2(64, 64, n_blocks=2, name_prefix="s2_c3k2")

        # Stage 3 (P4)
        self.conv3 = ConvBNSiLU(
            64, 128, kernel_size=3, stride=2, padding="same", name_prefix="s3_conv"
        )
        self.c3k2_3 = C3k2(128, 128, n_blocks=2, name_prefix="s3_c3k2")

        # Stage 4 (P5)
        self.conv4 = ConvBNSiLU(
            128, 256, kernel_size=3, stride=2, padding="same", name_prefix="s4_conv"
        )
        self.c3k2_4 = C3k2(256, 256, n_blocks=1, name_prefix="s4_c3k2")

        # SPPF
        self.sppf = SPPF(256, 256, pool_size=5, name_prefix="sppf")

        # C2PSA
        self.c2psa = C2PSA(256, 256, name_prefix="c2psa")

        # Collect parameters
        self.params = []
        self.bn_stats = []
        for module in [
            self.conv0,
            self.conv1,
            self.c3k2_1,
            self.conv2,
            self.c3k2_2,
            self.conv3,
            self.c3k2_3,
            self.conv4,
            self.c3k2_4,
            self.sppf,
            self.c2psa,
        ]:
            self.params.extend(module.params)
            self.bn_stats.extend(module.bn_stats)

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : TensorVariable
            Input (batch, 3, 320, 320)

        Returns
        -------
        p3, p4, p5 : TensorVariables
            Features at 3 scales:
            - p3: (batch, 64, 40, 40)
            - p4: (batch, 128, 20, 20)
            - p5: (batch, 256, 10, 10)
        """
        # Stem
        x = self.conv0(x)  # 160x160

        # Stage 1
        x = self.conv1(x)  # 80x80
        x = self.c3k2_1(x)

        # Stage 2 (P3)
        x = self.conv2(x)  # 40x40
        p3 = self.c3k2_2(x)

        # Stage 3 (P4)
        x = self.conv3(p3)  # 20x20
        p4 = self.c3k2_3(x)

        # Stage 4 (P5)
        x = self.conv4(p4)  # 10x10
        x = self.c3k2_4(x)
        x = self.sppf(x)
        p5 = self.c2psa(x)

        return p3, p4, p5


class YOLO11nHead:
    """
    YOLO11n detection head with FPN-PAN.

    Takes backbone features and produces detection predictions.
    """

    def __init__(self, num_classes=2):
        """
        Parameters
        ----------
        num_classes : int
            Number of detection classes
        """
        self.num_classes = num_classes

        # FPN upsampling path
        # P5 → P4
        self.c3k2_p4 = C3k2(256 + 128, 128, n_blocks=1, name_prefix="head_p4")

        # P4 → P3
        self.c3k2_p3 = C3k2(128 + 64, 64, n_blocks=1, name_prefix="head_p3")

        # PAN downsampling path
        # P3 → P4
        self.down1 = ConvBNSiLU(
            64, 64, kernel_size=3, stride=2, padding="same", name_prefix="head_down1"
        )
        self.c3k2_p4_final = C3k2(
            64 + 128, 128, n_blocks=1, name_prefix="head_p4_final"
        )

        # P4 → P5
        self.down2 = ConvBNSiLU(
            128, 128, kernel_size=3, stride=2, padding="same", name_prefix="head_down2"
        )
        self.c3k2_p5_final = C3k2(
            128 + 256, 256, n_blocks=1, name_prefix="head_p5_final"
        )

        # Detection heads (one per scale)
        # Each head outputs: [batch, 4 + num_classes, H, W]
        # where 4 = (x, y, w, h)
        self.detect_p3 = ConvBNSiLU(
            64,
            4 + num_classes,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix="detect_p3",
        )
        self.detect_p4 = ConvBNSiLU(
            128,
            4 + num_classes,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix="detect_p4",
        )
        self.detect_p5 = ConvBNSiLU(
            256,
            4 + num_classes,
            kernel_size=1,
            stride=1,
            padding="valid",
            name_prefix="detect_p5",
        )

        # Collect params
        self.params = []
        self.bn_stats = []
        for module in [
            self.c3k2_p4,
            self.c3k2_p3,
            self.down1,
            self.c3k2_p4_final,
            self.down2,
            self.c3k2_p5_final,
            self.detect_p3,
            self.detect_p4,
            self.detect_p5,
        ]:
            self.params.extend(module.params)
            self.bn_stats.extend(module.bn_stats)

    def __call__(self, p3, p4, p5):
        """
        Forward pass.

        Parameters
        ----------
        p3, p4, p5 : TensorVariables
            Backbone features

        Returns
        -------
        det_p3, det_p4, det_p5 : TensorVariables
            Detection predictions at 3 scales
        """
        # FPN path (top-down)
        # P5 → P4
        p5_up = self._upsample(p5, scale=2)
        p4_fused = pt.concatenate([p5_up, p4], axis=1)
        p4_out = self.c3k2_p4(p4_fused)

        # P4 → P3
        p4_up = self._upsample(p4_out, scale=2)
        p3_fused = pt.concatenate([p4_up, p3], axis=1)
        p3_out = self.c3k2_p3(p3_fused)

        # PAN path (bottom-up)
        # P3 → P4
        p3_down = self.down1(p3_out)
        p4_fused2 = pt.concatenate([p3_down, p4_out], axis=1)
        p4_final = self.c3k2_p4_final(p4_fused2)

        # P4 → P5
        p4_down = self.down2(p4_final)
        p5_fused = pt.concatenate([p4_down, p5], axis=1)
        p5_final = self.c3k2_p5_final(p5_fused)

        # Detection heads
        det_p3 = self.detect_p3(p3_out)  # (batch, 4+C, 40, 40)
        det_p4 = self.detect_p4(p4_final)  # (batch, 4+C, 20, 20)
        det_p5 = self.detect_p5(p5_final)  # (batch, 4+C, 10, 10)

        return det_p3, det_p4, det_p5

    def _upsample(self, x, scale=2):
        """Upsample using nearest neighbor (JAX-compatible)."""
        # x: (batch, C, H, W)
        # Use explicit reshaping to avoid dynamic shapes in JAX JIT

        # Get input shape using pt.shape() for symbolic computation
        input_shape = x.shape
        batch_size = input_shape[0]
        channels = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]

        # Strategy: expand dims, tile, then rearrange and flatten
        # (B, C, H, W) -> (B, C, H, 1, W, 1) -> (B, C, H, scale, W, scale)
        # -> (B, C, H, W, scale, scale) -> (B, C, H*scale, W*scale)

        # Step 1: Add singleton dimensions for tiling
        x_expanded = x.dimshuffle(0, 1, 2, "x", 3, "x")  # (B, C, H, 1, W, 1)

        # Step 2: Tile along the new dimensions
        x_tiled = pt.tile(
            x_expanded, (1, 1, 1, scale, 1, scale)
        )  # (B, C, H, scale, W, scale)

        # Step 3: Rearrange to interleave dimensions
        # (B, C, H, scale, W, scale) -> (B, C, H, W, scale, scale)
        x_rearranged = x_tiled.dimshuffle(0, 1, 2, 4, 3, 5)

        # Step 4: Flatten last 4 dims to (B, C, H*scale, W*scale)
        # Compute output shape from input shape
        out_height = height * scale
        out_width = width * scale

        x_upsampled = x_rearranged.reshape(
            (batch_size, channels, out_height, out_width)
        )

        return x_upsampled


class YOLO11n:
    """
    Complete YOLO11n model.

    Combines backbone and head for end-to-end object detection.
    """

    def __init__(self, num_classes=2, input_size=320):
        """
        Parameters
        ----------
        num_classes : int
            Number of detection classes
        input_size : int
            Input image size (square)
        """
        self.num_classes = num_classes
        self.input_size = input_size

        self.backbone = YOLO11nBackbone()
        self.head = YOLO11nHead(num_classes=num_classes)

        # Collect all parameters
        self.params = self.backbone.params + self.head.params
        self.bn_stats = self.backbone.bn_stats + self.head.bn_stats

        print("YOLO11n initialized:")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Num classes: {num_classes}")
        print(f"  Total params: {sum(p.get_value().size for p in self.params):,}")

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : TensorVariable
            Input (batch, 3, 320, 320)

        Returns
        -------
        predictions : dict
            Detection predictions at 3 scales
        """
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Head
        det_p3, det_p4, det_p5 = self.head(p3, p4, p5)

        return {
            "p3": det_p3,  # (batch, 4+C, 40, 40)
            "p4": det_p4,  # (batch, 4+C, 20, 20)
            "p5": det_p5,  # (batch, 4+C, 10, 10)
        }


def build_yolo11n(num_classes=2, input_size=320):
    """
    Build YOLO11n model.

    Parameters
    ----------
    num_classes : int
        Number of classes to detect (default: 2 for person, cellphone)
    input_size : int
        Input image size

    Returns
    -------
    model : YOLO11n
        Initialized model
    x : TensorVariable
        Input symbolic variable
    predictions : dict
        Output predictions
    """
    # Input
    x = pt.tensor4("x", dtype="float32")

    # Model
    model = YOLO11n(num_classes=num_classes, input_size=input_size)

    # Forward pass
    predictions = model(x)

    return model, x, predictions
