"""
Radio U-Net with Transformer Blocks (TxUNet).

A Transformer-based U-Net architecture for radio map prediction.

Main exports:
    - TxUNet: The main model class
    - TransformerBlock: Single transformer block
    - DConvBlock: 1×1 Conv + 3×3 Depthwise Conv building block
    - migrate_checkpoint: Utility to migrate old checkpoints to new format
"""

from .unet import TxUNet, Downsample, Upsample, make_blocks
from .blocks import (
    DConvBlock,
    LayerNorm2d,
    EfficientGlobalAttention,
    GatedDepthwiseFFN,
    TransformerBlock,
    migrate_checkpoint,
)

__all__ = [
    # Main model
    "TxUNet",
    # U-Net components
    "Downsample",
    "Upsample",
    "make_blocks",
    # Block components
    "DConvBlock",
    "LayerNorm2d",
    "EfficientGlobalAttention",
    "GatedDepthwiseFFN",
    "TransformerBlock",
    # Utilities
    "migrate_checkpoint",
]
