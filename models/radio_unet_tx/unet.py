"""
Radio U-Net with Transformer Blocks (TxUNet).

A Transformer-based U-Net architecture for radio map prediction.

Architecture:
    Input [B, 4, H, W] → Stem → Encoder (4 levels) → Decoder (4 levels) → Head → Output [B, 1, H, W]

Key features:
- 4-channel input: Reflection, Transmission, Distance, Pathloss samples
- Transformer blocks with efficient global attention at each level
- Skip connections: level 0 uses concat only, levels 1-2 use 1×1 conv after concat
- F₀ residual: stem output is added back before final output
- Decoder level 0: N₁ blocks + 1 extra single block
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from .blocks import TransformerBlock, WindowedTransformerBlock


def make_blocks(
    dim: int,
    num: int,
    heads: int,
    expand: float,
    ln_eps: float,
    block_cls: type[nn.Module] = TransformerBlock,
    block_kwargs: dict | None = None,
) -> nn.Sequential:
    """
    Create a sequence of TransformerBlocks.
    
    Args:
        dim: Channel dimension
        num: Number of blocks
        heads: Number of attention heads
        expand: FFN expansion ratio
        ln_eps: LayerNorm epsilon
        block_cls: Block class to instantiate (default: TransformerBlock)
        block_kwargs: Optional kwargs passed to each block
    
    Returns:
        nn.Sequential of TransformerBlocks
    """
    block_kwargs = block_kwargs or {}
    return nn.Sequential(*[
        block_cls(dim, heads, expand, ln_eps=ln_eps, **block_kwargs)
        for _ in range(num)
    ])


class Downsample(nn.Module):
    """
    2× spatial downsampling with channel expansion.
    
    Uses stride-2 3×3 convolution.
    
    Input: [B, in_ch, H, W]
    Output: [B, out_ch, H/2, W/2]
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    2× spatial upsampling with channel reduction.
    
    Uses nearest neighbor upsampling followed by 1×1 convolution.
    
    Input: [B, in_ch, H, W]
    Output: [B, out_ch, 2H, 2W]
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.up(x))


class TxUNet(nn.Module):
    """
    Transformer-based U-Net for radio map prediction.
    
    Architecture overview:
        - Stem: 3×3 conv (in_ch → C), produces F₀
        - Encoder: 4 levels with Transformer blocks and downsampling
            - Level 0: C channels, N₁ blocks
            - Level 1: 2C channels, N₂ blocks
            - Level 2: 4C channels, N₃ blocks
            - Bottleneck: 8C channels, N₄ blocks
        - Decoder: 4 levels with upsampling and skip connections
            - Level 2: concat + 1×1 fuse, N₃ blocks
            - Level 1: concat + 1×1 fuse, N₂ blocks
            - Level 0: concat only (no fuse), N₁ blocks + 1 extra block
        - Head: 3×3 conv (2C → C) → Add F₀ → 3×3 conv (C → out_ch)
    
    Args:
        in_ch: Input channels (default 4: R, T, D, P)
        out_ch: Output channels (default 1: predicted pathloss)
        base_ch: Base channel count C (default 48)
        depths: Transformer blocks per level [N₁, N₂, N₃, N₄] (default (4, 6, 6, 8))
        heads: Attention heads per level (default (4, 4, 8, 8))
        expand: FFN expansion ratio (default 2.66)
        use_checkpoint: Enable gradient checkpointing (default True)
        ln_eps: LayerNorm epsilon (default 1e-5)
    """
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 1,
        base_ch: int = 48,
        depths: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (4, 4, 8, 8),
        expand: float = 2.66,
        use_checkpoint: bool = True,
        ln_eps: float = 1e-5,
        window0: int | None = None,
        window0_stride: int | None = None,
        sra0_enabled: bool = False,
        sra0_stride: int = 2,
    ) -> None:
        super().__init__()
        c = base_ch
        self.use_checkpoint = use_checkpoint
        self.ln_eps = float(ln_eps)

        # ============ STEM ============
        # 3×3 conv: in_ch → C
        # Produces F₀ which is used for residual in head
        self.stem = nn.Conv2d(in_ch, c, kernel_size=3, padding=1)

        # ============ ENCODER ============
        # Level 0: [B, C, H, W]
        enc0_cls: type[nn.Module]
        enc0_kwargs: dict | None = None
        if sra0_enabled:
            enc0_cls = TransformerBlock
            enc0_kwargs = {"kv_stride": sra0_stride}
        elif window0:
            enc0_cls = WindowedTransformerBlock
            enc0_kwargs = {"window": window0, "stride": window0_stride}
        else:
            enc0_cls = TransformerBlock
        self.enc0 = make_blocks(c, depths[0], heads[0], expand, self.ln_eps, block_cls=enc0_cls, block_kwargs=enc0_kwargs)
        
        # Downsample: C → 2C, spatial /2
        self.down1 = Downsample(c, 2 * c)
        
        # Level 1: [B, 2C, H/2, W/2]
        self.enc1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps)
        
        # Downsample: 2C → 4C, spatial /2
        self.down2 = Downsample(2 * c, 4 * c)
        
        # Level 2: [B, 4C, H/4, W/4]
        self.enc2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps)
        
        # Downsample: 4C → 8C, spatial /2
        self.down3 = Downsample(4 * c, 8 * c)
        
        # Bottleneck (Level 3): [B, 8C, H/8, W/8]
        self.enc3 = make_blocks(8 * c, depths[3], heads[3], expand, self.ln_eps)

        # ============ SKIP CONNECTIONS ============
        # Level 0: NO skip conv (concat only)
        # Level 1: 1×1 conv on skip features BEFORE concat
        self.skip1 = nn.Conv2d(2 * c, 2 * c, kernel_size=1)
        # Level 2: 1×1 conv on skip features BEFORE concat
        self.skip2 = nn.Conv2d(4 * c, 4 * c, kernel_size=1)

        # ============ DECODER ============
        # Level 2 decoder
        # Upsample: 8C → 4C
        self.up3 = Upsample(8 * c, 4 * c)
        # Fuse after concat: 8C → 4C
        self.fuse2 = nn.Conv2d(8 * c, 4 * c, kernel_size=1)
        self.dec2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps)

        # Level 1 decoder
        # Upsample: 4C → 2C
        self.up2 = Upsample(4 * c, 2 * c)
        # Fuse after concat: 4C → 2C
        self.fuse1 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        self.dec1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps)

        # Level 0 decoder (special handling)
        # Upsample: 2C → C
        self.up1 = Upsample(2 * c, c)
        # NO fuse0 - just concat (C + C = 2C)
        # N₁ transformer blocks at 2C
        if sra0_enabled:
            dec0_cls = TransformerBlock
            dec0_kwargs = {"kv_stride": sra0_stride}
        elif window0:
            dec0_cls = WindowedTransformerBlock
            dec0_kwargs = {"window": window0, "stride": window0_stride}
        else:
            dec0_cls = TransformerBlock
            dec0_kwargs = None
        self.dec0 = make_blocks(2 * c, depths[0], heads[0], expand, self.ln_eps, block_cls=dec0_cls, block_kwargs=dec0_kwargs)
        # +1 extra single transformer block
        self.dec0_extra = dec0_cls(2 * c, heads[0], expand, ln_eps=self.ln_eps, **(dec0_kwargs or {}))

        # ============ HEAD ============
        # 3×3 conv: 2C → C
        self.head_conv1 = nn.Conv2d(2 * c, c, kernel_size=3, padding=1)
        # After adding F₀ residual: 3×3 conv: C → out_ch
        self.head_conv2 = nn.Conv2d(c, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, in_ch, H, W]
        
        Returns:
            Output tensor [B, out_ch, H, W]
        """
        # ============ STEM ============
        # Save F₀ for residual connection in head
        f0 = self.stem(x)  # [B, C, H, W]

        # ============ ENCODER ============
        # Level 0
        x0 = self._run_blocks(self.enc0, f0)  # [B, C, H, W]
        # Note: x0 is used directly for skip (no 1×1 conv)

        # Level 1
        x1 = self.down1(x0)  # [B, 2C, H/2, W/2]
        x1 = self._run_blocks(self.enc1, x1)
        s1 = self.skip1(x1)  # 1×1 on skip

        # Level 2
        x2 = self.down2(x1)  # [B, 4C, H/4, W/4]
        x2 = self._run_blocks(self.enc2, x2)
        s2 = self.skip2(x2)  # 1×1 on skip

        # Bottleneck (Level 3)
        x3 = self.down3(x2)  # [B, 8C, H/8, W/8]
        x3 = self._run_blocks(self.enc3, x3)

        # ============ DECODER ============
        # Level 2 decoder
        y2 = self.up3(x3)  # [B, 4C, H/4, W/4]
        y2 = torch.cat([y2, s2], dim=1)  # [B, 8C, H/4, W/4]
        y2 = self.fuse2(y2)  # [B, 4C, H/4, W/4]
        y2 = self._run_blocks(self.dec2, y2)

        # Level 1 decoder
        y1 = self.up2(y2)  # [B, 2C, H/2, W/2]
        y1 = torch.cat([y1, s1], dim=1)  # [B, 4C, H/2, W/2]
        y1 = self.fuse1(y1)  # [B, 2C, H/2, W/2]
        y1 = self._run_blocks(self.dec1, y1)

        # Level 0 decoder (special handling)
        y0 = self.up1(y1)  # [B, C, H, W]
        y0 = torch.cat([y0, x0], dim=1)  # [B, 2C, H, W] - concat only, no skip conv
        y0 = self._run_blocks(self.dec0, y0)  # N₁ transformer blocks
        y0 = self._run_single_block(self.dec0_extra, y0)  # +1 extra block

        # ============ HEAD ============
        y0 = self.head_conv1(y0)  # [B, C, H, W]
        y0 = y0 + f0  # Add F₀ residual
        out = self.head_conv2(y0)  # [B, out_ch, H, W]

        return out

    def _run_blocks(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """
        Run a sequence of blocks with optional gradient checkpointing.
        """
        if not self.use_checkpoint or not self.training:
            return seq(x)
        for m in seq:
            x = checkpoint.checkpoint(m, x, use_reentrant=False)
        return x

    def _run_single_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Run a single block with optional gradient checkpointing.
        """
        if not self.use_checkpoint or not self.training:
            return block(x)
        return checkpoint.checkpoint(block, x, use_reentrant=False)
