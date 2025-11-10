from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from .blocks import TransformerBlock


def make_blocks(dim: int, num: int, heads: int, expand: float, ln_eps: float,
                apply_dw_on_v: bool, residual_scale: float) -> nn.Sequential:
    return nn.Sequential(*[
        TransformerBlock(dim, heads, expand, ln_eps=ln_eps,
                         apply_dw_on_v=apply_dw_on_v, residual_scale=residual_scale)
        for _ in range(num)
    ])


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.up(x))


class TxUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base_ch: int = 48,
        depths: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (4, 4, 8, 8),
        expand: float = 2.66,
        use_checkpoint: bool = True,
        ln_eps: float = 1e-5,
        apply_dw_on_v: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        c = base_ch
        self.use_checkpoint = use_checkpoint
        self.ln_eps = float(ln_eps)
        self.apply_dw_on_v = bool(apply_dw_on_v)
        self.residual_scale = float(residual_scale)
        # Stem
        self.stem = nn.Conv2d(in_ch, c, kernel_size=3, padding=1)

        # Encoder levels 0..3
        self.enc0 = make_blocks(c, depths[0], heads[0], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)
        self.down1 = Downsample(c, 2 * c)
        self.enc1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)
        self.down2 = Downsample(2 * c, 4 * c)
        self.enc2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)
        self.down3 = Downsample(4 * c, 8 * c)
        self.enc3 = make_blocks(8 * c, depths[3], heads[3], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)  # bottleneck

        # Skip alignment (1x1 BEFORE concat)
        self.skip0 = nn.Conv2d(c, c, kernel_size=1)
        self.skip1 = nn.Conv2d(2 * c, 2 * c, kernel_size=1)
        self.skip2 = nn.Conv2d(4 * c, 4 * c, kernel_size=1)

        # Decoder
        self.up3 = Upsample(8 * c, 4 * c)
        self.fuse2 = nn.Conv2d(8 * c, 4 * c, kernel_size=1)
        self.dec2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)
        self.up2 = Upsample(4 * c, 2 * c)
        self.fuse1 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        self.dec1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)
        self.up1 = Upsample(2 * c, c)  # up to C
        self.fuse0 = nn.Conv2d(2 * c, 2 * c, kernel_size=1)  # keep 2C penultimate
        self.dec0 = make_blocks(2 * c, depths[0], heads[0], expand, self.ln_eps, self.apply_dw_on_v, self.residual_scale)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        x0 = self.stem(x)
        x0 = self._run_blocks(self.enc0, x0)
        s0 = self.skip0(x0)

        x1 = self.down1(x0)
        x1 = self._run_blocks(self.enc1, x1)
        s1 = self.skip1(x1)

        x2 = self.down2(x1)
        x2 = self._run_blocks(self.enc2, x2)
        s2 = self.skip2(x2)

        x3 = self.down3(x2)
        x3 = self._run_blocks(self.enc3, x3)

        # decoder
        y2 = self.up3(x3)
        y2 = torch.cat([y2, s2], dim=1)
        y2 = self.fuse2(y2)
        y2 = self._run_blocks(self.dec2, y2)

        y1 = self.up2(y2)
        y1 = torch.cat([y1, s1], dim=1)
        y1 = self.fuse1(y1)
        y1 = self._run_blocks(self.dec1, y1)

        y0 = self.up1(y1)
        y0 = torch.cat([y0, s0], dim=1)
        y0 = self.fuse0(y0)
        y0 = self._run_blocks(self.dec0, y0)

        out = self.head(y0)
        return out

    def _run_blocks(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if not self.use_checkpoint or not self.training:
            return seq(x)
        for m in seq:
            x = checkpoint.checkpoint(m, x, use_reentrant=False)
        return x



