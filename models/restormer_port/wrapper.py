from __future__ import annotations

import torch
from torch import nn

from .restormer_arch import RestormerPort


class RestormerRadio(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base_ch: int = 48,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        refinement_blocks: int = 4,
        expand: float = 2.66,
    ) -> None:
        super().__init__()
        self.body = RestormerPort(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            heads=list(heads),
            refinement_blocks=refinement_blocks,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


