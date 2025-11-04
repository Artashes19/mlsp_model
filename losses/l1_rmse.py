from __future__ import annotations

import torch
from torch import nn


class L1LossMasked(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(torch.abs(pred - target))
        diff = torch.abs(pred - target) * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        return diff.sum() / denom


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse)
    diff2 = ((pred - target) ** 2) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    mse = diff2.sum() / denom
    return torch.sqrt(mse)




