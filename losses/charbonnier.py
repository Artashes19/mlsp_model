from __future__ import annotations

import torch
from torch import nn


class CharbonnierLossMasked(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        diff = pred - target
        if mask is not None:
            diff = diff * mask
        loss = torch.sqrt(diff * diff + (self.epsilon * self.epsilon))
        if mask is None:
            return torch.mean(loss)
        denom = torch.clamp(mask.sum(), min=1.0)
        return loss.sum() / denom



