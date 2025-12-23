from typing import Any

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegFormerModel(nn.Module):
    
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None,
        in_chans: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()
                
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_chans,
            classes=1,
            activation=None,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.unet(x)
        return logits.squeeze(1)