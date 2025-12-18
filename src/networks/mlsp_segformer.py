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
        patch_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        
        encoder_entry = smp.encoders.encoders[encoder_name]
        encoder_params = dict(encoder_entry.get("params", {}))
        encoder_params["patch_size"] = patch_size
        encoder_entry["params"] = encoder_params
        
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