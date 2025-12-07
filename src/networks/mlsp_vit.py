import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ViTModel(nn.Module):
    def __init__(
        self, 
        encoder_name="mit_b2", 
        encoder_weights=None, 
        in_chans=6, 
        **kwargs
    ):
        super().__init__()
        
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_chans,
            classes=1,
            activation=None,
            **kwargs
        )

    def forward(self, x):
        logits = self.unet(x)
        return logits.squeeze(1)

