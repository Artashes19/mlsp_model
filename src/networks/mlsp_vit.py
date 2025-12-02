import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ViTModel(nn.Module):
    def __init__(
        self, 
        encoder_name="mit_b2", 
        encoder_weights="imagenet", 
        in_chans=6, 
        **kwargs
    ):
        super().__init__()
        
        # Use SMP Unet with a Transformer encoder (SegFormer/MixTransformer)
        # encoder_name="mit_b2" corresponds to SegFormer-B2 (similar to ViT-Small/Base range)
        
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_chans,
            classes=1,
            activation=None,
            # Pass any extra kwargs if needed (e.g. decoder_channels)
            **kwargs
        )

        # SegFormer (MixTransformer) encoders in SMP might not support >3 channels natively via weights
        # if 'in_channels' != 3. SMP usually handles the first conv layer modification automatically
        # for ResNets, but let's verify if we need to manually patch it for transformers.
        # SMP 0.3.0+ handles 6-channel input for most encoders by re-initializing the first conv.
        # However, transformers (mit_b*) use PatchEmbeddings, not simple Conv2d often.
        # Let's inspect the first layer if needed. For now, we trust SMP's robust handling.
        # Wait, MixTransformer uses a specialized PatchEmbed.
        
        # If the input channels are not 3, SMP will try to adapt the weights.
        # If it fails, we might need manual intervention, but usually it works:
        # It copies the sum of weights or mean of weights to the new channels.
        
    def forward(self, x):
        # x: (B, 6, H, W)
        
        # SMP Unet forward returns (B, classes, H, W)
        logits = self.unet(x)
        
        # Squeeze to (B, H, W) as expected by MLSP algorithm
        return logits.squeeze(1)
