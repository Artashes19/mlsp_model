from typing import List

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class UpsampleBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class ViTModel(nn.Module):
    
    def __init__(
        self,
        image_size: int,
        in_chans: int,
        patch_size: int,
        embed_dim: int,
        decoder_channels: List[int],
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.encoder = VisionTransformer(
            img_size=image_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=self.embed_dim,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            class_token=False,
            global_pool="",
        )
        
        blocks = []
        in_channels = self.embed_dim
        for out_channels in decoder_channels:
            blocks.append(
                UpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*blocks)
        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.GELU(),
        )
        self.head = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Input spatial size must be divisible by the patch size.")
        
        tokens = self.encoder.forward_features(x)
        feature_height = height // self.patch_size
        feature_width = width // self.patch_size
        
        spatial_tokens = tokens.transpose(1, 2)
        features = spatial_tokens.reshape(
            batch_size,
            self.embed_dim,
            feature_height,
            feature_width,
        )
        
        decoded = self.decoder(features)
        refined = self.refine(decoded)
        logits = self.head(refined)
        
        if logits.shape[-2:] != (height, width):
            raise ValueError("Upsampled logits shape mismatch.")
        
        return logits.squeeze(1)
