import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ASPPModule(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Dilated convolutions at different rates
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # Global pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        # Output projection
        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.size()
        
        # Process branches
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(x)))
        out3 = self.relu(self.bn3(self.conv3(x)))
        out4 = self.relu(self.bn4(self.conv4(x)))
        
        # Global pooling branch
        out5 = self.global_pool(x)
        out5 = self.relu(self.bn5(self.conv5(out5)))
        out5 = nn.functional.interpolate(out5, size=size[2:], mode='bilinear', align_corners=True)
        
        # Concatenate and project
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.relu(self.out_bn(self.out_conv(out)))
        
        return out


class UNetModel(nn.Module):
    
    def __init__(
        self,
        n_channels: int,
        use_aspp: bool,
        resnet_size: int,
        **kwargs,
    ):
        super(UNetModel, self).__init__()
        
        self.unet = smp.Unet(
            encoder_name=f"resnet{resnet_size}",
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
            activation=None
        )
        
        # ResNet34 bottleneck still has 512 channels like ResNet18,
        # but the network is deeper with more layers
        self.aspp = ASPPModule(in_channels=512, out_channels=512) if use_aspp else nn.Identity()
    
    def forward(self, x):
        features = self.unet.encoder(x)
        features[-1] = self.aspp(features[-1])
        decoder_output = self.unet.decoder(features)
        logits = self.unet.segmentation_head(decoder_output)
        
        return logits
