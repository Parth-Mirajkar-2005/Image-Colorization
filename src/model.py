import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    """Two consecutive 3x3 convolutions with BatchNorm and ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UnetUpBlock(nn.Module):
    """Upsampling block with skip connection concatenation"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNetColorizationNet(nn.Module):
    """
    U-Net with Pretrained ResNet-18 Encoder for Image Colorization:
    - Input : L channel (1 x 256 x 256), grayscale lightness
    - Output: ab channels (2 x 256 x 256), color hues bounded to [-1, 1] with Tanh
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Modify first layer: change in_channels from 3 (RGB) to 1 (L)
        # We initialize it by summing the weights of R, G, B channels
        w = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = w.sum(dim=1, keepdim=True)

        # Encoder (ResNet stages)
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64, 128x128
        self.maxpool = resnet.maxpool                                         # 64, 64x64
        self.layer1 = resnet.layer1                                           # 64, 64x64
        self.layer2 = resnet.layer2                                           # 128, 32x32
        self.layer3 = resnet.layer3                                           # 256, 16x16
        self.layer4 = resnet.layer4                                           # 512, 8x8

        # Decoder (Upsampling + Skip connections)
        self.up4 = UnetUpBlock(512 + 256, 256)                                # 8x8 -> 16x16
        self.up3 = UnetUpBlock(256 + 128, 128)                                # 16x16 -> 32x32
        self.up2 = UnetUpBlock(128 + 64, 64)                                  # 32x32 -> 64x64
        self.up1 = UnetUpBlock(64 + 64, 32)                                   # 64x64 -> 128x128

        # Final upsampling block: 128x128 -> 256x256
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),                                 # Output: 2 channels (a, b)
            nn.Tanh()                                                         # Constrain to [-1, 1]
        )

    def forward(self, L):
        # 1. Forward through Encoder
        x0 = self.init_conv(L)             # shape: (B, 64, 128, 128)
        x1 = self.layer1(self.maxpool(x0)) # shape: (B, 64, 64, 64)
        x2 = self.layer2(x1)               # shape: (B, 128, 32, 32)
        x3 = self.layer3(x2)               # shape: (B, 256, 16, 16)
        x4 = self.layer4(x3)               # shape: (B, 512, 8, 8)

        # 2. Forward through Decoder with Skip Connections
        d4 = self.up4(x4, x3)              # shape: (B, 256, 16, 16)
        d3 = self.up3(d4, x2)              # shape: (B, 128, 32, 32)
        d2 = self.up2(d3, x1)              # shape: (B, 64, 64, 64)
        d1 = self.up1(d2, x0)              # shape: (B, 32, 128, 128)
        out_ab = self.final_up(d1)         # shape: (B, 2, 256, 256)

        return out_ab
