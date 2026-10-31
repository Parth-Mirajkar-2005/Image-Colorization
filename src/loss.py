"""
Loss Functions for Image Colorization
Includes:
1. DifferentiableLab2RGB: Differentiable conversion from CIE Lab to sRGB
2. VGGPerceptualLoss: High-level feature loss using frozen VGG-16
3. ColorizationLoss: Combined L1 Loss + Perceptual Loss
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DifferentiableLab2RGB(nn.Module):
    """
    Differentiable CIE Lab to sRGB conversion in PyTorch.
    Allows loss gradients to backpropagate from RGB perceptual losses into ab predictions.
    """
    def __init__(self):
        super().__init__()
        # Standard XYZ to sRGB conversion matrix (D65 illuminant)
        M = torch.tensor([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=torch.float32)
        self.register_buffer("M", M)

    def forward(self, L_norm, ab_norm):
        # Denormalize L: [-1, 1] -> [0, 100]
        L = (L_norm + 1.0) * 50.0
        # Denormalize ab: [-1, 1] -> [-128, 128]
        ab = ab_norm * 128.0

        y = (L + 16.0) / 116.0
        x = ab[:, 0:1] / 500.0 + y
        z = y - ab[:, 1:2] / 200.0

        eps = 6.0 / 29.0
        x3 = torch.where(x > eps, x ** 3, (x - 4.0 / 29.0) * (3.0 * (eps ** 2)))
        y3 = torch.where(y > eps, y ** 3, (y - 4.0 / 29.0) * (3.0 * (eps ** 2)))
        z3 = torch.where(z > eps, z ** 3, (z - 4.0 / 29.0) * (3.0 * (eps ** 2)))

        # D65 white point scaling
        X = x3 * 0.95047
        Y = y3 * 1.00000
        Z = z3 * 1.08883

        XYZ = torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)
        B, C, H, W = XYZ.shape
        XYZ_flat = XYZ.permute(0, 2, 3, 1).reshape(-1, 3)
        RGB_flat = torch.matmul(XYZ_flat, self.M.t())
        RGB = RGB_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return torch.clamp(RGB, 0.0, 1.0)


class ColorizationLoss(nn.Module):
    """
    Combined Colorization Loss:
    Total Loss = L1_Loss(pred_ab, target_ab) + lambda * VGG_Perceptual_Loss(pred_rgb, target_rgb)
    
    - L1 Loss guarantees structural color correctness.
    - VGG Perceptual Loss encourages rich, vibrant textures and realistic scene harmony.
    """
    def __init__(self, vgg_weight=0.1):
        super().__init__()
        self.vgg_weight = vgg_weight
        self.l1 = nn.L1Loss()
        self.lab2rgb = DifferentiableLab2RGB()

        # Load pretrained VGG16 features and freeze parameters
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg_slice = nn.Sequential(*[vgg[x] for x in range(16)]).eval()
        for p in self.vgg_slice.parameters():
            p.requires_grad = False

        # ImageNet normalization constants for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, L, pred_ab, target_ab):
        l1_loss = self.l1(pred_ab, target_ab)

        if self.vgg_weight > 0:
            pred_rgb = self.lab2rgb(L, pred_ab)
            gt_rgb = self.lab2rgb(L, target_ab)

            p_norm = (pred_rgb - self.mean) / self.std
            g_norm = (gt_rgb - self.mean) / self.std

            p_feat = self.vgg_slice(p_norm)
            g_feat = self.vgg_slice(g_norm)

            vgg_loss = self.l1(p_feat, g_feat)
            total_loss = l1_loss + (self.vgg_weight * vgg_loss)
            return total_loss, l1_loss, vgg_loss

        return l1_loss, l1_loss, torch.tensor(0.0, device=L.device)
