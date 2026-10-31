import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for Pix2Pix Conditional GAN.
    Conditioned on Grayscale L channel (1 ch) concatenated with color ab channels (2 ch).
    Total input channels = 3.
    Output: Grid of patch logits (shape: B x 1 x 30 x 30 for 256x256 input).
    """
    def __init__(self, in_channels=3, num_filters=64):
        super().__init__()
        
        # Layer 1: No BatchNorm in first layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3: 128 -> 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 4: 256 -> 512 with stride 1
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 5: Output 1-channel prediction map
        self.conv5 = nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, L, ab):
        # Concatenate grayscale input L and color ab along channel dimension
        x = torch.cat([L, ab], dim=1) # (B, 3, 256, 256)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv5(x)           # (B, 1, 30, 30)
        return out


class GANLoss(nn.Module):
    """
    Adversarial loss supporting LSGAN (MSE) or Vanilla GAN (BCEWithLogits).
    Includes one-sided label smoothing (real=0.9) to stabilize discriminator.
    """
    def __init__(self, gan_mode="vanilla", real_label=0.9, fake_label=0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.gan_mode = gan_mode
        
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")

    def get_target_tensor(self, prediction, is_real):
        target_value = self.real_label if is_real else self.fake_label
        return torch.full_like(prediction, target_value)

    def forward(self, prediction, is_real):
        target_tensor = self.get_target_tensor(prediction, is_real)
        return self.loss(prediction, target_tensor)
