import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """Lightweight ResNet-like generator for deblurring (placeholder for DeblurGAN-v2 FPN)."""
    def __init__(self, in_ch: int = 3, base_ch: int = 64, blocks: int = 8):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, 1, 3),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1),
            nn.InstanceNorm2d(base_ch * 4, affine=True),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResidualBlock(base_ch * 4) for _ in range(blocks)])
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, in_ch, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xr = self.res(x3)
        xu = self.up2(xr)
        xu = self.up1(xu)
        out = self.head(xu)
        # Map from [-1,1] to [0,1] space by residual on input
        return torch.clamp((out + 1) * 0.5, 0, 1)
