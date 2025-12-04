import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1)]
            if norm:
                layers += [nn.InstanceNorm2d(oc, affine=True)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch, base_ch, norm=False),
            block(base_ch, base_ch*2),
            block(base_ch*2, base_ch*4),
            nn.Conv2d(base_ch*4, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
