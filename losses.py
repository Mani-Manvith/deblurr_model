import torch
import torch.nn as nn
import torchvision.models as models

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real: bool):
        if self.use_lsgan:
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.loss(pred, target)
        else:
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.loss(pred, target)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.crit = nn.L1Loss()

    def forward(self, x, y):
        return self.crit(self.slice(x), self.slice(y))
