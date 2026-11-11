"""Training loss = L1 + SSIM + perceptual (VGG).

  * L1          - average pixel difference. Gets colours right.
  * 1 - SSIM    - compares local structure (edges, texture), not just colour.
  * Perceptual  - compares VGG16 feature maps, so outputs look sharp instead
                  of blurry.

Fixes vs the original notebook:
  * The perceptual loss used to run fully inside torch.no_grad(), so it sent
    zero learning signal to the model. Now only the *target* features skip
    gradients; the prediction's features keep them.
  * VGG expects ImageNet-normalised input; that normalisation is now applied.
  * All losses are computed in float32 even under mixed precision, because
    SSIM's variance terms (E[x^2] - E[x]^2) go wrong in float16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[None, :]
    k = (g.t() @ g)[None, None]                       # (1, 1, size, size)
    return k.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Differentiable mean SSIM (Gaussian 11x11 window, same as the paper/skimage default)."""
    C = pred.shape[1]
    k = _gaussian_kernel(channels=C, device=pred.device)
    conv = lambda t: F.conv2d(t, k, groups=C)
    mu_p, mu_t = conv(pred), conv(target)
    var_p = conv(pred * pred) - mu_p ** 2
    var_t = conv(target * target) - mu_t ** 2
    cov = conv(pred * target) - mu_p * mu_t
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return ssim_map.mean()


class PerceptualLoss(nn.Module):
    """L1 distance between VGG16 relu2_2 features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(vgg.features.children())[:9]).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()                      # VGG must stay frozen in eval mode
        return self

    def forward(self, pred, target):
        norm = lambda t: (t.clamp(0, 1) - self.mean) / self.std
        with torch.no_grad():
            target_feat = self.features(norm(target))
        return F.l1_loss(self.features(norm(pred)), target_feat)


class RestorationLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_perceptual=0.05, pretrained_vgg=True):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_perceptual = w_l1, w_ssim, w_perceptual
        self.perceptual = PerceptualLoss(pretrained_vgg) if w_perceptual > 0 else None

    def forward(self, pred, target):
        device_type = pred.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pred, target = pred.float(), target.float()
            l1 = F.l1_loss(pred, target)
            s = 1.0 - ssim(pred.clamp(0, 1), target)
            perc = self.perceptual(pred, target) if self.perceptual is not None else pred.new_zeros(())
            total = self.w_l1 * l1 + self.w_ssim * s + self.w_perceptual * perc
        parts = {"l1": l1.detach(), "ssim_loss": s.detach(), "perceptual": perc.detach()}
        return total, parts
