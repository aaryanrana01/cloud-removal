import numpy as np
import pytest
import torch

from cloudremoval.losses import PerceptualLoss, RestorationLoss, ssim
from cloudremoval.metrics import image_metrics, summarise


def test_perceptual_loss_trains_the_model():
    """Regression test for the original bug: the loss ran under no_grad and gave no gradient."""
    loss_fn = PerceptualLoss(pretrained=False)
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss = loss_fn(pred, torch.rand(1, 3, 32, 32))
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in loss_fn.features.parameters())   # VGG itself stays frozen


def test_ssim_is_one_for_identical_images():
    x = torch.rand(1, 3, 32, 32)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)
    assert ssim(x, torch.rand(1, 3, 32, 32)).item() < 0.5


def test_restoration_loss_parts():
    loss_fn = RestorationLoss(pretrained_vgg=False)
    x = torch.rand(2, 3, 32, 32)
    total, parts = loss_fn(x, x)
    assert total.item() == pytest.approx(0.0, abs=1e-4)
    assert set(parts) == {"l1", "ssim_loss", "perceptual"}


def test_metrics_use_only_real_rgb_channels():
    rng = np.random.default_rng(0)
    gt = rng.random((64, 64, 3)).astype(np.float32)
    perfect = image_metrics(gt, gt)
    assert perfect["psnr"] == 100.0 and perfect["ssim"] == pytest.approx(1.0) and perfect["cc"] == pytest.approx(1.0)

    noisy = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, 1).astype(np.float32)
    m = image_metrics(noisy, gt)
    assert 24 < m["psnr"] < 28          # sigma=0.05 noise is ~26 dB; the old 4-band code reported ~35 dB
    with pytest.raises(ValueError):
        image_metrics(np.zeros((64, 64, 4)), np.zeros((64, 64, 3)))


def test_summarise():
    s = summarise([{"psnr": 20.0}, {"psnr": 30.0}])
    assert s["psnr"]["mean"] == 25.0 and s["psnr"]["std"] == 5.0
