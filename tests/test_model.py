import pytest
import torch

from cloudremoval.inference import load_checkpoint, save_checkpoint
from cloudremoval.model import build_model, shifted_window_mask


def test_untrained_model_returns_input_unchanged(tiny_cfg):
    model = build_model(tiny_cfg).eval()
    x = torch.rand(2, 3, 64, 64)
    assert torch.equal(model(x), x)


@pytest.mark.parametrize("h,w", [(64, 64), (50, 77), (33, 128)])
def test_any_input_size_gives_same_output_size(tiny_cfg, h, w):
    model = build_model(tiny_cfg).eval()
    torch.nn.init.normal_(model.tail.weight, std=0.01)          # make the output depend on the network
    with torch.no_grad():
        assert model(torch.rand(1, 3, h, w)).shape == (1, 3, h, w)


def test_every_parameter_gets_a_gradient(tiny_cfg):
    model = build_model(tiny_cfg)
    torch.nn.init.normal_(model.tail.weight, std=0.01)
    model(torch.rand(1, 3, 64, 64)).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, missing


def test_gradient_checkpointing_matches_normal_forward(tiny_cfg):
    tiny_cfg.grad_checkpoint = True
    model = build_model(tiny_cfg).train()
    torch.nn.init.normal_(model.tail.weight, std=0.01)
    x = torch.rand(1, 3, 64, 64)
    a = model(x)
    model.encoders[0].grad_checkpoint = False
    assert torch.allclose(a, model(x), atol=1e-6)


def test_shift_mask_blocks_wraparound():
    mask = shifted_window_mask(16, 16, 4, 2, "cpu")
    assert mask.shape == (16, 16, 16)
    assert (mask[0] == 0).all()               # top-left window has no wrapped pixels
    assert (mask[-1] < 0).any()               # bottom-right window mixes wrapped regions


def test_checkpoint_round_trip(tmp_path, tiny_cfg):
    model = build_model(tiny_cfg)
    torch.nn.init.normal_(model.tail.weight, std=0.01)
    save_checkpoint(tmp_path / "m.pt", model, {"epoch": 3})
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    x = torch.rand(1, 3, 64, 64)
    assert meta["epoch"] == 3
    assert torch.allclose(model.eval()(x), loaded(x))
