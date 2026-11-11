"""Swin-Transformer U-Net for cloud removal.

Shape of the network (default config, 256x256 input):

    input RGB --conv--> 24ch @256 (full-res detail path) ------------------+
                 |                                                         |
            2x2 patch embed -> 48ch @128 --Swin stage--+ skip              |
                                    merge v            |                   |
                                 96ch @64 --Swin-------+ skip              |
                                    merge v            |                   |
                                192ch @32 --Swin-------+ skip              |
                                    merge v            |                   |
                                384ch @16  bottleneck  |                   |
                 expand ^ + skip -> Swin stage (x3, back up to 48ch @128)  |
                 expand ^ to 24ch @256 + full-res detail path <-----------+
                 conv -> residual;   output = input + residual

Design choices worth knowing for interviews:
  * Window attention: each token attends only inside an 8x8 window, and every
    second layer shifts the windows by 4 so information flows between windows.
    Cost grows linearly with image size instead of quadratically.
  * Attention runs from half resolution down (like the original Swin, which
    uses 4x4 patches). This cut training memory and CPU inference time ~4x.
    A cheap full-resolution conv path carries fine detail around it.
  * Residual output: the network predicts only the *change* to apply to the
    cloudy image. The last conv starts at zero, so an untrained model returns
    the input unchanged and training only has to learn the cloud correction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


# ---------------------------------------------------------------- window helpers
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * num_windows, ws*ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(windows: torch.Tensor, ws: int, B: int, H: int, W: int) -> torch.Tensor:
    """(B * num_windows, ws*ws, C) -> (B, H, W, C)"""
    C = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


_mask_cache: dict = {}


def shifted_window_mask(H: int, W: int, ws: int, shift: int, device) -> torch.Tensor:
    """Attention mask that stops pixels from attending across the wrap-around
    edge created by torch.roll. Cached per (size, device) because it only
    depends on the image size."""
    key = (H, W, ws, shift, str(device))
    if key not in _mask_cache:
        img = torch.zeros(1, H, W, 1)
        region = 0
        for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                img[:, hs, wsl, :] = region
                region += 1
        win = window_partition(img, ws).squeeze(-1)          # (nW, ws*ws)
        diff = win.unsqueeze(1) - win.unsqueeze(2)
        mask = torch.zeros_like(diff).masked_fill(diff != 0, -100.0)
        _mask_cache[key] = mask.to(device)
    return _mask_cache[key]


# ---------------------------------------------------------------- building blocks
class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with a learned relative
    position bias (the model learns how much 'up-left' or 'right' matters)."""

    def __init__(self, dim: int, ws: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.rel_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        Bw, N, C = x.shape
        q, k, v = self.qkv(x).view(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)                        # (Bw, heads, N, N)
        bias = self.rel_bias[self.rel_index.reshape(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0).to(attn.dtype)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.heads, N, N) + mask[None, :, None].to(attn.dtype)
            attn = attn.view(Bw, self.heads, N, N)
        out = attn.softmax(-1) @ v
        return self.proj(out.transpose(1, 2).reshape(Bw, N, C))


class SwinLayer(nn.Module):
    """LayerNorm -> (shifted) window attention -> LayerNorm -> MLP, both with residuals."""

    def __init__(self, dim: int, heads: int, ws: int, shift: int, mlp_ratio: float):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, H, W, C)
        B, H, W, C = x.shape
        shift = self.shift if min(H, W) > self.ws else 0          # one window -> nothing to shift
        y = self.norm1(x)
        if shift:
            y = torch.roll(y, (-shift, -shift), dims=(1, 2))
        mask = shifted_window_mask(H, W, self.ws, shift, x.device) if shift else None
        y = self.attn(window_partition(y, self.ws), mask)
        y = window_reverse(y, self.ws, B, H, W)
        if shift:
            y = torch.roll(y, (shift, shift), dims=(1, 2))
        x = x + y
        return x + self.mlp(self.norm2(x))


class SwinStage(nn.Module):
    """A group of Swin layers followed by a 3x3 conv, wrapped in a residual
    connection (the 'RSTB' block from SwinIR). The conv adds local detail
    that pure window attention can miss."""

    def __init__(self, dim, heads, depth, ws, mlp_ratio, grad_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList(
            SwinLayer(dim, heads, ws, 0 if i % 2 == 0 else ws // 2, mlp_ratio) for i in range(depth)
        )
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, C, H, W)
        y = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                y = checkpoint(layer, y, use_reentrant=False)
            else:
                y = layer(y)
        return x + self.conv(y.permute(0, 3, 1, 2))


class PatchMerging(nn.Module):
    """Halve resolution, double channels: 2x2 neighbours are stacked then projected."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):                                         # (B, C, H, W)
        x = F.pixel_unshuffle(x, 2).permute(0, 2, 3, 1)           # (B, H/2, W/2, 4C)
        return self.reduce(self.norm(x)).permute(0, 3, 1, 2)


class PatchExpand(nn.Module):
    """Double resolution, halve channels (inverse idea of PatchMerging)."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):                                         # (B, C, H, W)
        x = self.expand(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, 2)                                 # (B, C/2, 2H, 2W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------- full model
class SwinUNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        if len(cfg.depths) != len(cfg.num_heads):
            raise ValueError("depths and num_heads must have the same length")
        self.cfg = cfg
        n = len(cfg.depths)
        dims = [cfg.embed_dim * 2 ** i for i in range(n)]
        stage = lambda d, h, depth: SwinStage(d, h, depth, cfg.window_size, cfg.mlp_ratio, cfg.grad_checkpoint)

        if cfg.embed_dim % 2:
            raise ValueError("embed_dim must be even")
        half = cfg.embed_dim // 2
        self.head = nn.Conv2d(cfg.in_ch, half, 3, padding=1)                  # full-res detail path
        self.embed = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * half, cfg.embed_dim, 1))
        self.encoders = nn.ModuleList(stage(dims[i], cfg.num_heads[i], cfg.depths[i]) for i in range(n))
        self.downs = nn.ModuleList(PatchMerging(d) for d in dims)
        self.bottleneck = stage(dims[-1] * 2, cfg.num_heads[-1] * 2, cfg.bottleneck_depth)
        self.ups = nn.ModuleList(PatchExpand(d * 2) for d in reversed(dims))
        self.fuses = nn.ModuleList(nn.Conv2d(2 * d, d, 1) for d in reversed(dims))
        self.decoders = nn.ModuleList(
            stage(dims[i], cfg.num_heads[i], cfg.depths[i]) for i in reversed(range(n))
        )
        self.up_full = PatchExpand(cfg.embed_dim)                             # back to full res
        self.refine = nn.Sequential(nn.Conv2d(2 * half, half, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(half, half, 3, padding=1))
        self.tail = nn.Conv2d(half, cfg.in_ch, 3, padding=1)

        # input size must divide evenly through the patch embed, every merge, and into windows
        self.size_multiple = cfg.window_size * 2 ** (n + 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tail.weight)      # start as identity: output == input
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1], any H and W. Returns same shape (not clamped)."""
        H, W = x.shape[-2:]
        m = self.size_multiple
        ph, pw = (m - H % m) % m, (m - W % m) % m
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x

        detail = self.head(xp)
        f0 = self.embed(detail)
        f, skips = f0, []
        for enc, down in zip(self.encoders, self.downs):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.bottleneck(f)
        for up, fuse, dec, skip in zip(self.ups, self.fuses, self.decoders, reversed(skips)):
            f = dec(fuse(torch.cat([up(f), skip], dim=1)))
        f = self.up_full(f + f0)
        out = xp + self.tail(self.refine(torch.cat([f, detail], dim=1)))
        return out[..., :H, :W]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | dict | None = None) -> SwinUNet:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return SwinUNet(cfg)


def _self_check():  # pragma: no cover - handy for `python -m cloudremoval.model`
    model = build_model()
    x = torch.rand(1, 3, 200, 300)
    y = model(x)
    print(f"params={count_parameters(model):,}  in={tuple(x.shape)} out={tuple(y.shape)} "
          f"identity_at_init={torch.allclose(x, y)}  size_multiple={model.size_multiple}")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
