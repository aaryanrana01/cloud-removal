"""All settings in one place, loaded from a YAML file.

Keeping every number (model size, learning rate, crop size...) in a config file
means an experiment can be repeated exactly, and the config is saved next to
every checkpoint so we always know how a model was trained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    in_ch: int = 3                      # RGB in, RGB out (RICE1 has no NIR band)
    embed_dim: int = 48                 # channels at full resolution; doubles each stage
    depths: list[int] = field(default_factory=lambda: [2, 2, 2])      # Swin layers per encoder stage
    num_heads: list[int] = field(default_factory=lambda: [3, 6, 12])  # attention heads per stage
    bottleneck_depth: int = 2
    window_size: int = 8
    mlp_ratio: float = 2.0
    grad_checkpoint: bool = False       # trade compute for GPU memory if needed


@dataclass
class DataConfig:
    root: str = ""                      # folder that contains cloud/ and label/
    crop_size: int = 256                # random training crop (images are 512x512)
    repeats: int = 4                    # random crops taken from each image per epoch
    val_fraction: float = 0.1
    test_fraction: float = 0.2          # 100 of 500 RICE1 pairs, same size as papers use
    seed: int = 42


@dataclass
class TrainConfig:
    out_dir: str = "runs/rice1"
    epochs: int = 60
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.02
    warmup_epochs: int = 3
    patience: int = 15                  # early stop after this many epochs without val improvement
    w_l1: float = 1.0
    w_ssim: float = 0.5
    w_perceptual: float = 0.05
    amp: bool = True                    # mixed precision on GPU
    num_workers: int = 2
    seed: int = 42
    save_every: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        def build(dc, values):
            allowed = {f.name for f in fields(dc)}
            unknown = set(values or {}) - allowed
            if unknown:
                raise ValueError(f"Unknown {dc.__name__} keys: {sorted(unknown)}")
            return dc(**(values or {}))

        return cls(
            model=build(ModelConfig, raw.get("model")),
            data=build(DataConfig, raw.get("data")),
            train=build(TrainConfig, raw.get("train")),
        )

    def to_dict(self) -> dict:
        return asdict(self)
