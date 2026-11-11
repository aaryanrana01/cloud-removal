"""Server settings, read from environment variables so the same Docker image
works locally and on Hugging Face Spaces without code changes."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: _env("APP_NAME", "Cloud Removal"))
    # Weights: use a local file if it exists, otherwise download from a Hugging Face model repo.
    model_path: Path = field(default_factory=lambda: Path(_env("MODEL_PATH", str(REPO_ROOT / "weights" / "best.pt"))))
    model_repo: str = field(default_factory=lambda: _env("MODEL_REPO", ""))          # e.g. "user/cloud-removal-swin"
    model_file: str = field(default_factory=lambda: _env("MODEL_FILE", "best.pt"))
    # Limits that protect a small free CPU server.
    max_upload_mb: float = field(default_factory=lambda: float(_env("MAX_UPLOAD_MB", "10")))
    max_side: int = field(default_factory=lambda: int(_env("MAX_SIDE", "1024")))    # bigger images are scaled down
    rate_limit_per_minute: int = field(default_factory=lambda: int(_env("RATE_LIMIT_PER_MINUTE", "12")))
    torch_threads: int = field(default_factory=lambda: int(_env("TORCH_THREADS", "2")))
    # Files bundled with the app.
    results_path: Path = field(default_factory=lambda: Path(_env("RESULTS_PATH", str(REPO_ROOT / "results" / "metrics.json"))))
    samples_dir: Path = field(default_factory=lambda: Path(_env("SAMPLES_DIR", str(REPO_ROOT / "server" / "samples"))))
    static_dir: Path = field(default_factory=lambda: Path(_env("STATIC_DIR", str(REPO_ROOT / "web" / "dist"))))


def get_settings() -> Settings:
    return Settings()
