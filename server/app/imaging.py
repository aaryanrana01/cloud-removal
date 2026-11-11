"""Turning uploaded bytes into a safe RGB array, and arrays back into PNGs."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from cloudremoval.visualize import change_overlay, changed_fraction  # noqa: F401  (re-exported)

# Refuse 'decompression bombs': tiny files that expand to gigantic images in memory.
Image.MAX_IMAGE_PIXELS = 40_000_000


class BadImage(ValueError):
    """The upload is not an image we can read."""


def decode_upload(data: bytes, max_side: int) -> tuple[np.ndarray, tuple[int, int], bool]:
    """bytes -> (float32 RGB array in [0,1], original (w, h), was_resized)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            im = ImageOps.exif_transpose(im)          # respect phone-camera rotation
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                im = Image.fromarray((np.asarray(im, np.float32) / 257).clip(0, 255).astype(np.uint8))
            im = im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as e:
        raise BadImage("Could not read this file as an image. Use PNG, JPEG, TIFF or WebP.") from e

    original = im.size
    if min(original) < 32:
        raise BadImage("Image is too small (minimum 32x32 pixels).")
    resized = max(original) > max_side
    if resized:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0, original, resized


def to_png_data_url(arr: np.ndarray) -> str:
    """float [0,1] RGB or RGBA array -> 'data:image/png;base64,...'"""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
