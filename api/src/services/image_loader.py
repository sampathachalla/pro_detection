"""Decode uploaded image bytes into a BGR numpy array."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

_PIL_ERRORS = (UnidentifiedImageError, Image.DecompressionBombError, OSError)


def load_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("empty image")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a valid image")
    return image


def inspect_image(data: bytes) -> dict:
    """Read format and size from the file header without decoding the pixels.

    Returns {"format": "JPEG" | "PNG" | ..., "width": int, "height": int}.
    Width and height are as stored, before EXIF orientation is applied.
    """
    if not data:
        raise ValueError("empty image")
    try:
        with Image.open(BytesIO(data)) as img:
            return {"format": img.format, "width": img.width, "height": img.height}
    except _PIL_ERRORS as exc:
        raise ValueError("not a valid image") from exc


def load_image_upright(data: bytes) -> np.ndarray:
    """Decode bytes into a BGR image, rotated per the EXIF orientation tag (phone photos)."""
    if not data:
        raise ValueError("empty image")
    try:
        with Image.open(BytesIO(data)) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
    except _PIL_ERRORS as exc:
        raise ValueError("not a valid image") from exc
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
