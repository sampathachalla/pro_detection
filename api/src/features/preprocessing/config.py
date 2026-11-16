"""Preprocessing feature settings. Limits can be overridden with environment variables."""

import os
from dataclasses import dataclass, field

from api.src.services.preprocess_service import PreprocessOptions

PIPELINE_VERSION = "1"
KEY_PREFIX = "preprocessing"

# Pillow format name -> (file extension, content type)
FORMATS = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
}


@dataclass(frozen=True)
class PreprocessingConfig:
    # Hard limits: the upload is rejected.
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG")
    max_bytes: int = 20 * 1024 * 1024        # must be less than 20 MB
    max_pixels: int = 40_000_000             # guards against decompression bombs
    min_side: int = 64                       # tiny images are rejected

    # Soft limits: the user is warned and chooses retake or continue.
    min_blur: float = 50.0                   # Laplacian variance
    min_brightness: float = 60.0             # mean gray 0-255
    min_contrast: float = 10.0               # gray std dev

    retention_hours: float = 24.0
    options: PreprocessOptions = field(default_factory=PreprocessOptions)

    @classmethod
    def from_env(cls) -> "PreprocessingConfig":
        env = os.environ.get
        default = cls()
        return cls(
            max_bytes=int(env("PREPROCESS_MAX_BYTES", default.max_bytes)),
            max_pixels=int(env("PREPROCESS_MAX_PIXELS", default.max_pixels)),
            min_side=int(env("PREPROCESS_MIN_SIDE", default.min_side)),
            min_blur=float(env("PREPROCESS_MIN_BLUR", default.min_blur)),
            min_brightness=float(env("PREPROCESS_MIN_BRIGHTNESS", default.min_brightness)),
            min_contrast=float(env("PREPROCESS_MIN_CONTRAST", default.min_contrast)),
            retention_hours=float(env("PREPROCESS_RETENTION_HOURS", default.retention_hours)),
        )
