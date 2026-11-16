"""Result, status and error types for the preprocessing feature."""

from dataclasses import dataclass

import numpy as np


class Status:
    NEEDS_REVIEW = "needs_review"  # quality warnings: ask the user to retake or continue
    READY = "ready"                # processed image stored and ready for detection


class ErrorCode:
    EMPTY = "empty"
    TOO_LARGE = "too_large"
    UNREADABLE = "unreadable"
    UNSUPPORTED_TYPE = "unsupported_type"
    TOO_MANY_PIXELS = "too_many_pixels"
    TOO_SMALL = "too_small"
    NOT_FOUND = "not_found"
    INVALID_STATE = "invalid_state"


class WarningCode:
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    LOW_CONTRAST = "low_contrast"


class PreprocessingError(Exception):
    """The upload or request was rejected. `code` is one of ErrorCode."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PreprocessResult:
    image_id: str
    status: str                  # Status.NEEDS_REVIEW or Status.READY
    warnings: list[dict]         # [{"code": WarningCode, "message": str}]
    manifest: dict               # contents of manifest.json
    image: np.ndarray | None     # processed BGR image when READY, else None
