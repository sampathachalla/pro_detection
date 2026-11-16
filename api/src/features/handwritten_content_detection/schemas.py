"""Result and error types for handwritten content detection."""

from dataclasses import dataclass

import numpy as np


class Detector:
    PADDLE = "paddle"
    EASYOCR = "easyocr"
    NONE = "none"  # neither detector found handwriting


class DetectionErrorCode:
    NOT_FOUND = "not_found"   # unknown image id, or detection not run yet
    NOT_READY = "not_ready"   # preprocessing is not "ready" (e.g. waiting for retake/continue)


class DetectionError(Exception):
    """`code` is one of DetectionErrorCode; `message` is safe to show to the user."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class DetectionResult:
    image_id: str
    detector: str          # Detector.PADDLE / EASYOCR / NONE
    regions: list[dict]    # [{"points": [[x, y] x4], "bbox": [x0, y0, x1, y1], "score": float | None}]
    annotated: np.ndarray  # original upright photo with borders drawn (BGR)
    result: dict           # contents of handwriting_detection.json
