"""PaddleOCR text detection (detection only, PP-OCRv6_medium_det)."""

import threading

import numpy as np

MODEL_NAME = "PP-OCRv6_medium_det"

_detector = None
_lock = threading.Lock()


def get_detector():
    """Load the baked-in model once and reuse it."""
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                from paddleocr import TextDetection

                _detector = TextDetection(model_name=MODEL_NAME)
    return _detector


def detect(image: np.ndarray) -> list[dict]:
    """Detect text regions in a BGR image.

    Returns a list of {"points": [[x, y] x4], "score": float}.
    """
    boxes = []
    for result in get_detector().predict(image):
        for poly, score in zip(result["dt_polys"], result["dt_scores"]):
            boxes.append(
                {
                    "points": np.asarray(poly).astype(int).tolist(),
                    "score": float(score),
                }
            )
    return boxes
