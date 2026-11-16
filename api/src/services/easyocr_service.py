"""EasyOCR text detection (reader.detect, English model)."""

import os
import threading

import numpy as np

_reader = None
_lock = threading.Lock()


def get_reader():
    """Load the baked-in model once and reuse it. Never downloads."""
    global _reader
    if _reader is None:
        with _lock:
            if _reader is None:
                import easyocr

                _reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                    model_storage_directory=os.environ["EASYOCR_MODULE_PATH"],
                    download_enabled=False,
                )
    return _reader


def detect(image: np.ndarray, width_ths: float = 0.1) -> list[dict]:
    """Detect text regions in a BGR image.

    Returns a list of {"points": [[x, y] x4], "score": None}
    (EasyOCR detection does not return confidence scores).
    """
    horizontal, free = get_reader().detect(image, width_ths=width_ths)
    boxes = []
    for x_min, x_max, y_min, y_max in horizontal[0]:
        points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        boxes.append({"points": [[int(x), int(y)] for x, y in points], "score": None})
    for poly in free[0]:
        boxes.append({"points": [[int(x), int(y)] for x, y in poly], "score": None})
    return boxes
