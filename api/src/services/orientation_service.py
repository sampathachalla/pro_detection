"""Page orientation classification (PaddleOCR PP-LCNet_x1_0_doc_ori): is the page turned 0, 90, 180 or 270°?

Classification only; it does not rotate anything. The model is baked into the image under
PADDLE_PDX_CACHE_HOME, loads on the first call and is then reused (thread-safe).
"""

import threading

import numpy as np

MODEL_NAME = "PP-LCNet_x1_0_doc_ori"

_classifier = None
_lock = threading.Lock()


def get_classifier():
    """Load the baked-in model once and reuse it."""
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                from paddleocr import DocImgOrientationClassification

                _classifier = DocImgOrientationClassification(model_name=MODEL_NAME)
    return _classifier


def classify(image: np.ndarray) -> dict:
    """Classify a BGR page image. Returns {"angle": 0 | 90 | 180 | 270, "score": float}."""
    for result in get_classifier().predict(image):
        return {"angle": int(result["label_names"][0]), "score": float(result["scores"][0])}
    return {"angle": 0, "score": 0.0}
