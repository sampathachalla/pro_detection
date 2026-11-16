"""Image quality measurements (OpenCV). Measures only; callers decide what passes."""

import cv2
import numpy as np


def assess(image: np.ndarray) -> dict:
    """Measure a BGR or grayscale image.

    Returns {
        "blur": float,        # variance of the Laplacian; lower = blurrier
        "brightness": float,  # mean gray level 0-255; lower = darker
        "contrast": float,    # std dev of gray levels; lower = flatter
        "width": int, "height": int,
    }
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "width": int(width),
        "height": int(height),
    }
