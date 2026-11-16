"""Synthetic images shared by the service tests (no files or downloads needed)."""

import cv2
import numpy as np


def text_image(text: str = "Handwritten text 123", width: int = 720, height: int = 180) -> np.ndarray:
    """White BGR image with one line of script-style black text."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, text, (24, height * 2 // 3), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    return image


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()
