"""OpenCV preprocessing for handwriting before text detection.

Pipeline: resize -> grayscale -> denoise -> contrast (CLAHE) -> optional deskew.
The output stays 3-channel BGR because both PaddleOCR and EasyOCR expect it.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessOptions:
    max_side: int = 1600          # downscale large photos to keep CPU time down
    denoise: bool = True          # remove paper texture / camera noise
    clahe: bool = True            # boost faint pen strokes
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    deskew: bool = False          # straighten a rotated page (off: slanted handwriting can fool it)


def resize(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is at most max_side. Returns (image, scale)."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the page so text lines are horizontal. Returns (image, angle in degrees)."""
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(ink)
    if coords is None:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5:
        return gray, 0.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, float(angle)


def preprocess(image: np.ndarray, options: PreprocessOptions | None = None) -> tuple[np.ndarray, dict]:
    """Prepare a BGR image for detection.

    Returns (bgr_image, info). info["scale"] maps boxes back to the original
    image: original_xy = detected_xy / scale. If deskew ran, boxes are in the
    rotated frame (info["angle"]).
    """
    options = options or PreprocessOptions()
    image, scale = resize(image, options.max_side)
    gray = to_gray(image)
    if options.denoise:
        gray = denoise(gray)
    if options.clahe:
        gray = enhance_contrast(gray, options.clahe_clip_limit, options.clahe_tile_size)
    angle = 0.0
    if options.deskew:
        gray, angle = deskew(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), {"scale": scale, "angle": angle}
