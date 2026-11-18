"""Straighten detected line regions before handwriting recognition."""

import cv2
import numpy as np


def rectify_line(image: np.ndarray, points: list[list[int]]) -> np.ndarray | None:
    """Warp a horizontal text line's quadrilateral to a padded rectangular crop."""
    corners = np.asarray(points, dtype=np.float32)
    if corners.shape != (4, 2):
        return None

    by_x = corners[np.argsort(corners[:, 0], kind="stable")]
    left, right = by_x[:2], by_x[2:]
    top_left, bottom_left = left[np.argsort(left[:, 1])]
    top_right, bottom_right = right[np.argsort(right[:, 1])]
    source = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    if not cv2.isContourConvex(source):
        return None

    width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))) + 1
    height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))) + 1
    if width < 2 or height < 2:
        return None

    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    crop = cv2.warpPerspective(image, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)

    margin = max(2, round(height * 0.08))
    paper = np.median(crop, axis=(0, 1)).tolist()
    return cv2.copyMakeBorder(crop, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=paper)
