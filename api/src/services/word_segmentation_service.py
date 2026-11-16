"""Split one line of handwriting into word boxes (OpenCV, no model).

Ink is separated from paper with Otsu thresholding, then the empty columns
between ink are measured: gaps wider than `gap_ratio` x the text height are
word breaks, narrower gaps are spaces between letters.
"""

import cv2
import numpy as np


def ink_mask(crop: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of dark ink on light paper, with tiny specks removed."""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(4, int(0.0015 * gray.shape[0] * gray.shape[0]))
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def segment_words(crop: np.ndarray, gap_ratio: float = 0.35, min_ink_fraction: float = 0.002) -> list[list[int]]:
    """Word boxes inside one text-line crop.

    crop:      BGR or grayscale image of a single line (dark text on light background).
    gap_ratio: a gap wider than gap_ratio x text height starts a new word.
    Returns [[x0, y0, x1, y1], ...] in crop coordinates (inclusive), left to right.
    Returns [] when the crop has (almost) no ink.
    """
    ink = ink_mask(crop)
    if ink.mean() / 255 < min_ink_fraction:
        return []
    rows = np.flatnonzero(ink.any(axis=1))
    text_height = rows[-1] - rows[0] + 1
    min_gap = max(2, round(gap_ratio * text_height))

    columns = ink.any(axis=0)
    xs = np.flatnonzero(columns)
    # Split wherever the run of empty columns between two inked columns is >= min_gap.
    breaks = np.flatnonzero(np.diff(xs) > min_gap)
    starts = np.concatenate(([xs[0]], xs[breaks + 1]))
    ends = np.concatenate((xs[breaks], [xs[-1]]))

    boxes = []
    for x0, x1 in zip(starts, ends):
        word_rows = np.flatnonzero(ink[:, x0:x1 + 1].any(axis=1))
        boxes.append([int(x0), int(word_rows[0]), int(x1), int(word_rows[-1])])
    return boxes
