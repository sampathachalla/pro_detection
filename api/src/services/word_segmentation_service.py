"""Split one line of handwriting into word boxes (OpenCV, no model).

Ink is separated from paper with Otsu thresholding, then the empty columns
between ink are measured: gaps wider than `gap_ratio` x the small-letter height
are word breaks, narrower gaps are spaces between letters.

The small-letter height (the band where a, e, o, n sit) is used instead of the
full ink height because tall letters (T, h, l) and tails (y, g, f) make the full
height ~3x larger in cursive, which made the gap limit too big to split words.
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


def remove_ruled_lines(ink: np.ndarray) -> np.ndarray:
    """Drop ruled paper lines from an ink mask so they can't bridge the gaps between words.

    A ruled line is a thin stroke longer than any letter. It is found with a horizontal opening
    (kernel as long as the crop is tall) on a mask thickened by a few rows, so a slightly slanted line
    still forms one long horizontal run. Ink inside that band is removed; letters only lose the few
    rows where they cross the line. A longer kernel (2x) missed the short pieces where a slanted line
    enters or leaves the crop, and those pieces still joined words.
    """
    height, width = ink.shape
    length = height
    if width <= length:
        return ink
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    lines = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    if not lines.any():
        return ink
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def small_letter_height(ink: np.ndarray) -> int:
    """Height of the band of rows holding at least half the ink of the fullest row.

    Rows inked across most of the width (a ruled paper line) are skipped so they can't become the "fullest row".
    """
    per_row = np.count_nonzero(ink, axis=1)
    per_row[per_row > 0.6 * ink.shape[1]] = 0
    if not per_row.any():
        return 0
    band = np.flatnonzero(per_row >= 0.5 * per_row.max())
    return int(band[-1] - band[0] + 1)


def segment_words(crop: np.ndarray, gap_ratio: float = 0.45, min_ink_fraction: float = 0.002) -> list[list[int]]:
    """Word boxes inside one text-line crop.

    crop:      BGR or grayscale image of a single line (dark text on light background).
    gap_ratio: a gap wider than gap_ratio x small-letter height starts a new word.
    Returns [[x0, y0, x1, y1], ...] in crop coordinates (inclusive), left to right.
    Returns [] when the crop has (almost) no ink.
    """
    ink = remove_ruled_lines(ink_mask(crop))
    if ink.mean() / 255 < min_ink_fraction:
        return []
    letter_height = small_letter_height(ink)
    if not letter_height:  # only ruled-line rows: fall back to the full ink height
        rows = np.flatnonzero(ink.any(axis=1))
        letter_height = rows[-1] - rows[0] + 1
    min_gap = max(2, round(gap_ratio * letter_height))

    columns = ink.any(axis=0)
    xs = np.flatnonzero(columns)
    # Split wherever the run of empty columns between two inked columns is >= min_gap.
    breaks = np.flatnonzero(np.diff(xs) > min_gap)
    starts = np.concatenate(([xs[0]], xs[breaks + 1]))
    ends = np.concatenate((xs[breaks], [xs[-1]]))

    boxes = []
    width = ink.shape[1]
    for x0, x1 in zip(starts, ends):
        word_rows = np.flatnonzero(ink[:, x0:x1 + 1].any(axis=1))
        y0, y1 = word_rows[0], word_rows[-1]
        # At the crop edge, a thin sliver (paper edge) or a sparse box (ruled-line pieces cut off by the
        # line polygon, ~4% ink) is not a word. Real words fill ~10-15% of their box.
        if x0 <= 2 or x1 >= width - 3:
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            if x1 - x0 + 1 < 0.3 * letter_height or np.count_nonzero(ink[y0:y1 + 1, x0:x1 + 1]) < 0.06 * area:
                continue
        boxes.append([int(x0), int(y0), int(x1), int(y1)])
    return boxes
