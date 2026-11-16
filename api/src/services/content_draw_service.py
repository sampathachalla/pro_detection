"""Draw borders around content (blocks, lines or words) on an image (OpenCV).

Box-level agnostic: any list of {"points": [[x, y], ...]} is drawn, so the same
service draws line boxes today and word boxes later.
"""

import cv2
import numpy as np

DEFAULT_COLOR = (0, 200, 0)  # BGR green


def default_thickness(image: np.ndarray) -> int:
    """Border width that stays visible when a large photo is shown at phone width.

    About 1/250 of the longest side, at least 3 px (a 4032 px photo gets 16 px,
    which is ~4 px when displayed ~1000 px tall).
    """
    return max(3, round(max(image.shape[:2]) / 250))


def draw_boxes(
    image: np.ndarray,
    boxes: list[dict],
    color: tuple[int, int, int] = DEFAULT_COLOR,
    thickness: int | None = None,
    labels: bool = False,
) -> np.ndarray:
    """Return a copy of `image` (BGR or grayscale) with each box drawn as a closed polygon.

    boxes:  [{"points": [[x, y], ...] (3+ points), ...}, ...]; other keys are ignored.
    labels: number the boxes 1, 2, 3 ... in the order given.
    The input image is not modified. Output is always BGR.
    """
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    width = thickness or default_thickness(out)
    for number, box in enumerate(boxes, start=1):
        points = np.asarray(box["points"], dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"box {number} needs 3 or more [x, y] points")
        cv2.polylines(out, [points.reshape(-1, 1, 2)], True, color, width, cv2.LINE_AA)
        if labels:
            x, y = points.min(axis=0)
            scale = 0.4 * width
            cv2.putText(out, str(number), (int(x), int(max(y - width, 12 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, width // 2), cv2.LINE_AA)
    return out
