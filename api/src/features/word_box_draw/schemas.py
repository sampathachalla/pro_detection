"""Result type for word box drawing.

Errors: an unknown image or an image whose handwriting has not been detected
yet raises the detection feature's DetectionError("not_found").
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WordBoxResult:
    image_id: str
    words: list[dict]      # [{"line": int, "points": [[x, y] x4], "bbox": [x0, y0, x1, y1]}]
    annotated: np.ndarray  # original upright photo with word borders drawn (BGR)
    result: dict           # contents of word_boxes.json
