"""Word box drawing: border every word inside the detected handwritten lines.

draw(image_id)                       (handwritten content detection must have run)
  1. load      detected line regions + the original upright photo
  2. crop      each line's bounding box, masked to the line polygon (neighbours removed)
  3. segment   split the crop into words at wide gaps (word_segmentation_service)
  4. map back  crop coordinates -> photo coordinates; drop tiny boxes
  5. draw      word borders on the original photo (content_draw_service)
  6. store     words_annotated.jpg + word_boxes.json next to the other files
"""

import json
from datetime import datetime, timezone

import cv2
import numpy as np

from api.src.features.handwritten_content_detection import HandwrittenContentDetector
from api.src.features.preprocessing.config import KEY_PREFIX
from api.src.services.content_draw_service import default_thickness, draw_boxes
from api.src.services.image_loader import load_image, load_image_upright
from api.src.services.word_segmentation_service import segment_words

from .config import ANNOTATED_FILENAME, ANNOTATED_JPEG_QUALITY, RESULT_FILENAME, WORDS_VERSION, WordBoxConfig
from .schemas import WordBoxResult


def crop_line(image: np.ndarray, points: list[list[int]]) -> tuple[np.ndarray, int, int]:
    """Crop the line's bounding box; pixels outside the line polygon become the paper colour.

    The paper colour (median inside the polygon) is used rather than white: pure white
    corners on a tilted line make Otsu split "white vs paper" and treat paper as ink.
    Returns (crop, x_offset, y_offset).
    """
    poly = np.asarray(points, dtype=np.int32)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    crop = image[y0:y1 + 1, x0:x1 + 1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    inside = crop[mask > 0]
    if inside.size:
        crop[mask == 0] = np.median(inside, axis=0).astype(crop.dtype)
    return crop, int(x0), int(y0)


class WordBoxDrawer:
    def __init__(self, detector: HandwrittenContentDetector | None = None, config: WordBoxConfig | None = None):
        self.detector = detector or HandwrittenContentDetector()
        self.storage = self.detector.storage
        self.config = config or WordBoxConfig.from_env()

    def draw(self, image_id: str) -> WordBoxResult:
        cfg = self.config

        # 1. load (raises DetectionError not_found if detection hasn't run)
        detection = self.detector.get_result(image_id)
        manifest = self.detector.preprocessing.get_manifest(image_id)
        original = load_image_upright(self.storage.get(manifest["source"]["key"]))

        # 2-4. crop, segment, map back
        words = []
        for line_index, region in enumerate(detection["regions"]):
            crop, dx, dy = crop_line(original, region["points"])
            if crop.size == 0:
                continue
            for x0, y0, x1, y1 in segment_words(crop, cfg.gap_ratio):
                if (x1 - x0 + 1) * (y1 - y0 + 1) < cfg.min_word_area:
                    continue
                bx0, by0, bx1, by1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
                words.append({
                    "line": line_index,
                    "points": [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
                    "bbox": [bx0, by0, bx1, by1],
                })

        # 5. draw: green PaddleOCR line borders first (if enabled), then thinner blue word borders on top
        base = original
        if cfg.show_lines:
            base = draw_boxes(original, detection["regions"], color=self.detector.config.color)
        thickness = max(2, default_thickness(original) // 2)
        annotated = draw_boxes(base, words, color=cfg.color, thickness=thickness)

        # 6. store
        prefix = f"{KEY_PREFIX}/{image_id}/"
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("failed to encode word image")
        self.storage.put(prefix + ANNOTATED_FILENAME, encoded.tobytes(), "image/jpeg")
        result = {
            "image_id": image_id,
            "words_version": WORDS_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": len(words),
            "lines": len(detection["regions"]),
            "annotated_key": prefix + ANNOTATED_FILENAME,
            "words": words,
        }
        self.storage.put(prefix + RESULT_FILENAME, json.dumps(result, indent=2).encode(), "application/json")
        return WordBoxResult(image_id, words, annotated, result)

    def get_result(self, image_id: str) -> dict | None:
        """Stored word_boxes.json, or None if words haven't been drawn for this image."""
        self.detector.get_result(image_id)  # validates the id / detection exists
        try:
            return json.loads(self.storage.get(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}"))
        except FileNotFoundError:
            return None

    def load_annotated(self, image_id: str) -> np.ndarray | None:
        result = self.get_result(image_id)
        return load_image(self.storage.get(result["annotated_key"])) if result else None
