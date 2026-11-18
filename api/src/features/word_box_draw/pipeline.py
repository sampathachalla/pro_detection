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
from api.src.services.image_loader import load_image
from api.src.services.word_segmentation_service import segment_words

from .config import ANNOTATED_FILENAME, ANNOTATED_JPEG_QUALITY, RESULT_FILENAME, WORDS_VERSION, WordBoxConfig
from .schemas import WordBoxResult

SPELLING_BOX_COLOR = (0, 0, 220)  # BGR red


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
        original = self.detector.preprocessing.load_original(image_id)

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
        annotated = self._draw_preview(
            original, words, detection["regions"] if cfg.show_lines else [],
            word_boxes=True, numbers=True,
        )

        # 6. store
        prefix = f"{KEY_PREFIX}/{image_id}/"
        self.storage.put(prefix + ANNOTATED_FILENAME, self._encode_preview(annotated), "image/jpeg")
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

    def preview(
        self, image_id: str, words: list[dict], *, word_boxes: bool, line_boxes: bool, numbers: bool,
        spelling_boxes: set[int], crop_to_text: bool = False,
    ) -> bytes:
        original = self.detector.preprocessing.load_original(image_id)
        regions = self.detector.get_result(image_id)["regions"] if line_boxes or crop_to_text else []
        colors = {number: SPELLING_BOX_COLOR for number in spelling_boxes}
        image = self._draw_preview(
            original, words, regions if line_boxes else [], word_boxes=word_boxes, numbers=numbers,
            border_colors=colors,
        )
        if crop_to_text and regions:
            bounds = [region["bbox"] for region in regions]
            margin = max(16, default_thickness(original) * 3)
            x0 = max(0, min(box[0] for box in bounds) - margin)
            y0 = max(0, min(box[1] for box in bounds) - margin)
            x1 = min(image.shape[1], max(box[2] for box in bounds) + margin + 1)
            y1 = min(image.shape[0], max(box[3] for box in bounds) + margin + 1)
            image = image[y0:y1, x0:x1]
        return self._encode_preview(image)

    def _draw_preview(
        self, image: np.ndarray, words: list[dict], regions: list[dict], *, word_boxes: bool, numbers: bool,
        border_colors: dict[int, tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        if regions:
            image = draw_boxes(image, regions, color=self.detector.config.color)
        if word_boxes or numbers:
            image = draw_boxes(
                image,
                words,
                color=self.config.color,
                thickness=max(2, default_thickness(image) // 2),
                labels=numbers,
                borders=word_boxes,
                border_colors=border_colors,
            )
        return image

    @staticmethod
    def _encode_preview(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("failed to encode word image")
        return encoded.tobytes()

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
