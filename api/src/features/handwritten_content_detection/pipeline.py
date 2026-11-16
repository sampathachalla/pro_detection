"""Handwritten content detection: draw borders around the handwriting in an uploaded image.

detect(image_id)
  1. check     preprocessing status is "ready"
  2. detect    PaddleOCR on the processed image; nothing found -> EasyOCR fallback
  3. map back  processed coordinates -> original upright photo (undo deskew, divide by scale)
  4. clean up  drop tiny boxes, sort into reading order
  5. draw      borders on the original photo (content_draw_service)
  6. store     annotated.jpg + handwriting_detection.json next to the preprocessing files
"""

import json
from datetime import datetime, timezone

import cv2
import numpy as np

from api.src.features.preprocessing import PreprocessingError, PreprocessingPipeline, Status
from api.src.features.preprocessing.config import KEY_PREFIX
from api.src.services import easyocr_service, paddle_service
from api.src.services.content_draw_service import draw_boxes
from api.src.services.image_loader import load_image, load_image_upright

from .config import ANNOTATED_FILENAME, ANNOTATED_JPEG_QUALITY, DETECTION_VERSION, RESULT_FILENAME, DetectionConfig
from .schemas import DetectionError, DetectionErrorCode, DetectionResult, Detector


def map_to_original(points, processed: dict, source_width: int, source_height: int) -> list[list[int]]:
    """Map points from the processed image back to the original upright image.

    processed: manifest["processed"] -> {"width", "height", "scale", "angle"}.
    Preprocessing resized (scale) and then optionally deskewed (angle about the centre),
    so this undoes the rotation first, then the scale, and clips to the image.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if processed.get("angle"):
        center = (processed["width"] / 2, processed["height"] / 2)
        matrix = cv2.getRotationMatrix2D(center, processed["angle"], 1.0)
        pts = cv2.transform(pts, cv2.invertAffineTransform(matrix))
    pts = pts.reshape(-1, 2) / processed["scale"]
    pts[:, 0] = np.clip(pts[:, 0], 0, source_width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, source_height - 1)
    return np.rint(pts).astype(int).tolist()


def reading_order(regions: list[dict], line_tolerance: float) -> list[dict]:
    """Sort top-to-bottom by line, then left-to-right within a line."""
    if not regions:
        return []
    heights = sorted(r["bbox"][3] - r["bbox"][1] for r in regions)
    tolerance = max(1, heights[len(heights) // 2]) * line_tolerance
    lines: list[tuple[float, list[dict]]] = []
    for region in sorted(regions, key=lambda r: (r["bbox"][1] + r["bbox"][3]) / 2):
        center_y = (region["bbox"][1] + region["bbox"][3]) / 2
        if lines and abs(center_y - lines[-1][0]) < tolerance:
            lines[-1][1].append(region)
        else:
            lines.append((center_y, [region]))
    return [r for _, line in lines for r in sorted(line, key=lambda r: r["bbox"][0])]


class HandwrittenContentDetector:
    def __init__(self, preprocessing: PreprocessingPipeline | None = None, config: DetectionConfig | None = None):
        self.preprocessing = preprocessing or PreprocessingPipeline()
        self.storage = self.preprocessing.storage
        self.config = config or DetectionConfig.from_env()

    # ---- public API -------------------------------------------------------

    def detect(self, image_id: str) -> DetectionResult:
        cfg = self.config

        # 1. check
        manifest = self._manifest(image_id)
        if manifest["status"] != Status.READY:
            raise DetectionError(DetectionErrorCode.NOT_READY, "The image is waiting for retake or continue.")
        processed = self.preprocessing.load_processed(image_id)

        # 2. detect: Paddle first, EasyOCR fallback
        boxes, detector = paddle_service.detect(processed), Detector.PADDLE
        if not boxes:
            boxes, detector = easyocr_service.detect(processed, width_ths=cfg.easyocr_width_ths), Detector.EASYOCR
        if not boxes:
            detector = Detector.NONE

        # 3. map back + 4. clean up
        source = manifest["source"]
        regions = []
        for box in boxes:
            points = map_to_original(box["points"], manifest["processed"], source["width"], source["height"])
            if cv2.contourArea(np.asarray(points, dtype=np.float32)) < cfg.min_box_area:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            regions.append({"points": points, "bbox": [min(xs), min(ys), max(xs), max(ys)], "score": box["score"]})
        regions = reading_order(regions, cfg.line_tolerance)

        # 5. draw on the original upright photo
        original = load_image_upright(self.storage.get(source["key"]))
        annotated = draw_boxes(original, regions, color=cfg.color, labels=cfg.labels)

        # 6. store
        prefix = f"{KEY_PREFIX}/{image_id}/"
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("failed to encode annotated image")
        self.storage.put(prefix + ANNOTATED_FILENAME, encoded.tobytes(), "image/jpeg")
        result = {
            "image_id": image_id,
            "detection_version": DETECTION_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "detector": detector,
            "count": len(regions),
            "image": {"width": source["width"], "height": source["height"]},
            "annotated_key": prefix + ANNOTATED_FILENAME,
            "regions": regions,
        }
        self.storage.put(prefix + RESULT_FILENAME, json.dumps(result, indent=2).encode(), "application/json")
        return DetectionResult(image_id, detector, regions, annotated, result)

    def get_result(self, image_id: str) -> dict:
        self._manifest(image_id)
        try:
            return json.loads(self.storage.get(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}"))
        except FileNotFoundError:
            raise DetectionError(DetectionErrorCode.NOT_FOUND, "Detection has not been run for this image.") from None

    def load_annotated(self, image_id: str) -> np.ndarray:
        return load_image(self.storage.get(self.get_result(image_id)["annotated_key"]))

    # ---- internals --------------------------------------------------------

    def _manifest(self, image_id: str) -> dict:
        try:
            return self.preprocessing.get_manifest(image_id)
        except PreprocessingError:
            raise DetectionError(DetectionErrorCode.NOT_FOUND, "unknown image id") from None
