"""Handwritten content detection: draw borders around the handwriting in an uploaded image.

detect(image_id)
  1. check     preprocessing status is "ready"
  2. detect    PaddleOCR on the processed image (no fallback; nothing found -> detector "none")
     sideways  most boxes taller than wide -> orientation model picks the turn,
               preprocessing turns the page, detect again
  3. map back  processed coordinates -> original upright photo (undo deskew, divide by scale)
  4. clean up  drop tiny boxes, sort into reading order (column by column)
  5. draw      borders on the original photo (content_draw_service)
  6. store     annotated.jpg + handwriting_detection.json next to the preprocessing files
"""

import json
from datetime import datetime, timezone

import cv2
import numpy as np

from api.src.features.preprocessing import PreprocessingError, PreprocessingPipeline, Status
from api.src.features.preprocessing.config import KEY_PREFIX
from api.src.services import orientation_service, paddle_service
from api.src.services.content_draw_service import draw_boxes
from api.src.services.image_loader import load_image

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


def is_sideways(boxes: list[dict], min_boxes: int = 2, tall_ratio: float = 1.5, min_share: float = 0.6) -> bool:
    """True when most detected text is in tall, narrow boxes: lines running up/down a turned page."""
    if len(boxes) < min_boxes:
        return False
    tall = 0
    for box in boxes:
        pts = np.asarray(box["points"])
        width, height = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        tall += height > tall_ratio * width
    return tall >= min_share * len(boxes)


def split_columns(regions: list[dict], max_crossing: float = 0.1, min_lines: int = 3) -> list[list[dict]]:
    """Split regions into side-by-side columns, left to right. One column if there is no clear gutter.

    A gutter is an x range that at most max_crossing × len(regions) boxes cross. A split needs
    at least min_lines regions on each side, and at least 2 of them standing side by side
    (overlapping in y), so word gaps and indented short lines don't count as columns.
    """
    if len(regions) < 2 * min_lines:
        return [regions]
    left_edge = min(r["bbox"][0] for r in regions)
    width = max(r["bbox"][2] for r in regions) - left_edge + 1
    cover = np.zeros(width + 1, dtype=int)
    for r in regions:
        cover[r["bbox"][0] - left_edge] += 1
        cover[r["bbox"][2] - left_edge + 1] -= 1
    cover = np.cumsum(cover)[:width]

    for limit in range(int(len(regions) * max_crossing) + 1):  # cleanest gutter first
        runs, start = [], None
        for x, open_ in enumerate(np.append(cover <= limit, False)):
            if open_ and start is None:
                start = x
            elif not open_ and start is not None:
                runs.append((x - start, start + (x - start) // 2))
                start = None
        for _, mid in sorted(runs, reverse=True):  # widest gutter first
            split = left_edge + mid
            left = [r for r in regions if (r["bbox"][0] + r["bbox"][2]) / 2 < split]
            right = [r for r in regions if (r["bbox"][0] + r["bbox"][2]) / 2 >= split]
            if len(left) < min_lines or len(right) < min_lines:
                continue
            side_by_side = sum(
                any(a["bbox"][1] < b["bbox"][3] and b["bbox"][1] < a["bbox"][3] for b in right) for a in left
            )
            if side_by_side >= 2:
                return split_columns(left, max_crossing, min_lines) + split_columns(right, max_crossing, min_lines)
    return [regions]


def reading_order(regions: list[dict], line_tolerance: float, max_crossing: float = 0.1, min_lines: int = 3) -> list[dict]:
    """Column by column (left to right); within a column top-to-bottom by line, then left-to-right."""
    columns = split_columns(regions, max_crossing, min_lines) if regions else []
    return [r for column in columns for r in _rows(column, line_tolerance)]


def _rows(regions: list[dict], line_tolerance: float) -> list[dict]:
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

        # 2. detect with Paddle; turn a sideways page and detect again
        boxes, detector = self._find_boxes(processed)
        if cfg.auto_rotate and is_sideways(boxes):
            turns = orientation_service.classify(processed)["angle"] // 90
            if turns in (1, 3):  # the model agrees the page is sideways
                processed = self.preprocessing.rotate(image_id, turns).image
                manifest = self._manifest(image_id)
                boxes, detector = self._find_boxes(processed)

        # 3. map back + 4. clean up
        source = manifest["source"]
        regions = []
        for box in boxes:
            points = map_to_original(box["points"], manifest["processed"], source["width"], source["height"])
            if cv2.contourArea(np.asarray(points, dtype=np.float32)) < cfg.min_box_area:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            regions.append({"points": points, "bbox": [min(xs), min(ys), max(xs), max(ys)], "score": box["score"]})
        regions = reading_order(regions, cfg.line_tolerance, cfg.column_max_crossing, cfg.column_min_lines)

        # 5. draw on the original upright photo
        original = self.preprocessing.load_original(image_id)
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
            "rotation": source.get("rotation", 0) * 90,  # degrees the page was turned counter-clockwise
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

    def _find_boxes(self, processed: np.ndarray) -> tuple[list[dict], str]:
        boxes = paddle_service.detect(processed)
        return boxes, (Detector.PADDLE if boxes else Detector.NONE)

    def _manifest(self, image_id: str) -> dict:
        try:
            return self.preprocessing.get_manifest(image_id)
        except PreprocessingError:
            raise DetectionError(DetectionErrorCode.NOT_FOUND, "unknown image id") from None
