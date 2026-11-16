"""Handwriting recognition: read the text inside each detected handwritten line (TrOCR).

read(image_id)                       (handwritten content detection must have run)
  1. load      detected line regions + the original upright photo
  2. crop      each line (masked to its polygon, outside = paper colour)
  3. read      trocr_service.recognize -> text + confidence, one line at a time (~1 s each)
  4. store     recognized_text.json next to the other files
"""

import json
import time
from datetime import datetime, timezone

from api.src.features.handwritten_content_detection import HandwrittenContentDetector
from api.src.features.preprocessing.config import KEY_PREFIX
from api.src.features.word_box_draw import crop_line
from api.src.services import trocr_service
from api.src.services.image_loader import load_image_upright

from .config import MODEL_NAME, RECOGNITION_VERSION, RESULT_FILENAME, RecognitionConfig


class HandwritingRecognizer:
    def __init__(self, detector: HandwrittenContentDetector | None = None, config: RecognitionConfig | None = None):
        self.detector = detector or HandwrittenContentDetector()
        self.storage = self.detector.storage
        self.config = config or RecognitionConfig.from_env()

    def read(self, image_id: str) -> dict:
        """Read every detected line. Returns the stored recognized_text.json contents.

        Raises the detection feature's DetectionError("not_found") if detection hasn't run.
        """
        started = time.perf_counter()
        detection = self.detector.get_result(image_id)
        manifest = self.detector.preprocessing.get_manifest(image_id)
        original = load_image_upright(self.storage.get(manifest["source"]["key"]))

        lines = []
        for index, region in enumerate(detection["regions"]):
            crop, _, _ = crop_line(original, region["points"])
            read = trocr_service.recognize(crop) if crop.size else {"text": "", "confidence": 0.0}
            lines.append({
                "line": index,
                "bbox": region["bbox"],
                "text": read["text"],
                "confidence": round(read["confidence"], 3),
                # TrOCR confidence alone doesn't catch drawings (an arrow read as "#" scored 0.54),
                # so text with no letters or digits is also treated as unclear.
                "low_confidence": read["confidence"] < self.config.min_confidence
                or not any(ch.isalnum() for ch in read["text"]),
            })

        result = {
            "image_id": image_id,
            "recognition_version": RECOGNITION_VERSION,
            "model": MODEL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.perf_counter() - started, 2),
            "lines": lines,
            "text": "\n".join(line["text"] for line in lines if not line["low_confidence"]),
        }
        self.storage.put(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}", json.dumps(result, indent=2).encode(), "application/json")
        return result

    def get_result(self, image_id: str) -> dict | None:
        """Stored recognized_text.json, or None if the text hasn't been read yet."""
        self.detector.get_result(image_id)  # validates the id / detection exists
        try:
            return json.loads(self.storage.get(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}"))
        except FileNotFoundError:
            return None
