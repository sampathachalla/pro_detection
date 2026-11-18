"""Turn handwriting into printed text with OpenAI, in one request per page.

read(image_id)                       (handwritten content detection must have run)
  1. load    the detected line regions + the original upright photo
  2. draw    numbered line borders (1, 2, 3 in reading order), cropped to the handwriting
  3. ask     the OpenAI vision model for the text of every numbered line (one request)
  4. store   recognized_text_openai.json next to the other files
"""

import json
import os
from time import perf_counter

from api.src.features.handwritten_content_detection import HandwrittenContentDetector
from api.src.features.preprocessing.config import KEY_PREFIX
from api.src.services import openai_LLM_call
from api.src.services.content_draw_service import default_thickness, draw_boxes

from .config import DEFAULT_MODEL, RESULT_FILENAME


class LLMHandwritingRecognizer:
    def __init__(self, detector: HandwrittenContentDetector):
        self.detector = detector
        self.storage = detector.storage

    def read(self, image_id: str) -> dict:
        started = perf_counter()
        regions = self.detector.get_result(image_id)["regions"]
        model, api_key = self._settings()

        texts, usage = {}, {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        if regions:
            original = self.detector.preprocessing.load_original(image_id)
            numbered = draw_boxes(original, regions, color=self.detector.config.color, labels=True)
            # Crop to the handwriting; the margin leaves room for the numbers drawn above each box.
            margin = default_thickness(original) * 12
            x0 = max(0, min(region["bbox"][0] for region in regions) - margin)
            y0 = max(0, min(region["bbox"][1] for region in regions) - margin)
            x1 = min(numbered.shape[1], max(region["bbox"][2] for region in regions) + margin + 1)
            y1 = min(numbered.shape[0], max(region["bbox"][3] for region in regions) + margin + 1)
            texts, usage = openai_LLM_call.transcribe_numbered_lines(
                numbered[y0:y1, x0:x1], set(range(1, len(regions) + 1)), model=model, api_key=api_key,
            )

        lines = [
            {"line": index, "bbox": region["bbox"], "text": texts.get(index + 1, "")}
            for index, region in enumerate(regions)
        ]
        result = {
            "image_id": image_id,
            "model": model,
            "lines": lines,
            "text": "\n".join(line["text"] for line in lines if line["text"]),
            "usage": usage,
            "request_count": int(bool(regions)),
            "processing_ms": round((perf_counter() - started) * 1000),
        }
        self.storage.put(
            f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}",
            json.dumps(result, indent=2).encode(),
            "application/json",
        )
        return result

    def get_result(self, image_id: str) -> dict | None:
        """Stored recognized_text_openai.json, or None if the text hasn't been read for this image."""
        try:
            return json.loads(self.storage.get(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}"))
        except FileNotFoundError:
            return None

    @staticmethod
    def _settings() -> tuple[str, str]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("LLM_API", "").strip()
        if not api_key:
            raise openai_LLM_call.OpenAIRecognitionError(
                "OpenAI handwriting recognition requires OPENAI_API_KEY or LLM_API.", status_code=503
            )
        return os.environ.get("OPENAI_HANDWRITING_MODEL", "").strip() or DEFAULT_MODEL, api_key
