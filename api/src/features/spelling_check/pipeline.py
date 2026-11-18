"""Spelling check: send the numbered word image to OpenAI and store the misspelled box numbers.

check(image_id)                      (word_box_draw must have run)
  1. load    the stored word boxes
  2. draw    one image: word borders + box numbers, cropped to the handwriting
  3. ask     the OpenAI vision model which numbered boxes are misspelled (one request)
  4. store   recognized_words_openai.json next to the other files
"""

import json
from time import perf_counter

from api.src.features.llm_handwritten_recognition import LLMHandwritingRecognizer
from api.src.features.preprocessing.config import KEY_PREFIX
from api.src.features.word_box_draw import WordBoxDrawer
from api.src.services import openai_LLM_call
from api.src.services.image_loader import load_image

from .config import RESULT_FILENAME


class SpellingChecker:
    def __init__(self, word_drawer: WordBoxDrawer):
        self.word_drawer = word_drawer
        self.storage = word_drawer.storage

    def check(self, image_id: str) -> dict:
        started = perf_counter()
        boxes = self.word_drawer.get_result(image_id)
        if boxes is None:
            raise openai_LLM_call.OpenAIRecognitionError(
                "Word boxes have not been drawn for this image.", status_code=404
            )
        llm, model, api_key = LLMHandwritingRecognizer._settings()

        mistakes, usage = [], {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        box_numbers = set(range(1, len(boxes["words"]) + 1))  # same 1-based order as the drawn labels
        if box_numbers:
            numbered = load_image(self.word_drawer.preview(
                image_id, boxes["words"], word_boxes=True, line_boxes=False, numbers=True,
                spelling_boxes=set(), crop_to_text=True,
            ))
            mistakes, usage = llm.check_numbered_spelling(
                numbered, box_numbers, model=model, api_key=api_key,
            )
        result = {
            "image_id": image_id,
            "model": model,
            "mistakes": sorted(mistakes, key=lambda mistake: mistake["box_number"]),
            "usage": usage,
            "request_count": int(bool(box_numbers)),
            "processing_ms": round((perf_counter() - started) * 1000),
        }
        self.storage.put(
            f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}",
            json.dumps(result, indent=2).encode(),
            "application/json",
        )
        return result

    def get_result(self, image_id: str) -> dict | None:
        """Stored recognized_words_openai.json, or None if spelling hasn't been checked for this image."""
        try:
            return json.loads(self.storage.get(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}"))
        except FileNotFoundError:
            return None

    def spelling_boxes(self, image_id: str) -> set[int]:
        try:
            result = json.loads(self.storage.get(f"{KEY_PREFIX}/{image_id}/{RESULT_FILENAME}"))
        except FileNotFoundError:
            return set()
        if "mistakes" in result:
            return {mistake["box_number"] for mistake in result["mistakes"]}
        return {word["number"] for word in result["words"] if word.get("correction")}
