"""LLM only: the photo goes straight to the LLM, with no preprocessing, PaddleOCR or OpenCV word boxes.

analyse(bytes)
  1. validate   same upload limits as preprocessing (size, JPEG/PNG, pixel count, minimum side)
  2. store      original.<ext> under preprocessing/<id>/, so the 24-hour cleanup removes it too
  3. ask        two LLM calls in parallel on the plain photo (shrunk to SEND_MAX_SIDE):
                  transcription   every handwritten line, copied letter by letter
                  spelling        every word with written / intended / box_2d; keeps the misspelled ones
  4. draw       numbered red boxes around the misspelled words on the photo -> llm_boxed.jpg
  5. store      llm_boxing.json (the full response, so a page reload shows it again)

Only Gemini can locate words (box_2d), so LLM_PROVIDER must be "gemini" (the default).
"""

import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import cv2
import numpy as np

from api.src.features.llm_handwritten_recognition import LLMHandwritingRecognizer
from api.src.features.preprocessing import ErrorCode, PreprocessingError, PreprocessingPipeline
from api.src.features.preprocessing.config import FORMATS, KEY_PREFIX
from api.src.services.content_draw_service import draw_boxes
from api.src.services.image_loader import inspect_image, load_image_upright
from api.src.services.llm_prompts import LLMRecognitionError

from .config import BOXED_FILENAME, BOXED_JPEG_QUALITY, MISTAKE_BOX_COLOR, RESULT_FILENAME, SEND_MAX_SIDE

log = logging.getLogger("uvicorn.error")
_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")


class LLMBoxing:
    def __init__(self, preprocessing: PreprocessingPipeline):
        self.storage = preprocessing.storage
        self.config = preprocessing.config

    def analyse(self, data: bytes, filename: str | None = None) -> dict:
        ext, content_type = self._validate(data)
        llm, model, api_key = LLMHandwritingRecognizer._settings()
        if not hasattr(llm, "check_spelling_with_boxes"):
            raise LLMRecognitionError("LLM only mode needs LLM_PROVIDER=gemini (it locates the words).", status_code=503)
        try:
            image = load_image_upright(data)
        except ValueError:
            raise PreprocessingError(ErrorCode.UNREADABLE, "The file is not a readable image.") from None

        image_id = uuid.uuid4().hex
        self.storage.put(self._key(image_id, f"original.{ext}"), data, content_type)

        height, width = image.shape[:2]
        scale = min(1.0, SEND_MAX_SIDE / max(height, width))
        sent = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else image

        with ThreadPoolExecutor(max_workers=2) as pool:
            transcription = pool.submit(self._safe, "Transcription", self._transcribe, llm, sent, model, api_key)
            spelling = pool.submit(self._safe, "Spelling check", self._spelling, llm, sent, model, api_key, width, height)
            transcription, spelling = transcription.result(), spelling.result()

        mistakes = spelling.get("mistakes", [])
        boxed = draw_boxes(image, mistakes, color=MISTAKE_BOX_COLOR, labels=True) if mistakes else image
        ok, encoded = cv2.imencode(".jpg", boxed, [cv2.IMWRITE_JPEG_QUALITY, BOXED_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("failed to encode the boxed image")
        self.storage.put(self._key(image_id, BOXED_FILENAME), encoded.tobytes(), "image/jpeg")

        result = {
            "image_id": image_id,
            "status": "done",
            "mode": "llm_only",
            "filename": filename,
            "warnings": [],
            "detector": "llm",
            "count": len(transcription.get("lines", [])),
            "image": {"width": width, "height": height},
            "regions": [],
            "annotated_url": f"/api/llm-images/{image_id}/boxed.jpg",
            "word_count": spelling.get("word_count"),
            "words": [],
            "words_annotated_url": None,
            "llm": {"transcription": transcription, "spelling": spelling},
        }
        self.storage.put(self._key(image_id, RESULT_FILENAME), json.dumps(result, indent=2).encode(), "application/json")
        return result

    def analyse_stored(self, image_id: str) -> dict:
        """Run analyse() again on a photo already uploaded (OCR or LLM only), using its stored original.

        Returns a new LLM-only result with its own image_id; the earlier analysis is kept.
        A page turned with rotate() is sent as originally uploaded (EXIF orientation only).
        """
        return self.analyse(self.load_original(image_id))

    def load_original(self, image_id: str) -> bytes:
        """The uploaded file's bytes (original.<ext>) for any image id, OCR or LLM only."""
        if not _IMAGE_ID.match(image_id or ""):
            raise PreprocessingError(ErrorCode.NOT_FOUND, "unknown image id")
        originals = [key for key in self.storage.list(f"{KEY_PREFIX}/{image_id}/")
                     if key.rsplit("/", 1)[-1].startswith("original.")]
        if not originals:
            raise PreprocessingError(ErrorCode.NOT_FOUND, "This image is no longer stored. Upload it again.")
        return self.storage.get(originals[0])

    def get_result(self, image_id: str) -> dict | None:
        """Stored llm_boxing.json, or None if this id was not analysed in LLM only mode."""
        if not _IMAGE_ID.match(image_id or ""):
            return None
        try:
            return json.loads(self.storage.get(self._key(image_id, RESULT_FILENAME)))
        except FileNotFoundError:
            return None

    def load_boxed(self, image_id: str) -> bytes:
        if not _IMAGE_ID.match(image_id or ""):
            raise PreprocessingError(ErrorCode.NOT_FOUND, "unknown image id")
        try:
            return self.storage.get(self._key(image_id, BOXED_FILENAME))
        except FileNotFoundError:
            raise PreprocessingError(ErrorCode.NOT_FOUND, "unknown image id") from None

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _key(image_id: str, name: str) -> str:
        return f"{KEY_PREFIX}/{image_id}/{name}"

    def _validate(self, data: bytes) -> tuple[str, str]:
        """Same hard limits as PreprocessingPipeline.submit, without its quality check or enhancement."""
        cfg = self.config
        if not data:
            raise PreprocessingError(ErrorCode.EMPTY, "The file is empty.")
        if len(data) >= cfg.max_bytes:
            raise PreprocessingError(ErrorCode.TOO_LARGE, f"The file must be smaller than {cfg.max_bytes // (1024 * 1024)} MB.")
        try:
            header = inspect_image(data)
        except ValueError:
            raise PreprocessingError(ErrorCode.UNREADABLE, "The file is not a readable image.") from None
        if header["format"] not in cfg.allowed_formats or header["format"] not in FORMATS:
            raise PreprocessingError(ErrorCode.UNSUPPORTED_TYPE, "Only JPEG and PNG images are supported.")
        if header["width"] * header["height"] > cfg.max_pixels:
            raise PreprocessingError(ErrorCode.TOO_MANY_PIXELS, "The image resolution is too large.")
        if min(header["width"], header["height"]) < cfg.min_side:
            raise PreprocessingError(ErrorCode.TOO_SMALL, f"The image must be at least {cfg.min_side} px on each side.")
        return FORMATS[header["format"]]

    @staticmethod
    def _safe(name: str, call, *args) -> dict:
        """Run one LLM call. A failure becomes {"status": "failed"} so the other call's result is kept."""
        try:
            return {"status": "ok", **call(*args)}
        except LLMRecognitionError as exc:
            return {"status": "failed", "error": {"message": str(exc)}}
        except Exception:
            log.exception("LLM only %s failed", name)
            return {"status": "failed", "error": {"message": f"{name} failed. Try again."}}

    @staticmethod
    def _transcribe(llm, image: np.ndarray, model: str, api_key: str) -> dict:
        started = perf_counter()
        texts, usage = llm.transcribe_page(image, model=model, api_key=api_key)
        return {
            "model": model,
            "lines": [{"line": index, "text": text} for index, text in enumerate(texts)],
            "text": "\n".join(texts),
            "usage": usage,
            "request_count": 1,
            "processing_ms": round((perf_counter() - started) * 1000),
        }

    @staticmethod
    def _spelling(llm, image: np.ndarray, model: str, api_key: str, width: int, height: int) -> dict:
        started = perf_counter()
        found, word_count, usage = llm.check_spelling_with_boxes(image, model=model, api_key=api_key)
        mistakes = []
        for number, mistake in enumerate(found, start=1):
            ymin, xmin, ymax, xmax = mistake["box_2d"]  # 0-1000 of the image, so the photo size doesn't matter
            x0, y0 = round(xmin * width / 1000), round(ymin * height / 1000)
            x1, y1 = round(xmax * width / 1000), round(ymax * height / 1000)
            mistakes.append({
                "box_number": number,
                "wrong_spelling": mistake["wrong_spelling"],
                "correct_spelling": mistake["correct_spelling"],
                "bbox": [x0, y0, x1, y1],
                "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            })
        return {
            "model": model,
            "mistakes": mistakes,
            "word_count": word_count,
            "usage": usage,
            "request_count": 1,
            "processing_ms": round((perf_counter() - started) * 1000),
        }
