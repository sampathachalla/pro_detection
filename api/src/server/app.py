"""HTTP API: web app upload -> preprocessing -> handwritten content detection.

Run (inside the container):
    uvicorn api.src.server.app:create_app --factory --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from api.src.features.handwritten_content_detection import (
    DetectionError,
    DetectionErrorCode,
    HandwrittenContentDetector,
)
from api.src.features.preprocessing import ErrorCode, PreprocessingError, PreprocessingPipeline, Status
from api.src.features.llm_handwritten_recognition import LLMHandwritingRecognizer
from api.src.features.spelling_check import SpellingChecker
from api.src.features.word_box_draw import WordBoxDrawer
from api.src.services import paddle_service
from api.src.services.openai_LLM_call import OpenAIRecognitionError

log = logging.getLogger("uvicorn.error")  # uvicorn's logger, so INFO lines appear in `docker logs`

HTTP_STATUS = {
    ErrorCode.EMPTY: 400,
    ErrorCode.UNREADABLE: 400,
    ErrorCode.TOO_SMALL: 400,
    ErrorCode.TOO_LARGE: 413,
    ErrorCode.TOO_MANY_PIXELS: 413,
    ErrorCode.UNSUPPORTED_TYPE: 415,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.INVALID_STATE: 409,
    DetectionErrorCode.NOT_FOUND: 404,
    DetectionErrorCode.NOT_READY: 409,
}
MULTIPART_OVERHEAD = 64 * 1024  # form boundaries and headers around the file


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=HTTP_STATUS.get(code, 400))


def _done(result: dict, warnings: list[dict], words: dict | None = None) -> dict:
    """Response for an image whose handwriting has been detected and drawn (lines and words)."""
    image_id = result["image_id"]
    return {
        "image_id": result["image_id"],
        "status": "done",
        "warnings": warnings,
        "detector": result["detector"],
        "count": result["count"],
        "image": result["image"],
        "regions": result["regions"],
        "annotated_url": f"/api/images/{image_id}/annotated.jpg",
        "word_count": words["count"] if words else None,
        "words": words["words"] if words else [],
        "words_annotated_url": f"/api/images/{image_id}/words.jpg" if words else None,
    }


def _needs_review(image_id: str, warnings: list[dict]) -> dict:
    return {"image_id": image_id, "status": "needs_review", "warnings": warnings}


class _Timer:
    """Times the server-side stages of a request for logs and the Server-Timing header."""

    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.stages: list[tuple[str, float]] = []

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages.append((stage, (now - self.last) * 1000))
        self.last = now

    def respond(self, body: dict, image_id: str) -> JSONResponse:
        total = (time.perf_counter() - self.start) * 1000
        parts = [f"{name};dur={ms:.0f}" for name, ms in self.stages] + [f"total;dur={total:.0f}"]
        log.info("image %s %s in %.0f ms (%s)", image_id, body["status"], total,
                 ", ".join(f"{name} {ms:.0f} ms" for name, ms in self.stages))
        body = {**body, "timings_ms": {**{name: round(ms) for name, ms in self.stages}, "total": round(total)}}
        return JSONResponse(body, headers={"Server-Timing": ", ".join(parts)})


def create_app(
    preprocessing: PreprocessingPipeline | None = None,
    detector: HandwrittenContentDetector | None = None,
    word_drawer: WordBoxDrawer | None = None,
    warm_up: bool = True,
    cleanup_interval_seconds: float = 3600,
) -> FastAPI:
    preprocessing = preprocessing or PreprocessingPipeline()
    detector = detector or HandwrittenContentDetector(preprocessing)
    word_drawer = word_drawer or WordBoxDrawer(detector)
    llm_recognizer = LLMHandwritingRecognizer(detector)
    spelling_checker = SpellingChecker(word_drawer)
    max_bytes = preprocessing.config.max_bytes
    # One image at a time: the container has 1 CPU and the model predictors are not thread-safe.
    work_lock = threading.Lock()

    def run_llm(image_id: str) -> dict:
        """Run transcription and the spelling check in parallel. A failure in one doesn't stop the other."""
        def safe(name: str, call) -> dict:
            try:
                return {"status": "ok", **call(image_id)}
            except (OpenAIRecognitionError, DetectionError, PreprocessingError) as exc:
                return {"status": "failed", "error": {"message": str(exc)}}
            except Exception:
                log.exception("%s failed", name)
                return {"status": "failed", "error": {"message": f"{name} failed. Try again."}}

        with ThreadPoolExecutor(max_workers=2) as pool:
            transcription = pool.submit(safe, "Transcription", llm_recognizer.read)
            spelling = pool.submit(safe, "Spelling check", spelling_checker.check)
            return {"transcription": transcription.result(), "spelling": spelling.result()}

    async def cleanup_loop():
        while True:
            try:
                removed = await asyncio.to_thread(preprocessing.cleanup)
                if removed:
                    log.info("cleanup removed %d expired image(s)", removed)
            except Exception:
                log.exception("cleanup failed")
            await asyncio.sleep(cleanup_interval_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:  # load PP-OCRv6 now so the first upload isn't slow
            await asyncio.to_thread(paddle_service.get_detector)
        task = asyncio.create_task(cleanup_loop()) if cleanup_interval_seconds else None
        yield
        if task:
            task.cancel()

    app = FastAPI(
        title="Pro Detection API",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.exception_handler(PreprocessingError)
    async def preprocessing_error(request: Request, exc: PreprocessingError):
        return _error(exc.code, exc.message)

    @app.exception_handler(DetectionError)
    async def detection_error(request: Request, exc: DetectionError):
        return _error(exc.code, exc.message)

    @app.exception_handler(OpenAIRecognitionError)
    async def openai_recognition_error(request: Request, exc: OpenAIRecognitionError):
        return JSONResponse(
            {"error": {"code": "openai_recognition", "message": str(exc)}},
            status_code=exc.status_code,
        )

    # Sync handlers run in FastAPI's thread pool, so model work doesn't block the event loop.

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/images")
    def upload(request: Request, file: UploadFile = File(...)):
        """Upload an image. Returns "done" with OCR + OpenAI results, or "needs_review" with warnings."""
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > max_bytes + MULTIPART_OVERHEAD:
            return _error(ErrorCode.TOO_LARGE, f"The file must be smaller than {max_bytes // (1024 * 1024)} MB.")
        timer = _Timer()
        data = file.file.read(max_bytes)  # never more than the limit; the pipeline rejects >= max_bytes
        filename = (file.filename or "")[:255] or None
        with work_lock:
            timer.mark("wait")  # time queued behind another image
            pre = preprocessing.submit(data, filename=filename)
            timer.mark("preprocess")
            if pre.status == Status.NEEDS_REVIEW:
                return timer.respond(_needs_review(pre.image_id, pre.warnings), pre.image_id)
            detection = detector.detect(pre.image_id)
            timer.mark("detect")
            words = word_drawer.draw(pre.image_id)
            timer.mark("words")
        # OpenAI calls run outside the lock: they wait on the network, not the CPU.
        llm = run_llm(pre.image_id)
        timer.mark("llm")
        return timer.respond({**_done(detection.result, pre.warnings, words.result), "llm": llm}, pre.image_id)

    @app.post("/api/images/{image_id}/confirm")
    def confirm(image_id: str):
        """User chose "continue anyway" after quality warnings."""
        timer = _Timer()
        with work_lock:
            timer.mark("wait")
            pre = preprocessing.confirm(image_id)
            timer.mark("preprocess")
            detection = detector.detect(image_id)
            timer.mark("detect")
            words = word_drawer.draw(image_id)
            timer.mark("words")
        llm = run_llm(image_id)
        timer.mark("llm")
        return timer.respond({**_done(detection.result, pre.warnings, words.result), "llm": llm}, image_id)

    @app.delete("/api/images/{image_id}", status_code=204)
    def retake(image_id: str):
        """User chose "retake": delete everything stored for this image."""
        preprocessing.retake(image_id)
        return Response(status_code=204)

    @app.get("/api/images/{image_id}")
    def get_image(image_id: str):
        manifest = preprocessing.get_manifest(image_id)
        warnings = manifest["quality"]["warnings"]
        if manifest["status"] == Status.NEEDS_REVIEW:
            return _needs_review(image_id, warnings)
        body = _done(detector.get_result(image_id), warnings, word_drawer.get_result(image_id))
        # Saved OpenAI results, so a page reload shows the text and mistakes again.
        saved = {"transcription": llm_recognizer.get_result(image_id), "spelling": spelling_checker.get_result(image_id)}
        llm = {name: {"status": "ok", **result} for name, result in saved.items() if result is not None}
        return {**body, "llm": llm} if llm else body

    @app.get("/api/images/{image_id}/words.jpg")
    def words_image(
        image_id: str, word_boxes: bool | None = None, line_boxes: bool | None = None, numbers: bool | None = None,
        spelling: bool = False,
        crop_to_text: bool = False,
    ):
        words = word_drawer.get_result(image_id)
        if words is None:
            raise DetectionError(DetectionErrorCode.NOT_FOUND, "Words have not been drawn for this image.")
        if word_boxes is None and line_boxes is None and numbers is None and not spelling and not crop_to_text:
            data = preprocessing.storage.get(words["annotated_key"])
        else:
            data = word_drawer.preview(
                image_id,
                words["words"],
                word_boxes=word_boxes is not False,
                line_boxes=word_drawer.config.show_lines if line_boxes is None else line_boxes,
                numbers=numbers is not False,
                spelling_boxes=spelling_checker.spelling_boxes(image_id) if spelling else set(),
                crop_to_text=crop_to_text,
            )
        cache = "no-store" if spelling else "private, max-age=3600"
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": cache})

    @app.get("/api/images/{image_id}/annotated.jpg")
    def annotated(image_id: str):
        result = detector.get_result(image_id)
        key = result["annotated_key"]
        media_type = "image/png" if key.endswith(".png") else "image/jpeg"  # older results were PNG
        data = preprocessing.storage.get(key)
        return Response(data, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})

    return app
