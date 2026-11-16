"""HTTP API: web app upload -> preprocessing -> handwritten content detection.

Run (inside the container):
    uvicorn api.src.server.app:create_app --factory --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from api.src.features.handwritten_content_detection import (
    DetectionError,
    DetectionErrorCode,
    HandwrittenContentDetector,
)
from api.src.features.preprocessing import ErrorCode, PreprocessingError, PreprocessingPipeline, Status
from api.src.features.handwriting_recognition import HandwritingRecognizer
from api.src.features.word_box_draw import WordBoxDrawer
from api.src.services import paddle_service

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


def _done(result: dict, warnings: list[dict], words: dict | None = None, recognition: dict | None = None) -> dict:
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
        "recognition": recognition,  # TrOCR text per line once POST /read has run, else None
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
        return JSONResponse(body, headers={"Server-Timing": ", ".join(parts)})


def create_app(
    preprocessing: PreprocessingPipeline | None = None,
    detector: HandwrittenContentDetector | None = None,
    word_drawer: WordBoxDrawer | None = None,
    recognizer: HandwritingRecognizer | None = None,
    warm_up: bool = True,
    cleanup_interval_seconds: float = 3600,
) -> FastAPI:
    preprocessing = preprocessing or PreprocessingPipeline()
    detector = detector or HandwrittenContentDetector(preprocessing)
    word_drawer = word_drawer or WordBoxDrawer(detector)
    recognizer = recognizer or HandwritingRecognizer(detector)
    max_bytes = preprocessing.config.max_bytes
    # One image at a time: the container has 1 CPU and the model predictors are not thread-safe.
    work_lock = threading.Lock()

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

    # Sync handlers run in FastAPI's thread pool, so model work doesn't block the event loop.

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/images")
    def upload(request: Request, file: UploadFile = File(...)):
        """Upload an image. Returns "done" with the drawn result, or "needs_review" with warnings."""
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
        return timer.respond(_done(detection.result, pre.warnings, words.result), pre.image_id)

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
        return timer.respond(_done(detection.result, pre.warnings, words.result), image_id)

    @app.post("/api/images/{image_id}/read")
    def read_text(image_id: str):
        """Read the handwriting in every detected line with TrOCR (~1 s per line)."""
        timer = _Timer()
        with work_lock:
            timer.mark("wait")
            recognition = recognizer.read(image_id)
            timer.mark("read")
        return timer.respond({"image_id": image_id, "status": "read", **recognition}, image_id)

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
        return _done(
            detector.get_result(image_id), warnings, word_drawer.get_result(image_id), recognizer.get_result(image_id)
        )

    @app.get("/api/images/{image_id}/words.jpg")
    def words_image(image_id: str):
        words = word_drawer.get_result(image_id)
        if words is None:
            raise DetectionError(DetectionErrorCode.NOT_FOUND, "Words have not been drawn for this image.")
        data = preprocessing.storage.get(words["annotated_key"])
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/images/{image_id}/annotated.jpg")
    def annotated(image_id: str):
        result = detector.get_result(image_id)
        key = result["annotated_key"]
        media_type = "image/png" if key.endswith(".png") else "image/jpeg"  # older results were PNG
        data = preprocessing.storage.get(key)
        return Response(data, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})

    return app
