"""Preprocessing workflow: get an uploaded image ready for text detection.

submit(bytes)
  1. validate   size, type (JPEG/PNG from file header), pixel count, minimum side
  2. identify   image_id + sha256
  3. normalize  decode upright (EXIF orientation) to BGR
  4. quality    blur / brightness / contrast -> warnings
  5. store      original + manifest
  6. if no warnings: enhance + store processed  -> READY
     else                                       -> NEEDS_REVIEW (user picks retake or confirm)
confirm(image_id)  user chose "continue anyway" -> enhance + store processed -> READY
retake(image_id)   user chose "retake"          -> delete everything for that image
cleanup()          delete images older than retention_hours
"""

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import cv2
import numpy as np

from api.src.services import quality_service
from api.src.services.image_loader import inspect_image, load_image, load_image_upright
from api.src.services.preprocess_service import preprocess
from api.src.services.storage import Storage, get_storage

from .config import FORMATS, KEY_PREFIX, PIPELINE_VERSION, PreprocessingConfig
from .schemas import ErrorCode, PreprocessingError, PreprocessResult, Status, WarningCode

_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PreprocessingPipeline:
    def __init__(self, storage: Storage | None = None, config: PreprocessingConfig | None = None):
        self.storage = storage or get_storage()
        self.config = config or PreprocessingConfig.from_env()

    # ---- keys -------------------------------------------------------------

    def _prefix(self, image_id: str) -> str:
        if not _IMAGE_ID.match(image_id or ""):
            raise PreprocessingError(ErrorCode.NOT_FOUND, "unknown image id")
        return f"{KEY_PREFIX}/{image_id}/"

    def _manifest_key(self, image_id: str) -> str:
        return self._prefix(image_id) + "manifest.json"

    # ---- public API -------------------------------------------------------

    def submit(self, data: bytes, filename: str | None = None) -> PreprocessResult:
        cfg = self.config

        # 1. validate
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

        # 2. identify
        image_id = uuid.uuid4().hex
        ext, content_type = FORMATS[header["format"]]

        # 3. normalize
        try:
            image = load_image_upright(data)
        except ValueError:
            raise PreprocessingError(ErrorCode.UNREADABLE, "The file is not a readable image.") from None

        # 4. quality
        quality = quality_service.assess(image)
        warnings = self._warnings(quality)

        # 5. store original + manifest
        original_key = self._prefix(image_id) + f"original.{ext}"
        self.storage.put(original_key, data, content_type)
        now = _now()
        manifest = {
            "image_id": image_id,
            "pipeline_version": PIPELINE_VERSION,
            "status": Status.NEEDS_REVIEW,
            "created_at": now,
            "updated_at": now,
            "source": {
                "key": original_key,
                "filename": filename,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": quality["width"],    # upright (after EXIF rotation)
                "height": quality["height"],
            },
            "quality": {**quality, "warnings": warnings},
            "options": asdict(cfg.options),
            "processed": None,
            "accepted_warnings": False,
        }

        # 6. process now, or wait for the user's choice
        if warnings:
            self._save_manifest(manifest)
            return PreprocessResult(image_id, Status.NEEDS_REVIEW, warnings, manifest, None)
        return self._process(image, manifest)

    def confirm(self, image_id: str) -> PreprocessResult:
        """User chose to continue despite the quality warnings."""
        manifest = self.get_manifest(image_id)
        if manifest["status"] != Status.NEEDS_REVIEW:
            raise PreprocessingError(ErrorCode.INVALID_STATE, f"Image is already {manifest['status']}.")
        image = load_image_upright(self.storage.get(manifest["source"]["key"]))
        manifest["accepted_warnings"] = True
        return self._process(image, manifest)

    def retake(self, image_id: str) -> None:
        """User chose to retake: delete everything stored for this image."""
        self.get_manifest(image_id)  # raises NOT_FOUND if unknown
        for key in self.storage.list(self._prefix(image_id)):
            self.storage.delete(key)

    def get_manifest(self, image_id: str) -> dict:
        try:
            return json.loads(self.storage.get(self._manifest_key(image_id)))
        except FileNotFoundError:
            raise PreprocessingError(ErrorCode.NOT_FOUND, "unknown image id") from None

    def load_processed(self, image_id: str) -> np.ndarray:
        manifest = self.get_manifest(image_id)
        if manifest["status"] != Status.READY:
            raise PreprocessingError(ErrorCode.INVALID_STATE, f"Image is {manifest['status']}, not ready.")
        return load_image(self.storage.get(manifest["processed"]["key"]))

    def cleanup(self, now: float | None = None) -> int:
        """Delete images whose newest file is older than retention_hours. Returns how many."""
        cutoff = (now if now is not None else time.time()) - self.config.retention_hours * 3600
        groups: dict[str, list[str]] = {}
        for key in self.storage.list(f"{KEY_PREFIX}/"):
            groups.setdefault(key.split("/")[1], []).append(key)
        removed = 0
        for keys in groups.values():
            if max(self.storage.modified_at(k) for k in keys) < cutoff:
                for key in keys:
                    self.storage.delete(key)
                removed += 1
        return removed

    # ---- internals --------------------------------------------------------

    def _warnings(self, quality: dict) -> list[dict]:
        cfg = self.config
        warnings = []
        if quality["blur"] < cfg.min_blur:
            warnings.append({"code": WarningCode.BLURRY, "message": "The image looks blurry."})
        if quality["brightness"] < cfg.min_brightness:
            warnings.append({"code": WarningCode.TOO_DARK, "message": "The image is too dark."})
        if quality["contrast"] < cfg.min_contrast:
            warnings.append({"code": WarningCode.LOW_CONTRAST, "message": "The text may be too faint."})
        return warnings

    def _process(self, image: np.ndarray, manifest: dict) -> PreprocessResult:
        processed, info = preprocess(image, self.config.options)
        ok, encoded = cv2.imencode(".png", processed)
        if not ok:
            raise RuntimeError("failed to encode processed image")
        processed_key = self._prefix(manifest["image_id"]) + "processed.png"
        self.storage.put(processed_key, encoded.tobytes(), "image/png")
        manifest["processed"] = {
            "key": processed_key,
            "width": int(processed.shape[1]),
            "height": int(processed.shape[0]),
            "scale": info["scale"],   # original_xy = processed_xy / scale
            "angle": info["angle"],
        }
        manifest["status"] = Status.READY
        manifest["updated_at"] = _now()
        self._save_manifest(manifest)
        return PreprocessResult(manifest["image_id"], Status.READY, manifest["quality"]["warnings"], manifest, processed)

    def _save_manifest(self, manifest: dict) -> None:
        self.storage.put(
            self._manifest_key(manifest["image_id"]),
            json.dumps(manifest, indent=2).encode(),
            "application/json",
        )
