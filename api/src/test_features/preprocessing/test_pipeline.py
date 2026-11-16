import json
import os
import tempfile
import time
import unittest

import cv2
import numpy as np

from api.src.features.preprocessing import (
    ErrorCode,
    PreprocessingConfig,
    PreprocessingError,
    PreprocessingPipeline,
    Status,
    WarningCode,
)
from api.src.services.storage import LocalFileStorage
from api.src.test_services.sample_images import png_bytes, text_image


def jpeg_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = LocalFileStorage(self._tmp.name)
        self.pipeline = PreprocessingPipeline(self.storage, PreprocessingConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def files(self, image_id):
        return [k.rsplit("/", 1)[1] for k in self.storage.list(f"preprocessing/{image_id}/")]

    def assert_error(self, code, data):
        with self.assertRaises(PreprocessingError) as ctx:
            self.pipeline.submit(data)
        self.assertEqual(ctx.exception.code, code)

    # ---- happy path ---------------------------------------------------------

    def test_clean_png_is_ready(self):
        result = self.pipeline.submit(png_bytes(text_image()), filename="note.png")
        self.assertEqual(result.status, Status.READY)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.image.ndim, 3)
        self.assertEqual(sorted(self.files(result.image_id)), ["manifest.json", "original.png", "processed.png"])

        manifest = json.loads(self.storage.get(f"preprocessing/{result.image_id}/manifest.json"))
        self.assertEqual(manifest["status"], Status.READY)
        self.assertEqual(manifest["source"]["filename"], "note.png")
        self.assertEqual(manifest["source"]["content_type"], "image/png")
        self.assertEqual(len(manifest["source"]["sha256"]), 64)
        self.assertEqual(manifest["processed"]["scale"], 1.0)

    def test_clean_jpeg_is_accepted(self):
        result = self.pipeline.submit(jpeg_bytes(text_image()))
        self.assertEqual(result.status, Status.READY)
        self.assertIn("original.jpg", self.files(result.image_id))

    def test_load_processed(self):
        result = self.pipeline.submit(png_bytes(text_image()))
        self.assertTrue(np.array_equal(self.pipeline.load_processed(result.image_id), result.image))

    # ---- quality warnings: retake or continue -------------------------------

    def test_blurry_image_needs_review(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        result = self.pipeline.submit(png_bytes(blurry))
        self.assertEqual(result.status, Status.NEEDS_REVIEW)
        self.assertIn(WarningCode.BLURRY, [w["code"] for w in result.warnings])
        self.assertIsNone(result.image)
        self.assertEqual(sorted(self.files(result.image_id)), ["manifest.json", "original.png"])

    def test_dark_image_warns(self):
        dark = (text_image() * 0.15).astype(np.uint8)
        codes = [w["code"] for w in self.pipeline.submit(png_bytes(dark)).warnings]
        self.assertIn(WarningCode.TOO_DARK, codes)

    def test_confirm_processes_after_warning(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        image_id = self.pipeline.submit(png_bytes(blurry)).image_id
        result = self.pipeline.confirm(image_id)
        self.assertEqual(result.status, Status.READY)
        self.assertTrue(result.manifest["accepted_warnings"])
        self.assertIn("processed.png", self.files(image_id))

    def test_confirm_twice_is_invalid(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        image_id = self.pipeline.submit(png_bytes(blurry)).image_id
        self.pipeline.confirm(image_id)
        with self.assertRaises(PreprocessingError) as ctx:
            self.pipeline.confirm(image_id)
        self.assertEqual(ctx.exception.code, ErrorCode.INVALID_STATE)

    def test_retake_deletes_everything(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        image_id = self.pipeline.submit(png_bytes(blurry)).image_id
        self.pipeline.retake(image_id)
        self.assertEqual(self.files(image_id), [])
        with self.assertRaises(PreprocessingError) as ctx:
            self.pipeline.get_manifest(image_id)
        self.assertEqual(ctx.exception.code, ErrorCode.NOT_FOUND)

    # ---- rejections ---------------------------------------------------------

    def test_empty_rejected(self):
        self.assert_error(ErrorCode.EMPTY, b"")

    def test_too_large_rejected(self):
        self.pipeline = PreprocessingPipeline(self.storage, PreprocessingConfig(max_bytes=1000))
        self.assert_error(ErrorCode.TOO_LARGE, png_bytes(text_image()) + b"\0" * 1000)

    def test_unreadable_rejected(self):
        self.assert_error(ErrorCode.UNREADABLE, b"not an image at all")

    def test_unsupported_type_rejected(self):
        ok, bmp = cv2.imencode(".bmp", text_image())
        self.assert_error(ErrorCode.UNSUPPORTED_TYPE, bmp.tobytes())

    def test_too_many_pixels_rejected(self):
        self.pipeline = PreprocessingPipeline(self.storage, PreprocessingConfig(max_pixels=10_000))
        self.assert_error(ErrorCode.TOO_MANY_PIXELS, png_bytes(text_image()))

    def test_tiny_image_rejected(self):
        self.assert_error(ErrorCode.TOO_SMALL, png_bytes(text_image(width=50, height=30)))

    def test_rejected_upload_stores_nothing(self):
        with self.assertRaises(PreprocessingError):
            self.pipeline.submit(b"garbage")
        self.assertEqual(self.storage.list(), [])

    def test_bad_image_id_not_found(self):
        for image_id in ("", "../etc", "f" * 32):
            with self.subTest(image_id=image_id), self.assertRaises(PreprocessingError) as ctx:
                self.pipeline.get_manifest(image_id)
            self.assertEqual(ctx.exception.code, ErrorCode.NOT_FOUND)

    # ---- retention ----------------------------------------------------------

    def test_cleanup_removes_only_expired(self):
        old_id = self.pipeline.submit(png_bytes(text_image())).image_id
        new_id = self.pipeline.submit(png_bytes(text_image())).image_id
        two_days_ago = time.time() - 48 * 3600
        for key in self.storage.list(f"preprocessing/{old_id}/"):
            path = os.path.join(self._tmp.name, *key.split("/"))
            os.utime(path, (two_days_ago, two_days_ago))

        self.assertEqual(self.pipeline.cleanup(), 1)
        self.assertEqual(self.files(old_id), [])
        self.assertNotEqual(self.files(new_id), [])


if __name__ == "__main__":
    unittest.main()
