import tempfile
import unittest
from unittest import mock

import cv2
from fastapi.testclient import TestClient

from api.src.features.handwritten_content_detection import DetectionConfig, HandwrittenContentDetector
from api.src.features.preprocessing import PreprocessingConfig, PreprocessingPipeline
from api.src.server.app import create_app
from api.src.services.storage import LocalFileStorage
from api.src.test_services.sample_images import png_bytes, text_image

PADDLE = "api.src.services.paddle_service.detect"
BOX = {"points": [[20, 60], [300, 60], [300, 130], [20, 130]], "score": 0.9}


class TestServer(unittest.TestCase):
    """Fake PaddleOCR (fast); real preprocessing, drawing and storage in a temp folder."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        storage = LocalFileStorage(self._tmp.name)
        self.preprocessing = PreprocessingPipeline(storage, PreprocessingConfig(max_bytes=2 * 1024 * 1024))
        detector = HandwrittenContentDetector(self.preprocessing, DetectionConfig())
        app = create_app(self.preprocessing, detector, warm_up=False, cleanup_interval_seconds=0)
        self.client = TestClient(app)
        self.client.__enter__()
        patcher = mock.patch(PADDLE, return_value=[BOX])
        self.paddle = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self._tmp.cleanup()

    def upload(self, data: bytes, name="page.png", content_type="image/png"):
        return self.client.post("/api/images", files={"file": (name, data, content_type)})

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})

    def test_upload_clean_image_is_done(self):
        response = self.upload(png_bytes(text_image()))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "done")
        self.assertEqual(body["detector"], "paddle")
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["regions"][0]["bbox"], [20, 60, 300, 130])
        self.assertEqual(body["image"], {"width": 720, "height": 180})
        self.assertEqual(body["annotated_url"], f"/api/images/{body['image_id']}/annotated.jpg")
        self.assertEqual(body["words_annotated_url"], f"/api/images/{body['image_id']}/words.jpg")
        self.assertEqual(body["word_count"], len(body["words"]))
        self.assertGreater(body["word_count"], 0)
        self.assertTrue(all(w["line"] == 0 for w in body["words"]))

    def test_annotated_jpeg_is_served(self):
        body = self.upload(png_bytes(text_image())).json()
        response = self.client.get(body["annotated_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertTrue(response.content.startswith(b"\xff\xd8\xff"))

    def test_words_jpeg_is_served(self):
        body = self.upload(png_bytes(text_image())).json()
        response = self.client.get(body["words_annotated_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_server_timing_header(self):
        response = self.upload(png_bytes(text_image()))
        timing = response.headers["server-timing"]
        for stage in ("preprocess", "detect", "words", "total"):
            self.assertIn(f"{stage};dur=", timing)

    def test_get_image_returns_same_result(self):
        body = self.upload(png_bytes(text_image())).json()
        self.assertEqual(self.client.get(f"/api/images/{body['image_id']}").json(), body)

    def test_blurry_upload_needs_review_then_confirm(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        body = self.upload(png_bytes(blurry)).json()
        self.assertEqual(body["status"], "needs_review")
        self.assertIn("blurry", [w["code"] for w in body["warnings"]])
        self.paddle.assert_not_called()

        confirmed = self.client.post(f"/api/images/{body['image_id']}/confirm")
        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(confirmed.json()["status"], "done")
        self.assertIn("blurry", [w["code"] for w in confirmed.json()["warnings"]])

    def test_retake_deletes(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        image_id = self.upload(png_bytes(blurry)).json()["image_id"]
        self.assertEqual(self.client.delete(f"/api/images/{image_id}").status_code, 204)
        response = self.client.get(f"/api/images/{image_id}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "not_found")

    def test_confirm_twice_conflicts(self):
        image_id = self.upload(png_bytes(text_image())).json()["image_id"]
        response = self.client.post(f"/api/images/{image_id}/confirm")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "invalid_state")

    def test_rejections_map_to_http_status(self):
        ok, bmp = cv2.imencode(".bmp", text_image())
        cases = [
            (b"", 400, "empty"),
            (b"not an image", 400, "unreadable"),
            (bmp.tobytes(), 415, "unsupported_type"),
            (png_bytes(text_image(width=40, height=30)), 400, "too_small"),
            (png_bytes(text_image()) + b"\0" * (2 * 1024 * 1024), 413, "too_large"),
        ]
        for data, status, code in cases:
            with self.subTest(code=code):
                response = self.upload(data)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], code)
                self.assertTrue(response.json()["error"]["message"])

    def test_missing_file_field_is_422(self):
        self.assertEqual(self.client.post("/api/images").status_code, 422)

    def test_unknown_or_unsafe_ids_are_404(self):
        for image_id in ("f" * 32, "..%2F..%2Fetc", "abc"):
            with self.subTest(image_id=image_id):
                self.assertEqual(self.client.get(f"/api/images/{image_id}").status_code, 404)
                self.assertEqual(self.client.get(f"/api/images/{image_id}/annotated.jpg").status_code, 404)
                self.assertEqual(self.client.get(f"/api/images/{image_id}/words.jpg").status_code, 404)


if __name__ == "__main__":
    unittest.main()
