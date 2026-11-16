import tempfile
import unittest
from unittest import mock

from api.src.features.handwriting_recognition import HandwritingRecognizer, RecognitionConfig
from api.src.features.handwritten_content_detection import (
    DetectionConfig,
    DetectionError,
    DetectionErrorCode,
    HandwrittenContentDetector,
)
from api.src.features.preprocessing import PreprocessingConfig, PreprocessingPipeline
from api.src.services.storage import LocalFileStorage
from api.src.test_services.sample_images import png_bytes
from api.src.test_services.test_word_segmentation_service import line_image

PADDLE = "api.src.services.paddle_service.detect"
TROCR = "api.src.services.trocr_service.recognize"
LINES = [
    {"points": [[5, 10], [890, 10], [890, 50], [5, 50]], "score": 0.9},
    {"points": [[5, 55], [890, 55], [890, 95], [5, 95]], "score": 0.9},
]


class TestHandwritingRecognizer(unittest.TestCase):
    """Fake PaddleOCR (two lines) and fake TrOCR: fast, no models needed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = LocalFileStorage(self._tmp.name)
        lenient = PreprocessingConfig(min_blur=0, min_brightness=0, min_contrast=0)
        self.preprocessing = PreprocessingPipeline(self.storage, lenient)
        self.detector = HandwrittenContentDetector(self.preprocessing, DetectionConfig())
        self.recognizer = HandwritingRecognizer(self.detector, RecognitionConfig(min_confidence=0.5))

    def tearDown(self):
        self._tmp.cleanup()

    def detected_image(self):
        image, _ = line_image(["alpha", "beta"])
        image_id = self.preprocessing.submit(png_bytes(image)).image_id
        with mock.patch(PADDLE, return_value=LINES):
            self.detector.detect(image_id)
        return image_id

    def test_reads_every_line_in_reading_order(self):
        image_id = self.detected_image()
        reads = [{"text": "first line", "confidence": 0.9}, {"text": "second line", "confidence": 0.8}]
        with mock.patch(TROCR, side_effect=reads) as trocr:
            result = self.recognizer.read(image_id)
        self.assertEqual(trocr.call_count, 2)
        self.assertEqual([line["text"] for line in result["lines"]], ["first line", "second line"])
        self.assertEqual([line["line"] for line in result["lines"]], [0, 1])
        self.assertEqual(result["text"], "first line\nsecond line")
        self.assertEqual(result["model"], "microsoft/trocr-small-handwritten")
        self.assertIn("recognized_text.json", [k.rsplit("/", 1)[1] for k in self.storage.list()])

    def test_low_confidence_lines_flagged_and_left_out_of_text(self):
        image_id = self.detected_image()
        reads = [{"text": "#", "confidence": 0.2}, {"text": "real words", "confidence": 0.9}]
        with mock.patch(TROCR, side_effect=reads):
            result = self.recognizer.read(image_id)
        self.assertEqual([line["low_confidence"] for line in result["lines"]], [True, False])
        self.assertEqual(result["text"], "real words")

    def test_symbol_only_text_is_unclear_even_with_high_confidence(self):
        image_id = self.detected_image()
        reads = [{"text": "#", "confidence": 0.9}, {"text": "box", "confidence": 0.9}]
        with mock.patch(TROCR, side_effect=reads):
            result = self.recognizer.read(image_id)
        self.assertEqual([line["low_confidence"] for line in result["lines"]], [True, False])
        self.assertEqual(result["text"], "box")

    def test_get_result(self):
        image_id = self.detected_image()
        self.assertIsNone(self.recognizer.get_result(image_id))
        with mock.patch(TROCR, return_value={"text": "x", "confidence": 0.9}):
            result = self.recognizer.read(image_id)
        self.assertEqual(self.recognizer.get_result(image_id), result)

    def test_needs_detection_first(self):
        image, _ = line_image(["alpha"])
        image_id = self.preprocessing.submit(png_bytes(image)).image_id
        with self.assertRaises(DetectionError) as ctx:
            self.recognizer.read(image_id)
        self.assertEqual(ctx.exception.code, DetectionErrorCode.NOT_FOUND)

    def test_retake_removes_text(self):
        image_id = self.detected_image()
        with mock.patch(TROCR, return_value={"text": "x", "confidence": 0.9}):
            self.recognizer.read(image_id)
        self.preprocessing.retake(image_id)
        self.assertEqual(self.storage.list(f"preprocessing/{image_id}/"), [])


if __name__ == "__main__":
    unittest.main()
