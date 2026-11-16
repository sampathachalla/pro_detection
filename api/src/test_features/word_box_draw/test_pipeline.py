import tempfile
import unittest
from unittest import mock

import numpy as np

from api.src.features.handwritten_content_detection import (
    DetectionConfig,
    DetectionError,
    DetectionErrorCode,
    HandwrittenContentDetector,
)
from api.src.features.preprocessing import PreprocessingConfig, PreprocessingPipeline
from api.src.features.word_box_draw import WordBoxConfig, WordBoxDrawer, crop_line
from api.src.services.storage import LocalFileStorage
from api.src.test_services.sample_images import png_bytes
from api.src.test_services.test_word_segmentation_service import line_image

PADDLE = "api.src.services.paddle_service.detect"


class TestWordBoxDrawer(unittest.TestCase):
    """Fake PaddleOCR returns one line box around three drawn words."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = LocalFileStorage(self._tmp.name)
        lenient = PreprocessingConfig(min_blur=0, min_brightness=0, min_contrast=0)
        self.preprocessing = PreprocessingPipeline(self.storage, lenient)
        self.detector = HandwrittenContentDetector(self.preprocessing, DetectionConfig())
        self.drawer = WordBoxDrawer(self.detector, WordBoxConfig())
        self.image, self.starts = line_image(["alpha", "beta", "gamma"])

    def tearDown(self):
        self._tmp.cleanup()

    def detected_image(self, line_box):
        image_id = self.preprocessing.submit(png_bytes(self.image)).image_id
        with mock.patch(PADDLE, return_value=[{"points": line_box, "score": 0.9}]):
            self.detector.detect(image_id)
        return image_id

    def files(self, image_id):
        return [k.rsplit("/", 1)[1] for k in self.storage.list(f"preprocessing/{image_id}/")]

    def test_draws_one_box_per_word(self):
        image_id = self.detected_image([[5, 15], [890, 15], [890, 95], [5, 95]])
        result = self.drawer.draw(image_id)
        self.assertEqual(result.result["count"], 3)
        self.assertEqual(result.result["lines"], 1)
        self.assertEqual([w["line"] for w in result.words], [0, 0, 0])
        for word, start in zip(result.words, self.starts):
            self.assertLess(abs(word["bbox"][0] - start), 8)  # photo coordinates
        self.assertEqual(result.annotated.shape, self.image.shape)
        blue = np.array(WordBoxConfig().color, np.uint8)
        green = np.array(DetectionConfig().color, np.uint8)
        self.assertTrue(np.all(result.annotated == blue, axis=-1).any(), "word borders")
        self.assertTrue(np.all(result.annotated == green, axis=-1).any(), "line borders underneath")
        self.assertIn("words_annotated.jpg", self.files(image_id))
        self.assertIn("word_boxes.json", self.files(image_id))

    def test_show_lines_off_draws_words_only(self):
        image_id = self.detected_image([[5, 15], [890, 15], [890, 95], [5, 95]])
        drawer = WordBoxDrawer(self.detector, WordBoxConfig(show_lines=False))
        annotated = drawer.draw(image_id).annotated
        green = np.array(DetectionConfig().color, np.uint8)
        self.assertFalse(np.all(annotated == green, axis=-1).any())

    def test_tilted_line_box_still_splits_words(self):
        # A slanted polygon leaves corners outside the line; they must not count as ink.
        image_id = self.detected_image([[5, 5], [890, 25], [890, 99], [5, 85]])
        self.assertEqual(self.drawer.draw(image_id).result["count"], 3)

    def test_get_result_and_load_annotated(self):
        image_id = self.detected_image([[5, 15], [890, 15], [890, 95], [5, 95]])
        result = self.drawer.draw(image_id)
        self.assertEqual(self.drawer.get_result(image_id), result.result)
        self.assertEqual(self.drawer.load_annotated(image_id).shape, self.image.shape)

    def test_get_result_before_draw_is_none(self):
        image_id = self.detected_image([[5, 15], [890, 15], [890, 95], [5, 95]])
        self.assertIsNone(self.drawer.get_result(image_id))

    def test_needs_detection_first(self):
        image_id = self.preprocessing.submit(png_bytes(self.image)).image_id
        with self.assertRaises(DetectionError) as ctx:
            self.drawer.draw(image_id)
        self.assertEqual(ctx.exception.code, DetectionErrorCode.NOT_FOUND)

    def test_retake_removes_word_files(self):
        image_id = self.detected_image([[5, 15], [890, 15], [890, 95], [5, 95]])
        self.drawer.draw(image_id)
        self.preprocessing.retake(image_id)
        self.assertEqual(self.files(image_id), [])


class TestCropLine(unittest.TestCase):
    def test_outside_polygon_gets_paper_colour(self):
        image = np.full((100, 200, 3), 180, np.uint8)
        crop, dx, dy = crop_line(image, [[10, 10], [190, 30], [190, 90], [10, 70]])
        self.assertEqual((dx, dy), (10, 10))
        self.assertEqual(crop.shape, (81, 181, 3))
        self.assertTrue(np.all(crop == 180), "corners filled with the paper colour, not white")


if __name__ == "__main__":
    unittest.main()
