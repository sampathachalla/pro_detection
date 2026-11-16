import unittest

from api.src.services import paddle_service
from api.src.services.preprocess_service import preprocess
from api.src.test_services.sample_images import text_image


class TestPaddleService(unittest.TestCase):
    """Needs the baked-in PP-OCRv6_medium_det weights (run inside the container)."""

    def assert_valid_boxes(self, boxes):
        self.assertIsInstance(boxes, list)
        self.assertGreater(len(boxes), 0, "no text detected")
        for box in boxes:
            self.assertEqual(set(box), {"points", "score"})
            self.assertEqual(len(box["points"]), 4)
            for x, y in box["points"]:
                self.assertIsInstance(x, int)
                self.assertIsInstance(y, int)
            self.assertIsInstance(box["score"], float)
            self.assertGreaterEqual(box["score"], 0.0)
            self.assertLessEqual(box["score"], 1.0)

    def test_detects_text(self):
        self.assert_valid_boxes(paddle_service.detect(text_image()))

    def test_detects_text_after_preprocess(self):
        clean, _ = preprocess(text_image())
        self.assert_valid_boxes(paddle_service.detect(clean))

    def test_detector_is_loaded_once(self):
        self.assertIs(paddle_service.get_detector(), paddle_service.get_detector())


if __name__ == "__main__":
    unittest.main()
