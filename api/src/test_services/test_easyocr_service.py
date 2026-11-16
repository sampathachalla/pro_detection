import unittest

from api.src.services import easyocr_service
from api.src.services.preprocess_service import preprocess
from api.src.test_services.sample_images import text_image


class TestEasyOCRService(unittest.TestCase):
    """Needs the baked-in EasyOCR "en" weights (run inside the container)."""

    def assert_valid_boxes(self, boxes):
        self.assertIsInstance(boxes, list)
        self.assertGreater(len(boxes), 0, "no text detected")
        for box in boxes:
            self.assertEqual(set(box), {"points", "score"})
            self.assertEqual(len(box["points"]), 4)
            for x, y in box["points"]:
                self.assertIsInstance(x, int)
                self.assertIsInstance(y, int)
            self.assertIsNone(box["score"])

    def test_detects_text(self):
        self.assert_valid_boxes(easyocr_service.detect(text_image()))

    def test_detects_text_after_preprocess(self):
        clean, _ = preprocess(text_image())
        self.assert_valid_boxes(easyocr_service.detect(clean))

    def test_reader_is_loaded_once(self):
        self.assertIs(easyocr_service.get_reader(), easyocr_service.get_reader())


if __name__ == "__main__":
    unittest.main()
