import unittest

import cv2
import numpy as np

from api.src.services.quality_service import assess
from api.src.test_services.sample_images import text_image


class TestQualityService(unittest.TestCase):
    def test_returns_all_metrics(self):
        result = assess(text_image(width=720, height=180))
        self.assertEqual(set(result), {"blur", "brightness", "contrast", "width", "height"})
        self.assertEqual((result["width"], result["height"]), (720, 180))

    def test_blurred_image_scores_lower(self):
        sharp = text_image()
        blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
        self.assertLess(assess(blurred)["blur"], assess(sharp)["blur"])

    def test_dark_image_scores_lower_brightness(self):
        image = text_image()
        dark = (image * 0.15).astype(np.uint8)
        self.assertLess(assess(dark)["brightness"], assess(image)["brightness"])

    def test_flat_image_has_zero_contrast(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        self.assertEqual(assess(flat)["contrast"], 0.0)

    def test_accepts_grayscale(self):
        gray = np.full((50, 80), 200, dtype=np.uint8)
        self.assertEqual(assess(gray)["width"], 80)


if __name__ == "__main__":
    unittest.main()
