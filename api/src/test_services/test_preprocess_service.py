import unittest

import cv2
import numpy as np

from api.src.services.preprocess_service import (
    PreprocessOptions,
    deskew,
    preprocess,
    resize,
    to_gray,
)
from api.src.test_services.sample_images import text_image


class TestPreprocessService(unittest.TestCase):
    def test_output_is_bgr_uint8_same_size(self):
        image = text_image()
        out, info = preprocess(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(info, {"scale": 1.0, "angle": 0.0})

    def test_large_image_is_downscaled(self):
        image = text_image(width=3200, height=800)
        out, info = preprocess(image, PreprocessOptions(max_side=1600, denoise=False))
        self.assertEqual(max(out.shape[:2]), 1600)
        self.assertAlmostEqual(info["scale"], 0.5)

    def test_small_image_is_not_upscaled(self):
        image = text_image(width=400, height=100)
        resized, scale = resize(image, 1600)
        self.assertEqual(resized.shape, image.shape)
        self.assertEqual(scale, 1.0)

    def test_steps_can_be_disabled(self):
        image = text_image()
        out, _ = preprocess(image, PreprocessOptions(denoise=False, clahe=False))
        expected = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        self.assertTrue(np.array_equal(out, expected))

    def test_to_gray_accepts_gray_and_bgr(self):
        gray = to_gray(text_image())
        self.assertEqual(gray.ndim, 2)
        self.assertIs(to_gray(gray), gray)

    def test_deskew_detects_rotation(self):
        gray = to_gray(text_image(width=900, height=400))
        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
        rotated = cv2.warpAffine(gray, matrix, (w, h), borderValue=255)
        _, angle = deskew(rotated)
        self.assertNotEqual(angle, 0.0)

    def test_deskew_blank_image_is_unchanged(self):
        blank = np.full((100, 100), 255, dtype=np.uint8)
        out, angle = deskew(blank)
        self.assertEqual(angle, 0.0)
        self.assertTrue(np.array_equal(out, blank))


if __name__ == "__main__":
    unittest.main()
