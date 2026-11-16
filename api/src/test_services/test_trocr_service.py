import os
import unittest

import cv2
import numpy as np

from api.src.services import trocr_service

MODEL_DIR = os.environ.get("TROCR_MODEL_DIR", trocr_service.DEFAULT_MODEL_DIR)


def line(text: str) -> np.ndarray:
    image = np.full((80, 60 + 34 * len(text), 3), 255, dtype=np.uint8)
    cv2.putText(image, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    return image


@unittest.skipUnless(os.path.isdir(MODEL_DIR), f"TrOCR weights not found at {MODEL_DIR}")
class TestTrOCRService(unittest.TestCase):
    """Slow (~2 s): uses the baked-in TrOCR-small weights. Run inside the container."""

    def test_reads_a_clear_line(self):
        result = trocr_service.recognize(line("hello world"))
        self.assertEqual(set(result), {"text", "confidence"})
        self.assertIn("hello", result["text"].lower())
        self.assertGreater(result["confidence"], 0.5)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_grayscale_input(self):
        gray = cv2.cvtColor(line("box"), cv2.COLOR_BGR2GRAY)
        self.assertIsInstance(trocr_service.recognize(gray)["text"], str)

    def test_model_loaded_once(self):
        self.assertIs(trocr_service.get_model()[1], trocr_service.get_model()[1])


if __name__ == "__main__":
    unittest.main()
