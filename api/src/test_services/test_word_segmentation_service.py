import unittest

import cv2
import numpy as np

from api.src.services.word_segmentation_service import ink_mask, segment_words


def line_image(words, gap=60, width=900, height=100, background=255):
    """One line: each word drawn separately, `gap` px apart. Returns (image, x-start of each word)."""
    image = np.full((height, width, 3), background, dtype=np.uint8)
    x, starts = 20, []
    for word in words:
        starts.append(x)
        (w, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(image, word, (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
        x += w + gap
    return image, starts


class TestWordSegmentation(unittest.TestCase):
    def test_splits_three_words_left_to_right(self):
        image, starts = line_image(["alpha", "beta", "gamma"])
        boxes = segment_words(image)
        self.assertEqual(len(boxes), 3)
        self.assertEqual([b[0] for b in boxes], sorted(b[0] for b in boxes))
        for box, start in zip(boxes, starts):
            self.assertLess(abs(box[0] - start), 8)

    def test_letters_of_one_word_stay_together(self):
        image, _ = line_image(["handwriting"])
        self.assertEqual(len(segment_words(image)), 1)

    def test_large_gap_ratio_merges_words(self):
        image, _ = line_image(["alpha", "beta"], gap=40)
        self.assertEqual(len(segment_words(image, gap_ratio=5.0)), 1)

    def test_blank_crop_has_no_words(self):
        self.assertEqual(segment_words(np.full((80, 400, 3), 255, np.uint8)), [])

    def test_grey_paper_is_not_ink(self):
        image, _ = line_image(["alpha", "beta", "gamma"], background=190)
        self.assertEqual(len(segment_words(image)), 3)

    def test_specks_are_ignored(self):
        image, _ = line_image(["alpha", "beta"])
        for x in range(300, 900, 37):
            image[10, x] = 0  # single-pixel dust
        self.assertEqual(len(segment_words(image)), 2)
        self.assertEqual(int(ink_mask(image)[10, 300]), 0)

    def test_boxes_inside_crop(self):
        image, _ = line_image(["alpha", "beta"])
        h, w = image.shape[:2]
        for x0, y0, x1, y1 in segment_words(image):
            self.assertTrue(0 <= x0 <= x1 < w and 0 <= y0 <= y1 < h)

    def test_grayscale_input(self):
        image, _ = line_image(["alpha", "beta"])
        self.assertEqual(len(segment_words(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))), 2)


if __name__ == "__main__":
    unittest.main()
