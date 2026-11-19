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

    def test_tall_letters_and_tails_do_not_merge_close_words(self):
        # Cursive-like: long strokes above and below each word make the full ink height ~6x the small letters.
        # The old rule (0.35 x full height = ~45 px) merged these 14 px word gaps; small-letter height keeps them apart.
        image, starts = line_image(["alpha", "beta", "gamma"], gap=14, height=170)
        image = np.roll(image, 25, axis=0)  # move the letters to the middle so strokes fit above and below
        for start in starts:
            cv2.line(image, (start + 6, 15), (start + 6, 160), (20, 20, 20), 3)
        boxes = segment_words(image)
        self.assertEqual(len(boxes), 3)
        for box, start in zip(boxes, starts):
            self.assertLess(abs(box[0] - start), 8)

    def test_slanted_ruled_line_does_not_join_words(self):
        # Ruled paper: a thin line crossing the whole crop at a slight slant inks every column.
        image, starts = line_image(["alpha", "beta", "gamma"], gap=40)
        cv2.line(image, (0, 76), (image.shape[1], 86), (90, 90, 90), 2)
        boxes = segment_words(image)
        self.assertEqual(len(boxes), 3)
        for box, start in zip(boxes, starts):
            self.assertLess(abs(box[0] - start), 8)

    def test_ruled_line_pieces_at_crop_edge_are_not_a_word(self):
        # Short ruled-line pieces left at the edge of a line crop: sparse ink, no letters.
        image, _ = line_image(["alpha", "beta"])
        image = cv2.copyMakeBorder(image, 0, 0, 60, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.line(image, (0, 12), (30, 12), (20, 20, 20), 2)
        cv2.line(image, (0, 92), (25, 92), (20, 20, 20), 2)
        boxes = segment_words(image)
        self.assertEqual(len(boxes), 2)
        self.assertGreater(boxes[0][0], 60)

    def test_grayscale_input(self):
        image, _ = line_image(["alpha", "beta"])
        self.assertEqual(len(segment_words(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))), 2)


if __name__ == "__main__":
    unittest.main()
