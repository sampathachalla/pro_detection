import unittest

import numpy as np

from api.src.services.content_draw_service import DEFAULT_COLOR, default_thickness, draw_boxes

BOX = {"points": [[20, 20], [120, 20], [120, 60], [20, 60]]}


def has_color(image, color):
    return bool(np.all(image == np.array(color, dtype=np.uint8), axis=-1).any())


class TestContentDrawService(unittest.TestCase):
    def setUp(self):
        self.image = np.full((200, 300, 3), 255, dtype=np.uint8)

    def test_draws_border_in_color(self):
        out = draw_boxes(self.image, [BOX])
        self.assertTrue(has_color(out, DEFAULT_COLOR))
        self.assertTrue(np.array_equal(out[40, 70], [255, 255, 255]), "inside of the box stays untouched")

    def test_does_not_modify_input(self):
        before = self.image.copy()
        draw_boxes(self.image, [BOX])
        self.assertTrue(np.array_equal(self.image, before))

    def test_no_boxes_returns_equal_copy(self):
        out = draw_boxes(self.image, [])
        self.assertTrue(np.array_equal(out, self.image))
        self.assertIsNot(out, self.image)

    def test_custom_color(self):
        self.assertTrue(has_color(draw_boxes(self.image, [BOX], color=(0, 0, 255)), (0, 0, 255)))

    def test_grayscale_input_becomes_bgr(self):
        gray = np.full((200, 300), 255, dtype=np.uint8)
        out = draw_boxes(gray, [BOX])
        self.assertEqual(out.shape, (200, 300, 3))

    def test_polygon_with_extra_points_and_ignored_keys(self):
        box = {"points": [[10, 10], [50, 5], [90, 10], [90, 40], [10, 40]], "score": 0.9, "bbox": [10, 5, 90, 40]}
        self.assertTrue(has_color(draw_boxes(self.image, [box]), DEFAULT_COLOR))

    def test_labels_add_more_ink(self):
        plain = draw_boxes(self.image, [BOX])
        labelled = draw_boxes(self.image, [BOX], labels=True)
        self.assertGreater(int((labelled != 255).sum()), int((plain != 255).sum()))

    def test_default_thickness_scales_with_size(self):
        self.assertEqual(default_thickness(np.zeros((100, 100, 3))), 3)
        self.assertEqual(default_thickness(np.zeros((4032, 3024, 3))), 16)

    def test_bad_points_raise(self):
        for points in ([[1, 1], [2, 2]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3, 4]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                draw_boxes(self.image, [{"points": points}])


if __name__ == "__main__":
    unittest.main()
