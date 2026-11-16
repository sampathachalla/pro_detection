import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from api.src.services.image_loader import inspect_image, load_image, load_image_upright
from api.src.test_services.sample_images import png_bytes, text_image


def jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation  # EXIF Orientation tag
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, "JPEG", exif=exif)
    return buf.getvalue()


class TestImageLoader(unittest.TestCase):
    def test_decodes_png_to_bgr(self):
        original = text_image()
        image = load_image(png_bytes(original))
        self.assertEqual(image.shape, original.shape)
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(np.array_equal(image, original))

    def test_empty_bytes_raise(self):
        with self.assertRaisesRegex(ValueError, "empty image"):
            load_image(b"")

    def test_invalid_bytes_raise(self):
        with self.assertRaisesRegex(ValueError, "not a valid image"):
            load_image(b"this is not an image")


class TestInspectImage(unittest.TestCase):
    def test_reads_format_and_size(self):
        self.assertEqual(inspect_image(png_bytes(text_image(width=300, height=120))), {"format": "PNG", "width": 300, "height": 120})

    def test_invalid_bytes_raise(self):
        for data in (b"", b"not an image"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                inspect_image(data)


class TestLoadImageUpright(unittest.TestCase):
    def test_matches_png_pixels(self):
        original = text_image()
        self.assertTrue(np.array_equal(load_image_upright(png_bytes(original)), original))

    def test_applies_exif_rotation(self):
        # Orientation 6 = stored sideways, displayed rotated 90°: width and height swap.
        image = load_image_upright(jpeg_with_orientation(200, 100, 6))
        self.assertEqual(image.shape, (200, 100, 3))

    def test_invalid_bytes_raise(self):
        with self.assertRaises(ValueError):
            load_image_upright(b"not an image")


if __name__ == "__main__":
    unittest.main()
