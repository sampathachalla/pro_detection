import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from api.src.features.handwritten_content_detection import (
    DetectionConfig,
    DetectionError,
    DetectionErrorCode,
    Detector,
    HandwrittenContentDetector,
    map_to_original,
    reading_order,
)
from api.src.features.handwritten_content_detection.pipeline import is_sideways, split_columns
from api.src.features.preprocessing import PreprocessingConfig, PreprocessingPipeline
from api.src.services.content_draw_service import DEFAULT_COLOR
from api.src.services.storage import LocalFileStorage
from api.src.test_services.sample_images import png_bytes, text_image

PADDLE = "api.src.services.paddle_service.detect"
EASYOCR = "api.src.services.easyocr_service.detect"
ORIENTATION = "api.src.services.orientation_service.classify"


def box(x0, y0, x1, y1, score=0.9):
    return {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "score": score}


def region(x0, y0, x1, y1):
    return {"bbox": [x0, y0, x1, y1]}


class DetectorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = LocalFileStorage(self._tmp.name)
        # Quality warnings off, so every synthetic image goes straight to "ready".
        lenient = PreprocessingConfig(min_blur=0, min_brightness=0, min_contrast=0)
        self.preprocessing = PreprocessingPipeline(self.storage, lenient)
        self.detector = HandwrittenContentDetector(self.preprocessing, DetectionConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def ready_image(self, image=None):
        result = self.preprocessing.submit(png_bytes(image if image is not None else text_image()))
        self.assertEqual(result.status, "ready")
        return result.image_id

    def files(self, image_id):
        return [k.rsplit("/", 1)[1] for k in self.storage.list(f"preprocessing/{image_id}/")]


class TestDetectWithFakeModels(DetectorTestCase):
    """Fast: the two detectors are replaced with fakes."""

    def test_paddle_boxes_are_drawn_and_stored(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[box(20, 60, 300, 130)]), mock.patch(EASYOCR) as easy:
            result = self.detector.detect(image_id)
        easy.assert_not_called()
        self.assertEqual(result.detector, Detector.PADDLE)
        self.assertEqual(result.regions[0]["bbox"], [20, 60, 300, 130])
        self.assertEqual(result.regions[0]["score"], 0.9)
        self.assertEqual(result.annotated.shape, text_image().shape)
        self.assertTrue(np.all(result.annotated == np.array(DEFAULT_COLOR, np.uint8), axis=-1).any())
        self.assertIn("annotated.jpg", self.files(image_id))
        self.assertIn("handwriting_detection.json", self.files(image_id))

    def test_easyocr_fallback_when_paddle_finds_nothing(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[]), mock.patch(EASYOCR, return_value=[box(20, 60, 300, 130, None)]) as easy:
            result = self.detector.detect(image_id)
        easy.assert_called_once()
        self.assertEqual(easy.call_args.kwargs["width_ths"], 0.1)
        self.assertEqual(result.detector, Detector.EASYOCR)
        self.assertIsNone(result.regions[0]["score"])

    def test_nothing_found(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[]), mock.patch(EASYOCR, return_value=[]):
            result = self.detector.detect(image_id)
        self.assertEqual(result.detector, Detector.NONE)
        self.assertEqual(result.result["count"], 0)
        self.assertTrue(np.array_equal(result.annotated, text_image()), "nothing drawn")

    def test_boxes_mapped_back_to_original_size(self):
        # 3200 px wide -> preprocessing shrinks to 1600 (scale 0.5); boxes must double.
        image_id = self.ready_image(text_image(width=3200, height=800))
        with mock.patch(PADDLE, return_value=[box(100, 100, 200, 150)]):
            result = self.detector.detect(image_id)
        self.assertEqual(result.regions[0]["bbox"], [200, 200, 400, 300])
        self.assertEqual(result.annotated.shape[:2], (800, 3200))

    def test_tiny_boxes_dropped(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[box(10, 10, 14, 14), box(20, 60, 300, 130)]):
            result = self.detector.detect(image_id)
        self.assertEqual(len(result.regions), 1)

    def test_regions_in_reading_order(self):
        image_id = self.ready_image()
        shuffled = [box(400, 100, 600, 140), box(20, 20, 200, 60), box(20, 100, 200, 140), box(400, 22, 600, 62)]
        with mock.patch(PADDLE, return_value=shuffled):
            result = self.detector.detect(image_id)
        self.assertEqual([r["bbox"][:2] for r in result.regions], [[20, 20], [400, 22], [20, 100], [400, 100]])

    def test_sideways_page_is_turned_and_detected_again(self):
        image_id = self.ready_image()  # 720 × 180 (w × h)
        tall = [box(20, 10, 60, 170), box(100, 10, 140, 170)]
        with mock.patch(PADDLE, side_effect=[tall, [box(20, 60, 150, 100)]]) as paddle, \
                mock.patch(ORIENTATION, return_value={"angle": 270, "score": 0.99}):
            result = self.detector.detect(image_id)
        self.assertEqual(paddle.call_count, 2)
        self.assertEqual(paddle.call_args_list[1].args[0].shape[:2], (720, 180), "detected on the turned page")
        self.assertEqual(result.result["rotation"], 270)
        self.assertEqual(result.result["image"], {"width": 180, "height": 720})
        self.assertEqual(result.annotated.shape[:2], (720, 180))
        self.assertEqual(self.preprocessing.load_original(image_id).shape[:2], (720, 180))

    def test_sideways_boxes_but_model_says_upright_keeps_page(self):
        image_id = self.ready_image()
        tall = [box(20, 10, 60, 170), box(100, 10, 140, 170)]
        with mock.patch(PADDLE, return_value=tall) as paddle, mock.patch(ORIENTATION, return_value={"angle": 0, "score": 0.6}):
            result = self.detector.detect(image_id)
        self.assertEqual(paddle.call_count, 1)
        self.assertEqual(result.result["rotation"], 0)

    def test_upright_page_skips_orientation_model(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[box(20, 60, 300, 130)]), mock.patch(ORIENTATION) as orientation:
            self.detector.detect(image_id)
        orientation.assert_not_called()

    def test_get_result_and_load_annotated(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[box(20, 60, 300, 130)]):
            result = self.detector.detect(image_id)
        self.assertEqual(self.detector.get_result(image_id), result.result)
        loaded = self.detector.load_annotated(image_id)  # JPEG: close, not byte-identical
        self.assertEqual(loaded.shape, result.annotated.shape)
        self.assertLess(float(np.abs(loaded.astype(int) - result.annotated.astype(int)).mean()), 3.0)

    def test_retake_removes_detection_files(self):
        image_id = self.ready_image()
        with mock.patch(PADDLE, return_value=[box(20, 60, 300, 130)]):
            self.detector.detect(image_id)
        self.preprocessing.retake(image_id)
        self.assertEqual(self.files(image_id), [])


class TestDetectErrors(DetectorTestCase):
    def assert_code(self, code, func, *args):
        with self.assertRaises(DetectionError) as ctx:
            func(*args)
        self.assertEqual(ctx.exception.code, code)

    def test_unknown_image(self):
        self.assert_code(DetectionErrorCode.NOT_FOUND, self.detector.detect, "f" * 32)
        self.assert_code(DetectionErrorCode.NOT_FOUND, self.detector.detect, "../etc")

    def test_needs_review_is_not_ready(self):
        blurry = cv2.GaussianBlur(text_image(), (31, 31), 0)
        image_id = PreprocessingPipeline(self.storage, PreprocessingConfig()).submit(png_bytes(blurry)).image_id
        self.assert_code(DetectionErrorCode.NOT_READY, self.detector.detect, image_id)

    def test_result_before_detect(self):
        self.assert_code(DetectionErrorCode.NOT_FOUND, self.detector.get_result, self.ready_image())


class TestHelpers(unittest.TestCase):
    def test_map_scale_only(self):
        processed = {"width": 100, "height": 50, "scale": 0.5, "angle": 0.0}
        self.assertEqual(map_to_original([[10, 20], [30, 40]], processed, 200, 100), [[20, 40], [60, 80]])

    def test_map_undoes_deskew(self):
        processed = {"width": 400, "height": 200, "scale": 1.0, "angle": 10.0}
        original = np.array([[[100, 50]], [[300, 150]]], dtype=np.float32)
        matrix = cv2.getRotationMatrix2D((200, 100), 10.0, 1.0)
        rotated = cv2.transform(original, matrix).reshape(-1, 2).tolist()
        self.assertEqual(map_to_original(rotated, processed, 400, 200), [[100, 50], [300, 150]])

    def test_map_clips_to_image(self):
        processed = {"width": 100, "height": 100, "scale": 1.0, "angle": 0.0}
        self.assertEqual(map_to_original([[-5, 150]], processed, 100, 100), [[0, 99]])

    def test_reading_order_empty(self):
        self.assertEqual(reading_order([], 0.5), [])

    def test_reading_order_lines_then_left_to_right(self):
        regions = [region(300, 0, 400, 40), region(0, 60, 100, 100), region(0, 5, 100, 45)]
        ordered = reading_order(regions, 0.5)
        self.assertEqual([r["bbox"][:2] for r in ordered], [[0, 5], [300, 0], [0, 60]])

    def test_two_columns_read_one_after_the_other(self):
        left = [region(0, y, 400, y + 40) for y in (0, 60, 120, 180)]
        right = [region(500, y + 5, 900, y + 45) for y in (0, 60, 120)]
        ordered = reading_order(right + left, 0.5)
        self.assertEqual(ordered, left + right)

    def test_column_gutter_may_be_crossed_by_a_few_lines(self):
        left = [region(0, y, 400, y + 40) for y in range(0, 600, 60)]
        right = [region(500, y, 900, y + 40) for y in range(0, 600, 60)]
        wide = region(0, 300, 700, 340)  # one long line runs into the right column
        columns = split_columns(left + right + [wide])
        self.assertEqual(len(columns), 2)
        self.assertIn(wide, columns[0])

    def test_single_column_with_short_lines_is_not_split(self):
        regions = [region(0, 0, 900, 40), region(0, 60, 400, 100), region(600, 62, 900, 102), region(0, 120, 300, 160)]
        self.assertEqual(len(split_columns(regions)), 1)

    def test_columns_need_lines_side_by_side(self):
        top = [region(0, y, 400, y + 40) for y in (0, 60, 120)]
        bottom = [region(500, y, 900, y + 40) for y in (300, 360, 420)]
        self.assertEqual(len(split_columns(top + bottom)), 1)

    def test_is_sideways(self):
        self.assertTrue(is_sideways([box(0, 0, 40, 300), box(60, 0, 100, 280), box(0, 0, 200, 40)]))
        self.assertFalse(is_sideways([box(0, 0, 300, 40), box(0, 60, 280, 100)]))
        self.assertFalse(is_sideways([box(0, 0, 40, 300)]), "one tall box is not enough")


class TestDetectWithRealModels(DetectorTestCase):
    """Slow (~20 s): uses the baked-in PP-OCRv6 weights. Run inside the container."""

    def test_real_handwriting_sample(self):
        image_id = self.ready_image()
        result = self.detector.detect(image_id)
        self.assertIn(result.detector, (Detector.PADDLE, Detector.EASYOCR))
        self.assertGreater(result.result["count"], 0)
        h, w = text_image().shape[:2]
        for r in result.regions:
            x0, y0, x1, y1 = r["bbox"]
            self.assertTrue(0 <= x0 < x1 < w and 0 <= y0 < y1 < h)


if __name__ == "__main__":
    unittest.main()
