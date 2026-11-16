"""Handwritten content detection settings. Can be overridden with environment variables."""

import os
from dataclasses import dataclass

DETECTION_VERSION = "1"
RESULT_FILENAME = "handwriting_detection.json"
ANNOTATED_FILENAME = "annotated.jpg"  # JPEG: a PNG of a 12 MP photo is ~16 MB
ANNOTATED_JPEG_QUALITY = 90


@dataclass(frozen=True)
class DetectionConfig:
    min_box_area: int = 50          # px² in the original image; smaller boxes are dropped as noise
    easyocr_width_ths: float = 0.1  # EasyOCR fallback merge threshold (lower keeps words apart)
    line_tolerance: float = 0.5     # boxes whose centres differ by < this × median box height share a line
    color: tuple[int, int, int] = (0, 200, 0)  # border colour (BGR)
    labels: bool = False            # number the boxes in reading order

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        env = os.environ.get
        default = cls()
        return cls(
            min_box_area=int(env("DETECT_MIN_BOX_AREA", default.min_box_area)),
            easyocr_width_ths=float(env("DETECT_EASYOCR_WIDTH_THS", default.easyocr_width_ths)),
            line_tolerance=float(env("DETECT_LINE_TOLERANCE", default.line_tolerance)),
            labels=env("DETECT_LABELS", "false").lower() == "true",
        )
