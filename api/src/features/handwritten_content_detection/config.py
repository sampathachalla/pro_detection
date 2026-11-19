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
    line_tolerance: float = 0.5     # boxes whose centres differ by < this × median box height share a line
    column_max_crossing: float = 0.1  # a column gutter may be crossed by at most this share of the boxes
    column_min_lines: int = 3       # each column needs at least this many boxes
    auto_rotate: bool = True        # turn sideways pages (most boxes taller than wide) before reading
    color: tuple[int, int, int] = (0, 200, 0)  # border colour (BGR)
    labels: bool = True             # number the boxes 1, 2, 3 in reading order

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        env = os.environ.get
        default = cls()
        return cls(
            min_box_area=int(env("DETECT_MIN_BOX_AREA", default.min_box_area)),
            line_tolerance=float(env("DETECT_LINE_TOLERANCE", default.line_tolerance)),
            column_max_crossing=float(env("DETECT_COLUMN_MAX_CROSSING", default.column_max_crossing)),
            column_min_lines=int(env("DETECT_COLUMN_MIN_LINES", default.column_min_lines)),
            auto_rotate=env("DETECT_AUTO_ROTATE", "true").lower() == "true",
            labels=env("DETECT_LABELS", "true").lower() == "true",
        )
