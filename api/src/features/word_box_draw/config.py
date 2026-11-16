"""Word box drawing settings. Can be overridden with environment variables."""

import os
from dataclasses import dataclass

WORDS_VERSION = "1"
RESULT_FILENAME = "word_boxes.json"
ANNOTATED_FILENAME = "words_annotated.jpg"
ANNOTATED_JPEG_QUALITY = 90


@dataclass(frozen=True)
class WordBoxConfig:
    gap_ratio: float = 0.35         # gap > this x text height = new word (lower splits more)
    min_word_area: int = 30         # px² on the original photo; smaller boxes are dropped
    color: tuple[int, int, int] = (230, 120, 0)  # border colour (BGR blue), distinct from the green lines
    show_lines: bool = True         # also draw the green PaddleOCR line borders under the word borders

    @classmethod
    def from_env(cls) -> "WordBoxConfig":
        env = os.environ.get
        default = cls()
        return cls(
            gap_ratio=float(env("WORDS_GAP_RATIO", default.gap_ratio)),
            min_word_area=int(env("WORDS_MIN_AREA", default.min_word_area)),
            show_lines=env("WORDS_SHOW_LINES", "true").lower() == "true",
        )
