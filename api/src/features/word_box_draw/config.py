"""Word box drawing settings. Can be overridden with environment variables."""

import os
from dataclasses import dataclass

WORDS_VERSION = "4"  # 2: gaps vs small-letter height; 3: ruled lines removed; 4: edge slivers dropped
# Saved analyses with another version are not reused for a re-upload of the same photo (see /api/image-analysis).
RESULT_FILENAME = "word_boxes.json"
ANNOTATED_FILENAME = "words_annotated.jpg"
ANNOTATED_JPEG_QUALITY = 90


@dataclass(frozen=True)
class WordBoxConfig:
    gap_ratio: float = 0.45         # gap > this x small-letter height = new word (lower splits more)
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
