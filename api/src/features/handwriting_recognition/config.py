"""Handwriting recognition settings. Can be overridden with environment variables."""

import os
from dataclasses import dataclass

RECOGNITION_VERSION = "1"
RESULT_FILENAME = "recognized_text.json"
MODEL_NAME = "microsoft/trocr-small-handwritten"


@dataclass(frozen=True)
class RecognitionConfig:
    # Lines read with lower confidence are kept but flagged "low_confidence"
    # (e.g. a drawn arrow read as "#"); they are left out of the joined text.
    min_confidence: float = 0.5

    @classmethod
    def from_env(cls) -> "RecognitionConfig":
        return cls(min_confidence=float(os.environ.get("RECOGNIZE_MIN_CONFIDENCE", cls().min_confidence)))
