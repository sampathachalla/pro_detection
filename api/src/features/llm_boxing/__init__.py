"""LLM only: read a photo and box its spelling mistakes with two parallel LLM calls (no OCR models)."""

from .pipeline import LLMBoxing

__all__ = ["LLMBoxing"]
