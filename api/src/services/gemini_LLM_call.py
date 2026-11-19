"""Read handwriting through the Google Gemini API (generateContent). Prompts and schemas come from llm_prompts.

Same functions and return values as openai_LLM_call, so a feature can use either provider.
"""

import base64
import json
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from .llm_prompts import (
    BOXED_SPELLING_INSTRUCTIONS,
    BOXED_SPELLING_SCHEMA,
    PAGE_TRANSCRIBE_INSTRUCTIONS,
    PAGE_TRANSCRIBE_SCHEMA,
    SPELLING_INSTRUCTIONS,
    SPELLING_SCHEMA,
    TRANSCRIBE_INSTRUCTIONS,
    TRANSCRIBE_SCHEMA,
    LLMRecognitionError,
    encode_png,
    parse_boxed_spelling,
    parse_page_lines,
    parse_spelling,
    parse_transcription,
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def transcribe_numbered_lines(
    image: np.ndarray, line_numbers: set[int], *, model: str, api_key: str
) -> tuple[dict[int, str], dict]:
    """Transcribe one image whose handwritten lines are boxed and numbered 1..N. Returns {number: text}."""
    text, usage = _generate(model, api_key, TRANSCRIBE_INSTRUCTIONS, image, TRANSCRIBE_SCHEMA, thinking="low")
    return parse_transcription(text, line_numbers, "Gemini"), usage


def check_numbered_spelling(
    image: np.ndarray, box_numbers: set[int], *, model: str, api_key: str
) -> tuple[list[dict], dict]:
    """Find spelling mistakes in one image whose words are boxed and numbered 1..N."""
    # "low": at "medium" Gemini 3.8 Flash spent ~10k thinking tokens and ~90 s on an 11-word page.
    text, usage = _generate(model, api_key, SPELLING_INSTRUCTIONS, image, SPELLING_SCHEMA, thinking="low")
    return parse_spelling(text, box_numbers, "Gemini"), usage


def transcribe_page(image: np.ndarray, *, model: str, api_key: str) -> tuple[list[str], dict]:
    """Transcribe a plain photo (no boxes drawn). Returns (lines top to bottom, usage)."""
    text, usage = _generate(model, api_key, PAGE_TRANSCRIBE_INSTRUCTIONS, image, PAGE_TRANSCRIBE_SCHEMA, thinking="low")
    return parse_page_lines(text, "Gemini"), usage


def check_spelling_with_boxes(image: np.ndarray, *, model: str, api_key: str) -> tuple[list[dict], int, dict]:
    """Find misspelled words on a plain photo; Gemini also locates them.

    Returns (mistakes with box_2d [ymin, xmin, ymax, xmax] scaled 0-1000, number of words read, usage).
    """
    text, usage = _generate(model, api_key, BOXED_SPELLING_INSTRUCTIONS, image, BOXED_SPELLING_SCHEMA, thinking="low")
    mistakes, word_count = parse_boxed_spelling(text, "Gemini")
    return mistakes, word_count, usage


def _generate(
    model: str, api_key: str, instructions: str, image: np.ndarray, schema: dict, *, thinking: str,
    timeout: int = 120,
) -> tuple[str, dict]:
    """One generateContent call. Returns (JSON text, {input/cached/output tokens, latency_ms})."""
    payload = {
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(encode_png(image)).decode("ascii")}},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "mediaResolution": "MEDIA_RESOLUTION_HIGH",  # small handwriting needs the detail
            "thinkingConfig": {"thinkingLevel": thinking},
        },
    }
    request = Request(
        API_URL.format(model=model),
        data=json.dumps(payload).encode(),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        try:
            reason = json.load(error).get("error", {}).get("message", "")[:200]
        except (OSError, ValueError):
            reason = ""
        raise LLMRecognitionError(f"The Gemini request failed (HTTP {error.code}). {reason}".strip()) from None
    except (OSError, ValueError):
        raise LLMRecognitionError("The Gemini handwriting request failed.") from None
    latency_ms = round((perf_counter() - started) * 1000)

    usage = result.get("usageMetadata") or {}
    thoughts = usage.get("thoughtsTokenCount") or 0
    metrics = {
        "input_tokens": usage.get("promptTokenCount"),
        "cached_tokens": usage.get("cachedContentTokenCount") or 0,
        # Like OpenAI's output_tokens, this includes the thinking tokens (both are billed as output).
        "output_tokens": (usage.get("candidatesTokenCount") or 0) + thoughts,
        "thinking_tokens": thoughts,
        "latency_ms": latency_ms,
    }

    candidates = result.get("candidates") or []
    if not candidates:
        block = (result.get("promptFeedback") or {}).get("blockReason", "no answer")
        raise LLMRecognitionError(f"Gemini did not answer ({block}).")
    candidate = candidates[0]
    if candidate.get("finishReason") not in (None, "STOP"):
        raise LLMRecognitionError(f"Gemini stopped early ({candidate['finishReason']}).")
    text = "".join(
        part.get("text", "") for part in (candidate.get("content") or {}).get("parts", []) if not part.get("thought")
    ).strip()
    if not text:
        raise LLMRecognitionError("Gemini did not return a handwriting transcript.")
    return text, metrics
