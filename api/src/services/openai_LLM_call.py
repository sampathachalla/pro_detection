"""Read handwriting through the OpenAI Responses API. Prompts and schemas come from llm_prompts."""

import base64
import json
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np

from .llm_prompts import (
    SPELLING_INSTRUCTIONS,
    SPELLING_SCHEMA,
    TRANSCRIBE_INSTRUCTIONS,
    TRANSCRIBE_SCHEMA,
    LLMRecognitionError,
    encode_png,
    parse_spelling,
    parse_transcription,
)

OpenAIRecognitionError = LLMRecognitionError  # older name, still imported by the server


def _image_input(image: np.ndarray) -> dict:
    image_url = "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")
    return {"type": "input_image", "image_url": image_url, "detail": "high"}


def transcribe_numbered_lines(
    image: np.ndarray, line_numbers: set[int], *, model: str, api_key: str
) -> tuple[dict[int, str], dict]:
    """Transcribe one image whose handwritten lines are boxed and numbered 1..N. Returns {number: text}."""
    text, usage = _request_text({
        "model": model,
        **({"reasoning": {"effort": "low"}} if model == "gpt-6-luna" else {}),
        "store": False,
        "instructions": TRANSCRIBE_INSTRUCTIONS,
        "input": [{"role": "user", "content": [_image_input(image)]}],
        "text": {"format": {"type": "json_schema", "name": "line_transcripts", "strict": True,
                            "schema": TRANSCRIBE_SCHEMA}},
    }, api_key, timeout=120)
    return parse_transcription(text, line_numbers, "OpenAI"), usage


def check_numbered_spelling(
    image: np.ndarray, box_numbers: set[int], *, model: str, api_key: str
) -> tuple[list[dict], dict]:
    """Find spelling mistakes in one image whose words are boxed and numbered 1..N."""
    text, usage = _request_text({
        "model": model,
        **({"reasoning": {"effort": "medium"}} if model == "gpt-6-luna" else {}),
        "store": False,
        "instructions": SPELLING_INSTRUCTIONS,
        "input": [{"role": "user", "content": [_image_input(image)]}],
        "text": {"format": {"type": "json_schema", "name": "word_spelling", "strict": True,
                            "schema": SPELLING_SCHEMA}},
    }, api_key, timeout=120)
    return parse_spelling(text, box_numbers, "OpenAI"), usage


def _request_text(payload: dict, api_key: str, *, timeout: int = 30) -> tuple[str, dict]:
    """POST to the Responses API. Returns (output text, {input/cached/output tokens, latency_ms})."""
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (OSError, ValueError):
        raise LLMRecognitionError("The OpenAI handwriting request failed.") from None
    latency_ms = round((perf_counter() - started) * 1000)

    if result.get("status") != "completed":
        raise LLMRecognitionError("OpenAI did not complete the handwriting transcription.")
    usage = result.get("usage") or {}
    metrics = {
        "input_tokens": usage.get("input_tokens"),
        "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "latency_ms": latency_ms,
    }
    for item in result.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content["text"].strip(), metrics
    raise LLMRecognitionError("OpenAI did not return a handwriting transcript.")
