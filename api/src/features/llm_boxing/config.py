"""LLM-only analysis settings."""

RESULT_FILENAME = "llm_boxing.json"
BOXED_FILENAME = "llm_boxed.jpg"
BOXED_JPEG_QUALITY = 90
# Longest side of the photo sent to the LLM. The web app already uploads at most 2048 px.
SEND_MAX_SIDE = 2048
MISTAKE_BOX_COLOR = (0, 0, 220)  # BGR red, same as the spelling boxes in the OCR flow
