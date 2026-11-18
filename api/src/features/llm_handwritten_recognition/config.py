"""Defaults for LLM handwriting recognition. LLM_PROVIDER, GEMINI_MODEL and OPENAI_HANDWRITING_MODEL override them."""

DEFAULT_PROVIDER = "gemini"                  # "gemini" or "openai"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_MODEL = "gpt-6-luna"                 # OpenAI
RESULT_FILENAME = "recognized_text_openai.json"
