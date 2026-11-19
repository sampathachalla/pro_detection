# features

Workflow-level features. Each subfolder is one feature that combines the base services in [`../services/`](../services/README.md) for a specific use case.

Calls only go downward: features call `services/`, and `services/` never calls features.

| Path | What it is | What it does |
|---|---|---|
| `__init__.py` | Package marker | Makes `features` importable as `api.src.features`. It has no code, input or output. |
| `preprocessing/` | Preprocessing feature | Checks an uploaded image, warns about poor quality (retake or continue), cleans it for detection, and stores the original, processed image and manifest. See [`preprocessing/README.md`](preprocessing/README.md). |
| `word_box_draw/` | Word border feature | Splits each detected line into words (OpenCV gaps, no model) and draws blue word borders. See [`word_box_draw/README.md`](word_box_draw/README.md). |
| `llm_handwritten_recognition/` | Handwriting-to-text feature | Turns the handwriting into printed text: draws the numbered line borders on one image, sends it to the OpenAI vision model in a single request, and stores the text of each line. |
| `llm_boxing/` | LLM-only analysis | Sends the plain photo (no preprocessing, PaddleOCR or word boxes) to Gemini in two parallel requests: one transcribes the page, the other finds the misspelled words and their boxes. Draws numbered red boxes around the mistakes. See [`llm_boxing/README.md`](llm_boxing/README.md). |
| `spelling_check/` | Spelling check feature | Finds wrongly written words: draws the word boxes with their numbers on one image, sends it to the OpenAI vision model in a single request, and stores the misspelled box numbers with their corrections. |
| `handwritten_content_detection/` | Handwritten content detection feature | Draws borders around the handwriting in a ready image (PaddleOCR, falling back to EasyOCR), and stores the annotated image and box coordinates. See [`handwritten_content_detection/README.md`](handwritten_content_detection/README.md). |
