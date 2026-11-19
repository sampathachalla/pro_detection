# llm_boxing

**Feature goal:** read a handwriting photo and box its spelling mistakes using only the LLM. The photo is **not** preprocessed and never goes to PaddleOCR or the OpenCV word segmenter. The web app's **Run LLM only** button uses it, through `POST /api/llm-images`.

```
analyse(bytes)
 1. Validate   same hard limits as preprocessing: size, JPEG/PNG, pixel count, minimum side (no quality check)
 2. Store      preprocessing/<id>/original.<ext>   (same folder as other uploads, so the 24-hour cleanup removes it)
 3. Ask        two Gemini calls at the same time, on the plain photo shrunk to 2048 px:
                 transcription   every handwritten line, copied letter by letter (no spelling fixes)
                 spelling        every word: written, intended, is_misspelled, box_2d → keeps the misspelled ones
 4. Draw       numbered red boxes around the misspelled words on the photo → llm_boxed.jpg
 5. Store      llm_boxing.json (the full response, so a page reload shows it again)
```

A failed call comes back as `{"status": "failed", "error": {"message"}}`. The other call's result is kept, like in the OCR flow.

**Provider:** only Gemini returns word locations (`box_2d`: `[ymin, xmin, ymax, xmax]` on a 0–1000 scale of the image), so `LLM_PROVIDER` must be `gemini` (the default). With `openai` the request fails with `503`.

**Why every word is listed, not only the mistakes:** writing out each word before saying what was meant stops the model from silently correcting the spelling. It costs some extra output tokens.

## `pipeline.py`

| Method | Input | Output |
|---|---|---|
| `LLMBoxing(preprocessing)` | a `PreprocessingPipeline` (for its storage and upload limits) | the feature |
| `analyse(data, filename=None)` | image bytes | the **done** body (below). Raises `PreprocessingError` for a bad upload, `LLMRecognitionError` (`503`) when the provider can't locate words or has no key. |
| `analyse_stored(image_id)` | id of a photo already uploaded | `analyse()` on its stored `original.*`, as a **new** LLM-only result (new `image_id`). A page turned with `rotate()` is sent as uploaded. Raises `PreprocessingError("not_found")` if the photo was cleaned up. |
| `get_result(image_id)` | `str` | stored `llm_boxing.json`, or `None` |
| `load_boxed(image_id)` | `str` | `llm_boxed.jpg` bytes. Raises `PreprocessingError("not_found")`. |

## `config.py`

| Setting | Value | Effect |
|---|---|---|
| `SEND_MAX_SIDE` | 2048 | Longest side of the photo sent to the LLM |
| `MISTAKE_BOX_COLOR` | `(0, 0, 220)` red (BGR) | Mistake box colour |
| `RESULT_FILENAME`, `BOXED_FILENAME` | `llm_boxing.json`, `llm_boxed.jpg` | Stored files |

## Response (`done`)

Same shape as the OCR flow's **done** body, so the web app shows it the same way:

```json
{
  "image_id": "…", "status": "done", "mode": "llm_only", "detector": "llm",
  "count": 4, "word_count": 20, "image": {"width": 1536, "height": 2048},
  "regions": [], "words": [], "words_annotated_url": null,
  "annotated_url": "/api/llm-images/<id>/boxed.jpg",
  "llm": {
    "transcription": {"status": "ok", "model": "…", "text": "…", "lines": [{"line": 0, "text": "…"}],
                      "usage": {…}, "request_count": 1, "processing_ms": 4200},
    "spelling": {"status": "ok", "model": "…", "word_count": 20,
                 "mistakes": [{"box_number": 1, "wrong_spelling": "fracure", "correct_spelling": "fracture",
                               "bbox": [x0, y0, x1, y1], "points": [[x0, y0], …]}],
                 "usage": {…}, "request_count": 1, "processing_ms": 5100}
  },
  "timings_ms": {"llm": 5300, "total": 5300}
}
```
