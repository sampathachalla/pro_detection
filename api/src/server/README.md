# server

HTTP API (FastAPI and uvicorn) that connects the web app to the features:

```
web app (app/) ──/api/*──▶ Vite proxy ──▶ this server :8000 ──▶ features/preprocessing ──▶ features/handwritten_content_detection ──▶ features/word_box_draw
                                                                    then in parallel ──▶ features/llm_handwritten_recognition + features/spelling_check
                                                  /api/llm-images ──▶ features/llm_boxing (LLM only: two parallel LLM calls, no OCR models)
```

It only handles HTTP: reading uploads, mapping errors to HTTP status codes and serving files. All the logic lives in `features/`.

| File | What it is |
|---|---|
| `__init__.py` | Package marker. |
| `app.py` | `create_app(preprocessing=None, detector=None, word_drawer=None, warm_up=True, cleanup_interval_seconds=3600)` builds the FastAPI app. Tests pass in a temporary storage folder and turn warm-up off. |

**Run** (this is the container's default command):

```bash
uvicorn api.src.server.app:create_app --factory --host 0.0.0.0 --port 8000
```

**At startup** it loads PP-OCRv6 (warm-up), so `/api/health` answers only once the model is ready. It then runs `preprocessing.cleanup()` every hour (24-hour retention).

**Timing:** every upload and confirm returns a `Server-Timing` header and a `timings_ms` body field (`wait`, `preprocess`, `detect`, `words`, `llm`, `total` in ms), and logs one line per image.

**OpenAI:** after the word boxes, transcription and the spelling check run in parallel (one request each), outside the model lock. A failed call is returned as `{"status": "failed", "error": {"message"}}` without failing the upload.

**One image at a time:** a lock serialises model work. The container has 1 CPU and the model predictors aren't thread-safe.

## Endpoints

| Method and path | Input | Output |
|---|---|---|
| `GET /api/health` | none | `{"status": "ok"}` |
| `POST /api/images` | `multipart/form-data` with field `file` (JPEG or PNG, smaller than 20 MB) | `200` with a **done** or **needs_review** body (below) |
| `POST /api/images/{id}/confirm` | none. The user chose **Continue anyway**. | `200` **done** |
| `POST /api/images/{id}/analyse` | none. The web app's **Run** on an image already shown. | `200` **done**: detection, word boxes and both LLM calls run again. For an LLM-only image, its stored photo goes through the full flow as a new upload (**new** `image_id`, may be **needs_review**). |
| `DELETE /api/images/{id}` | none. The user chose **Retake**. | `204`, and every file for that image is deleted |
| `GET /api/images/{id}` | none | the current **done** or **needs_review** body (for an LLM-only image, its saved LLM-only body) |
| `POST /api/llm-images` | `multipart/form-data` with field `file` (same limits) | `200` **done** body with `"mode": "llm_only"`, `"detector": "llm"`, `llm.transcription` and `llm.spelling` (each mistake has `bbox`); no preprocessing, PaddleOCR or word boxes, and no model lock. See [`features/llm_boxing`](../features/llm_boxing/README.md). |
| `POST /api/images/{id}/llm-only` | none | LLM only on a photo already uploaded (OCR or LLM-only), using the stored original. `200` **done** body like `POST /api/llm-images`, with a **new** `image_id`. |
| `GET /api/llm-images/{id}/boxed.jpg` | none | `image/jpeg`: the photo with numbered **red** boxes around the misspelled words |
| `GET /api/images/{id}/annotated.jpg` | none | `image/jpeg`: the photo with **green line** borders |
| `GET /api/images/{id}/words.jpg` | none | `image/jpeg`: the photo with **green line** borders and **blue word** borders inside them |
| `GET /api/docs` | none | Interactive API docs |

**done**
```json
{
  "image_id": "…", "status": "done", "warnings": [],
  "detector": "paddle", "count": 3,
  "image": {"width": 3024, "height": 4032},
  "regions": [{"points": [[x, y] x4], "bbox": [x0, y0, x1, y1], "score": 0.91}],
  "annotated_url": "/api/images/<id>/annotated.jpg",
  "word_count": 31,
  "words": [{"line": 10, "points": [[x, y] x4], "bbox": [x0, y0, x1, y1]}],
  "words_annotated_url": "/api/images/<id>/words.jpg",
  "llm": {
    "transcription": {"status": "ok", "model": "…", "text": "…", "lines": [{"line", "bbox", "text"}],
                      "usage": {"input_tokens", "cached_tokens", "output_tokens", "latency_ms"},
                      "request_count": 1, "processing_ms": 3300},
    "spelling": {"status": "ok", "model": "…", "mistakes": [{"box_number", "wrong_spelling", "correct_spelling"}],
                 "usage": {"input_tokens", "cached_tokens", "output_tokens", "latency_ms"},
                 "request_count": 1, "processing_ms": 4200}
  },
  "timings_ms": {"wait": 0, "preprocess": 800, "detect": 4200, "words": 150, "llm": 4250, "total": 9400}
}
```

`GET /api/images/{id}` returns the same body from the saved files: `llm` holds whichever OpenAI results were saved, and `timings_ms` is left out.

**needs_review** (show the warnings, then call `confirm` or `DELETE`)
```json
{"image_id": "…", "status": "needs_review", "warnings": [{"code": "blurry", "message": "The image looks blurry."}]}
```

**Errors:** `{"error": {"code": "...", "message": "..."}}`. The `message` can be shown to the user.

| Code | HTTP |
|---|---|
| `empty`, `unreadable`, `too_small` | 400 |
| `not_found` | 404 |
| `invalid_state`, `not_ready` | 409 |
| `too_large`, `too_many_pixels` | 413 |
| `unsupported_type` | 415 |
| missing `file` field | 422 |

## Network

- **Published on `127.0.0.1:8000` only.** It isn't reachable from your local network directly.
- **The web app reaches it through Vite's `/api` proxy** (`app/vite.config.js`, `API_PROXY_TARGET`).
- **The mobile tunnel URL exposes `/api` publicly through that proxy, with no authentication.** That's fine for development; add authentication before sharing the URL widely.
- **The container now has outbound network access** (needed to publish a port). It still never downloads models: EasyOCR has `download_enabled=False`, and PaddleX uses the built-in weights ("Model files already exist. Using cached files").
