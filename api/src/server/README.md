# server

HTTP API (FastAPI and uvicorn) that connects the web app to the features:

```
web app (app/) ──/api/*──▶ Vite proxy ──▶ this server :8000 ──▶ features/preprocessing ──▶ features/handwritten_content_detection ──▶ features/word_box_draw
                                                                                                   └─ on demand (/read) ──▶ features/handwriting_recognition
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

**Timing:** every upload and confirm returns a `Server-Timing` header (`wait`, `preprocess`, `detect`, `words`, `total` in ms; `read` for `/read`) and logs one line per image.

**One image at a time:** a lock serialises model work. The container has 1 CPU and the model predictors aren't thread-safe.

## Endpoints

| Method and path | Input | Output |
|---|---|---|
| `GET /api/health` | none | `{"status": "ok"}` |
| `POST /api/images` | `multipart/form-data` with field `file` (JPEG or PNG, smaller than 20 MB) | `200` with a **done** or **needs_review** body (below) |
| `POST /api/images/{id}/confirm` | none. The user chose **Continue anyway**. | `200` **done** |
| `POST /api/images/{id}/read` | none. The user tapped **Read text**. | `200` `{"status": "read", "lines": [{"line", "bbox", "text", "confidence", "low_confidence"}], "text": "…", "seconds", "model"}`. TrOCR, about 1 s per line. |
| `DELETE /api/images/{id}` | none. The user chose **Retake**. | `204`, and every file for that image is deleted |
| `GET /api/images/{id}` | none | the current **done** or **needs_review** body |
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
  "recognition": null
}
```

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
