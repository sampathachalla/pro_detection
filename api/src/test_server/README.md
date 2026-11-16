# test_server

Tests for [`server/app.py`](../server/README.md). They use FastAPI's `TestClient` (via `httpx`, already in the image) with a fake PaddleOCR, and real preprocessing, drawing and storage in a temporary folder.

| File | What it tests | Passes when |
|---|---|---|
| `__init__.py` | Package marker | none |
| `test_app.py` | Every endpoint | `/api/health` is ok; a clean upload returns **done** with regions, image size and an `annotated_url` that serves a PNG; `GET /api/images/{id}` matches; a blurry upload returns **needs_review** without running detection, and `confirm` then gives **done**; `DELETE` removes everything (then `404`); confirming twice returns `409`; `empty`, `unreadable`, `unsupported_type`, `too_small` and `too_large` give 400, 400, 415, 400 and 413 with a message; a missing file returns `422`; unknown or unsafe IDs return `404` |

## How to run

From the project root, inside the container:

```bash
python -m unittest discover -s api/src/test_server -t . -v
```
