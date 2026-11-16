# api

Backend for handwriting text detection (CPU only).

| Path | What it is | What it does |
|---|---|---|
| `__init__.py` | Package marker | Makes `api` importable, so you can write `from api.src.services import ...` and the tests run with `python -m unittest`. It has no code, input or output. |
| `healthcheck/` | Health check | Checks the container's models, CPU-only Paddle and torch, the OCR libraries, and the non-root user. See [`healthcheck/README.md`](healthcheck/README.md). |
| `src/` | Source code | Detection services (image loading, OpenCV preprocessing, PaddleOCR and EasyOCR) and their tests in `test_services/`. See [`src/README.md`](src/README.md). |

**Status:** `api/` isn't copied into the Docker image yet. The `Dockerfile` needs a `COPY api/ ./api/` line and a rebuild.
