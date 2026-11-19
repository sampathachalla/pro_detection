# proDetection

CPU-only handwriting detection with PaddleOCR (PP-OCRv6_medium_det), word borders with OpenCV, and handwriting reading with an OpenAI vision model, packaged in Docker. No GPU and no CUDA.

| Path | What it is | Input | Output |
|---|---|---|---|
| `Dockerfile` | Image build: `python:3.13-slim`, CPU build of Paddle, PaddleOCR, OpenCV headless, non-root user `app`, model weights baked in under `/app/models` | Build context | Image `handwriting-ocr:cpu` |
| `compose.yaml` | Runs the API container: 1 CPU, `OMP_NUM_THREADS=1`, port `127.0.0.1:8000`, HTTP health check on `/api/health`, named volume `ocr-data` at `/app/data` for the temporary local file storage | `docker compose up -d` | Container `prodetection-handwriting-ocr-1` serving the API on `http://localhost:8000` |
| `requirements.txt` | Pinned Python package versions | none | none |
| `download_models.py` | Runs at build time and downloads the PaddleOCR (text detection, page orientation) weights into the image | The model directory environment variables | Weights under `/app/models/paddlex` |
| `test_ocr.py` | Smoke test: draws a sample image and runs PaddleOCR on it | none | Prints the PaddleOCR result count |
| `.dockerignore` | Keeps `.git`, `.env` and Python caches (including those under `api/`) out of the image | none | none |
| `.env` | Local environment settings. It's kept out of the image and shouldn't be committed. | none | none |
| `api/` | Backend: health check and detection services | See [`api/README.md`](api/README.md) | |
| `app/` | Web app (React and Vite). Its `/api` requests are proxied to the API server. | See [`app/README.md`](app/README.md) | |

## Pipeline

```
image bytes → features/preprocessing (validate → quality check → retake or continue → enhance → store) → PaddleOCR detect → boxes [{"points", "score"}]
```

## Commands

```bash
docker compose up -d --build                                                  # build and start
docker compose ps                                                             # status / health
docker exec -i prodetection-handwriting-ocr-1 python - < api/healthcheck/healthcheck.py   # full health check
```
