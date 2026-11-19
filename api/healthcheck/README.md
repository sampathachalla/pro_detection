# healthcheck

## `healthcheck.py`

**What:** Container health check script.
**Does:** Confirms the CPU-only OCR stack inside the container is usable. It runs five checks and prints `[PASS]` or `[FAIL]` for each.

| Check | Passes when |
|---|---|
| `models` | `PADDLE_PDX_CACHE_HOME` (default `/app/models/paddlex`) is not empty |
| `paddle` | Paddle imports, runs on CPU, and was not built with CUDA |
| `ocr libs` | `paddleocr` and `cv2` import |
| `runtime` | The process is not running as root (uid ≠ 0). It also reports `OMP_NUM_THREADS`. |

**Input:** None. It reads the environment variables above.
**Output:** One line per check plus a final `HEALTHY` or `UNHEALTHY (n failed)` on stdout.
**Exit code:** `0` if all checks pass, `1` otherwise.

It only checks the setup; it doesn't run OCR on an image. A run takes about 35 seconds, mostly import time.

**Run it against the running container** (the script isn't in the image yet):

```bash
docker exec -i prodetection-handwriting-ocr-1 python - < api/healthcheck/healthcheck.py
```
