# test_features/handwriting_recognition

Tests for [`features/handwriting_recognition/`](../../features/handwriting_recognition/README.md). They use a fake PaddleOCR (two line boxes) and a fake `trocr_service.recognize`, so they're fast and need no models. The real TrOCR model is tested in `test_services/test_trocr_service.py`.

| File | What it tests | Passes when |
|---|---|---|
| `__init__.py` | Package marker | none |
| `test_pipeline.py` | `HandwritingRecognizer` | **Every line** is read once, in reading order, with `line` indexes and the joined `text`, and `recognized_text.json` is stored. **Low-confidence lines** are flagged and left out of `text`. **`get_result`** is `None` before `read`, then equals it. **Without detection:** `DetectionError("not_found")`. **`retake`** deletes the text file. |

## How to run

```bash
python -m unittest discover -s api/src/test_features/handwriting_recognition -t . -v
```
