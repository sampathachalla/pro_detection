# test_features

Tests for `api/src/features/`, with **one folder per feature**, mirroring `features/`.

| Path | What it is | What it does |
|---|---|---|
| `__init__.py` | Package marker | Makes the tests importable as `api.src.test_features`. |
| `handwritten_content_detection/` | Handwritten content detection tests | The fallback order, mapping boxes back, drawing, storage, errors and reading order, plus one real-model test. See [`handwritten_content_detection/README.md`](handwritten_content_detection/README.md). |
| `handwriting_recognition/` | Text reading tests | Reading order, low-confidence lines, storage, errors (fake TrOCR). See [`handwriting_recognition/README.md`](handwriting_recognition/README.md). |
| `word_box_draw/` | Word border tests | Word splitting per line, tilted boxes, storage, errors. See [`word_box_draw/README.md`](word_box_draw/README.md). |
| `preprocessing/` | Preprocessing feature tests | The full workflow (validate, quality warnings, retake or continue, store, cleanup) and the config. See [`preprocessing/README.md`](preprocessing/README.md). |

## How to run

From the project root, inside the container:

```bash
python -m unittest discover -s api/src/test_features -t . -v
```
