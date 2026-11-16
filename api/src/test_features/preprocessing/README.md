# test_features/preprocessing

Tests for [`features/preprocessing/`](../../features/preprocessing/README.md). They use the built-in `unittest` module and need no OCR models. Storage is a temporary folder deleted after each test, so nothing touches `/app/data`.

| File | What it tests | Input | Passes when |
|---|---|---|---|
| `__init__.py` | Package marker | none | Makes the tests importable as `api.src.test_features.preprocessing` |
| `test_pipeline.py` | `PreprocessingPipeline`, the full workflow | Synthetic PNG, JPEG, blurred, dark, BMP, tiny and garbage uploads | See the table below |
| `test_config.py` | `PreprocessingConfig` | Defaults and `PREPROCESS_*` environment variables | Defaults are JPEG and PNG, 20 MB, 24 hours and deskew off; environment variables override them; with no variables set, `from_env()` equals the defaults |

## What `test_pipeline.py` covers

| Area | Case | Expected |
|---|---|---|
| Happy path | Clean PNG or JPEG | `ready`; `original` + `processed.png` + `manifest.json` stored; manifest has filename, content type, sha256 and scale |
| | `load_processed(id)` | Returns the same image |
| Retake or continue | Blurred image | `needs_review` with `blurry`; no processed file |
| | Dark image | `too_dark` warning |
| | `confirm(id)` | `ready`, `accepted_warnings = true` |
| | `confirm` twice | `invalid_state` |
| | `retake(id)` | All files deleted, then `not_found` |
| Rejections | Empty, too large, garbage, BMP, too many pixels, tiny | `empty`, `too_large`, `unreadable`, `unsupported_type`, `too_many_pixels`, `too_small`; nothing stored |
| | Bad IDs (`""`, `../etc`, an unknown ID) | `not_found` |
| Retention | One image older than 24 hours, one new | `cleanup()` removes only the old one |

## How to run

From the project root, inside the container:

```bash
# just this feature
python -m unittest discover -s api/src/test_features/preprocessing -t . -v

# one file
python -m unittest api.src.test_features.preprocessing.test_pipeline -v
```
