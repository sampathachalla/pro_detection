# preprocessing

**Feature:** gets an uploaded handwriting image ready for text detection. It checks the upload, warns the user about poor quality so they can **retake or continue**, cleans the image, and stores the original, the processed image and a manifest.

It only calls base services: `image_loader`, `quality_service`, `preprocess_service` and `storage`.

## Workflow

```
submit(bytes)
 1. Validate    empty? ≥ 20 MB? JPEG/PNG (checked from the file header, not the name)?
                more than 40 MP? smaller than 64 px?            → reject (PreprocessingError)
 2. Identify    image_id (uuid4 hex) + sha256
 3. Normalize   decode upright (EXIF orientation) → BGR
 4. Quality     blur / brightness / contrast → warnings
 5. Store       original.{jpg|png}
 6a. no warnings → enhance → processed.png + manifest.json       → status "ready"
 6b. warnings    → manifest.json only                           → status "needs_review"
                   the UI shows the warnings and asks: Retake or Continue?
                     Continue → confirm(image_id) → enhance → "ready"
                     Retake   → retake(image_id)  → everything deleted; the user uploads again
cleanup()  deletes images whose files are older than 24 hours
```

Storage layout (the same keys will work on MinIO later):

```
preprocessing/<image_id>/original.jpg | original.png
preprocessing/<image_id>/processed.png      (only once ready)
preprocessing/<image_id>/manifest.json
```

---

## `pipeline.py`

**What:** `PreprocessingPipeline(storage=None, config=None)`, the workflow. Storage defaults to `get_storage()` and config to `PreprocessingConfig.from_env()`. Tests pass in a temporary storage and custom limits.

| Method | Input | Output |
|---|---|---|
| `submit(data, filename=None)` | upload `bytes`, optional original file name | `PreprocessResult`: `status` is `"ready"` (with `image`) or `"needs_review"` (with `warnings`, `image=None`). Raises `PreprocessingError` when rejected. |
| `confirm(image_id)` | `str` | `PreprocessResult` with `status="ready"` and `manifest["accepted_warnings"] = True`. Raises `invalid_state` if not `needs_review`. |
| `retake(image_id)` | `str` | `None`. Deletes every file for that image. |
| `get_manifest(image_id)` | `str` | the manifest `dict`. Raises `not_found`. |
| `load_processed(image_id)` | `str` | processed BGR `np.ndarray`. Raises `invalid_state` if not ready. |
| `cleanup(now=None)` | optional Unix time | `int`, the number of images deleted because they're older than `retention_hours` |

Image IDs must be 32 lowercase hex characters. Anything else is treated as `not_found`, so an ID can't be used to reach other files.

## `config.py`

**What:** `PreprocessingConfig`, a frozen dataclass of limits, plus constants.

| Setting | Default | Env override | Effect |
|---|---|---|---|
| `allowed_formats` | `("JPEG", "PNG")` | none | Any other type is rejected. |
| `max_bytes` | 20 MB | `PREPROCESS_MAX_BYTES` | The file must be smaller than this, or it's rejected. |
| `max_pixels` | 40,000,000 | `PREPROCESS_MAX_PIXELS` | Rejects larger images (guards against decompression bombs). |
| `min_side` | 64 px | `PREPROCESS_MIN_SIDE` | Rejects smaller images. |
| `min_blur` | 50.0 | `PREPROCESS_MIN_BLUR` | Below this, the user gets a `blurry` warning. |
| `min_brightness` | 60.0 | `PREPROCESS_MIN_BRIGHTNESS` | Below this, the user gets a `too_dark` warning. |
| `min_contrast` | 10.0 | `PREPROCESS_MIN_CONTRAST` | Below this, the user gets a `low_contrast` warning. |
| `retention_hours` | 24 | `PREPROCESS_RETENTION_HOURS` | `cleanup()` age limit. |
| `options` | `PreprocessOptions()` | none | Enhancement settings (deskew is off). |

The quality limits come from synthetic samples: a sharp sparse page scores about 131 for blur and 15.6 for contrast. **Tune them with real handwriting photos.**

## `schemas.py`

**What:** Types shared by callers.

- `Status`: `NEEDS_REVIEW = "needs_review"`, `READY = "ready"`.
- `ErrorCode` (rejections): `empty`, `too_large`, `unreadable`, `unsupported_type`, `too_many_pixels`, `too_small`, `not_found`, `invalid_state`.
- `WarningCode` (retake or continue): `blurry`, `too_dark`, `low_contrast`.
- `PreprocessingError(code, message)`: the exception for rejections. `message` can be shown to the user.
- `PreprocessResult`: `image_id`, `status`, `warnings` (`[{"code", "message"}]`), `manifest`, and `image` (`np.ndarray` or `None`).

## `__init__.py`

**What:** Re-exports `PreprocessingPipeline`, `PreprocessingConfig`, `PreprocessResult`, `PreprocessingError`, `Status`, `ErrorCode` and `WarningCode`.

## `manifest.json` (stored per image)

```json
{
  "image_id": "…", "pipeline_version": "1", "status": "ready",
  "created_at": "…", "updated_at": "…",
  "source":    {"key", "filename", "content_type", "bytes", "sha256", "width", "height"},
  "quality":   {"blur", "brightness", "contrast", "width", "height", "warnings": [...]},
  "options":   {...PreprocessOptions},
  "processed": {"key", "width", "height", "scale", "angle"} | null,
  "accepted_warnings": false
}
```

To map detection boxes back to the upright original, use `original_xy = processed_xy / processed.scale`.

## Notes

- **Nothing calls `cleanup()` automatically yet.** Call it from the API's startup or on a timer. With MinIO, bucket lifecycle rules can replace it.
- **The original is always kept,** so detection can compare the raw and processed versions.
