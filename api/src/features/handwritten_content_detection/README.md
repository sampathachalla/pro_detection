# handwritten_content_detection

**Feature goal:** draw borders around the handwritten content in an uploaded image. It returns the annotated image and the box coordinates.

It runs after the [`preprocessing`](../preprocessing/README.md) feature: the image must be `ready`. It uses these base services: `paddle_service`, `easyocr_service`, `content_draw_service` and `image_loader`, plus storage through the preprocessing pipeline.

Uploads are assumed to be **handwriting only**, so every text region found counts as handwritten content. The two detectors can't tell handwriting from print.

## Workflow

```
detect(image_id)
 1. Check     preprocessing status is "ready"   → else not_ready / not_found
 2. Detect    PaddleOCR PP-OCRv6 on the processed image
              → nothing found? EasyOCR (width_ths=0.1)
              → still nothing? detector = "none", 0 boxes
 3. Map back  processed coordinates → original upright photo (undo deskew, divide by scale, clip)
 4. Clean up  drop boxes smaller than min_box_area, sort into reading order
 5. Draw      green polygon borders on the ORIGINAL photo (content_draw_service)
 6. Store     preprocessing/<id>/annotated.png
              preprocessing/<id>/handwriting_detection.json
```

Results are stored next to the preprocessing files, so `retake()` and the 24-hour `cleanup()` remove them too. Running `detect` again overwrites them.

---

## `pipeline.py`

**What:** `HandwrittenContentDetector(preprocessing=None, config=None)`. It defaults to a new `PreprocessingPipeline()` and `DetectionConfig.from_env()`, and uses the same storage as the preprocessing pipeline.

| Method / function | Input | Output |
|---|---|---|
| `detect(image_id)` | `str`, a `ready` image | `DetectionResult` (annotated image, regions, detector used, stored JSON). Raises `DetectionError` `not_found` or `not_ready`. |
| `get_result(image_id)` | `str` | the stored `handwriting_detection.json` as a `dict`. Raises `not_found` if detection hasn't run. |
| `load_annotated(image_id)` | `str` | the annotated BGR `np.ndarray` |
| `map_to_original(points, processed, source_width, source_height)` | points on the processed image, `manifest["processed"]`, original size | integer points on the original upright image |
| `reading_order(regions, line_tolerance)` | regions with `bbox` | the same regions sorted top to bottom by line, then left to right |

## `config.py`

**What:** `DetectionConfig` (frozen dataclass).

| Setting | Default | Env override | Effect |
|---|---|---|---|
| `min_box_area` | 50 px² | `DETECT_MIN_BOX_AREA` | Smaller boxes on the original image are dropped as noise. |
| `easyocr_width_ths` | 0.1 | `DETECT_EASYOCR_WIDTH_THS` | EasyOCR fallback merge threshold. |
| `line_tolerance` | 0.5 | `DETECT_LINE_TOLERANCE` | Boxes whose centres are less than this × the median box height apart count as the same line, for reading order. |
| `color` | `(0, 200, 0)` green (BGR) | none | Border colour. |
| `labels` | `True` | `DETECT_LABELS=false` | Number the boxes 1, 2, 3 in reading order. |

## `schemas.py`

- `Detector`: `"paddle"`, `"easyocr"` or `"none"`.
- `DetectionErrorCode`: `not_found` (unknown image, or detection not run yet) and `not_ready` (waiting for retake or continue).
- `DetectionError(code, message)`: the exception. `message` can be shown to the user.
- `DetectionResult`: `image_id`, `detector`, `regions`, `annotated` (`np.ndarray`), and `result` (the stored JSON).

## `__init__.py`

Re-exports everything above.

## `handwriting_detection.json`

```json
{
  "image_id": "…", "detection_version": "1", "created_at": "…",
  "detector": "paddle", "count": 3,
  "image": {"width": 3200, "height": 800},
  "annotated_key": "preprocessing/<id>/annotated.png",
  "regions": [
    {"points": [[x, y], [x, y], [x, y], [x, y]], "bbox": [x0, y0, x1, y1], "score": 0.93}
  ]
}
```

All coordinates are on the **original upright photo**. `score` is `null` when EasyOCR found the box.

## Later: word borders

`content_draw_service` already draws any list of boxes. Word borders only need a word-level box source, such as EasyOCR with a low `width_ths` or splitting Paddle's line boxes at the gaps between words. That would be added as a detection option here.
