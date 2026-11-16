# services

**Base-level model services.** These are the lowest layer: each one wraps a single model or a single OpenCV step. Higher-level workflow services call them; the base services never call workflows.

Rules for this layer:
- One responsibility per file: load or call one model, or do one processing step.
- No business or workflow logic, such as combining detectors, filtering results, or choosing preprocessing settings for a use case. That belongs in the workflow services.
- Plain Python with no web framework, so workflows, an API, a CLI or tests can all call it.
- The same simple input and output for every service: a BGR `np.ndarray` in, plain dicts and lists out.

Typical flow:

```
image bytes → load_image → preprocess → paddle_service.detect / easyocr_service.detect → boxes
```

Both detectors return boxes in the same shape:

```python
[{"points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], "score": float | None}, ...]
```

Points are integer pixel coordinates, clockwise from top-left.

---

## `__init__.py`

**What:** The package entry point.
**Does:** Re-exports the services so callers can write `from api.src.services import preprocess, paddle_service`.
**Exports:** `content_draw_service`, `easyocr_service`, `paddle_service`, `preprocess_service`, `quality_service`, `inspect_image`, `load_image`, `load_image_upright`, `PreprocessOptions`, `preprocess`, `Storage`, `LocalFileStorage`, `get_storage`.

---

## `image_loader.py`

**What:** Image decoder.
**Does:** Reads image headers and turns raw uploaded bytes into an OpenCV image. It uses OpenCV and Pillow.

| Function | Input | Output |
|---|---|---|
| `load_image(data)` | `data: bytes`, the raw file contents | `np.ndarray`, a BGR image of shape `(H, W, 3)`, `uint8`, decoded as stored |
| `inspect_image(data)` | `data: bytes` | `{"format": "JPEG" \| "PNG" \| ..., "width": int, "height": int}`, read from the header without decoding the pixels (size is before EXIF rotation) |
| `load_image_upright(data)` | `data: bytes` | BGR `np.ndarray`, rotated according to the EXIF orientation tag so phone photos come out upright |

**Errors:** All three raise `ValueError("empty image")` for empty bytes and `ValueError("not a valid image")` when the bytes can't be read.

---

## `preprocess_service.py`

**What:** OpenCV preprocessing for handwriting.
**Does:** Cleans an image before detection. It uses no model and downloads nothing.

Pipeline: resize → grayscale → denoise → contrast boost (CLAHE) → optional deskew → back to 3-channel BGR.

### `PreprocessOptions` (settings)

| Field | Default | Meaning |
|---|---|---|
| `max_side` | `1600` | Shrink so the longest side is at most this many pixels. It never enlarges. |
| `denoise` | `True` | Remove paper texture and camera noise. This is the slowest step. |
| `clahe` | `True` | Boost faint pen strokes. |
| `clahe_clip_limit` | `2.0` | CLAHE contrast limit. |
| `clahe_tile_size` | `8` | CLAHE grid size. |
| `deskew` | `False` | Rotate a tilted page level. Off by default because slanted handwriting can fool it. |

### Functions

| Function | Input | Output |
|---|---|---|
| `preprocess(image, options=None)` | BGR `np.ndarray` and optional `PreprocessOptions` | `(bgr_image, info)`, where `info = {"scale": float, "angle": float}` |
| `resize(image, max_side)` | image, int | `(image, scale)` |
| `to_gray(image)` | BGR or grayscale image | grayscale image |
| `denoise(gray)` | grayscale image | denoised grayscale image |
| `enhance_contrast(gray, clip_limit, tile_size)` | grayscale image, float, int | contrast-boosted grayscale image |
| `deskew(gray)` | grayscale image | `(rotated_gray, angle_degrees)` |

**Mapping boxes back to the original image:** `original_xy = detected_xy / info["scale"]`. If deskew ran, the boxes are in the rotated frame (`info["angle"]`).

---

## `quality_service.py`

**What:** Image quality measurements (OpenCV).
**Does:** Measures how usable an image is. It only measures; the calling feature decides what counts as too blurry or too dark.

| Function | Input | Output |
|---|---|---|
| `assess(image)` | BGR or grayscale `np.ndarray` | `{"blur": float, "brightness": float, "contrast": float, "width": int, "height": int}` |

- `blur`: variance of the Laplacian. Lower means blurrier. A sharp but sparse handwriting page scores about 130; a heavily blurred one scores under 15.
- `brightness`: mean gray level from 0 to 255. Lower means darker.
- `contrast`: standard deviation of the gray levels. Lower means flatter or fainter.

---

## `content_draw_service.py`

**What:** Border drawing (OpenCV).
**Does:** Draws a closed polygon border around each box on a copy of the image. It doesn't care what a box represents (a block, line or word), so the same service will draw word borders later.

| Function | Input | Output |
|---|---|---|
| `draw_boxes(image, boxes, color=(0, 200, 0), thickness=None, labels=False)` | BGR or grayscale image; `[{"points": [[x, y], ...] (3 or more points), ...}]`, where other keys are ignored; BGR colour; width in px; `labels` numbers the boxes 1, 2, 3 | a **new** BGR image with borders drawn; the input isn't changed |
| `default_thickness(image)` | image | `int`: 2 px, plus 1 for each 500 px of the longest side |

**Errors:** Raises `ValueError` if a box has fewer than 3 points or the points aren't `[x, y]` pairs.

---

## `trocr_service.py`

**What:** Handwriting recognition with **TrOCR-small-handwritten** (Microsoft; `transformers` + PyTorch).
**Does:** Reads **one line** of handwriting and returns its text. Weights are built into the image at `TROCR_MODEL_DIR` (`/app/models/trocr/trocr-small-handwritten`, 236 MB) and loaded with `local_files_only`. `HF_HUB_OFFLINE=1` in `compose.yaml` means it never downloads. It loads once and is reused (thread-safe). About 1 s per line.

| Function | Input | Output |
|---|---|---|
| `recognize(image)` | BGR or grayscale crop of **one line** | `{"text": str, "confidence": float}` (0 to 1, the mean probability of the generated tokens; low for junk such as drawings) |
| `get_model()` | none | `(TrOCRProcessor, VisionEncoderDecoderModel)` |

**Dependency note:** TrOCR-small's tokenizer needs **protobuf 7.36 or later**. PaddlePaddle installs 6.33, so the Dockerfile pins `protobuf==7.36.2`. PaddleOCR was checked with it: all 108 tests passed.

---

## `word_segmentation_service.py`

**What:** Word splitting for one line of handwriting (OpenCV, no model).
**Does:** Separates ink from paper (Otsu thresholding, specks removed), then splits at runs of empty columns wider than `gap_ratio` × the text height.

| Function | Input | Output |
|---|---|---|
| `segment_words(crop, gap_ratio=0.35, min_ink_fraction=0.002)` | BGR or grayscale crop of **one line** (dark text on light paper) | `[[x0, y0, x1, y1], ...]` word boxes in crop coordinates, left to right; `[]` if there's almost no ink |
| `ink_mask(crop)` | crop | `uint8` mask (255 = ink) with specks smaller than about 0.15% of the height² removed |

---

## `storage/`

**What:** File storage behind one interface. It's local disk for now and will be swapped for object storage (likely MinIO) later. See [`storage/README.md`](storage/README.md).

---

## `paddle_service.py`

**What:** PaddleOCR text detector.
**Does:** Finds text regions with `PP-OCRv6_medium_det`. It only detects; it doesn't read the text. The model is baked into the image under `PADDLE_PDX_CACHE_HOME`, loads on the first call and is then reused (thread-safe).

| Function | Input | Output |
|---|---|---|
| `detect(image)` | BGR `np.ndarray` | list of `{"points": [[x, y] x4], "score": float}` |
| `get_detector()` | none | the shared `paddleocr.TextDetection` instance |

---

## `easyocr_service.py`

**What:** EasyOCR text detector.
**Does:** Finds text regions with EasyOCR's `reader.detect` and the English model. It only detects; it doesn't read the text. Downloads are off (`download_enabled=False`), so it fails instead of downloading if the weights are missing. The weights come from `EASYOCR_MODULE_PATH`. The model loads on the first call and is then reused (thread-safe).

| Function | Input | Output |
|---|---|---|
| `detect(image, width_ths=0.1)` | BGR `np.ndarray`; `width_ths` is EasyOCR's box-merge threshold (lower keeps words apart) | list of `{"points": [[x, y] x4], "score": None}` |
| `get_reader()` | none | the shared `easyocr.Reader` instance |

`score` is always `None` because EasyOCR detection doesn't return confidence values.
