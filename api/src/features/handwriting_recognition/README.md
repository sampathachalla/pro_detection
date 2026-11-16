# handwriting_recognition

**Feature goal:** read what's written inside each handwritten line found by [`handwritten_content_detection`](../handwritten_content_detection/README.md), using **TrOCR-small-handwritten** (Microsoft).

It runs on demand (the page's **Read text** button, `POST /api/images/{id}/read`), not on every upload, because TrOCR takes about **1 second per line** on the 1-CPU container (17.5 s for a 16-line photo). Reading lines in a batch was no faster.

## Workflow

```
read(image_id)                       ← handwritten content detection must have run
 1. Load    detected line regions + original upright photo
 2. Crop    each line (word_box_draw.crop_line: masked to the polygon, outside = paper colour)
 3. Read    trocr_service.recognize → {"text", "confidence"}, one line at a time
 4. Flag    confidence < min_confidence, or no letters/digits (e.g. an arrow read as "#",
            which scored 0.54) → low_confidence (left out of "text")
 5. Store   preprocessing/<id>/recognized_text.json
```

The file sits next to the image's other files, so `retake()` and the 24-hour cleanup remove it.

## `pipeline.py`

| Method | Input | Output |
|---|---|---|
| `HandwritingRecognizer(detector=None, config=None)` | a `HandwrittenContentDetector` (shares its storage) and `RecognitionConfig` | the recognizer |
| `read(image_id)` | `str` | the stored result `dict` (below). Raises `DetectionError("not_found")` if detection hasn't run. |
| `get_result(image_id)` | `str` | stored `recognized_text.json`, or `None` if not read yet |

## `config.py`

| Setting | Default | Env override | Effect |
|---|---|---|---|
| `min_confidence` | 0.5 | `RECOGNIZE_MIN_CONFIDENCE` | Lines below this mean token probability are flagged `low_confidence` (for example a drawn arrow read as `#`) and left out of `text`. |

Constants: `MODEL_NAME = "microsoft/trocr-small-handwritten"` and `RESULT_FILENAME = "recognized_text.json"`.

## `recognized_text.json`

```json
{
  "image_id": "…", "recognition_version": "1", "model": "microsoft/trocr-small-handwritten",
  "created_at": "…", "seconds": 17.5,
  "lines": [{"line": 10, "bbox": [x0, y0, x1, y1], "text": "This is the test-I to", "confidence": 0.81, "low_confidence": false}],
  "text": "container\n…\nthe photo taken"
}
```

`line` is the index into the detection's `regions` (reading order).

## Accuracy on a real photo (16 lines)

TrOCR-small read **7 lines exactly right** ("container", "services", "handwritten", "box", "detect handwritings in", "the photo taken", and "This is the test-I to" apart from 1 → I). It got **5 close**: "detect nondwitting and", "used for production", "software affiliation to", and "disservices" for "Services". It got **3 wrong** on short or unusual words: "monters" for "models", "spc" for "src", "over-performing" for "preprocessing". The arrow came out as `#`.

**To improve:** `trocr-base-handwritten` (about 1.3 GB, slower) is more accurate. Fine-tuning on your own handwriting helps most.
