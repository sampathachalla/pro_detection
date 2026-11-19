# word_box_draw

**Feature goal:** draw a border around **every word** inside the handwritten lines found by [`handwritten_content_detection`](../handwritten_content_detection/README.md).

It uses no model. OpenCV splits each line at the gaps between words (`services/word_segmentation_service.py`), so it takes milliseconds (88 ms for a 16-line photo).

## Workflow

```
draw(image_id)                       ← handwritten content detection must have run
 1. Load      detected line regions + original upright photo
 2. Crop      each line's bounding box; outside the line polygon → paper colour
 3. Segment   remove ruled paper lines, then gap > gap_ratio × small-letter height = new word (word_segmentation_service)
 4. Map back  crop coordinates → photo coordinates; drop boxes < min_word_area
 5. Draw      green PaddleOCR line borders, then thinner blue word borders on top (content_draw_service)
 6. Store     preprocessing/<id>/words_annotated.jpg + word_boxes.json
```

Files sit next to the image's other files, so `retake()` and the 24-hour cleanup remove them. The server runs this automatically after line detection.

**Why the paper colour, not white (step 2):** a tilted line box leaves corners outside the polygon. Pure-white corners make Otsu thresholding split white from paper and treat the paper as ink, so a whole line becomes one "word". Filling with the median paper colour avoids that.

## `pipeline.py`

| Method / function | Input | Output |
|---|---|---|
| `WordBoxDrawer(detector=None, config=None)` | a `HandwrittenContentDetector` (shares its storage) and `WordBoxConfig` | the drawer |
| `draw(image_id)` | `str` | `WordBoxResult`. Raises `DetectionError("not_found")` if detection hasn't run. |
| `get_result(image_id)` | `str` | stored `word_boxes.json` as a `dict`, or `None` if words haven't been drawn |
| `load_annotated(image_id)` | `str` | BGR `np.ndarray` with word borders, or `None` |
| `crop_line(image, points)` | photo, line polygon | `(crop, x_offset, y_offset)`, with outside the polygon filled with the paper colour |

## `config.py`

| Setting | Default | Env override | Effect |
|---|---|---|---|
| `gap_ratio` | 0.45 | `WORDS_GAP_RATIO` | Gap wider than this × small-letter height starts a new word. Lower splits more. |
| `min_word_area` | 30 px² | `WORDS_MIN_AREA` | Smaller boxes are dropped. |
| `color` | `(230, 120, 0)` blue (BGR) | none | Word border colour, distinct from the green line borders. |
| `show_lines` | `True` | `WORDS_SHOW_LINES=false` | Also draw the green PaddleOCR line borders under the word borders. |

**Calibrated on a cursive page** (4 lines on ruled paper): small letters are 23–26 px tall, gaps between letters are at most 6 px, and gaps between words are 15–37 px. `0.45` puts the limit at 10–12 px and finds every word (5 per line). The old rule (`0.35` × full ink height) put the limit at about 34 px, because tall letters and tails made the height ~100 px, so each line split into only 2 boxes.

## `schemas.py`

`WordBoxResult`: `image_id`, `words` (`[{"line", "points", "bbox"}]`), `annotated` (`np.ndarray`), and `result` (the stored JSON).

## `word_boxes.json`

```json
{
  "image_id": "…", "words_version": "4", "created_at": "…",
  "count": 31, "lines": 16,
  "annotated_key": "preprocessing/<id>/words_annotated.jpg",
  "words": [{"line": 10, "points": [[x, y] x4], "bbox": [x0, y0, x1, y1]}]
}
```

`line` is the index into the detection's `regions` (reading order). Coordinates are on the original upright photo.

## Limits

- **Joined-up handwriting** with tiny gaps between words can merge words. Very wide letter spacing can split one.
- **Words that overlap vertically** (a slanted letter reaching over the next word) remove the gap between them.
- **Non-text boxes** (like a drawn arrow) still get one "word" box.
