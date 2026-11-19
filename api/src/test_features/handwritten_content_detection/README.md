# test_features/handwritten_content_detection

Tests for [`features/handwritten_content_detection/`](../../features/handwritten_content_detection/README.md). They use the built-in `unittest` module, with a temporary folder as storage.

| File | What it tests | Input | Passes when |
|---|---|---|---|
| `__init__.py` | Package marker | none | Makes the tests importable |
| `test_pipeline.py` | `HandwrittenContentDetector`, `map_to_original`, `reading_order` | Synthetic images that preprocessing has marked ready; fake detectors (`unittest.mock`) for the fast tests; the real models for one test | See the table below |

| Group | Case | Expected |
|---|---|---|
| Fake models (fast) | Paddle finds a box | `detector="paddle"`, green border drawn, `annotated.png` and JSON stored |
| | Neither finds anything | `detector="none"`, count 0, image unchanged |
| | 3200 px image (scale 0.5) | Boxes doubled back to original coordinates; annotated image is the original size |
| | Tiny box | Dropped |
| | Shuffled boxes on two lines | Returned in reading order |
| | `get_result` / `load_annotated` | Match what `detect` returned |
| | `retake` after detection | Detection files are deleted too |
| Errors | Unknown or unsafe ID | `not_found` |
| | Image in `needs_review` | `not_ready` |
| | `get_result` before `detect` | `not_found` |
| Helpers | `map_to_original` | Undoes scale and deskew; clips to the image |
| | `reading_order` | Empty list works; sorts by line, then left to right |
| Real models (slow, ~20 s) | Synthetic handwriting sample | At least one box, inside the image, from Paddle |

Most of these tests turn the preprocessing quality warnings off so synthetic images go straight to `ready`. The `not_ready` test uses the default limits.

## How to run

From the project root, inside the container:

```bash
python -m unittest discover -s api/src/test_features/handwritten_content_detection -t . -v
```
