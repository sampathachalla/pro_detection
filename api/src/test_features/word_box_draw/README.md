# test_features/word_box_draw

Tests for [`features/word_box_draw/`](../../features/word_box_draw/README.md). They use the built-in `unittest` module, a temporary folder as storage, and a fake PaddleOCR that returns one line box around three drawn words.

| File | What it tests | Passes when |
|---|---|---|
| `__init__.py` | Package marker | none |
| `test_pipeline.py` | `WordBoxDrawer`, `crop_line` | **Three words** give three blue word boxes on line 0 with the green line border underneath at the right x positions in photo coordinates, plus stored `words_annotated.jpg` and `word_boxes.json`. **`show_lines=False`** draws no green. **A tilted line box** still gives 3 words (regression test for the white-corner bug). **`get_result` / `load_annotated`** match `draw`. **Before `draw`,** `get_result` is `None`. **Without detection,** `DetectionError("not_found")`. **`retake`** deletes the word files. **`crop_line`** fills outside the polygon with the paper colour, not white. |

## How to run

```bash
python -m unittest discover -s api/src/test_features/word_box_draw -t . -v
```
