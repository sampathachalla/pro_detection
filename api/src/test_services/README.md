# test_services

Tests for `api/src/services/`, one file per service. They use Python's built-in `unittest`, so the image needs no extra packages. Test images are drawn in code, so no sample files or downloads are needed.

| File | What it tests | Input | Output / passes when |
|---|---|---|---|
| `__init__.py` | Package marker | none | Makes the tests importable as `api.src.test_services` |
| `sample_images.py` | Helper, not a test. `text_image()` draws a white BGR image with black script text; `png_bytes(image)` encodes it as PNG. | width, height, text | `np.ndarray` / `bytes` |
| `test_image_loader.py` | `image_loader`: `load_image`, `inspect_image`, `load_image_upright` | PNG bytes, a JPEG with an EXIF rotation tag, empty bytes, garbage bytes | Decodes the PNG exactly; reads format and size from the header; rotates the EXIF image upright; raises `ValueError` for empty or invalid input |
| `test_quality_service.py` | `quality_service.assess` | Sharp, blurred, dark, flat and grayscale images | Returns all metrics with the correct size; blurred scores lower blur, dark scores lower brightness, flat has zero contrast |
| `test_content_draw_service.py` | `content_draw_service.draw_boxes` | White images and boxes | Draws a border in the right colour and leaves the inside untouched; doesn't change the input; no boxes gives an equal copy; custom colour works; grayscale input becomes BGR; polygons with more than 4 points and extra keys work; labels add ink; thickness grows with size; bad points raise `ValueError` |
| `test_word_segmentation_service.py` | `word_segmentation_service` | Lines drawn word by word | Three words give three boxes, left to right, at the right x; letters of one word stay together; a large `gap_ratio` merges words; a blank crop gives `[]`; grey paper isn't ink; dust specks are ignored; boxes stay inside the crop; grayscale works |
| `test_trocr_service.py` | `trocr_service` (needs the TrOCR weights; skipped elsewhere) | A drawn line "hello world" | Text contains "hello" with confidence above 0.5; grayscale works; the model loads once |
| `test_local_storage.py` | `storage.LocalFileStorage` | Keys and bytes in a temporary folder | put/get round trip, overwrite, exists/delete (removes empty folders), missing key raises `FileNotFoundError`, listing by prefix, no temporary files left behind, unsafe keys such as `../` rejected |
| `test_preprocess_service.py` | `preprocess_service` | Synthetic text images | Output is BGR `uint8`; large images shrink to `max_side` with the correct `scale`; small images aren't enlarged; steps can be turned off; `to_gray` accepts gray or colour; `deskew` notices a 10° tilt and leaves a blank image unchanged |
| `test_paddle_service.py` | `paddle_service.detect` (needs model weights) | Synthetic text image, raw and preprocessed | At least one box; each has 4 integer points and a `float` score between 0 and 1; the detector loads only once |
| `test_easyocr_service.py` | `easyocr_service.detect` (needs model weights) | Synthetic text image, raw and preprocessed | At least one box; each has 4 integer points and `score` is `None`; the reader loads only once |

## How to run

Run them from the project root (the folder that contains `api/`). The Paddle and EasyOCR tests need the baked-in models, so run them inside the container:

```bash
# all service tests
python -m unittest discover -s api/src/test_services -t . -v

# one service
python -m unittest api.src.test_services.test_preprocess_service -v
```

`api/` has to be in the container first, either through a `COPY api/ ./api/` line in the `Dockerfile` or a volume mount. The container has no network, so a test that tries to download a model fails instead of downloading quietly.
