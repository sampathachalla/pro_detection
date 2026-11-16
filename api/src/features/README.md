# features

Workflow-level features. Each subfolder is one feature that combines the base services in [`../services/`](../services/README.md) for a specific use case.

Calls only go downward: features call `services/`, and `services/` never calls features.

| Path | What it is | What it does |
|---|---|---|
| `__init__.py` | Package marker | Makes `features` importable as `api.src.features`. It has no code, input or output. |
| `preprocessing/` | Preprocessing feature | Checks an uploaded image, warns about poor quality (retake or continue), cleans it for detection, and stores the original, processed image and manifest. See [`preprocessing/README.md`](preprocessing/README.md). |
| `handwriting_recognition/` | Text reading feature | Reads each detected line with TrOCR-small-handwritten (on demand, about 1 s per line). See [`handwriting_recognition/README.md`](handwriting_recognition/README.md). |
| `word_box_draw/` | Word border feature | Splits each detected line into words (OpenCV gaps, no model) and draws blue word borders. See [`word_box_draw/README.md`](word_box_draw/README.md). |
| `handwritten_content_detection/` | Handwritten content detection feature | Draws borders around the handwriting in a ready image (PaddleOCR, falling back to EasyOCR), and stores the annotated image and box coordinates. See [`handwritten_content_detection/README.md`](handwritten_content_detection/README.md). |
