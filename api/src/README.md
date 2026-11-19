# src

Application source code for the detection API.

**Layers:**

```
server/             (HTTP API: web app → features)
     │
features/           (workflow level: combine base services for a use case)
     │
services/           (base level: one model or one OpenCV step per file)
     │
models              (PP-OCRv6_medium_det, OpenCV)
```

Calls only go downward. `features/` calls `services/`; `services/` never calls `features/`.

| Path | What it is | What it does |
|---|---|---|
| `__init__.py` | Package marker | Makes `src` importable as `api.src`. It has no code, input or output. |
| `features/` | Workflow features | One subfolder per feature, built on top of `services/`. See [`features/README.md`](features/README.md). |
| `server/` | HTTP API | FastAPI app that the web app calls: upload, then retake or continue, then the annotated image. See [`server/README.md`](server/README.md). |
| `services/` | Service modules | Image loading, OpenCV preprocessing, and PaddleOCR detection. See [`services/README.md`](services/README.md) for each file's input and output. |
| `test_features/` | Feature tests | One `unittest` file per feature. See [`test_features/README.md`](test_features/README.md). |
| `test_server/` | Server tests | Endpoint tests with FastAPI's `TestClient`. See [`test_server/README.md`](test_server/README.md). |
| `test_services/` | Service tests | One `unittest` file per service. See [`test_services/README.md`](test_services/README.md). |
