import os

import easyocr
from huggingface_hub import snapshot_download
from paddleocr import TextDetection


TextDetection(model_name="PP-OCRv6_medium_det")
easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory=os.environ["EASYOCR_MODULE_PATH"],
    download_enabled=True,
)
snapshot_download(
    "microsoft/trocr-small-handwritten",
    local_dir=os.environ["TROCR_MODEL_DIR"],
    allow_patterns=["*.json", "*.bin", "*.model"],
)
