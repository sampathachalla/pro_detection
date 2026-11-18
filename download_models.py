import os

import easyocr
from paddleocr import DocImgOrientationClassification, TextDetection


TextDetection(model_name="PP-OCRv6_medium_det")
DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory=os.environ["EASYOCR_MODULE_PATH"],
    download_enabled=True,
)
