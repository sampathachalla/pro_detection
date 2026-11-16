import os

import cv2
import easyocr
import numpy as np
from paddleocr import TextDetection


image = np.full((180, 720, 3), 255, dtype=np.uint8)
cv2.putText(
    image,
    "Handwritten text 123",
    (24, 110),
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    1.8,
    (0, 0, 0),
    3,
    cv2.LINE_AA,
)
sample_path = "/tmp/handwriting-sample.png"
cv2.imwrite(sample_path, image)

paddle_detector = TextDetection(model_name="PP-OCRv6_medium_det")
paddle_results = list(paddle_detector.predict(sample_path))

easy_reader = easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory=os.environ["EASYOCR_MODULE_PATH"],
    download_enabled=False,
)
easy_horizontal, easy_free = easy_reader.detect(image, width_ths=0.1)

print(f"PaddleOCR detection results: {len(paddle_results)}")
print(
    "EasyOCR detection boxes: "
    f"{sum(map(len, easy_horizontal)) + sum(map(len, easy_free))}"
)

