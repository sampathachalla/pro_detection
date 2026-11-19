import cv2
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

print(f"PaddleOCR detection results: {len(paddle_results)}")

