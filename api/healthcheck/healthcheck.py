"""Container health check: verifies the CPU-only OCR stack is usable.

Exits 0 when every check passes, 1 otherwise.
Usage (inside the container): python healthcheck.py
"""

import os
import sys
from pathlib import Path

MODEL_DIRS = {
    "paddlex models": os.environ.get("PADDLE_PDX_CACHE_HOME", "/app/models/paddlex"),
    "easyocr models": os.environ.get("EASYOCR_MODULE_PATH", "/app/models/easyocr"),
}


def check_models():
    for name, path in MODEL_DIRS.items():
        if not any(Path(path).rglob("*")):
            raise RuntimeError(f"{name} missing or empty: {path}")
    return ", ".join(MODEL_DIRS.values())


def check_paddle():
    import paddle

    if paddle.device.is_compiled_with_cuda():
        raise RuntimeError("paddle was built with CUDA")
    return f"paddle {paddle.__version__}, device={paddle.device.get_device()}"


def check_torch():
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("torch reports CUDA available")
    return f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"


def check_ocr_libs():
    import cv2
    import easyocr
    import paddleocr

    return f"easyocr {easyocr.__version__}, paddleocr {paddleocr.__version__}, cv2 {cv2.__version__}"


def check_runtime():
    if os.getuid() == 0:
        raise RuntimeError("running as root")
    return f"uid={os.getuid()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"


CHECKS = [
    ("models", check_models),
    ("paddle", check_paddle),
    ("torch", check_torch),
    ("ocr libs", check_ocr_libs),
    ("runtime", check_runtime),
]


def main():
    failed = 0
    for name, check in CHECKS:
        try:
            print(f"[PASS] {name}: {check()}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {exc}")
    print("HEALTHY" if not failed else f"UNHEALTHY ({failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
