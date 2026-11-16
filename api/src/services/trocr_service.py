"""TrOCR handwriting recognition (microsoft/trocr-small-handwritten): image of one text line -> text.

The model is baked into the image at TROCR_MODEL_DIR and loaded with local_files_only,
so it never downloads at runtime. ~1 s per line on the 1-CPU container.
"""

import os
import threading

import cv2
import numpy as np

DEFAULT_MODEL_DIR = "/app/models/trocr/trocr-small-handwritten"
MAX_NEW_TOKENS = 48

_model = None
_processor = None
_lock = threading.Lock()


def get_model():
    """Load the processor and model once and reuse them. Returns (processor, model)."""
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                path = os.environ.get("TROCR_MODEL_DIR", DEFAULT_MODEL_DIR)
                _processor = TrOCRProcessor.from_pretrained(path, local_files_only=True)
                _model = VisionEncoderDecoderModel.from_pretrained(path, local_files_only=True).eval()
    return _processor, _model


def recognize(image: np.ndarray) -> dict:
    """Read one line of handwriting.

    image: BGR (or grayscale) crop of a single text line.
    Returns {"text": str, "confidence": float 0-1}; confidence is the mean
    probability of the generated tokens (low for junk such as drawings).
    """
    import torch
    from PIL import Image

    processor, model = get_model()
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
    with torch.inference_mode():
        out = model.generate(
            pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    confidence = float(torch.exp(scores[0]).mean()) if scores.numel() else 0.0
    return {"text": text, "confidence": confidence}
