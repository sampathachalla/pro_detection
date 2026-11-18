FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    PADDLE_PDX_CACHE_HOME=/app/models/paddlex \
    PADDLEOCR_HOME=/app/models/paddleocr \
    EASYOCR_MODULE_PATH=/app/models/easyocr \
    STORAGE_BACKEND=local \
    STORAGE_ROOT=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.15.0+cpu" \
        "torchvision==0.29.2+cpu" \
    && pip install --no-cache-dir \
        --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
        --extra-index-url https://pypi.org/simple \
        "paddlepaddle==3.2.0" \
    && pip install --no-cache-dir "paddleocr==3.7.0" \
    && pip uninstall -y opencv-contrib-python \
    && pip install --no-cache-dir "easyocr==1.7.2" \
    && pip install --no-cache-dir "opencv-python-headless==5.0.0.93" \
    && pip install --no-cache-dir "numpy==2.3.5" "pillow==13.0.0" \
    && pip install --no-cache-dir "fastapi==0.163.1" "uvicorn==0.54.0" "python-multipart==0.0.34"

RUN find /usr/local/lib/python3.13/site-packages/paddlex \
        -type f -name '*.py' \
        -exec sed -i 's/opencv-contrib-python/opencv-python-headless/g' {} +

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/models/paddlex /app/models/paddleocr /app/models/easyocr /app/data \
    && chown -R app:app /app

COPY --chown=app:app download_models.py test_ocr.py ./

USER app

RUN python download_models.py

# App code last, so code changes do not re-download the models.
COPY --chown=app:app api ./api

EXPOSE 8000

CMD ["uvicorn", "api.src.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

