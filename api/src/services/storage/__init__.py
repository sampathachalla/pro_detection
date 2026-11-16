"""Storage backends. Choose with STORAGE_BACKEND (only "local" for now; MinIO later)."""

import os
import threading

from .base import Storage
from .local_storage import LocalFileStorage

_storage = None
_lock = threading.Lock()


def get_storage() -> Storage:
    """Shared storage instance configured from the environment."""
    global _storage
    if _storage is None:
        with _lock:
            if _storage is None:
                backend = os.environ.get("STORAGE_BACKEND", "local")
                if backend != "local":
                    raise ValueError(f"unsupported STORAGE_BACKEND: {backend!r}")
                _storage = LocalFileStorage(os.environ.get("STORAGE_ROOT", "/app/data"))
    return _storage


__all__ = ["Storage", "LocalFileStorage", "get_storage"]
