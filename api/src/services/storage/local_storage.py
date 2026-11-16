"""Local filesystem storage. Temporary stand-in for object storage (MinIO/S3)."""

import os
import tempfile
from pathlib import Path

_TMP_PREFIX = ".tmp-"


class LocalFileStorage:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Map a key to a path under root, rejecting anything that could escape it."""
        if not key or key.startswith("/") or "\\" in key:
            raise ValueError(f"invalid storage key: {key!r}")
        parts = key.split("/")
        if any(part in ("", ".", "..") or part.startswith(_TMP_PREFIX) for part in parts):
            raise ValueError(f"invalid storage key: {key!r}")
        return self.root.joinpath(*parts)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        # content_type is kept for interface parity with object storage; files don't store it.
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=_TMP_PREFIX)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic: readers never see a half-written file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        path = self._path(key)
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != self.root:  # drop empty folders left behind
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def list(self, prefix: str = "") -> list[str]:
        keys = []
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.startswith(_TMP_PREFIX):
                key = path.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def modified_at(self, key: str) -> float:
        return self._path(key).stat().st_mtime
