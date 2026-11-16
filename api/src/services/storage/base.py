"""Storage interface. Keys are object-storage style paths like "preprocessing/<id>/original.png"."""

from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Write (or overwrite) an object."""

    def get(self, key: str) -> bytes:
        """Read an object. Raises FileNotFoundError if missing."""

    def exists(self, key: str) -> bool:
        """True if the object exists."""

    def delete(self, key: str) -> None:
        """Delete an object. No error if missing."""

    def list(self, prefix: str = "") -> list[str]:
        """All keys starting with prefix, sorted."""

    def modified_at(self, key: str) -> float:
        """Last write time as a Unix timestamp. Raises FileNotFoundError if missing."""
