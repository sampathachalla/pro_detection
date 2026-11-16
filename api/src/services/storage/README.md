# storage

Base-level file storage behind one interface. It uses **local disk for now**; later an object storage backend (likely **MinIO**, which speaks the S3 API) will implement the same interface, and callers won't change.

Keys look like object storage paths, for example `preprocessing/<image_id>/original.png`. On local disk, the key becomes a file path under `STORAGE_ROOT`.

| File | What it is | What it does |
|---|---|---|
| `__init__.py` | Entry point | `get_storage()` returns one shared storage instance chosen by `STORAGE_BACKEND` (only `local` for now) and `STORAGE_ROOT` (default `/app/data`). It also exports `Storage` and `LocalFileStorage`. |
| `base.py` | `Storage` interface | The methods every backend must provide (table below). |
| `local_storage.py` | `LocalFileStorage(root)` | Stores objects as files under `root`. |

## `Storage` methods

| Method | Input | Output |
|---|---|---|
| `put(key, data, content_type="application/octet-stream")` | `str`, `bytes`, `str` | `None`. Creates or overwrites. |
| `get(key)` | `str` | `bytes`. Raises `FileNotFoundError` if missing. |
| `exists(key)` | `str` | `bool` |
| `delete(key)` | `str` | `None`. No error if missing. |
| `list(prefix="")` | `str` | `list[str]`, the matching keys in sorted order |
| `modified_at(key)` | `str` | `float`, the Unix time of the last write. Raises `FileNotFoundError` if missing. |

## `LocalFileStorage` details

- **Safe keys:** it rejects empty keys, absolute paths, `..`, `.`, empty segments and backslashes with `ValueError`, so no key can reach outside `root`.
- **Atomic writes:** it writes to a hidden `.tmp-*` file and then renames it, so a reader never sees a half-written file.
- **Tidy deletes:** deleting a file also removes folders left empty, up to `root`.
- **`content_type`** is accepted for parity with object storage but isn't saved on disk.

## Moving to MinIO later

Add `minio_storage.py` with the same six methods, using the `minio` or `boto3` client. Add a `STORAGE_BACKEND=minio` branch in `get_storage()`, with endpoint, bucket and credentials from environment variables, never in code. MinIO's bucket lifecycle rules can then handle the 24-hour expiry instead of `cleanup()`.
