import tempfile
import unittest
from pathlib import Path

from api.src.services.storage import LocalFileStorage


class TestLocalFileStorage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.storage = LocalFileStorage(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_put_get_roundtrip(self):
        self.storage.put("a/b/file.bin", b"hello")
        self.assertEqual(self.storage.get("a/b/file.bin"), b"hello")
        self.assertTrue((self.root / "a" / "b" / "file.bin").is_file())

    def test_put_overwrites(self):
        self.storage.put("k.txt", b"one")
        self.storage.put("k.txt", b"two")
        self.assertEqual(self.storage.get("k.txt"), b"two")

    def test_exists_and_delete(self):
        self.storage.put("x/y.txt", b"1")
        self.assertTrue(self.storage.exists("x/y.txt"))
        self.storage.delete("x/y.txt")
        self.assertFalse(self.storage.exists("x/y.txt"))
        self.assertFalse((self.root / "x").exists(), "empty folder should be removed")
        self.storage.delete("x/y.txt")  # deleting again is not an error

    def test_get_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.storage.get("missing.txt")

    def test_list_by_prefix(self):
        for key in ("p/1/a", "p/1/b", "p/2/a", "q/1/a"):
            self.storage.put(key, b"")
        self.assertEqual(self.storage.list("p/1/"), ["p/1/a", "p/1/b"])
        self.assertEqual(len(self.storage.list()), 4)

    def test_no_temp_files_left_behind(self):
        self.storage.put("d/f", b"data")
        self.assertEqual([p.name for p in (self.root / "d").iterdir()], ["f"])

    def test_modified_at(self):
        self.storage.put("m", b"")
        self.assertGreater(self.storage.modified_at("m"), 0)

    def test_rejects_unsafe_keys(self):
        for key in ("", "/abs", "../escape", "a/../b", "a//b", "a\\b", "./a", "a/.tmp-x"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.storage.put(key, b"x")


if __name__ == "__main__":
    unittest.main()
