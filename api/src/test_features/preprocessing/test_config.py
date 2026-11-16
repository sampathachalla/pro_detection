import os
import unittest
from unittest import mock

from api.src.features.preprocessing import PreprocessingConfig


class TestPreprocessingConfig(unittest.TestCase):
    def test_defaults_match_requirements(self):
        cfg = PreprocessingConfig()
        self.assertEqual(cfg.allowed_formats, ("JPEG", "PNG"))
        self.assertEqual(cfg.max_bytes, 20 * 1024 * 1024)
        self.assertEqual(cfg.retention_hours, 24.0)
        self.assertFalse(cfg.options.deskew)

    def test_env_overrides(self):
        env = {
            "PREPROCESS_MAX_BYTES": "1000",
            "PREPROCESS_MIN_BLUR": "75.5",
            "PREPROCESS_RETENTION_HOURS": "2",
        }
        with mock.patch.dict(os.environ, env):
            cfg = PreprocessingConfig.from_env()
        self.assertEqual(cfg.max_bytes, 1000)
        self.assertEqual(cfg.min_blur, 75.5)
        self.assertEqual(cfg.retention_hours, 2.0)

    def test_from_env_without_overrides_uses_defaults(self):
        keys = [k for k in os.environ if k.startswith("PREPROCESS_")]
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in keys:
                del os.environ[key]
            self.assertEqual(PreprocessingConfig.from_env(), PreprocessingConfig())


if __name__ == "__main__":
    unittest.main()
