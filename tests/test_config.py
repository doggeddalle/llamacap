import tempfile
import unittest
from pathlib import Path

from llamacap.config import load_config
from llamacap.errors import LlamacapError


class ConfigTests(unittest.TestCase):
    def _config(self, text: str):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def test_defaults_load_from_empty_config(self):
        config = self._config("")
        self.assertEqual(config.generation.timeout_seconds, 300)
        self.assertEqual(config.output.default_dir, "captions")

    def test_rejects_negative_resize_target(self):
        with self.assertRaisesRegex(LlamacapError, "zero or greater"):
            self._config("[preprocessing]\nresize_megapixels = -1\n")

    def test_rejects_wrong_value_types_cleanly(self):
        with self.assertRaisesRegex(LlamacapError, "Malformed config"):
            self._config('[generation]\ntimeout_seconds = "slow"\n')

    def test_rejects_non_positive_timeout(self):
        with self.assertRaisesRegex(LlamacapError, "must be greater than zero"):
            self._config("[generation]\ntimeout_seconds = 0\n")


if __name__ == "__main__":
    unittest.main()
