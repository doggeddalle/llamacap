import tempfile
import unittest
from pathlib import Path

from llamacap.sidecar import sidecar_path_for, write_caption


class SidecarTests(unittest.TestCase):
    def test_output_directory_preserves_relative_layout(self):
        root = Path("dataset")
        self.assertEqual(
            sidecar_path_for(root / "people" / "one.jpg", root, Path("captions"), ".txt"),
            Path("captions/people/one.txt"),
        )

    def test_write_caption_replaces_contents_without_temp_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "image.txt"
            write_caption(path, "first")
            write_caption(path, "replacement")
            self.assertEqual(path.read_text(encoding="utf-8"), "replacement\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
