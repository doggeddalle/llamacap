import tempfile
import unittest
from pathlib import Path

from PIL import Image

from llamacap.resize import resize_for_analysis


class ResizeTests(unittest.TestCase):
    def test_applies_exif_orientation_before_resizing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "phone.jpg"
            image = Image.new("RGB", (40, 20), "red")
            exif = image.getexif()
            exif[274] = 6  # Rotate 90 degrees clockwise for display.
            image.save(source, exif=exif)

            output_dir = root / "out"
            output_dir.mkdir()
            output = resize_for_analysis(source, 0.0002, output_dir)
            with Image.open(output) as resized:
                self.assertGreater(resized.height, resized.width)


if __name__ == "__main__":
    unittest.main()
