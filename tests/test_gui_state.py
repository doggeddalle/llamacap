import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from llamacap.gui.state import (
    GuiPreferences,
    format_duration,
    summarize_dataset,
    validate_number,
)


class GuiStateTests(unittest.TestCase):
    def test_preferences_round_trip_and_ignore_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui.json"
            GuiPreferences(profile="krea2", recursive=True).save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["future_option"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = GuiPreferences.load(path)
            self.assertEqual(loaded.profile, "krea2")
            self.assertTrue(loaded.recursive)

    def test_dataset_summary_respects_separate_output_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "captions"
            (dataset / "nested").mkdir(parents=True)
            (output / "nested").mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(dataset / "nested" / "one.png")
            (output / "nested" / "one.txt").write_text("caption", encoding="utf-8")
            summary = summarize_dataset(dataset, True, output_dir=output)
            self.assertEqual(summary.existing, 1)
            self.assertEqual(summary.pending, 0)

    def test_inline_numeric_validation(self):
        self.assertIsNone(validate_number("0", "Resize", integer=False, allow_zero=True))
        self.assertIn("greater", validate_number("0", "Limit", integer=True, allow_zero=False))
        self.assertIn("finite", validate_number("nan", "Resize", integer=False, allow_zero=True))

    def test_duration_formatting(self):
        self.assertEqual(format_duration(65), "1:05")
        self.assertEqual(format_duration(3661), "1:01:01")


if __name__ == "__main__":
    unittest.main()
