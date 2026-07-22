import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llamacap.cli import build_parser, main
from llamacap.config import load_config
from llamacap.runner import BatchResult


class CliParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_boolean_options_are_unset_by_default(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.overwrite)
        self.assertIsNone(args.recursive)

    def test_boolean_options_can_explicitly_disable_config_defaults(self):
        args = self.parser.parse_args(["--no-overwrite", "--no-recursive"])
        self.assertFalse(args.overwrite)
        self.assertFalse(args.recursive)

    def test_limit_must_be_positive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--limit", "0"])

    def test_size_cannot_be_negative(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--size", "-0.1"])

    def test_size_must_be_finite(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--size", "nan"])

    def test_exact_model_pair_options_are_available_to_the_gui(self):
        args = self.parser.parse_args(
            ["--model-gguf", "main.gguf", "--model-mmproj", "mmproj.gguf"]
        )
        self.assertEqual(args.model_gguf.name, "main.gguf")
        self.assertEqual(args.model_mmproj.name, "mmproj.gguf")

    def test_single_image_and_exact_pair_reach_batch_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.png"
            gguf, mmproj = root / "main.gguf", root / "mmproj.gguf"
            for path in (image, gguf, mmproj):
                path.touch()
            with patch("llamacap.cli.load_config", return_value=load_config()), patch(
                "llamacap.cli.run_batch", return_value=BatchResult()
            ) as run_batch:
                result = main([
                    "--profile", "krea2", "--input", str(root), "--image", str(image),
                    "--model-gguf", str(gguf), "--model-mmproj", str(mmproj),
                ])
            self.assertEqual(result, 0)
            options = run_batch.call_args.args[1]
            self.assertEqual(options.single_image, image)
            self.assertEqual(options.model_files, (gguf, mmproj))


if __name__ == "__main__":
    unittest.main()
