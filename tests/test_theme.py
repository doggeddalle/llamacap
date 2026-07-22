import unittest

from llamacap.gui.theme import PALETTES


def _luminance(color: str) -> float:
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class ThemeTests(unittest.TestCase):
    def test_tab_text_contrast_in_both_themes(self):
        for name, palette in PALETTES.items():
            with self.subTest(theme=name, state="selected"):
                self.assertGreaterEqual(_contrast(palette["fg"], palette["surface"]), 4.5)
            with self.subTest(theme=name, state="unselected"):
                self.assertGreaterEqual(_contrast(palette["muted"], palette["bg"]), 4.5)

    def test_accent_button_text_contrast(self):
        for name, palette in PALETTES.items():
            with self.subTest(theme=name):
                self.assertGreaterEqual(_contrast(palette["accent_fg"], palette["accent"]), 4.5)


if __name__ == "__main__":
    unittest.main()
