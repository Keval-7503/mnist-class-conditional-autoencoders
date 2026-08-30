from __future__ import annotations

import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8].hex() != "89504e470d0a1a0a":
        raise ValueError(f"{path} is not a valid PNG")
    return struct.unpack(">II", data[16:24])


class MediaAssetTests(unittest.TestCase):
    def test_shareable_figures_have_expected_dimensions(self) -> None:
        self.assertEqual(
            _png_dimensions(ROOT / "results" / "results_dashboard.png"),
            (2560, 1600),
        )
        self.assertEqual(
            _png_dimensions(ROOT / "results" / "paired_effects_by_digit.png"),
            (2040, 1156),
        )

    def test_linkedin_video_is_packaged_with_captions(self) -> None:
        video = ROOT / "assets" / "linkedin-project-video.mp4"
        header = video.read_bytes()[:32]

        self.assertGreater(video.stat().st_size, 100_000)
        self.assertIn(b"ftyp", header)

        captions = (ROOT / "assets" / "linkedin-project-video.srt").read_text(encoding="utf-8")
        self.assertEqual(captions.count(" --> "), 6)
        self.assertIn("00:00:00,000 --> 00:00:04,000", captions)
        self.assertIn("00:00:23,000 --> 00:00:27,500", captions)


if __name__ == "__main__":
    unittest.main()
