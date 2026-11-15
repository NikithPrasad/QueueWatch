from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze import analyze_video  # noqa: E402


class AnalyzeInputTests(unittest.TestCase):
    def test_invalid_video_fails_before_loading_detector(self):
        with tempfile.TemporaryDirectory() as folder:
            missing_video = Path(folder) / "missing.mp4"
            with patch("analyze.YOLO") as detector:
                with self.assertRaisesRegex(ValueError, "Could not open video"):
                    analyze_video(missing_video)
            detector.assert_not_called()

    def test_invalid_confidence_fails_before_opening_video(self):
        with patch("analyze.cv2.VideoCapture") as capture:
            with self.assertRaisesRegex(ValueError, "Confidence"):
                analyze_video("unused.mp4", confidence=1.5)
        capture.assert_not_called()

    def test_valid_clip_returns_frame_and_zone_summaries(self):
        with tempfile.TemporaryDirectory() as folder:
            video_path = Path(folder) / "empty_scene.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is unavailable on this machine")
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()

            detector = SimpleNamespace(track=lambda *args, **kwargs: [SimpleNamespace(boxes=None)])
            with patch("analyze.YOLO", return_value=detector):
                summary, frame_rows, written_video, events, zone, heatmap = analyze_video(
                    video_path,
                    line=(0.05, 0.5, 0.95, 0.5),
                    zone=(0.1, 0.1, 0.9, 0.9),
                )

            self.assertEqual(summary["frames"], 3)
            self.assertEqual(len(frame_rows), 3)
            self.assertEqual(summary["total_crossings"], 0)
            self.assertEqual(events, [])
            self.assertIsNone(written_video)
            self.assertTrue(zone["enabled"])
            self.assertEqual(zone["visitors"], 0)
            self.assertIsNone(heatmap)


if __name__ == "__main__":
    unittest.main()
