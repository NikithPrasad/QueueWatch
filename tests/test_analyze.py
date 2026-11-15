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

    def test_live_source_requires_a_finite_duration(self):
        with patch("analyze.cv2.VideoCapture") as capture:
            with self.assertRaisesRegex(ValueError, "finite max_seconds"):
                analyze_video(0)
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

            class EmptyResult:
                boxes = None

                @staticmethod
                def plot():
                    return np.zeros((48, 64, 3), dtype=np.uint8)

            detector = SimpleNamespace(track=lambda *args, **kwargs: [EmptyResult()])
            output_path = Path(folder) / "annotated.mp4"
            with patch("analyze.YOLO", return_value=detector):
                summary, frame_rows, written_video, events, zone, heatmap = analyze_video(
                    video_path,
                    output_path=output_path,
                    line=(0.05, 0.5, 0.95, 0.5),
                    zone=(0.1, 0.1, 0.9, 0.9),
                )

            self.assertEqual(summary["frames"], 3)
            self.assertEqual(len(frame_rows), 3)
            self.assertEqual(summary["total_crossings"], 0)
            self.assertEqual(events, [])
            self.assertEqual(written_video, output_path)
            self.assertTrue(output_path.exists())
            self.assertIn(summary["annotated_video_browser_compatible"], (True, False))
            self.assertIsNotNone(summary["annotated_video_codec"])
            self.assertTrue(zone["enabled"])
            self.assertEqual(zone["visitors"], 0)
            self.assertIsNone(heatmap)

    def test_direction_labels_are_added_to_crossing_events(self):
        class TensorLike:
            def __init__(self, values):
                self.values = values

            def int(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class FakeModel:
            def __init__(self):
                self.frame = 0

            def track(self, *args, **kwargs):
                self.frame += 1
                y = 20 if self.frame == 1 else 28
                class DetectionBoxes:
                    id = TensorLike([4])
                    xyxy = TensorLike([[28, y - 8, 36, y]])

                    def __len__(self):
                        return 1

                boxes = DetectionBoxes()
                return [SimpleNamespace(boxes=boxes)]

        with tempfile.TemporaryDirectory() as folder:
            video_path = Path(folder) / "crossing.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is unavailable on this machine")
            for _ in range(2):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            with patch("analyze.YOLO", return_value=FakeModel()):
                _summary, _rows, _output, events, _zone, _heatmap = analyze_video(
                    video_path,
                    line=(0, 0.5, 1, 0.5),
                    direction_labels=("In", "Out"),
                )
        self.assertEqual(events[0]["direction_label"], "In")


if __name__ == "__main__":
    unittest.main()
