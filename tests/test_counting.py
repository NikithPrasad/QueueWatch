import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from counting import LineCounter, segments_intersect  # noqa: E402


class LineCrossingTests(unittest.TestCase):
    def test_finite_segment_intersection(self):
        self.assertTrue(segments_intersect((5, -2), (5, 2), (0, 0), (10, 0)))
        self.assertFalse(segments_intersect((15, -2), (15, 2), (0, 0), (10, 0)))

    def test_direction_and_reverse_crossing(self):
        counter = LineCounter((0, 0), (10, 0), debounce_frames=1)
        counter.update(1, (5, -2), 1)
        first = counter.update(1, (5, 2), 2)
        self.assertEqual(first["direction"], "negative_to_positive")
        reverse = counter.update(1, (5, -2), 3)
        self.assertEqual(reverse["direction"], "positive_to_negative")

    def test_jitter_is_debounced(self):
        counter = LineCounter((0, 0), (10, 0), debounce_frames=4)
        counter.update(1, (5, -2), 1)
        self.assertIsNotNone(counter.update(1, (5, 2), 2))
        self.assertIsNone(counter.update(1, (5, -2), 3))

    def test_exact_line_frame_does_not_lose_crossing(self):
        counter = LineCounter((0, 0), (10, 0), debounce_frames=1)
        counter.update(1, (5, -2), 1)
        self.assertIsNone(counter.update(1, (5, 0), 2))
        event = counter.update(1, (5, 2), 3)
        self.assertEqual(event["direction"], "negative_to_positive")


if __name__ == "__main__":
    unittest.main()
