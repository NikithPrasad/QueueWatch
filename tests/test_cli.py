import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cli  # noqa: E402


class CommandLineTests(unittest.TestCase):
    def test_cli_exports_all_analyzer_results_and_zone_options(self):
        summary = {"total_crossings": 2}
        zone = {"enabled": True, "visitors": 1}
        analyzer_result = (
            summary,
            [{"frame": 1, "people_in_frame": 1}],
            None,
            [{"track_id": 8, "frame": 2, "direction": "negative_to_positive", "x": 10, "y": 20, "second": 0.04}],
            zone,
            None,
        )
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.argv", ["cli.py", "sample.mp4", "--output-dir", folder, "--zone", "0.2", "0.2", "0.8", "0.9", "--no-video"]):
                with patch.object(cli, "analyze_video", return_value=analyzer_result) as analyze:
                    with patch("builtins.print"):
                        cli.main()

            analyze.assert_called_once_with("sample.mp4", None, 0.35, (0.02, 0.5, 0.98, 0.5), (0.2, 0.2, 0.8, 0.9))
            self.assertEqual(json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))["zone"], zone)
            self.assertTrue((Path(folder) / "frame_counts.csv").exists())
            self.assertTrue((Path(folder) / "crossings.csv").exists())


if __name__ == "__main__":
    unittest.main()
