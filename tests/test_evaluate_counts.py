import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_counts import evaluate  # noqa: E402


class CountEvaluationTests(unittest.TestCase):
    def test_reports_mae_smape_and_measured_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / "predictions.csv"
            truth = root / "truth.csv"
            summary = root / "summary.json"
            pd.DataFrame({"frame": [1, 2], "people_in_frame": [0, 1]}).to_csv(predictions, index=False)
            pd.DataFrame({"frame": [1, 2], "people_in_frame": [0, 2]}).to_csv(truth, index=False)
            summary.write_text(json.dumps({"processing_fps": 12.5}), encoding="utf-8")
            result = evaluate(predictions, truth, summary)
        self.assertEqual(result["compared_frames"], 2)
        self.assertEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["smape"], 100 / 3)
        self.assertEqual(result["processing_fps"], 12.5)


if __name__ == "__main__":
    unittest.main()
