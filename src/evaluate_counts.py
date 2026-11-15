"""Compare frame counts with manual labels and optionally report measured inference FPS."""

import argparse
import json
from pathlib import Path

import pandas as pd


def evaluate(predictions_path: str | Path, ground_truth_path: str | Path, summary_path: str | Path | None = None) -> dict[str, float | int | None]:
    predicted = pd.read_csv(predictions_path)
    truth = pd.read_csv(ground_truth_path)
    required = {"frame", "people_in_frame"}
    for label, frame in (("predictions", predicted), ("ground truth", truth)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} CSV is missing columns: {', '.join(sorted(missing))}")
    predicted = predicted[["frame", "people_in_frame"]].rename(columns={"people_in_frame": "predicted"})
    truth = truth[["frame", "people_in_frame"]].rename(columns={"people_in_frame": "actual"})
    if predicted["frame"].duplicated().any() or truth["frame"].duplicated().any():
        raise ValueError("Each CSV must contain at most one count per frame.")
    joined = truth.merge(predicted, on="frame", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No matching frame numbers. Both files need shared frame values.")

    errors = (joined["predicted"] - joined["actual"]).abs()
    denominator = joined["predicted"].abs() + joined["actual"].abs()
    smape = (2 * errors / denominator.where(denominator != 0, 1)).mean() * 100
    fps = None
    if summary_path:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        fps = float(summary["processing_fps"])
    return {"compared_frames": int(len(joined)), "mae": float(errors.mean()), "smape": float(smape), "processing_fps": fps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="CSV exported by QueueWatch")
    parser.add_argument("ground_truth", help="CSV with columns frame, people_in_frame")
    parser.add_argument("--summary", help="Optional analysis summary JSON to report measured processing FPS")
    args = parser.parse_args()
    result = evaluate(args.predictions, args.ground_truth, args.summary)
    print(f"Compared frames: {result['compared_frames']}")
    print(f"Count MAE: {result['mae']:.3f} people/frame")
    print(f"Count sMAPE: {result['smape']:.2f}%")
    if result["processing_fps"] is not None:
        print(f"Processing FPS: {result['processing_fps']:.2f} frames/second")


if __name__ == "__main__":
    main()
