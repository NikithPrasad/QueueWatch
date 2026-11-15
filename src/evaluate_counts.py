"""Compare predicted frame-level counts with manually annotated counts."""

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="CSV exported by QueueWatch")
    parser.add_argument("ground_truth", help="CSV with columns frame, people_in_frame")
    args = parser.parse_args()
    predicted = pd.read_csv(args.predictions)[["frame", "people_in_frame"]].rename(columns={"people_in_frame": "predicted"})
    truth = pd.read_csv(args.ground_truth)[["frame", "people_in_frame"]].rename(columns={"people_in_frame": "actual"})
    joined = truth.merge(predicted, on="frame", how="inner")
    if joined.empty:
        raise SystemExit("No matching frame numbers. Both files need shared frame values.")
    mae = (joined["predicted"] - joined["actual"]).abs().mean()
    smape = (2 * (joined["predicted"] - joined["actual"]).abs() / (joined["predicted"].abs() + joined["actual"].abs()).replace(0, 1)).mean() * 100
    print(f"Compared frames: {len(joined)}")
    print(f"Count MAE: {mae:.3f} people/frame")
    print(f"Count sMAPE: {smape:.2f}%")


if __name__ == "__main__":
    main()
