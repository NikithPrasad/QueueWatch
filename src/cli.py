"""Run QueueWatch without the Streamlit interface."""

import argparse
import json
from pathlib import Path
import pandas as pd

from analyze import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local video file path")
    parser.add_argument("--output-dir", default="output", help="Folder for CSV and annotated video outputs")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--line", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=(0.02, 0.5, 0.98, 0.5), help="Line endpoints as fractions of frame width/height")
    parser.add_argument("--zone", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="Optional rectangular zone as frame fractions")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated video output")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = None if args.no_video else output_dir / "annotated.mp4"
    summary, frames, _written_video, events, zone_summary, heatmap = analyze_video(
        args.source,
        video_path,
        args.confidence,
        tuple(args.line),
        tuple(args.zone) if args.zone else None,
    )
    summary["zone"] = zone_summary
    pd.DataFrame(frames).to_csv(output_dir / "frame_counts.csv", index=False)
    pd.DataFrame(events, columns=["track_id", "frame", "direction", "x", "y", "second"]).to_csv(output_dir / "crossings.csv", index=False)
    if heatmap is not None:
        import cv2

        if not cv2.imwrite(str(output_dir / "footfall_heatmap.png"), heatmap):
            raise OSError("Could not write footfall_heatmap.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
