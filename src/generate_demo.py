"""Download and prepare the attributed real-footage demo and manual frame labels."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/d/d2/People_waiting_to_cross_the_street.webm"
SOURCE_SHA1 = "635373fdb0ccb21b9bd716666893faee18598669"
FPS = 10
MANUAL_PEOPLE_COUNTS = [13, 12, 12, 12, 11, 12, 12, 12, 14, 14, 14, 14, 13, 14, 14, 14]


def _download_source(destination: Path) -> None:
    request = Request(SOURCE_URL, headers={"User-Agent": "QueueWatch portfolio evaluation demo"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    digest = hashlib.sha1(destination.read_bytes()).hexdigest()
    if digest != SOURCE_SHA1:
        raise ValueError(f"The downloaded Commons source checksum did not match: {digest}")


def generate() -> tuple[Path, Path]:
    EVALUATION.mkdir(parents=True, exist_ok=True)
    video_output = EVALUATION / "sample_demo.mp4"
    labels_output = EVALUATION / "sample_manual_counts.csv"

    with tempfile.TemporaryDirectory(prefix="queuewatch-demo-") as folder:
        source = Path(folder) / "pedestrian_source.webm"
        _download_source(source)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-vf", f"scale=960:-2,fps={FPS}", "-an", "-c:v", "libx264", "-preset", "fast",
                "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_output),
            ],
            check=True,
        )

    with labels_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "people_in_frame"])
        for second, people in enumerate(MANUAL_PEOPLE_COUNTS):
            writer.writerow([1 + second * FPS, people])

    return video_output, labels_output


if __name__ == "__main__":
    video, labels = generate()
    print(f"Generated {video.relative_to(ROOT)} from Wikimedia Commons real pedestrian footage")
    print(f"Generated {labels.relative_to(ROOT)} with {len(MANUAL_PEOPLE_COUNTS)} manual one-second sample counts")
