"""Generate a short, deterministic synthetic pedestrian clip and manual labels."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
VIDEO_PATH = EVALUATION / "sample_demo.mp4"
LABELS_PATH = EVALUATION / "sample_manual_counts.csv"
WIDTH, HEIGHT, FPS, FRAME_COUNT = 320, 240, 10, 80


def _background() -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:150] = (190, 184, 166)
    frame[150:] = (105, 112, 115)
    for y in range(25, 145, 30):
        cv2.line(frame, (0, y), (WIDTH, y), (177, 172, 156), 1)
    cv2.line(frame, (0, 150), (WIDTH, 150), (78, 82, 85), 2)
    for x in range(24, WIDTH, 64):
        cv2.line(frame, (x, 151), (x - 20, HEIGHT), (125, 131, 133), 1)
    return frame


def _draw_person(frame: np.ndarray, center_x: int, foot_y: int, phase: float) -> None:
    cv2.ellipse(frame, (center_x, foot_y + 2), (22, 6), 0, 0, 360, (60, 63, 64), -1)
    head_y = foot_y - 86
    skin = (122, 171, 207)
    shirt = (38, 100, 213)
    pants = (61, 55, 42)
    cv2.ellipse(frame, (center_x, head_y), (11, 13), 0, 0, 360, skin, -1)
    cv2.ellipse(frame, (center_x, head_y - 4), (11, 6), 0, 180, 360, (38, 42, 47), -1)
    cv2.rectangle(frame, (center_x - 4, head_y + 11), (center_x + 4, head_y + 17), skin, -1)

    shoulder_y = foot_y - 68
    waist_y = foot_y - 37
    torso = np.array([
        [center_x - 11, shoulder_y], [center_x + 11, shoulder_y],
        [center_x + 15, waist_y], [center_x - 15, waist_y],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [torso], shirt)
    arm_swing = int(7 * np.sin(phase))
    cv2.line(frame, (center_x - 10, shoulder_y + 3), (center_x - 20, foot_y - 50 + arm_swing), shirt, 9, cv2.LINE_AA)
    cv2.line(frame, (center_x - 20, foot_y - 50 + arm_swing), (center_x - 17, foot_y - 33 + arm_swing), skin, 6, cv2.LINE_AA)
    cv2.line(frame, (center_x + 10, shoulder_y + 3), (center_x + 20, foot_y - 51 - arm_swing), shirt, 9, cv2.LINE_AA)
    cv2.line(frame, (center_x + 20, foot_y - 51 - arm_swing), (center_x + 17, foot_y - 34 - arm_swing), skin, 6, cv2.LINE_AA)

    stride = int(8 * np.sin(phase))
    hip_y = foot_y - 37
    cv2.line(frame, (center_x - 6, hip_y), (center_x - 12 - stride, foot_y - 18), pants, 10, cv2.LINE_AA)
    cv2.line(frame, (center_x - 12 - stride, foot_y - 18), (center_x - 15 - stride, foot_y - 3), pants, 8, cv2.LINE_AA)
    cv2.line(frame, (center_x + 6, hip_y), (center_x + 12 + stride, foot_y - 18), pants, 10, cv2.LINE_AA)
    cv2.line(frame, (center_x + 12 + stride, foot_y - 18), (center_x + 15 + stride, foot_y - 3), pants, 8, cv2.LINE_AA)
    cv2.line(frame, (center_x - 15 - stride, foot_y - 2), (center_x - 6 - stride, foot_y - 2), (35, 37, 39), 5, cv2.LINE_AA)
    cv2.line(frame, (center_x + 15 + stride, foot_y - 2), (center_x + 24 + stride, foot_y - 2), (35, 37, 39), 5, cv2.LINE_AA)


def generate() -> tuple[Path, Path]:
    EVALUATION.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 demo clip.")
    try:
        for index in range(FRAME_COUNT):
            frame = _background()
            foot_y = round(110 + 80 * index / (FRAME_COUNT - 1))
            _draw_person(frame, WIDTH // 2, foot_y, index * 0.42)
            writer.write(frame)
    finally:
        writer.release()

    with LABELS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "people_in_frame"])
        writer.writerows((frame, 1) for frame in range(1, FRAME_COUNT + 1))
    return VIDEO_PATH, LABELS_PATH


if __name__ == "__main__":
    video, labels = generate()
    print(f"Generated {video.relative_to(ROOT)} ({FRAME_COUNT} frames at {FPS} FPS)")
    print(f"Generated {labels.relative_to(ROOT)}")
