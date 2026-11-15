"""Person detection, short-term tracking, line-crossing counts, and CSV export."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import math
import cv2
import numpy as np
from ultralytics import YOLO

from counting import LineCounter


MODEL_NAME = "yolo11n.pt"


def analyze_video(video_path: str | Path, output_path: str | Path | None = None, confidence: float = 0.35, line=(0.02, 0.5, 0.98, 0.5), zone=None):
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if len(line) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in line):
        raise ValueError("Line endpoints must be four finite frame fractions between 0 and 1.")
    if (line[0], line[1]) == (line[2], line[3]):
        raise ValueError("The counting line needs two distinct endpoints.")
    if zone is not None and (len(zone) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in zone)):
        raise ValueError("Zone coordinates must be four finite frame fractions between 0 and 1.")
    if zone is not None and (zone[0] == zone[2] or zone[1] == zone[3]):
        raise ValueError("The measurement zone must have positive width and height.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video has invalid frame dimensions.")
    line_a = (int(width * line[0]), int(height * line[1]))
    line_b = (int(width * line[2]), int(height * line[3]))
    if line_a == line_b:
        capture.release()
        raise ValueError("The counting line is too small for this video's resolution.")
    try:
        # Validate the input before loading YOLO weights, which may require a large download.
        model = YOLO(MODEL_NAME)
    except Exception:
        capture.release()
        raise
    writer = None
    if output_path:
        output_path = Path(output_path)
        if Path(video_path).resolve() == output_path.resolve():
            capture.release()
            raise ValueError("The annotated output must not overwrite the input video.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise ValueError("Could not create the annotated MP4. Try a supported input video or disable export.")

    counter = LineCounter(line_a, line_b)
    zone_px = None
    if zone:
        x1, y1, x2, y2 = zone
        zone_px = (int(width * min(x1, x2)), int(height * min(y1, y2)), int(width * max(x1, x2)), int(height * max(y1, y2)))
    zone_started: dict[int, int] = {}
    zone_last_inside: dict[int, int] = {}
    dwell_seconds = []
    zone_entries = 0
    peak_zone_occupancy = 0
    zone_grace_frames = max(1, int(1.0 * fps))
    heatmap = np.zeros((height, width), dtype=np.float32) if zone_px else None
    first_frame = None
    events = []
    frame_rows = []
    frame_no = 0
    peak_people = 0
    started = perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_no += 1
            if first_frame is None:
                first_frame = frame.copy()
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=confidence, verbose=False)[0]
            boxes = result.boxes
            people = 0 if boxes is None else len(boxes)
            peak_people = max(peak_people, people)
            inside_now = set()
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()
                for track_id, box in zip(ids, xyxy):
                    foot = ((box[0] + box[2]) / 2, box[3])
                    event = counter.update(track_id, foot, frame_no)
                    if event:
                        events.append({**event, "second": (frame_no - 1) / fps})
                    if zone_px and zone_px[0] <= foot[0] <= zone_px[2] and zone_px[1] <= foot[1] <= zone_px[3]:
                        inside_now.add(track_id)
                        zone_last_inside[track_id] = frame_no
                        if track_id not in zone_started:
                            zone_started[track_id] = frame_no
                            zone_entries += 1
                        cv2.circle(heatmap, (int(foot[0]), int(foot[1])), 9, 1.0, -1)
            zone_occupancy = len(inside_now)
            peak_zone_occupancy = max(peak_zone_occupancy, zone_occupancy)
            for track_id, last_frame in list(zone_last_inside.items()):
                if frame_no - last_frame > zone_grace_frames:
                    dwell_seconds.append(max(0.0, (last_frame - zone_started[track_id]) / fps))
                    zone_last_inside.pop(track_id, None)
                    zone_started.pop(track_id, None)
            second = int((frame_no - 1) / fps)
            frame_rows.append({
                "frame": frame_no,
                "second": second,
                "people_in_frame": people,
                "negative_to_positive_total": counter.counts["negative_to_positive"],
                "positive_to_negative_total": counter.counts["positive_to_negative"],
            })
            if writer:
                annotated = result.plot()
                cv2.line(annotated, line_a, line_b, (0, 220, 255), 2)
                cv2.putText(annotated, f"Negative to positive: {counter.counts['negative_to_positive']}  Reverse: {counter.counts['positive_to_negative']}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                if zone_px:
                    cv2.rectangle(annotated, (zone_px[0], zone_px[1]), (zone_px[2], zone_px[3]), (255, 160, 40), 2)
                    cv2.putText(annotated, f"Zone occupancy: {zone_occupancy}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 40), 2)
                writer.write(annotated)
    finally:
        capture.release()
        if writer:
            writer.release()
    if frame_no == 0:
        raise ValueError("The video opened, but no frames could be read.")
    for track_id, last_frame in zone_last_inside.items():
        dwell_seconds.append(max(0.0, (last_frame - zone_started[track_id]) / fps))
    heatmap_image = None
    if heatmap is not None and first_frame is not None and float(heatmap.max()) > 0:
        normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        heatmap_image = cv2.addWeighted(first_frame, 0.58, colored, 0.42, 0)
    return {
        "total_crossings": sum(counter.counts.values()),
        "negative_to_positive": counter.counts["negative_to_positive"],
        "positive_to_negative": counter.counts["positive_to_negative"],
        "peak_people": peak_people,
        "frames": frame_no,
        "video_fps": fps,
        "processing_fps": frame_no / max(perf_counter() - started, 1e-6),
        "model_name": MODEL_NAME,
    }, frame_rows, Path(output_path) if output_path else None, events, {
        "enabled": bool(zone_px),
        "visitors": zone_entries,
        "peak_occupancy": peak_zone_occupancy,
        "mean_dwell_seconds": float(np.mean(dwell_seconds)) if dwell_seconds else 0.0,
        "max_dwell_seconds": max(dwell_seconds, default=0.0),
    }, heatmap_image
