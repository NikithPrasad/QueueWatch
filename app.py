from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from analyze import analyze_video  # noqa: E402


st.set_page_config(page_title="QueueWatch | People Flow Analytics", page_icon="📹", layout="wide")
st.title("QueueWatch")
st.caption("Pedestrian flow analytics for uploaded video, a webcam, or an RTSP stream")
st.info("People are detected and tracked locally. The app does not identify faces. Review privacy rules and obtain consent before analyzing footage.")

with st.sidebar:
    st.header("Video source")
    source_mode = st.radio("Input", ["Video file", "Webcam", "RTSP stream"])
    webcam_index = st.number_input("Webcam device index", min_value=0, max_value=10, value=0, step=1, disabled=source_mode != "Webcam")
    rtsp_url = st.text_input("RTSP URL", type="password", help="The URL is masked because it may contain a username or password.", disabled=source_mode != "RTSP stream")
    stream_seconds = st.slider("Live capture duration (seconds)", 5, 120, 30, 5, disabled=source_mode == "Video file")

    st.header("Detection and counting")
    confidence = st.slider("Detection confidence", 0.1, 0.9, 0.1, 0.05)
    st.caption("Set the counting line endpoints as percentages of the frame. The endpoint order determines the signed crossing direction.")
    x1 = st.slider("Line start X", 0, 100, 2, 1, key="line_start_x") / 100
    y1 = st.slider("Line start Y", 0, 100, 50, 1, key="line_start_y") / 100
    x2 = st.slider("Line end X", 0, 100, 98, 1, key="line_end_x") / 100
    y2 = st.slider("Line end Y", 0, 100, 50, 1, key="line_end_y") / 100
    entered_label = st.text_input("Label for negative → positive", "Entered", key="entered_label").strip() or "Entered"
    exited_label = st.text_input("Label for positive → negative", "Exited", key="exited_label").strip() or "Exited"

    enable_zone = st.toggle("Measure zone dwell + heatmap", value=False)
    zone = None
    if enable_zone:
        zx1 = st.slider("Zone left (%)", 0, 100, 35, 1, key="zone_left_pct") / 100
        zy1 = st.slider("Zone top (%)", 0, 100, 35, 1, key="zone_top_pct") / 100
        zx2 = st.slider("Zone right (%)", 0, 100, 70, 1, key="zone_right_pct") / 100
        zy2 = st.slider("Zone bottom (%)", 0, 100, 90, 1, key="zone_bottom_pct") / 100
        if zx1 == zx2 or zy1 == zy2:
            st.warning("Choose different left/right and top/bottom values to create a zone.")
        else:
            zone = (min(zx1, zx2), min(zy1, zy2), max(zx1, zx2), max(zy1, zy2))

    save_annotated = st.toggle("Create annotated video", value=True, disabled=source_mode != "Video file")

clip = None
if source_mode == "Video file":
    clip = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
    if clip:
        st.video(clip.getvalue())

live_source_ready = source_mode == "Webcam" or (source_mode == "RTSP stream" and bool(rtsp_url.strip()))
source_ready = bool(clip) if source_mode == "Video file" else live_source_ready
button_label = "Analyze video" if source_mode == "Video file" else "Start live analysis"

if source_ready and st.button(button_label, type="primary"):
    live_mode = source_mode != "Video file"
    preview = st.empty() if live_mode else None
    progress = st.progress(0.0) if live_mode else None

    def update_live_preview(frame, frame_number, total_frames):
        if frame_number % max(1, int(25 / 3)) == 0 or frame_number == total_frames:
            preview.image(frame, channels="BGR", caption=f"Live analysis · frame {frame_number:,} of {total_frames:,}")
            progress.progress(min(1.0, frame_number / max(1, total_frames)))

    with tempfile.TemporaryDirectory(prefix="queuewatch-") as folder:
        folder_path = Path(folder)
        if source_mode == "Video file":
            source = folder_path / Path(clip.name).name
            source.write_bytes(clip.getvalue())
            output = folder_path / "annotated.mp4" if save_annotated else None
            max_seconds = None
        elif source_mode == "Webcam":
            source = int(webcam_index)
            output = None
            max_seconds = float(stream_seconds)
        else:
            source = rtsp_url.strip()
            output = None
            max_seconds = float(stream_seconds)

        try:
            with st.spinner("Loading the person detector and processing frames locally…"):
                summary, frame_data, annotated, events, zone_summary, heatmap = analyze_video(
                    source,
                    output,
                    confidence,
                    (x1, y1, x2, y2),
                    zone,
                    direction_labels=(entered_label, exited_label),
                    max_seconds=max_seconds,
                    on_frame=update_live_preview if live_mode else None,
                )
            annotated_bytes = annotated.read_bytes() if annotated and annotated.exists() else None
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            st.stop()

    if live_mode:
        progress.empty()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total crossings", summary["total_crossings"])
    c2.metric(entered_label, summary["negative_to_positive"])
    c3.metric(exited_label, summary["positive_to_negative"])
    c4.metric("Peak people / frame", summary["peak_people"])
    st.caption(f"Processed {summary['frames']:,} frames at {summary['video_fps']:.1f} source FPS ({summary['processing_fps']:.1f} inference FPS). Model: {summary['model_name']}.")

    if annotated_bytes:
        if summary["annotated_video_browser_compatible"]:
            st.video(annotated_bytes, format="video/mp4")
        else:
            st.warning("This system could not encode browser-compatible H.264. The annotated MP4 is available to download; playback depends on your browser's codec support.")
        st.download_button("Download annotated video", annotated_bytes, file_name="queuewatch-annotated.mp4", mime="video/mp4")

    if zone_summary["enabled"]:
        z1, z2, z3 = st.columns(3)
        z1.metric("Zone visitors", zone_summary["visitors"])
        z2.metric("Peak zone occupancy", zone_summary["peak_occupancy"])
        z3.metric("Mean dwell", f"{zone_summary['mean_dwell_seconds']:.1f} s")
        if heatmap is not None:
            st.image(heatmap, channels="BGR", caption=f"Footfall heatmap · longest measured dwell {zone_summary['max_dwell_seconds']:.1f} s")

    table = pd.DataFrame(frame_data)
    if not table.empty:
        st.subheader("Flow over time")
        series = table.groupby("second")["people_in_frame"].mean()
        st.line_chart(series, width="stretch")
        st.download_button("Download frame-level CSV", table.to_csv(index=False), "queuewatch-frame-counts.csv", "text/csv")
    if events:
        st.subheader("Crossing events")
        st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
        st.download_button("Download crossing events", pd.DataFrame(events).to_csv(index=False), "queuewatch-crossings.csv", "text/csv")
    st.download_button("Download analysis summary", pd.Series(summary).to_json(indent=2), "queuewatch-summary.json", "application/json")
    st.caption("Crossing labels depend on the endpoint order you choose. Counts vary with camera angle, lighting, and tracker stability; compare against manually counted footage before operational use.")

st.markdown("---")
st.markdown("**How it works:** YOLO detects people, ByteTrack assigns temporary track IDs, and a virtual line counts crossings. A webcam or RTSP stream is analyzed for the selected time window with a live preview. Results are aggregate counts; no face recognition is used.")
