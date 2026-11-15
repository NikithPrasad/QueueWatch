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
st.caption("Local video analytics for pedestrian flow and queue monitoring")
st.info("People are detected and tracked locally. The app does not identify faces. Review privacy rules and obtain consent before analyzing footage.")
st.caption("Crossing directions use the signed sides of the line; reversing the endpoint order reverses the labels.")

clip = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
with st.sidebar:
    confidence = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
    st.caption("Set the counting line endpoints as percentages of the frame.")
    x1 = st.slider("Line start X", 0, 100, 2, 1) / 100
    y1 = st.slider("Line start Y", 0, 100, 50, 1) / 100
    x2 = st.slider("Line end X", 0, 100, 98, 1) / 100
    y2 = st.slider("Line end Y", 0, 100, 50, 1) / 100
    enable_zone = st.toggle("Measure rectangular zone dwell + heatmap", value=False)
    zone = None
    if enable_zone:
        zx1 = st.slider("Zone left", 0, 95, 35, 1) / 100
        zy1 = st.slider("Zone top", 0, 95, 35, 1) / 100
        zx2 = st.slider("Zone right", int(zx1 * 100) + 1, 100, max(int(zx1 * 100) + 1, 70), 1) / 100
        zy2 = st.slider("Zone bottom", int(zy1 * 100) + 1, 100, max(int(zy1 * 100) + 1, 90), 1) / 100
        zone = (zx1, zy1, zx2, zy2)
    save_annotated = st.toggle("Create annotated video", value=True)

if clip:
    st.video(clip.getvalue())

if clip and st.button("Analyze video", type="primary"):
    with tempfile.TemporaryDirectory(prefix="queuewatch-") as folder:
        source = Path(folder) / clip.name
        source.write_bytes(clip.getvalue())
        output = Path(folder) / "annotated.mp4" if save_annotated else None
        try:
            with st.spinner("Loading the person detector and processing frames locally…"):
                summary, frame_data, annotated, events, zone_summary, heatmap = analyze_video(source, output, confidence, (x1, y1, x2, y2), zone)
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            st.stop()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("People counted", summary["total_crossings"])
        c2.metric("Negative side → positive side", summary["negative_to_positive"])
        c3.metric("Positive side → negative side", summary["positive_to_negative"])
        c4.metric("Peak people / frame", summary["peak_people"])
        st.caption(f"Processed {summary['frames']:,} frames at {summary['video_fps']:.1f} source FPS ({summary['processing_fps']:.1f} inference FPS). Model: {summary['model_name']}.")
        if zone_summary["enabled"]:
            z1, z2, z3 = st.columns(3)
            z1.metric("Zone visitors", zone_summary["visitors"])
            z2.metric("Peak zone occupancy", zone_summary["peak_occupancy"])
            z3.metric("Mean dwell", f"{zone_summary['mean_dwell_seconds']:.1f} s")
            if heatmap is not None:
                st.image(heatmap, channels="BGR", caption=f"Footfall heatmap · longest measured dwell {zone_summary['max_dwell_seconds']:.1f} s")
        if annotated and annotated.exists():
            st.video(annotated.read_bytes())
            st.download_button("Download annotated video", annotated.read_bytes(), file_name="queuewatch-annotated.mp4", mime="video/mp4")
        table = pd.DataFrame(frame_data)
        if not table.empty:
            st.subheader("Flow over time")
            series = table.groupby("second")["people_in_frame"].mean()
            st.line_chart(series)
            st.download_button("Download frame-level CSV", table.to_csv(index=False), "queuewatch-frame-counts.csv", "text/csv")
        if events:
            st.subheader("Crossing events")
            st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)
            st.download_button("Download crossing events", pd.DataFrame(events).to_csv(index=False), "queuewatch-crossings.csv", "text/csv")
        st.caption("Crossing totals depend on the selected line, camera angle, lighting, and tracker stability. Compare with manually counted footage before operational use.")

st.markdown("---")
st.markdown("**How it works:** YOLO detects people, ByteTrack assigns temporary track IDs, and a virtual line counts a tracked person when they cross it. The result is an aggregate count; no face recognition is used.")
