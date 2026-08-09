"""
Real-Time Object Detection System
----------------------------------
YOLOv8 + OpenCV + Streamlit web interface for live video stream processing.

Run with:
    streamlit run app.py
"""

import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from utils.detector import ObjectDetector
from utils.video_stream import VideoStream


# ----------------------------- Page Config ----------------------------- #
st.set_page_config(
    page_title="Real-Time Object Detection | YOLOv8",
    page_icon="🎯",
    layout="wide",
)


# ----------------------------- Caching ---------------------------------- #
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(model_name: str) -> YOLO:
    """Load and cache the YOLOv8 model so it isn't reloaded on every rerun."""
    return YOLO(model_name)


# ----------------------------- Sidebar ----------------------------------- #
st.sidebar.title("⚙️ Detection Settings")

model_choice = st.sidebar.selectbox(
    "Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="Nano (n) is fastest — recommended for real-time webcam use on CPU (e.g. macOS).",
)

confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.45, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

frame_width = st.sidebar.selectbox("Frame width", [320, 480, 640, 960], index=2)

show_fps = st.sidebar.checkbox("Show FPS counter", value=True)
show_labels = st.sidebar.checkbox("Show class labels", value=True)

st.sidebar.markdown("---")
source_type = st.sidebar.radio("Input source", ["Webcam", "Upload video", "Upload image"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: on macOS, grant Terminal/iTerm camera permission in "
    "System Settings → Privacy & Security → Camera."
)


# ----------------------------- Header ------------------------------------ #
st.title("🎯 Real-Time Object Detection System")
st.markdown(
    "Real-time object detection using **YOLOv8**, **OpenCV**, and a **Streamlit** "
    "web interface — optimized for smooth live-video performance."
)

col_video, col_stats = st.columns([3, 1])

with col_stats:
    st.subheader("📊 Live Stats")
    fps_placeholder = st.empty()
    objects_placeholder = st.empty()
    detected_classes_placeholder = st.empty()


# ----------------------------- Detector ----------------------------------- #
model = load_model(model_choice)
detector = ObjectDetector(model=model, conf=confidence, iou=iou_threshold)


def render_stats(fps: float, results) -> None:
    if show_fps:
        fps_placeholder.metric("FPS", f"{fps:.1f}")
    names = results.names
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        counts[label] = counts.get(label, 0) + 1
    objects_placeholder.metric("Objects detected", sum(counts.values()))
    if counts:
        detected_classes_placeholder.write(
            "\n".join(f"- **{k}**: {v}" for k, v in sorted(counts.items()))
        )
    else:
        detected_classes_placeholder.write("_No objects detected yet._")


# ----------------------------- Webcam mode --------------------------------- #
if source_type == "Webcam":
    with col_video:
        run = st.toggle("▶️ Start webcam detection", value=False)
        frame_placeholder = st.empty()

    if run:
        stream = VideoStream(src=0, width=frame_width).start()
        prev_time = time.time()
        try:
            while run:
                frame = stream.read()
                if frame is None:
                    st.warning("Could not read from webcam. Check camera permissions.")
                    break

                annotated, results = detector.detect(frame, draw_labels=show_labels)

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                render_stats(fps, results)

                # Streamlit reruns the whole script on widget interaction;
                # this small sleep keeps CPU usage reasonable between frames.
                time.sleep(0.01)

                # Re-check the toggle state each loop without a full rerun
                run = st.session_state.get("▶️ Start webcam detection", run)
        finally:
            stream.stop()
    else:
        with col_video:
            st.info("Toggle the switch above to start the live webcam feed.")


# ----------------------------- Upload video mode ---------------------------- #
elif source_type == "Upload video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        tmp_path = f"/tmp/{uploaded_video.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_video.read())

        with col_video:
            frame_placeholder = st.empty()

        cap = cv2.VideoCapture(tmp_path)
        prev_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, results = detector.detect(frame, draw_labels=show_labels)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            render_stats(fps, results)
        cap.release()


# ----------------------------- Upload image mode ---------------------------- #
else:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, results = detector.detect(frame, draw_labels=show_labels)

        with col_video:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
        render_stats(0.0, results)
