import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import time


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Real-Time Object Detection System")
st.write("YOLOv11 + OpenCV | Live Object Detection")


# -----------------------------
# LOAD YOLOv11 MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# -----------------------------
# VIDEO PROCESSOR
# -----------------------------

class ObjectDetector(VideoProcessorBase):

    def __init__(self):
        self.prev_time = time.time()

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV
        img = frame.to_ndarray(format="bgr24")

        # YOLOv11 detection
        results = model(
            img,
            conf=0.5,
            verbose=False
        )

        # Draw bounding boxes
        output = results[0].plot()

        # -----------------------------
        # FPS
        # -----------------------------

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        cv2.putText(
            output,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # RETURN FRAME
        # -----------------------------

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# -----------------------------
# START LIVE CAMERA
# -----------------------------

st.subheader("📷 Live Camera")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


st.info(
    "Camera ke saamne object lao. "
    "YOLOv11 automatically live detection karega."
)
