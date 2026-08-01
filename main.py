import time
import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Real-Time Object Detection System")
st.write("YOLOv11 + OpenCV | Live Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

class ObjectDetector(VideoProcessorBase):
    def __init__(self):
        self.prev_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(
            img,
            conf=0.5,
            imgsz=640,
            verbose=False
        )

        output = results[0].plot()

        current_time = time.time()
        elapsed = current_time - self.prev_time
        fps = 1 / elapsed if elapsed > 0 else 0
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

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )

rtc_configuration = {
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"]
        }
    ]
}

st.subheader("📷 Live Camera")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetector,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.info(
    "Camera permission ko Allow karein. "
    "Camera ke saamne object laane par YOLOv11 automatically live detection karega."
)
