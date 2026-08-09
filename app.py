import cv2
import av
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


st.set_page_config(
    page_title="Real-Time Object Detection | YOLOv8",
    page_icon="🎯",
    layout="wide"
)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# ---------------- PAGE ----------------

st.title("🎯 Real-Time Object Detection System")

st.write(
    "YOLOv8 + OpenCV + Streamlit"
)


# ---------------- SETTINGS ----------------

st.sidebar.title("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence",
    0.1,
    1.0,
    0.45,
    0.05
)

show_labels = st.sidebar.checkbox(
    "Show class labels",
    value=True
)


# ---------------- VIDEO PROCESSOR ----------------

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.confidence = confidence
        self.show_labels = show_labels

    def recv(self, frame):

        # Browser frame → OpenCV
        img = frame.to_ndarray(format="bgr24")

        # YOLO detection
        results = model(
            img,
            conf=self.confidence,
            verbose=False
        )

        # Draw detections
        annotated = results[0].plot(
            labels=self.show_labels
        )

        # OpenCV → browser frame
        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ---------------- WEBRTC ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)


st.subheader("📷 Live Camera Detection")

webrtc_streamer(
    key="yolo-detection",
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
