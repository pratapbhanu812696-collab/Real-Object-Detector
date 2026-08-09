import os
import time
import av
import cv2
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Real-Time Object Detection | YOLOv8",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(model_name):
    return YOLO(model_name)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Detection Settings")

model_choice = st.sidebar.selectbox(
    "Model",
    [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
    ],
    index=0,
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    0.1,
    1.0,
    0.45,
    0.05,
)

show_labels = st.sidebar.checkbox(
    "Show class labels",
    value=True,
)

show_confidence = st.sidebar.checkbox(
    "Show confidence",
    value=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(model_choice)


# =========================================================
# TURN / STUN CONFIGURATION
# =========================================================

ice_servers = [
    {
        "urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
        ]
    }
]


# Optional TURN server
# Add these values in Render Environment Variables if required.

TURN_URL = os.getenv("TURN_URL")
TURN_USERNAME = os.getenv("TURN_USERNAME")
TURN_PASSWORD = os.getenv("TURN_PASSWORD")

if TURN_URL and TURN_USERNAME and TURN_PASSWORD:
    ice_servers.append(
        {
            "urls": TURN_URL,
            "username": TURN_USERNAME,
            "credential": TURN_PASSWORD,
        }
    )


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": ice_servers
    }
)


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = model
        self.confidence = confidence
        self.show_labels = show_labels
        self.show_confidence = show_confidence

        self.last_time = time.time()
        self.fps = 0.0
        self.object_counts = {}

    def recv(self, frame):

        # Browser camera frame -> OpenCV BGR
        img = frame.to_ndarray(format="bgr24")

        # YOLO detection
        results = self.model.predict(
            source=img,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]

        # =================================================
        # COUNT OBJECTS
        # =================================================

        counts = {}

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                counts[class_name] = (
                    counts.get(class_name, 0) + 1
                )

        self.object_counts = counts

        # =================================================
        # DRAW DETECTIONS
        # =================================================

        annotated = result.plot(
            labels=self.show_labels,
            conf=self.show_confidence,
        )

        # =================================================
        # FPS
        # =================================================

        current_time = time.time()

        elapsed = current_time - self.last_time

        if elapsed > 0:
            self.fps = 1.0 / elapsed

        self.last_time = current_time

        # =================================================
        # FPS ON VIDEO
        # =================================================

        cv2.putText(
            annotated,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # =================================================
        # OBJECT COUNT ON VIDEO
        # =================================================

        y = 80

        for name, count in counts.items():

            text = f"{name}: {count}"

            cv2.putText(
                annotated,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            y += 30

        # =================================================
        # RETURN FRAME TO BROWSER
        # =================================================

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# =========================================================
# HEADER
# =========================================================

st.title("🎯 Real-Time Object Detection System")

st.markdown(
    """
    YOLOv8 + OpenCV + Streamlit

    **Use your browser camera for real-time object detection.**
    """
)


# =========================================================
# CAMERA
# =========================================================

st.subheader("📷 Live Camera Detection")

st.info(
    "Click START below and allow camera permission when your browser asks."
)


ctx = webrtc_streamer(
    key="yolo-live-detection",

    video_processor_factory=YOLOVideoProcessor,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    async_processing=True,
)


# =========================================================
# LIVE STATS
# =========================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Detection Status")

    if ctx.state.playing:
        st.success("🟢 Camera is running")
    else:
        st.warning("🟡 Camera is stopped")


with col2:
    st.subheader("🌐 Connection")

    if TURN_URL:
        st.success("STUN + TURN configured")
    else:
        st.info("Using STUN")


# =========================================================
# TROUBLESHOOTING
# =========================================================

with st.expander("❗ Camera not connecting?"):

    st.write(
        """
        If you see:

        **Connection is taking longer than expected**

        your network may require a TURN server.

        Add these Environment Variables in Render:

        `TURN_URL`

        `TURN_USERNAME`

        `TURN_PASSWORD`

        Then redeploy the application.
        """
    )
