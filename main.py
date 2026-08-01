


import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Real-Time Object Detection System")
st.write("YOLOv11 + OpenCV | Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

st.subheader("📷 Choose Input")

option = st.radio(
    "Select an option:",
    ["Take Picture", "Upload Image"],
    horizontal=True
)

image = None

if option == "Take Picture":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
else:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

if image is not None:
    st.subheader("🔍 Detection Result")

    frame = np.array(image)

    results = model(
        frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    st.image(
        result_image,
        caption="YOLOv11 Detection Result",
        use_container_width=True
    )

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        st.subheader("📊 Detected Objects")

        detected = []

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            detected.append(
                f"**{class_name}** — {confidence * 100:.1f}%"
            )

        for item in list(dict.fromkeys(detected)):
            st.write("• " + item)

        st.success(f"Detected {len(boxes)} object(s).")
    else:
        st.warning("No object detected. Try another image.")

st.info(
    "Take Picture se photo capture karke YOLOv11 object detect karega. "
    "Live webcam/WebRTC ki zarurat nahi hai."
)
