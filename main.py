import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Real-Time Object Detection System")
st.write("YOLOv11 + OpenCV | Object Detection")

# Load YOLOv11
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

# Camera
if option == "Take Picture":

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# Upload
else:

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")


# Detection
if image is not None:

    st.subheader("🔍 Detection Result")

    frame = np.array(image)

    with st.spinner("Detecting objects..."):

        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

    result = results[0]

    # Draw boxes
    output = result.plot()

    # Display result
    st.image(
        output,
        channels="BGR",
        caption="YOLOv11 Detection Result",
        use_container_width=True
    )

    # Object information
    if result.boxes is not None and len(result.boxes) > 0:

        st.subheader("📊 Detected Objects")

        counts = {}

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = model.names[class_id]

            counts[class_name] = counts.get(class_name, 0) + 1

            st.write(
                f"**{class_name}** — {confidence * 100:.1f}%"
            )

        detected_text = ", ".join(
            f"{name} ({count})"
            for name, count in counts.items()
        )

        st.success(
            f"Detected: {detected_text}"
        )

    else:

        st.warning(
            "No object detected. Try a clearer image or move closer."
        )
