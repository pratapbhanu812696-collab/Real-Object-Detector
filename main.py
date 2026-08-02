import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(
    page_title="Bhanu's Detector",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Object Detection App")

# -------------------------------------------------
# Sidebar Controls (Confidence & Input Method)
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Camera", "Upload Image"]
)

# -------------------------------------------------
# Load YOLOv8 model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


try:
    model = load_model()
    st.success("Model Loaded!")

except Exception as e:
    st.error("Model load karne mein error aaya.")
    st.exception(e)
    st.stop()


# -------------------------------------------------
# Image Input (Camera or Uploader)
# -------------------------------------------------
img_file_buffer = None

if input_method == "Camera":
    img_file_buffer = st.camera_input("Click a photo")
else:
    img_file_buffer = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"]
    )


# -------------------------------------------------
# Object Detection
# -------------------------------------------------
if img_file_buffer is not None:

    try:
        # Read captured/uploaded image
        image = Image.open(img_file_buffer).convert("RGB")
        img_array = np.array(image)

        # Run YOLOv8 with dynamic confidence
        results = model.predict(
            source=img_array,
            conf=confidence_threshold,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes and labels
        res_plotted = result.plot()

        # BGR -> RGB
        res_rgb = cv2.cvtColor(
            res_plotted,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # Detection Result
        # -------------------------------------------------
        st.subheader("🔍 Detection Results")

        st.image(
            res_rgb,
            caption="YOLOv8 Detection",
            use_container_width=True
        )

        # -------------------------------------------------
        # Object Details
        # -------------------------------------------------
        if result.boxes is not None and len(result.boxes) > 0:

            st.success(
                f"Detected {len(result.boxes)} object(s)"
            )

            st.subheader("📊 Detected Objects")

            object_count = {}

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = model.names[class_id]

                object_count[object_name] = (
                    object_count.get(object_name, 0) + 1
                )

                st.write(
                    f"**{object_name}** — "
                    f"{confidence * 100:.1f}%"
                )

            # Summary
            summary = ", ".join(
                f"{name} ({count})"
                for name, count in object_count.items()
            )

            st.info(f"Objects found: {summary}")

        else:

            st.warning(
                "No object detected. "
                "Try lowering the confidence threshold from the sidebar, taking a clearer photo, or uploading a different image."
            )

    except Exception as e:

        st.error("Detection mein error aaya.")
        st.exception(e)
