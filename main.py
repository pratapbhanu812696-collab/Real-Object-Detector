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
# Camera input
# -------------------------------------------------
img_file_buffer = st.camera_input("Click a photo")


# -------------------------------------------------
# Object Detection
# -------------------------------------------------
if img_file_buffer is not None:

    try:
        # Read captured image
        image = Image.open(img_file_buffer).convert("RGB")
        img_array = np.array(image)

        # Run YOLOv8
        results = model.predict(
            source=img_array,
            conf=0.25,
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
                "Try taking a clearer photo or move closer."
            )

    except Exception as e:

        st.error("Detection mein error aaya.")
        st.exception(e)
