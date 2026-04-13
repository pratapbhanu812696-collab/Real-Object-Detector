import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Cloud settings
st.set_page_config(page_title="Bhanu's Detector", layout="centered")
st.title("🤖 Object Detection App")

# Model Loading with Cache
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
    st.success("Model Loaded!")
except Exception as e:
    st.error(f"Model load karne mein error: {e}")

# Camera Input
img_file_buffer = st.camera_input("Click a photo")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    # Run YOLO
    results = model(img_array)

    for r in results:
        res_plotted = r.plot()
        # Convert to RGB for Streamlit
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption='Detection Results', use_container_width=True)