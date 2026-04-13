import streamlit as st
import cv2
import numpy as np

# Import configuration and model
from src.config import MODEL_PATH, APP_TITLE, CAMERA_PROMPT
from src.model import ObjectDetector

# Set page layout and title
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Initialize the model efficiently using cache
@st.cache_resource
def load_detector():
    return ObjectDetector(MODEL_PATH)

def main():
    st.title(APP_TITLE)

    # Load our object detector
    detector = load_detector()

    # Webcam input
    img_file_buffer = st.camera_input(CAMERA_PROMPT)

    if img_file_buffer is not None:
        # Convert image to opencv format
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Predict and annotate
        with st.spinner('Analyzing...'):
            annotated_frame = detector.predict_and_annotate(cv2_img)

        # Show annotated image
        st.image(annotated_frame, caption="Detected Objects")

if __name__ == "__main__":
    from streamlit.runtime import exists
    if exists():
        main()
    else:
        import sys
        import subprocess
        print("Automatically routing to Streamlit runner...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]] + sys.argv[1:])
