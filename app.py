import gradio as gr
from ultralytics import YOLO
import cv2

# =========================
# LOAD MODEL
# =========================

# Agar custom model hai:
# model = YOLO("best.pt")

model = YOLO("yolov8n.pt")


# =========================
# YOLO DETECTION
# =========================

def detect(frame):
    if frame is None:
        return None

    # Gradio gives RGB image
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # YOLO prediction
    results = model.predict(
        frame_bgr,
        conf=0.45,
        verbose=False
    )

    # Draw bounding boxes
    output = results[0].plot()

    # BGR -> RGB
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return output


# =========================
# GRADIO UI
# =========================

with gr.Blocks(
    title="YOLOv8 Live Object Detection"
) as demo:

    gr.Markdown(
        """
        # 🎯 YOLOv8 Live Object Detection

        Start your webcam and the model will detect objects
        continuously.
        """
    )

    webcam = gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True,
        label="Live Camera"
    )

    output = gr.Image(
        type="numpy",
        streaming=True,
        label="Detection Result"
    )

    # Process every webcam frame
    webcam.stream(
        fn=detect,
        inputs=webcam,
        outputs=output,
        stream_every=0.1,
        concurrency_limit=1
    )


# =========================
# START APP
# =========================

demo.launch()
