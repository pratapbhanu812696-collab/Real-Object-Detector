# 🎯 Real-Time Object Detection System

A real-time object detection system built with **YOLOv8** and **OpenCV**,
served through a **Streamlit** web interface for live video stream
processing. Optimized for smooth performance on macOS.

## ✨ Features

- Real-time object detection from a live webcam feed
- Also supports uploaded video files and static images
- Adjustable confidence / IoU thresholds and frame resolution from the sidebar
- Live FPS counter and per-class object counts
- Threaded video capture for smoother, non-blocking real-time performance
- Choice of YOLOv8 model size (nano / small / medium) to balance speed vs. accuracy

## 🗂️ Project Structure

```
object-detection-yolov8/
├── app.py                  # Streamlit application (entry point)
├── requirements.txt        # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── detector.py         # YOLOv8 inference + bounding-box drawing
│   └── video_stream.py     # Threaded webcam capture helper
└── sample_images/          # (optional) put test images here
```

## 🚀 Getting Started

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (usually `http://localhost:8501`).

## 🍎 macOS Camera Permissions

If the webcam feed doesn't start, grant camera access to your terminal app:

**System Settings → Privacy & Security → Camera** → enable it for
Terminal / iTerm / VS Code (whichever app you launch `streamlit run` from).

## ⚙️ How It Works

1. `utils/video_stream.py` reads frames from the webcam on a background
   thread (via `cv2.VideoCapture`) so the UI never blocks waiting on frame
   capture.
2. `utils/detector.py` wraps an `ultralytics.YOLO` model, runs inference
   on each frame, and draws bounding boxes + labels with OpenCV.
3. `app.py` wires everything into a Streamlit UI: sidebar controls for
   model/confidence/resolution, a live image placeholder that's updated
   every frame, and a stats panel (FPS, object counts).

## 🧩 Model Options

| Model         | Speed  | Accuracy | Recommended use            |
|---------------|--------|----------|-----------------------------|
| `yolov8n.pt`  | Fastest| Good     | Real-time webcam on CPU (default) |
| `yolov8s.pt`  | Fast   | Better   | Balanced                    |
| `yolov8m.pt`  | Slower | Best     | Higher accuracy, GPU recommended |

Weights are downloaded automatically by `ultralytics` on first run.

## 🛠️ Tech Stack

- **YOLOv8** (Ultralytics) — object detection model
- **OpenCV** — frame capture and annotation
- **Streamlit** — web interface
- **NumPy / PyTorch** — supporting libraries

## 📌 Notes

- Lower the frame width in the sidebar for higher FPS on slower machines.
- Use `yolov8n.pt` for the smoothest real-time experience without a GPU.
