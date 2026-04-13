# Real-Time Object Detection

A real-time object detection web application built with Streamlit and YOLOv8 from Ultralytics. The application uses your device's webcam to capture an image and then runs object detection to identify and label multiple objects in the frame simultaneously.

## Features
- **Real-Time Webcam Analysis**: Directly capture pictures using your browser.
- **YOLOv8 Detection**: High performance and accurate object detection.
- **Modular Codebase**: Architected with best practices so you can easily modify or swap out models.

## Repository Structure
```
.
├── src/
│   ├── config.py    # Configuration constants (Model path, App Title, etc.)
│   └── model.py     # YOLO ObjectDetector abstraction layer
├── main.py          # Streamlit UI & Application Entry Point
├── yolov8n.pt       # Pre-trained Weights (downloaded automatically if missing)
├── requirements.txt # Python Dependencies
└── Dockerfile       # Instructions for Docker-based deployment
```

## Running Locally

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run main.py
   ```
   *Note: Because of our smart entrypoint script, you can also just hit "Run" in your IDE or run `python3 main.py` and it will automatically route to the Streamlit server!*

## Deployment

### Deploying to Streamlit Community Cloud
This repository is already perfectly structured for [Streamlit Community Cloud](https://streamlit.io/cloud). Just connect your GitHub repository and set `main.py` as your Main file Path.

### Deploying via Docker
You can also package this application as a Docker container using the included `Dockerfile`!
```bash
# Build the image
docker build -t object-detector .

# Run the container
docker run -p 8501:8501 object-detector
```
