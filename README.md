# Real-Time Object Detector

A lightweight, real-time object detection module built using Python, OpenCV, and the pre-trained YOLO model (Ultralytics).

## Project Structure

- `main.py`: The application entry point. Initializes the camera stream and coordinates between config and detection.
- `detector.py`: Contains the `RealTimeDetector` logic for encapsulating the YOLO model and single-frame predictions.
- `config.py`: Hardcoded settings and configurations to tweak the experience (e.g. camera index, model type).
- `requirements.txt`: Python package dependencies.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone <your-repository-url>
   cd Real-Time-Object-Detector
   ```

2. (Optional) Create and activate a Virtual Environment.

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Simply run the application using Python:
```bash
python main.py
```

Press `q` within the live video window to quit the application gracefully.

## Customization

You can change detection properties inside `config.py` easily. For example, if you want your default webcam to be substituted with an external camera, you can change `CAMERA_INDEX` to `1`.
