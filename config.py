# config.py

# Model Settings
MODEL_NAME = "yolo26n.pt"  # Pre-trained YOLO model filename
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence threshold to consider a detection valid

# Hardware Settings
CAMERA_INDEX = 0         # By default, 0 is the built-in webcam
STREAM_MODE = True       # Generator stream mode for better multi-threading performance with YOLO

# Application Settings
WINDOW_NAME = "Real-Time Object Detection (YOLO26)"
QUIT_KEY = 'q'           # Key to press to cleanly shut down the application
