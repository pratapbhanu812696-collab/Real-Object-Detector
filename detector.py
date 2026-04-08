import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_version="yolo26n.pt", conf=0.4):
        # Load the latest YOLO model
        self.model = YOLO(model_version)
        self.conf = conf
    
    def predict_and_annotate(self, frame):
        # Run Inference using generator for memory efficiency
        results = self.model.predict(source=frame, stream=True, conf=self.conf)
        
        # Process and visualize results
        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot()
            
        return annotated_frame
