import cv2
from ultralytics import YOLO

class RealTimeDetector:
    def __init__(self, model_name="yolo26n.pt"):
        """
        Initializes the YOLO model with the given model name.
        Defaults to 'yolo26n.pt'
        """
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Error loading YOLO model '{model_name}': {e}")
            raise

    def predict_and_plot(self, frame, stream=True, conf=0.4):
        """
        Runs the YOLO model on a single frame and returns the annotated frame.
        """
        # Run Inference
        results = self.model.predict(source=frame, stream=stream, conf=conf)
        
        # We need an initial frame in case results are empty
        annotated_frame = frame 

        for r in results:
            annotated_frame = r.plot()
        
        return annotated_frame
