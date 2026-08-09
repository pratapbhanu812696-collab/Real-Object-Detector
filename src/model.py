import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        """
        Initializes the YOLO model with the given model path.
        """
        self.model = YOLO(model_path)

    def predict_and_annotate(self, image):
        """
        Runs inference on the image and returns the annotated frame.
        """
        # Inference
        results = self.model(image)

        # Plot results
        for r in results:
            annotated_frame = r.plot()
            # Convert BGR to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            return annotated_frame
            
        return image
