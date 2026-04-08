import cv2
from detector import ObjectDetector

def main():
    detector = ObjectDetector()
    
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break

        # Get annotated frame from the detector
        annotated_frame = detector.predict_and_annotate(frame)

        # Display the resulting frame
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
