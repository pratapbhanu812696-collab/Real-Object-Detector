import cv2
import config
from detector import RealTimeDetector

def main():
    print(f"Loading object detector using {config.MODEL_NAME}...")
    try:
        detector = RealTimeDetector(model_name=config.MODEL_NAME)
    except Exception as e:
        print("Failed to initialize detector. Exiting.")
        return

    print("Initializing webcam...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {config.CAMERA_INDEX}.")
        return

    print(f"Webcam running. Press '{config.QUIT_KEY}' to quit.")

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to read frame from webcam.")
            break

        # Process frame to detect objects
        annotated_frame = detector.predict_and_plot(
            frame=frame, 
            stream=config.STREAM_MODE, 
            conf=config.CONFIDENCE_THRESHOLD
        )

        # Display the resulting frame
        cv2.imshow(config.WINDOW_NAME, annotated_frame)

        # Break the loop on the dedicated quit key press
        if cv2.waitKey(1) & 0xFF == ord(config.QUIT_KEY):
            break

    # Release resources cleanly
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()
