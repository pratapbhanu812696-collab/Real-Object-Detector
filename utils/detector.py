"""
ObjectDetector
--------------
Thin wrapper around a YOLOv8 model that runs inference on a single BGR
(OpenCV-style) frame and returns an annotated frame plus the raw results.
"""

from typing import Tuple

import cv2
import numpy as np


class ObjectDetector:
    def __init__(self, model, conf: float = 0.45, iou: float = 0.45):
        self.model = model
        self.conf = conf
        self.iou = iou

    def detect(self, frame: np.ndarray, draw_labels: bool = True) -> Tuple[np.ndarray, "Results"]:
        """
        Run YOLOv8 inference on a single frame.

        Returns:
            annotated_frame: frame with bounding boxes (and optionally labels) drawn
            results: the raw ultralytics Results object (first element of the batch)
        """
        outputs = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        results = outputs[0]

        annotated_frame = self._draw_boxes(frame.copy(), results, draw_labels)
        return annotated_frame, results

    @staticmethod
    def _draw_boxes(frame: np.ndarray, results, draw_labels: bool) -> np.ndarray:
        names = results.names
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            color = ObjectDetector._color_for_class(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if draw_labels:
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )
        return frame

    @staticmethod
    def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
        # Deterministic, visually distinct color per class id.
        np.random.seed(cls_id)
        return tuple(int(c) for c in np.random.randint(50, 255, size=3))
