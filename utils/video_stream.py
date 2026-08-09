"""
VideoStream
-----------
Threaded webcam reader. Reading frames on a background thread avoids
blocking the Streamlit main loop on cv2.VideoCapture.read(), which is
what actually makes the feed feel smooth/real-time (especially on macOS
where AVFoundation capture can be slow to hand back frames).
"""

import threading
import cv2


class VideoStream:
    def __init__(self, src: int = 0, width: int = 640):
        self.src = src
        self.width = width
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> "VideoStream":
        # AVFoundation backend is the reliable choice on macOS.
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_AVFOUNDATION) \
            if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.VideoCapture(self.src)

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        self.stopped = False
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                continue
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
