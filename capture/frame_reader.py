import queue
import threading
import time

import cv2


class FrameReader(threading.Thread):
    def __init__(self, video_path, frame_queue, fallback_fps=25):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.cap.release()
            raise FileNotFoundError(f"Không mở được video: {video_path}")
        self.frame_queue = frame_queue
        self._halt = threading.Event()

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps > 240:
            self.fps = fallback_fps
        self.frame_interval = 1.0 / self.fps

    def run(self):
        frame_nmr = -1
        next_frame_time = time.time()
        try:
            while not self._halt.is_set():#còn được phép duyệt thì tiếp tục
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_nmr += 1
                if not self._put((frame_nmr, frame)):
                    break

                next_frame_time += self.frame_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    if self._halt.wait(sleep_time):
                        break
                else:
                    next_frame_time = time.time()
        finally:
            self._put(None)
            self.cap.release()

    def _put(self, item):
        """Queue đầy thì bỏ frame cũ. None luôn phải lọt vào để consumer khỏi chờ mãi."""
        while True:
            if item is not None and self._halt.is_set():
                return False
            try:
                self.frame_queue.put_nowait(item)
                return True
            except queue.Full:
                try:
                    dropped = self.frame_queue.get_nowait()
                except queue.Empty:
                    continue
                if dropped is None:
                    self.frame_queue.put_nowait(None)
                    return item is None

    def stop(self):
        self._halt.set()
