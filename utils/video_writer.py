import cv2


class TimelineVideoWriter:
    """Keep source timestamps by repeating the last image across dropped frames."""

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Cannot open output video: {path}")
        self.next_frame = 0
        self.last_frame = None

    def write(self, frame_number, frame):
        if frame_number < self.next_frame:
            raise ValueError("Frame numbers must increase")
        filler = self.last_frame if self.last_frame is not None else frame
        while self.next_frame < frame_number:
            self.writer.write(filler)
            self.next_frame += 1
        self.writer.write(frame)
        self.next_frame += 1
        self.last_frame = frame.copy()

    def finish(self, source_frames):
        if self.last_frame is not None:
            while self.next_frame < source_frames:
                self.writer.write(self.last_frame)
                self.next_frame += 1

    def release(self):
        self.writer.release()
