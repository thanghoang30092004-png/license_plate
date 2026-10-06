import os 
import re
import threading 
import cv2 
from datetime import datetime 
import queue

class SnapshotSaver(threading.Thread):
    def __init__(self, output_dir, queue_max_size=50, on_saved = None):
        super().__init__(daemon=True)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=queue_max_size)
        self._halt = threading.Event()# tín hiệu thread dừng 
        self.on_saved = on_saved


    def submit(self, vehicle_crop, track_id, plate_text, frame_nmr):
        # Gửi ảnh crop của vehicle (có thể có plate) để lưu vào thư mục snapshot.
        # Nếu queue đầy, bỏ ảnh cũ nhất.
        item = (vehicle_crop, track_id, plate_text, frame_nmr)
        try:
            self.queue.put_nowait(item)

        except queue.Full:
            try:
                self.queue.get_nowait()  # Bỏ ảnh cũ nhất
            except:
                pass
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                print("SnapshotSaver queue is full, cannot save snapshot.")


    def run(self):
        while not self._halt.is_set():
            try :
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            vehicle_crop, track_id, plate_text, frame_nmr = item
            filepath = self._write_to_disk(vehicle_crop, track_id, plate_text, frame_nmr)

            if filepath and self.on_saved:
                self.on_saved(track_id, filepath)

    def _write_to_disk(self, vehicle_crop, track_id, plate_text, frame_nmr):
        if vehicle_crop is None or vehicle_crop.size == 0:
            print(f"[SnapshotWriter] Bỏ qua track {track_id}: crop rỗng.")
            return None

        safe_plate = self._sanitize_filename(plate_text) or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_plate}_track{track_id}_frame{frame_nmr}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        try:
            cv2.imwrite(filepath, vehicle_crop)  # I/O chậm, nhưng giờ chạy ở thread riêng
            return filepath
        except Exception as e:
            print(f"[SnapshotWriter] Lỗi khi lưu ảnh track {track_id}: {e}")
            return None

    @staticmethod
    def _sanitize_filename(text):
        if not text:
            return ""
        return re.sub(r'[^A-Za-z0-9\-\.]', '_', text)

    def stop(self):
        self._halt.set()
