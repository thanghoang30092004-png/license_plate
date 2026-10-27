"""Bounded snapshot writer. stop() seals submissions and drains accepted work."""
from collections import deque
from datetime import datetime
import logging
import os
import queue
import re
import threading
from uuid import uuid4

import cv2


class SnapshotSaver(threading.Thread):
    def __init__(self, output_dir, queue_max_size=50, on_saved=None):
        super().__init__(daemon=True)
        if queue_max_size < 1:
            raise ValueError('queue_max_size must be positive')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.queue = queue.Queue(maxsize=queue_max_size)
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._pending = set()
        self.on_saved = on_saved
        self.saved_count = self.failed_count = self.dropped_count = self.callback_error_count = 0
        self.errors = deque(maxlen=50)

    def submit(self, vehicle_crop, track_id, plate_text, frame_nmr):
        """Return False if stopped, empty or this track already has a pending image."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return False
        with self._lock:
            if self._halt.is_set() or track_id in self._pending:
                return False
            item = (vehicle_crop.copy(), track_id, plate_text, frame_nmr)
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                try:
                    dropped = self.queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self._pending.discard(dropped[1])
                    self.queue.task_done()
                    self.dropped_count += 1
                self.queue.put_nowait(item)
            self._pending.add(track_id)
            return True

    def run(self):
        while not self._halt.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            crop, track_id, plate_text, frame_nmr = item
            try:
                path = self._write_to_disk(crop, track_id, plate_text, frame_nmr)
                if path is None:
                    self.failed_count += 1
                else:
                    self.saved_count += 1
                    if self.on_saved is not None:
                        try:
                            self.on_saved(track_id, path)
                        except Exception as exc:
                            self.callback_error_count += 1
                            self._record_error('callback', track_id, exc)
            except Exception as exc:
                self.failed_count += 1
                self._record_error('write', track_id, exc)
            finally:
                with self._lock:
                    self._pending.discard(track_id)
                self.queue.task_done()

    def _record_error(self, stage, track_id, error):
        message = f'Snapshot {stage} failed for track {track_id}: {error}'
        self.errors.append(message)
        logging.getLogger(__name__).error(message)

    def _write_to_disk(self, crop, track_id, plate_text, frame_nmr):
        if crop is None or crop.size == 0:
            return None
        safe_plate = self._sanitize_filename(plate_text) or 'unknown'
        safe_track = self._sanitize_filename(str(track_id))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'{safe_plate}_track{safe_track}_frame{frame_nmr}_{timestamp}_{uuid4().hex[:8]}.jpg'
        path = os.path.join(self.output_dir, filename)
        try:
            if not cv2.imwrite(path, crop):
                raise OSError(f'cv2.imwrite returned False: {path}')
            return path
        except Exception as exc:
            self._record_error('write', track_id, exc)
            return None

    @staticmethod
    def _sanitize_filename(text):
        return re.sub(r'[^A-Za-z0-9\-.]', '_', str(text)) if text else ''

    def stop(self):
        with self._lock:
            self._halt.set()
