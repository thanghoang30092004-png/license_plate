import queue
import threading
import time

from tracking.deepsort_tracker import match_plate_to_tracks
from utils.snapshot_saver import SnapshotSaver

class InferenceWorker(threading.Thread):

    def __init__(self, frame_queue, result_queue, vehicle_detector,
                 plate_detector, tracker, ocr_engine, track_cache
                 , snapshot_writer = None):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.tracker = tracker
        self.ocr_engine = ocr_engine
        self.track_cache = track_cache
        self.snapshot_writer = snapshot_writer
        self.stopped = False
        self.error = None
        self.ocr_call_count = 0
        self.ocr_crop_count = 0
        self.ocr_total_ms = 0.0
        self.ocr_max_ms = 0.0
        self._halt = threading.Event()


    def run(self):
        try:
            while not self._halt.is_set():
                try:
                    item = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                self._process(item)
        except Exception as exc:
            self.error = exc
        finally:
            self._put_result(None)

    def _process(self, item):
        frame_nmr, frame = item
        started = time.perf_counter()

        mark = time.perf_counter()
        vehicle_detections = self.vehicle_detector.detect(frame)
        vehicle_ms = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        tracks = self.tracker.update(vehicle_detections, frame)
        track_ms = (time.perf_counter() - mark) * 1000

        active_track_ids = {t.track_id for t in tracks}
        self.track_cache.cleanup(active_track_ids)

        mark = time.perf_counter()
        plate_detections = self.plate_detector.detect(frame)
        plate_ms = (time.perf_counter() - mark) * 1000
        plate_boxes = []

        crops = []
        crop_metadata = []
        frame_results = {}
        # Mọi track đang active đều có dòng kết quả, kể cả khi frame này không OCR.
        for track in tracks:
            x1, y1, x2, y2 = track.to_ltrb()
            cached_text = self.track_cache.get_text(track.track_id)
            frame_results[track.track_id] = {
                "vehicle_bbox": [x1, y1, x2, y2],
                "plate_text": cached_text,
                "confidence": None,
                "valid": cached_text is not None,
                "confirmed": cached_text is not None,
                "source": "cache",
            }

        for plate_bbox in plate_detections:
            track = match_plate_to_tracks(plate_bbox, tracks)
            if track is None:
                continue
            plate_boxes.append(list(plate_bbox[:4]))
            track_id = track.track_id

            if not self.track_cache.should_run_ocr(track_id, frame_nmr):
                continue

            height, width = frame.shape[:2]
            x1, y1, x2, y2 = (int(value) for value in plate_bbox[:4])
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue
            crops.append(plate_crop)
            crop_metadata.append(track_id)

        ocr_ms = 0.0
        if crops:
            mark = time.perf_counter()
            ocr_results = self.ocr_engine.read_batch(crops)
            # read_batch includes preprocessing, synchronous inference and text parsing.
            ocr_ms = (time.perf_counter() - mark) * 1000
            self.ocr_call_count += 1
            self.ocr_crop_count += len(crops)
            self.ocr_total_ms += ocr_ms
            self.ocr_max_ms = max(self.ocr_max_ms, ocr_ms)
            if len(ocr_results) != len(crops):
                raise ValueError("OCR result count does not match crop count")
            for track_id, ocr_result in zip(crop_metadata, ocr_results):
                self.track_cache.update(track_id, ocr_result, frame_nmr)
                if track_id not in frame_results:
                    continue
                frame_results[track_id]["confidence"] = ocr_result["confidence"]
                confirmed_text = self.track_cache.get_text(track_id)
                frame_results[track_id]["valid"] = confirmed_text is not None
                frame_results[track_id]["ocr_text"] = ocr_result["text"]
                frame_results[track_id]["source"] = "ocr"
                frame_results[track_id]["plate_text"] = confirmed_text
                frame_results[track_id]["confirmed"] = confirmed_text is not None
                if confirmed_text is not None and confirmed_text != ocr_result["text"]:
                    frame_results[track_id]["confidence"] = None

        # Check after OCR updates the cache, including frames where OCR was skipped.
        if self.snapshot_writer is not None:
            for track in tracks:
                track_id = track.track_id
                if self.track_cache.should_save_snapshot(track_id):
                    confirmed_text = self.track_cache.get_text(track_id)
                    if confirmed_text is not None:
                        self._submit_snapshot(
                            frame, track.to_ltrb(), track_id, confirmed_text, frame_nmr)

        frame_results["__plates__"] = plate_boxes
        frame_results["__timing__"] = {
            "xe": vehicle_ms,
            "bien": plate_ms,
            "track": track_ms,
            "ocr": ocr_ms,
            "ocr_ran": bool(crops),
            "ocr_crops": len(crops),
            "tong": (time.perf_counter() - started) * 1000,
        }
        self._put_result((frame_nmr, frame, frame_results))

    def _put_result(self, item):
        """Queue đầy thì bỏ kết quả cũ. None luôn phải vào để phía nhận khỏi chờ mãi."""
        while True:
            try:
                self.result_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    dropped = self.result_queue.get_nowait()
                except queue.Empty:
                    continue
                if dropped is None:
                    self.result_queue.put_nowait(None)
                    return

    def stop(self):
        self.stopped = True
        self._halt.set()


    def _submit_snapshot(self, frame, vehicle_bbox, track_id, plate_text, frame_nmr):
        vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)
        height, width = frame.shape[:2]
        vx1, vx2 = max(0, min(vx1, width)), max(0, min(vx2, width))
        vy1, vy2 = max(0, min(vy1, height)), max(0, min(vy2, height))
        vehicle_crop = frame[vy1:vy2, vx1:vx2]

        if vehicle_crop.size == 0:
            #print(f"[InferenceWorker] Bỏ qua track {track_id}: vehicle crop rỗng.")
            return

        self.snapshot_writer.submit(vehicle_crop.copy(), track_id, plate_text, frame_nmr)

    
