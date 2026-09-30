from ultralytics import YOLO
import cv2
import numpy as np 

class VehicleDetection:
    VEHICLE_CLASSES = [2, 3, 5, 7]
    def __init__(self, model_path, conf_threshold = 0.3, iou_threshold = 0.4, device = "cuda"):
        self.model= YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
    

    def detect(self, frame):
        results = self.model(frame, device=self.device, conf=self.conf_threshold,
                             iou=self.iou_threshold, verbose=False)
        detections = []
        for x1, y1, x2, y2, conf, class_id in _box_rows(results):
            if int(class_id) in self.VEHICLE_CLASSES:
                detections.append((x1, y1, x2, y2, conf, class_id))
        return detections


def _box_rows(results):
    """YOLO trả về list Results. Không có box thì trả list rỗng."""
    if isinstance(results, (list, tuple)):
        if not results:
            return []
        results = results[0]
    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    return boxes.data.tolist()
