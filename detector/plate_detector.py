from ultralytics import YOLO 


class PlateDetector:
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.4, device="cuda",
                 min_ratio=0.8, max_ratio=5.0):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for x1, y1, x2, y2, conf, class_id in _box_rows(results):
            if self.is_valid_plate(x1, y1, x2, y2):
                detections.append((x1, y1, x2, y2, conf, class_id))
        return detections

    def is_valid_plate(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        ratio = width / height
        return self.min_ratio <= ratio <= self.max_ratio


def _box_rows(results):
    if isinstance(results, (list, tuple)):
        if not results:
            return []
        results = results[0]
    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    return boxes.data.tolist()
    
    
        
        
