from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

class ObjectTracker:
    def __init__(self, max_age=30, n_init=3, embedder="mobilenet",
                 max_cosine_distance=0.2, nn_budget=100, device="cuda"):
        use_gpu = device.startswith("cuda")
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder=embedder,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder_gpu=use_gpu,
            half=use_gpu,
            bgr=True,
        )

    def update(self, detections, frame):
        formated_detections = []
        # deep_sort_realtime yêu cầu format [([left, top, w, h], conf, class), ...]

        for x1, y1, x2, y2, conf, class_id in detections:
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            formated_detections.append(([x1, y1, width, height], conf, int(class_id)))

        # Vẫn gọi update_tracks khi không có detection để track cũ được age/xoá đúng
        #tracks = self.tracker.update_tracks(formated_detections, frame=frame)
        with torch.inference_mode():
            tracks = self.tracker.update_tracks(formated_detections, frame=frame)
        confirmed_tracks = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            confirmed_tracks.append(track)

        return confirmed_tracks



# Compatibility exports for existing DeepSORT callers.
from tracking.matching import compute_intersection, compute_iou, match_plate_to_tracks
