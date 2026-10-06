from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

class VehicleTracker:
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



def compute_intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_iou(box1, box2):
    inter_area = compute_intersection(box1, box2)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_plate_to_tracks(license_plate, vehicle_tracks, min_containment=0.9):
    """
    license_plate: [x1, y1, x2, y2, ...]
    vehicle_tracks: list track của DeepSort
    return: track khớp nhất, hoặc None nếu không có xe nào chứa biển số
    """
    x1, y1, x2, y2 = license_plate[:4]
    plate_box = [x1, y1, x2, y2]
    plate_area = (x2 - x1) * (y2 - y1)
    if plate_area <= 0:
        return None

    plate_center_x = (x1 + x2) / 2
    plate_center_y = (y1 + y2) / 2

    best_match = None
    best_iou = 0.0
    for track in vehicle_tracks:
        xcar1, ycar1, xcar2, ycar2 = track.to_ltrb()

        center_inside = xcar1 <= plate_center_x <= xcar2 and ycar1 <= plate_center_y <= ycar2
        if not center_inside:
            continue

        # Không đòi biển nằm lọt hoàn toàn: 2 model detect độc lập nên box hay lệch vài pixel
        car_box = [xcar1, ycar1, xcar2, ycar2]
        containment = compute_intersection(plate_box, car_box) / plate_area
        if containment < min_containment:
            continue

        # IoU nhỏ hơn với xe to hơn, nên tiêu chí này ưu tiên xe ôm sát biển số nhất
        iou = compute_iou(plate_box, car_box)
        if iou > best_iou:
            best_iou = iou
            best_match = track

    return best_match

