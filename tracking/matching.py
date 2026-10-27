"""Geometry-based plate-to-vehicle matching shared by tracker backends."""

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
        # Reject non-vehicle classes, including person (COCO 0).
        class_id = getattr(track, "class_id", getattr(track, "det_class", None))
        if class_id is not None and class_id not in (2, 3, 5, 7):
            continue
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

