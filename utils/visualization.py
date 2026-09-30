import cv2


def draw_results(frame, frame_results):
    """
    frame_results : dict track_id:{"vehicle_bbox": [...], "plate_text": str}
    Khóa __plates__ là các bbox biển nằm trong một xe đang được track.
    """
    for box in frame_results.get("__plates__", []):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for track_id, result in frame_results.items():
        if track_id in ("__plates__", "__timing__"):
            continue
        x1, y1, x2, y2 = map(int, result['vehicle_bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = result.get('plate_text') or ''
        cv2.putText(frame, f"ID:{track_id} {text}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame