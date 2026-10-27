"""So sanh detect bien toan anh voi detect batch tren cac crop xe (khong OCR).

Vi du:
    python tests/bench_plate_roi_batch.py --start 1000 --frames 200
    python tests/bench_plate_roi_batch.py --frames 300 --full-imgsz 640 --roi-imgsz 320

File CSV ghi thoi gian tung frame. Phan tong ket duoc in ra man hinh.
"""
import argparse
import csv
from pathlib import Path
import statistics
import sys
import time

import cv2
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.detection import ObjectDetector
from detector.plate_detector import PlateDetector


def project_path(config_path, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(config_path).resolve().parent.parent / path).resolve())


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(callable_):
    sync_cuda()
    started = time.perf_counter()
    result = callable_()
    sync_cuda()
    return result, (time.perf_counter() - started) * 1000.0


def intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box1, box2):
    inter = intersection(box1, box2)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def contained_in_vehicle(plate, vehicle, min_containment):
    area = max(0.0, plate[2] - plate[0]) * max(0.0, plate[3] - plate[1])
    if area <= 0:
        return False
    center_x = (plate[0] + plate[2]) / 2.0
    center_y = (plate[1] + plate[3]) / 2.0
    center_inside = vehicle[0] <= center_x <= vehicle[2] and vehicle[1] <= center_y <= vehicle[3]
    return center_inside and intersection(plate, vehicle) / area >= min_containment


def valid_full_frame_plates(plates, vehicles, min_containment):
    return [plate for plate in plates
            if any(contained_in_vehicle(plate, vehicle, min_containment) for vehicle in vehicles)]


def crop_vehicles(frame, vehicles, padding_ratio):
    height, width = frame.shape[:2]
    crops = []
    origins = []
    source_vehicles = []
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle[:4]
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        left = max(0, min(width, int(x1 - pad_x)))
        top = max(0, min(height, int(y1 - pad_y)))
        right = max(0, min(width, int(x2 + pad_x)))
        bottom = max(0, min(height, int(y2 + pad_y)))
        if right <= left or bottom <= top:
            continue
        crop = frame[top:bottom, left:right]
        if crop.size:
            crops.append(crop)
            origins.append((left, top))
            source_vehicles.append(vehicle)
    return crops, origins, source_vehicles


def box_rows(result):
    boxes = getattr(result, "boxes", None)
    return [] if boxes is None or len(boxes) == 0 else boxes.data.tolist()


def nms(boxes, threshold=0.5):
    """Gop cac bien trung nhau khi hai box xe bi chong lan."""
    ordered = sorted(boxes, key=lambda row: row[4], reverse=True)
    kept = []
    for candidate in ordered:
        if all(iou(candidate, previous) < threshold for previous in kept):
            kept.append(candidate)
    return kept


def detect_plate_batch(plate_detector, frame, vehicles, imgsz, padding_ratio,
                       min_containment):
    crops, origins, source_vehicles = crop_vehicles(frame, vehicles, padding_ratio)
    if not crops:
        return [], 0
    results = plate_detector.model.predict(
        crops,
        conf=plate_detector.conf_threshold,
        iou=plate_detector.iou_threshold,
        device=plate_detector.device,
        imgsz=imgsz,
        batch=len(crops),
        verbose=False,
    )
    global_boxes = []
    for result, (offset_x, offset_y), vehicle in zip(results, origins, source_vehicles):
        for x1, y1, x2, y2, confidence, class_id in box_rows(result):
            if not plate_detector.is_valid_plate(x1, y1, x2, y2):
                continue
            global_box = (x1 + offset_x, y1 + offset_y,
                          x2 + offset_x, y2 + offset_y,
                          confidence, class_id)
            if contained_in_vehicle(global_box, vehicle, min_containment):
                global_boxes.append(global_box)
    return nms(global_boxes), len(crops)


def compare_boxes(reference, candidate, threshold=0.5):
    pairs = sorted(((iou(a, b), ai, bi)
                    for ai, a in enumerate(reference)
                    for bi, b in enumerate(candidate)), reverse=True)
    used_a, used_b, matched_ious = set(), set(), []
    for overlap, ai, bi in pairs:
        if overlap < threshold or ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        matched_ious.append(overlap)
    return len(matched_ious), len(reference) - len(used_a), len(candidate) - len(used_b), matched_ious


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percent / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values):
    return statistics.median(values), percentile(values, 95), statistics.mean(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--video", help="Mac dinh dung video trong config")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--full-imgsz", type=int, default=640,
                        help="Input size detect bien toan anh, mac dinh 640")
    parser.add_argument("--roi-imgsz", type=int, default=640,
                        help="Input size cho moi crop xe trong batch, mac dinh 640")
    parser.add_argument("--crop-padding", type=float, default=0.02,
                        help="Padding quanh box xe, theo ti le kich thuoc box")
    parser.add_argument("--min-containment", type=float, default=0.9)
    parser.add_argument("--output", default=str(ROOT / "output" / "plate_roi_batch_timing.csv"))
    args = parser.parse_args()
    if args.start < 0 or args.frames < 1 or args.warmup < 1:
        parser.error("start >= 0; frames va warmup >= 1")
    if not 0 <= args.crop_padding <= 1 or not 0 <= args.min_containment <= 1:
        parser.error("crop-padding va min-containment phai nam trong [0, 1]")

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    video_path = args.video or project_path(args.config, cfg["video"]["source"])
    vehicle_detector = ObjectDetector(
        project_path(args.config, cfg["models"]["vehicle_detector"]),
        conf_threshold=cfg["detection"]["vehicle_conf_threshold"])
    plate_detector = PlateDetector(
        project_path(args.config, cfg["models"]["plate_detector"]),
        conf_threshold=cfg["detection"]["plate_conf_threshold"],
        min_ratio=cfg["detection"]["plate_min_ratio"],
        max_ratio=cfg["detection"]["plate_max_ratio"])

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, warm_frame = capture.read()
    if not ok:
        raise RuntimeError(f"Khong doc duoc frame {args.start}")

    print(f"Video: {video_path}")
    print(f"Bat dau frame {args.start}, so frame {args.frames}, "
          f"full imgsz {args.full_imgsz}, roi imgsz {args.roi_imgsz}")
    print("Warm-up model...", flush=True)
    for _ in range(args.warmup):
        vehicles = vehicle_detector.detect(warm_frame)
        plate_detector.model.predict(warm_frame, conf=plate_detector.conf_threshold,
                                     iou=plate_detector.iou_threshold,
                                     device=plate_detector.device, imgsz=args.full_imgsz,
                                     verbose=False)
        detect_plate_batch(plate_detector, warm_frame, vehicles,
                           args.roi_imgsz, args.crop_padding, args.min_containment)

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    rows = []
    all_match_ious = []
    for offset in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break
        frame_number = args.start + offset
        vehicles, vehicle_ms = timed(lambda: vehicle_detector.detect(frame))

        def full_call():
            raw = plate_detector.model.predict(
                frame, conf=plate_detector.conf_threshold,
                iou=plate_detector.iou_threshold,
                device=plate_detector.device, imgsz=args.full_imgsz, verbose=False)
            plates = []
            for values in box_rows(raw[0]):
                if plate_detector.is_valid_plate(*values[:4]):
                    plates.append(tuple(values))
            return valid_full_frame_plates(plates, vehicles, args.min_containment)

        def roi_call():
            return detect_plate_batch(plate_detector, frame, vehicles,
                                      args.roi_imgsz, args.crop_padding,
                                      args.min_containment)

        # Doi thu tu moi frame de giam sai lech do cache va nhiet GPU.
        if offset % 2 == 0:
            full_plates, full_ms = timed(full_call)
            (roi_plates, crop_count), roi_ms = timed(roi_call)
        else:
            (roi_plates, crop_count), roi_ms = timed(roi_call)
            full_plates, full_ms = timed(full_call)

        matched, missed, extra, match_ious = compare_boxes(full_plates, roi_plates)
        all_match_ious.extend(match_ious)
        row = {
            "frame": frame_number,
            "vehicles": len(vehicles),
            "vehicle_ms": round(vehicle_ms, 3),
            "full_plate_ms": round(full_ms, 3),
            "roi_batch_plate_ms": round(roi_ms, 3),
            "full_total_ms": round(vehicle_ms + full_ms, 3),
            "roi_batch_total_ms": round(vehicle_ms + roi_ms, 3),
            "vehicle_crops": crop_count,
            "full_valid_plates": len(full_plates),
            "roi_batch_plates": len(roi_plates),
            "matched_iou_05": matched,
            "missed_vs_full": missed,
            "extra_vs_full": extra,
        }
        rows.append(row)
        print(
            f"frame {frame_number}: xe={len(vehicles)} vehicle={vehicle_ms:.1f}ms | "
            f"full={full_ms:.1f}ms total={vehicle_ms + full_ms:.1f}ms plates={len(full_plates)} | "
            f"roi_batch={roi_ms:.1f}ms total={vehicle_ms + roi_ms:.1f}ms "
            f"crops={crop_count} plates={len(roi_plates)} match/miss/extra={matched}/{missed}/{extra}",
            flush=True)
    capture.release()
    if not rows:
        raise RuntimeError("Khong co frame nao duoc benchmark")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    vehicle_stat = stats([row["vehicle_ms"] for row in rows])
    full_stat = stats([row["full_total_ms"] for row in rows])
    roi_stat = stats([row["roi_batch_total_ms"] for row in rows])
    full_plate_stat = stats([row["full_plate_ms"] for row in rows])
    roi_plate_stat = stats([row["roi_batch_plate_ms"] for row in rows])
    gain = (full_stat[0] - roi_stat[0]) / full_stat[0] * 100.0 if full_stat[0] else 0.0
    total_missed = sum(row["missed_vs_full"] for row in rows)
    total_extra = sum(row["extra_vs_full"] for row in rows)
    mean_match_iou = statistics.mean(all_match_ious) if all_match_ious else 0.0

    print("\n=== TONG KET (median / p95 / mean) ===")
    print(f"Detect xe:             {vehicle_stat[0]:.1f} / {vehicle_stat[1]:.1f} / {vehicle_stat[2]:.1f} ms")
    print(f"Bien toan anh:         {full_plate_stat[0]:.1f} / {full_plate_stat[1]:.1f} / {full_plate_stat[2]:.1f} ms")
    print(f"Bien batch crop xe:    {roi_plate_stat[0]:.1f} / {roi_plate_stat[1]:.1f} / {roi_plate_stat[2]:.1f} ms")
    print(f"Tong cach hien tai:    {full_stat[0]:.1f} / {full_stat[1]:.1f} / {full_stat[2]:.1f} ms")
    print(f"Tong cach batch crop:  {roi_stat[0]:.1f} / {roi_stat[1]:.1f} / {roi_stat[2]:.1f} ms")
    print(f"Thay doi median:       {gain:+.1f}% (duong la nhanh hon)")
    print(f"So voi full-frame: missed={total_missed}, extra={total_extra}, mean matched IoU={mean_match_iou:.3f}")
    print(f"CSV: {output_path}")
    if gain > 0 and total_missed == 0:
        print("KET LUAN: batch crop nhanh hon tren mau nay va khong mat box so voi full-frame.")
    elif gain > 0:
        print("KET LUAN: batch crop nhanh hon, nhung co khac biet detection; can xem lai cac frame missed/extra.")
    else:
        print("KET LUAN: batch crop khong nhanh hon voi imgsz va so xe cua mau nay.")


if __name__ == "__main__":
    main()
