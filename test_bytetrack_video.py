"""Standalone vehicle ByteTrack benchmark (does not run OCR or modify the pipeline).

Examples (plate312 environment):
  python test_bytetrack_video.py --max-frames 300
  python test_bytetrack_video.py --video data/Download.mp4 --save-video
  python test_bytetrack_video.py --gt labels.csv --save-video

Ground truth CSV header: frame,id,x,y,w,h
Frames start at 1; coordinates are zero-based pixels in the original video.
Label ALL vehicles of the selected classes on EVERY evaluated frame, with stable
IDs. Missing rows mean no vehicles, NOT an unlabelled frame. Use a prepared CSV,
not raw MOTChallenge annotations (ignore regions/classes are not implemented).
Accuracy uses class-agnostic IoU matching; it measures detector + tracker together.
Optional accuracy dependency: python -m pip install motmetrics

Outputs: timings.csv, tracks.txt (MOT format), summary.json, optional annotated.mp4.
track_ms measures BYTETracker.update only (CPU association), detect_ms includes
GPU completion and transfer to CPU. Warm-up frames still update tracks and count
toward accuracy; they are excluded only from latency statistics. End-to-end FPS
includes decoding, evaluation, logging and optional video encoding.
"""

import argparse
import csv
from datetime import datetime
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def load_ground_truth(path):
    frames, seen = {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(("frame", "id", "x", "y", "w", "h")).issubset(reader.fieldnames or []):
            raise ValueError("GT CSV requires header: frame,id,x,y,w,h")
        for row in reader:
            frame, identity = int(row["frame"]), int(row["id"])
            box = [float(row[k]) for k in ("x", "y", "w", "h")]
            if frame < 1 or identity < 0 or not all(map(math.isfinite, box)) or min(box[2:]) <= 0:
                raise ValueError(f"Invalid GT row: {row}")
            if (frame, identity) in seen:
                raise ValueError(f"Duplicate GT identity {identity} on frame {frame}")
            seen.add((frame, identity))
            frames.setdefault(frame, []).append((identity, box))
    return frames


def latency_stats(values, np):
    if not values:
        return None
    return {"frames": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3, 5, 7], help="COCO vehicle classes")
    parser.add_argument("--conf", type=float, default=0.1, help="Detector threshold; must be <= low threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="Detector NMS IoU")
    parser.add_argument("--high-thresh", type=float, default=0.25)
    parser.add_argument("--low-thresh", type=float, default=0.1)
    parser.add_argument("--new-thresh", type=float, default=0.25)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--track-buffer", type=int, default=30, help="Frames to retain lost tracks")
    parser.add_argument("--warmup", type=int, default=20, help="Initial frames excluded from timing statistics")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing paths are never overwritten")
    parser.add_argument("--gt", type=Path, help="Fully annotated ground truth CSV; see module documentation")
    parser.add_argument("--eval-iou", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.conf <= args.low_thresh < args.high_thresh <= args.new_thresh <= 1:
        parser.error("Require 0 <= conf <= low-thresh < high-thresh <= new-thresh <= 1")
    if not (0 < args.iou <= 1 and 0 < args.eval_iou <= 1 and 0 < args.match_thresh <= 1):
        parser.error("IoU and match thresholds must be in (0, 1]")
    if args.warmup < 0 or args.track_buffer < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("warmup >= 0; track-buffer and max-frames must be positive")

    import cv2
    import numpy as np
    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent.parent
    video = args.video.resolve() if args.video else (base / cfg["video"]["source"]).resolve()
    model_path = args.model.resolve() if args.model else (base / cfg["models"]["vehicle_detector"]).resolve()
    for path in (video, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    gt, mm, accumulator = None, None, None
    if args.gt:
        gt = load_ground_truth(args.gt)
        try:
            import motmetrics as mm
        except ImportError as exc:
            raise RuntimeError("Accuracy evaluation requires: python -m pip install motmetrics") from exc
        accumulator = mm.MOTAccumulator(auto_id=True)

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(model_path), task="detect")
    tracker_args = SimpleNamespace(track_high_thresh=args.high_thresh, track_low_thresh=args.low_thresh,
                                  new_track_thresh=args.new_thresh, track_buffer=args.track_buffer,
                                  match_thresh=args.match_thresh, fuse_score=True)
    # Older releases scale track_buffer by frame_rate / 30; keep the CLI unit in frames.
    tracker = (BYTETracker(tracker_args, frame_rate=30) if "frame_rate" in inspect.signature(BYTETracker).parameters
               else BYTETracker(tracker_args))
    cap = cv2.VideoCapture(str(video))
    writer = None
    count, unique_ids, samples = 0, set(), []
    output = args.output_dir or ROOT / "output" / datetime.now().strftime("bytetrack-%Y%m%d-%H%M%S-%f")
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = float(cfg["video"].get("fallback_fps", 30))
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        stop_reason = "eof_or_decode_failure"
        with (output / "timings.csv").open("w", newline="", encoding="utf-8") as timing_file, \
                (output / "tracks.txt").open("w", encoding="utf-8") as track_file:
            timing = csv.writer(timing_file)
            timing.writerow(["frame", "warmup", "detections", "tracks", "read_ms", "detect_ms", "track_ms"])
            with torch.inference_mode():
                while args.max_frames is None or count < args.max_frames:
                    t0 = time.perf_counter()
                    ok, frame = cap.read()
                    t1 = time.perf_counter()
                    if not ok:
                        break
                    result = model.predict(frame, conf=args.conf, iou=args.iou, classes=args.classes,
                                           device=device, verbose=False)[0]
                    # CPU transfer completes GPU work before the tracking clock starts.
                    boxes = result.boxes.cpu().numpy()
                    t2 = time.perf_counter()
                    tracks = tracker.update(boxes, frame)
                    t3 = time.perf_counter()
                    count += 1
                    read_ms, detect_ms, track_ms = ((t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000)
                    measured = count > args.warmup
                    if measured:
                        samples.append((detect_ms, track_ms))
                    timing.writerow([count, int(not measured), len(boxes), len(tracks), read_ms, detect_ms, track_ms])
                    pred_ids, pred_boxes = [], []
                    for row in tracks:
                        x1, y1, x2, y2, identity, score, cls = row[:7]
                        identity = int(identity)
                        unique_ids.add(identity)
                        pred_ids.append(identity)
                        pred_boxes.append([x1, y1, x2-x1, y2-y1])
                        track_file.write(f"{count},{identity},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{score:.6f},-1,-1,-1\n")
                        if args.save_video:
                            color = ((identity*67)%200+55, (identity*37)%200+55, (identity*97)%200+55)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"ID {identity} {model.names[int(cls)]} {score:.2f}",
                                        (int(x1), max(20, int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                    if gt is not None:
                        truth = gt.get(count, [])
                        # Explicit IoU matrix avoids NumPy-version-specific helpers in motmetrics.
                        distances = iou_distances([box for _, box in truth], pred_boxes, args.eval_iou, np)
                        accumulator.update([identity for identity, _ in truth], pred_ids, distances)
                    if args.save_video:
                        if writer is None:
                            writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     fps, (frame.shape[1], frame.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError("Cannot open output video writer")
                        cv2.putText(frame, f"Frame {count} | track {track_ms:.2f} ms", (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                        writer.write(frame)
                    if count % 100 == 0:
                        print(f"frame={count} tracks={len(tracks)} detect={detect_ms:.2f}ms track={track_ms:.2f}ms", flush=True)
                else:
                    stop_reason = "max_frames"
        if writer is not None:
            writer.release()
            writer = None
        elapsed = time.perf_counter() - start
        if not count:
            raise RuntimeError("No frames decoded")
        accuracy = None
        if gt is not None:
            metrics = ["num_frames", "num_objects", "idf1", "mota", "precision", "recall",
                       "num_switches", "num_false_positives", "num_misses", "num_fragmentations"]
            values = mm.metrics.create().compute(accumulator, metrics=metrics, name="video").loc["video"]
            accuracy = {key: float(val) if math.isfinite(float(val)) else None for key, val in values.items()}
        detection = latency_stats([s[0] for s in samples], np)
        tracking = latency_stats([s[1] for s in samples], np)
        combined = latency_stats([sum(s) for s in samples], np)
        summary = {"video": str(video), "model": str(model_path), "device": device,
                   "ultralytics_version": ultralytics.__version__, "parameters": vars(tracker_args),
                   "detector_conf": args.conf, "detector_iou": args.iou, "classes": args.classes,
                   "source_fps": fps, "source_reported_frames": reported_frames, "processed_frames": count,
                   "stop_reason": stop_reason, "possible_early_decode_end": stop_reason != "max_frames" and count < reported_frames,
                   "warmup_frames": min(count, args.warmup), "unique_track_ids": len(unique_ids),
                   "detect": detection, "track": tracking, "detect_and_track": combined,
                   "compute_fps": 1000/combined["mean_ms"] if combined and combined["mean_ms"] > 0 else None,
                   "elapsed_seconds": elapsed, "end_to_end_fps": count/elapsed,
                   "ground_truth": str(args.gt) if args.gt else None, "eval_iou": args.eval_iou,
                   "accuracy": accuracy,
                   "accuracy_note": "Prepared CSV, class-agnostic matching; all processed frames included."
                   if gt is not None else "Not measured: no ground truth. Unique ID count is not accuracy."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f"Outputs: {output.resolve()}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def iou_distances(gt_boxes, pred_boxes, threshold, np):
    """MOT distances: 1-IoU, with forbidden matches represented by NaN."""
    a = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, :2]+a[:, None, 2:], b[None, :, :2]+b[None, :, 2:])
    intersection = np.maximum(br-tl, 0).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None]+b[:, 2:].prod(axis=1)[None, :]-intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return np.where(iou >= threshold, 1-iou, np.nan)


if __name__ == "__main__":
    main()
