import os       
import queue

import cv2
import yaml


from detector.detection import ObjectDetector
from detector.plate_detector import PlateDetector
from tracking.bytetrack_tracker import ObjectTracker
from ocr.read_ocr_engine import PlateOCR
from tracking.track_cache import TrackCache
from utils.visualization import draw_results
from capture.frame_reader import FrameReader
from pipeline.inference_worker import InferenceWorker
from utils.video_writer import TimelineVideoWriter
from utils.snapshot_saver import SnapshotSaver

def load_config(config_path=r"D:\license_plate_data\config\config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _project_path(config_path, path):
    if os.path.isabs(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.normpath(os.path.join(root, path))


def _write_frame_log(log_file, frame_nmr, frame_results):
    timing = frame_results.get("__timing__") or {}
    ocr_ran = timing.get('ocr_ran')
    ocr_ms = timing.get('ocr')
    if ocr_ran is False:
        ocr_timing = "ocr skipped"
    elif ocr_ms is None:
        ocr_timing = "ocr n/a"
    else:
        ocr_timing = f"ocr {ocr_ms:.3f} ms"
    timing_text = (
        f"detect_objects {timing.get('xe', 0):.3f} ms | "
        f"bien {timing.get('bien', 0):.3f} ms | "
        f"track {timing.get('track', 0):.3f} ms | "
        f"person_track {timing.get('person_track', 0):.3f} ms | "
        f"{ocr_timing} | ocr_crops={timing.get('ocr_crops', 'n/a')} | "
        f"tong {timing.get('tong', 0):.3f} ms"
    )
    parts = []
    for track_id, row in frame_results.items():
        if track_id in ("__plates__", "__timing__", "__persons__"):
            continue
        shown = row.get("plate_text") or "-"
        ocr_text = row.get("ocr_text") or "-"
        confidence = row.get("confidence")
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else "-"
        parts.append(
            f"track {track_id} hien={shown} ocr={ocr_text} "
            f"conf={confidence_text} valid={row.get('valid')} source={row.get('source')}"
        )
    for person_id, row in frame_results.get("__persons__", {}).items():
        parts.append(f"person P:{person_id} bbox={row['person_bbox']}")
    detail = " | ".join(parts) if parts else "khong co xe"
    log_file.write(f"frame {frame_nmr} | {timing_text} | {detail}\n")
    log_file.flush()


def _write_ocr_summary(log_file, worker):
    calls = worker.ocr_call_count
    mean_ms = worker.ocr_total_ms / calls if calls else 0.0
    # Worker counters also include frames discarded by the result queue.
    log_file.write(
        f"summary | ocr_calls={calls} | ocr_crops={worker.ocr_crop_count} | "
        f"ocr_total={worker.ocr_total_ms:.3f} ms | "
        f"ocr_mean_per_call={mean_ms:.3f} ms | "
        f"ocr_max_per_call={worker.ocr_max_ms:.3f} ms\n")
    log_file.flush()




def main(config_path=r"D:\license_plate_data\config\config.yaml", display=None):
    cfg = load_config(config_path)
    tracking_cfg = cfg['tracking']
    if tracking_cfg.get('type', 'bytetrack') != 'bytetrack':
        raise ValueError('This pipeline uses tracking.type: bytetrack')
    if cfg['detection']['vehicle_conf_threshold'] > tracking_cfg.get('track_low_thresh', 0.1):
        raise ValueError('vehicle_conf_threshold must be <= track_low_thresh for ByteTrack')
    gui_available = any(
        line.strip().startswith('GUI:') and 'NONE' not in line
        for line in cv2.getBuildInformation().splitlines())
    if display is None:
        display = cfg['video'].get('display', 'auto')
    if display == 'auto':
        display = gui_available
    elif not isinstance(display, bool):
        raise ValueError("video.display must be true, false, or 'auto'")
    if display and not gui_available:
        raise RuntimeError("OpenCV has no GUI support; use --no-display or video.display: false")
    if not display:
        print("Headless mode: processing video without an OpenCV window.")
    def on_snapshot_saved(track_id, filepath):
        track_cache.mark_snapshot_saved(track_id)


    vehicle_detector = ObjectDetector(
        model_path=_project_path(config_path, cfg['models']['vehicle_detector']),
        conf_threshold=cfg['detection']['vehicle_conf_threshold'])

    plate_detector = PlateDetector(
        model_path=_project_path(config_path, cfg['models']['plate_detector']),
        conf_threshold=cfg['detection']['plate_conf_threshold'],
        min_ratio=cfg['detection']['plate_min_ratio'],
        max_ratio=cfg['detection']['plate_max_ratio'])
    
    tracker = ObjectTracker(
        track_high_thresh=tracking_cfg.get('track_high_thresh', 0.25),
        track_low_thresh=tracking_cfg.get('track_low_thresh', 0.1),
        new_track_thresh=tracking_cfg.get('new_track_thresh', 0.25),
        track_buffer=tracking_cfg.get('track_buffer', 30),
        match_thresh=tracking_cfg.get('match_thresh', 0.8),
        fuse_score=tracking_cfg.get('fuse_score', True))
    
    person_tracker = ObjectTracker(
        track_high_thresh=tracking_cfg.get('track_high_thresh', 0.25),
        track_low_thresh=tracking_cfg.get('track_low_thresh', 0.1),
        new_track_thresh=tracking_cfg.get('new_track_thresh', 0.25),
        track_buffer=tracking_cfg.get('track_buffer', 30),
        match_thresh=tracking_cfg.get('match_thresh', 0.8),
        fuse_score=tracking_cfg.get('fuse_score', True))
    
    ocr_engine = PlateOCR(
        onnx_model_path=_project_path(config_path, cfg['ocr']['onnx_model_path']),
        plate_config_path=_project_path(config_path, cfg['ocr']['plate_config_path']),
        min_confidence=cfg['ocr']['min_confidence'],
        max_batch_cap=cfg['ocr']['max_batch_cap'],
        use_gpu=cfg['ocr'].get('use_gpu', True),
        )
    track_cache = TrackCache(
            min_frames_to_confirm=cfg['cache']['min_frames_to_confirm'],
            recheck_interval_found=cfg['cache']['recheck_interval_found'],
            recheck_interval_missing=cfg['cache']['recheck_interval_missing']
        )
    snapshot_writer = SnapshotSaver(
        output_dir = cfg['snapshot']['output_dir'],
        on_saved = on_snapshot_saved
    )
    frame_queue = queue.Queue(maxsize=int(cfg['queue']['frame_queue_size']))
    result_queue = queue.Queue(maxsize=int(cfg['queue']['result_queue_size']))

    reader = FrameReader(
        video_path=_project_path(config_path, cfg['video']['source']),
        frame_queue=frame_queue,
        fallback_fps=cfg['video']['fallback_fps'])
    
    
    worker = InferenceWorker(
        frame_queue=frame_queue,
        result_queue=result_queue,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        tracker=tracker,
        ocr_engine=ocr_engine,
        track_cache=track_cache,
        snapshot_writer=snapshot_writer,
        person_tracker=person_tracker)
    
    output_path = _project_path(
        config_path, cfg['video'].get('output', 'output/anpr_output.mp4'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_path = os.path.splitext(output_path)[0] + ".log"

    writer = None
    processed_frames = 0
    reached_end = False
    log_file = None
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        snapshot_writer.start()
        reader.start()
        worker.start()
        while True:
            try:
                item = result_queue.get(timeout=0.5)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if item is None:
                reached_end = True
                break

            frame_nmr, frame, frame_results = item
            processed_frames += 1
            _write_frame_log(log_file, frame_nmr, frame_results)
            frame = draw_results(frame, frame_results)
            if writer is None:
                height, width = frame.shape[:2]
                writer = TimelineVideoWriter(
                    output_path,
                    reader.fps,
                    (width, height),
                )
            writer.write(frame_nmr, frame)
            if display:
                cv2.imshow('ANPR - Real-time simulation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if worker.error is not None:
            raise RuntimeError("Inference worker failed") from worker.error
        if reader.error is not None:
            raise RuntimeError("Video reader failed") from reader.error
        if writer is None:
            raise RuntimeError("No video frames were processed")
        if reached_end:
            writer.finish(reader.frames_read)
    finally:
        reader.stop()
        worker.stop()
        if reader.ident is not None:
            reader.join(timeout=10)
        else:
            reader.cap.release()
        if worker.ident is not None:
            worker.join(timeout=10)
        snapshot_writer.stop()
        if snapshot_writer.ident is not None:
            snapshot_writer.join(timeout=10)
        if writer is not None:
            writer.release()
        if log_file is not None:
            if not worker.is_alive():
                _write_ocr_summary(log_file, worker)
            log_file.write(
                f'snapshot_summary | saved={snapshot_writer.saved_count} '
                f'failed={snapshot_writer.failed_count} dropped={snapshot_writer.dropped_count} '
                f'callback_errors={snapshot_writer.callback_error_count}\n')
            log_file.close()
        if display:
            cv2.destroyAllWindows()
    if worker.error is not None:
        raise RuntimeError("Inference worker failed") from worker.error
    if reader.is_alive() or worker.is_alive() or snapshot_writer.is_alive():
        raise RuntimeError("Processing threads did not stop within 10 seconds")

    return {"processed_frames": processed_frames, "source_frames": reader.frames_read,
            "completed": reached_end, "output": output_path, "log": log_path,
            "ocr_calls": worker.ocr_call_count, "ocr_crops": worker.ocr_crop_count,
            "ocr_total_ms": worker.ocr_total_ms,
            "snapshot_saved": snapshot_writer.saved_count,
            "snapshot_failed": snapshot_writer.failed_count,
            "snapshot_dropped": snapshot_writer.dropped_count,
            "snapshot_callback_errors": snapshot_writer.callback_error_count}
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ANPR on a video")
    parser.add_argument('--config', default=r"D:\license_plate_data\config\config.yaml")
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    main(args.config, display=False if args.no_display else None)
