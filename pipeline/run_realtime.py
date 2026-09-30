import os       
import queue

import cv2
import yaml


from detector.detection import VehicleDetection
from detector.plate_detector import PlateDetector
from tracking.deepsort_tracker import VehicleTracker
from ocr.paddle_ocr_engine import PlateOcr
from tracking.track_cache import TrackCache
from utils.visualization import draw_results
from capture.frame_reader import FrameReader
from pipeline.inference_worker import InferenceWorker


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
    timing_text = (
        f"xe {timing.get('xe', 0):.0f} ms | "
        f"bien {timing.get('bien', 0):.0f} ms | "
        f"track {timing.get('track', 0):.0f} ms | "
        f"ocr {timing.get('ocr', 0):.0f} ms | "
        f"tong {timing.get('tong', 0):.0f} ms"
    )
    parts = []
    for track_id, row in frame_results.items():
        if track_id in ("__plates__", "__timing__"):
            continue
        shown = row.get("plate_text") or "-"
        ocr_text = row.get("ocr_text") or "-"
        confidence = row.get("confidence")
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else "-"
        parts.append(
            f"track {track_id} hien={shown} ocr={ocr_text} "
            f"conf={confidence_text} valid={row.get('valid')} source={row.get('source')}"
        )
    detail = " | ".join(parts) if parts else "khong co xe"
    log_file.write(f"frame {frame_nmr} | {timing_text} | {detail}\n")
    log_file.flush()



def main():
    config_path = r"D:\license_plate_data\config\config.yaml"
    cfg = load_config(config_path)

    vehicle_detector = VehicleDetection(
        model_path=_project_path(config_path, cfg['models']['vehicle_detector']),
        conf_threshold=cfg['detection']['vehicle_conf_threshold'])

    plate_detector = PlateDetector(
        model_path=_project_path(config_path, cfg['models']['plate_detector']),
        conf_threshold=cfg['detection']['plate_conf_threshold'],
        min_ratio=cfg['detection']['plate_min_ratio'],
        max_ratio=cfg['detection']['plate_max_ratio'])
    
    tracker = VehicleTracker(
        max_age=cfg['tracking']['max_age'],
        n_init=cfg['tracking']['n_init'],
        embedder=cfg['tracking']['embedder'])
    
    ocr_engine = PlateOcr(min_confidence=cfg['ocr']['min_confidence'],
                    batch_size=cfg['ocr']['batch_size'],
                    use_gpu=cfg['ocr']['use_gpu'])
    
    track_cache = TrackCache(
            min_frames_to_confirm=cfg['cache']['min_frames_to_confirm'],
            recheck_interval_found=cfg['cache']['recheck_interval_found'],
            recheck_interval_missing=cfg['cache']['recheck_interval_missing']
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
        track_cache=track_cache)
    
    output_path = _project_path(
        config_path, cfg['video'].get('output', 'output/anpr_output.mp4'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_path = os.path.splitext(output_path)[0] + ".log"

    reader.start()
    worker.start()

    results = {}
    writer = None
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        while True:
            item = result_queue.get()
            if item is None:
                print("Stopping...")
                break

            frame_nmr, frame, frame_results = item
            results[frame_nmr] = frame_results
            _write_frame_log(log_file, frame_nmr, frame_results)
            frame = draw_results(frame, frame_results)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    reader.fps,
                    (width, height),
                )
            writer.write(frame)
            cv2.imshow('ANPR - Real-time simulation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        reader.stop()
        worker.stop()
        reader.join(timeout=2)
        worker.join(timeout=2)
        if writer is not None:
            writer.release()
        log_file.close()
        cv2.destroyAllWindows()
    print(f"Đã lưu video: {output_path}")
    print(f"Đã lưu log: {log_path}")
    print("Exiting...")
    
if __name__ == "__main__":
    main()
